"""Execute Blockly programs in KinDER environments."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import math
import threading

import kinder
import numpy as np
import pybullet as p
from kinder.envs.kinematic3d.base_motion3d import (
    BaseMotion3DEnvConfig,
    BaseMotion3DObjectCentricState,
    ObjectCentricBaseMotion3DEnv,
)
from numpy.typing import NDArray
from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.motion_planning import (
    run_single_arm_mobile_base_motion_planning,
)

kinder.register_all_environments()

MAX_STEPS = 500
FRAME_SKIP = 5

# Hide the target ball by making it fully transparent.
_ENV_CONFIG = BaseMotion3DEnvConfig(target_color=(0.0, 0.0, 0.0, 0.0))

# Trail visual geometry settings.
TRAIL_HEIGHT = 0.005  # z-centre of the flat box (just above the floor)
TRAIL_HALF_WIDTH = 0.030  # half-width of the line
TRAIL_HALF_THICKNESS = 0.001  # half-height — paper-thin


# Trail segment returned to the frontend for the top-down canvas.
TrailSegment = dict[str, float]  # keys: x1 y1 x2 y2 r g b

# Pen-up / pen-down event for the physics marker overlay.
PenEvent = dict[str, Any]  # keys: x y type('up'|'down') r g b

# Per-frame action label shown as an overlay in the 3-D view.
# None for frames with no associated block (e.g. the initial reset frame).
FrameLabel = dict[str, Any] | None  # keys: text str, r int, g int, b int


def _add_trail_box(
    x1: float, y1: float, x2: float, y2: float,
    color_01: tuple[float, float, float],
    client_id: int,
) -> None:
    """Place a thin visual-only box on the floor between two points."""
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    angle = math.atan2(dy, dx)

    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[length / 2.0, TRAIL_HALF_WIDTH, TRAIL_HALF_THICKNESS],
        rgbaColor=[*color_01, 1.0],
        physicsClientId=client_id,
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=vis,
        basePosition=[cx, cy, TRAIL_HEIGHT],
        baseOrientation=p.getQuaternionFromEuler([0, 0, angle]),
        physicsClientId=client_id,
    )


@dataclass
class _PenState:
    """Mutable pen state threaded through block execution."""

    down: bool = False
    color_rgb: tuple[int, int, int] = (255, 0, 0)
    prev_xy: list[float] | None = None
    trail: list[TrailSegment] = field(default_factory=list)
    events: list[PenEvent] = field(default_factory=list)

    def record_event(self, event_type: str) -> None:
        """Append a pen-up or pen-down event at the current position."""
        if self.prev_xy is None:
            return
        r, g, b = self.color_rgb
        self.events.append({
            "x": float(self.prev_xy[0]), "y": float(self.prev_xy[1]),
            "type": event_type, "r": r, "g": g, "b": b,
        })

    @property
    def color_01(self) -> tuple[float, float, float]:
        """Pen colour normalised to [0, 1] for pybullet."""
        r, g, b = self.color_rgb
        return (r / 255.0, g / 255.0, b / 255.0)


def _get_physics_client_id(env: Any) -> int | None:
    """Try to extract the pybullet physics client id from the env."""
    unwrapped = env.unwrapped
    for attr in ("_object_centric_env", "_env"):
        inner = getattr(unwrapped, attr, None)
        if inner is not None:
            robot = getattr(inner, "robot", None)
            if robot is not None and hasattr(robot, "physics_client_id"):
                return robot.physics_client_id  # type: ignore[no-any-return]
    robot = getattr(unwrapped, "robot", None)
    if robot is not None and hasattr(robot, "physics_client_id"):
        return robot.physics_client_id  # type: ignore[no-any-return]
    return None


def validate_program(program: dict[str, Any]) -> dict[str, Any]:
    """Abstract validator — catches OOB and infinite loops without physics.

    Returns {} on success, {"error": str} for OOB, or {"infinite_loop": True}.
    Runs in microseconds by tracking position symbolically.
    """
    blocks: list[dict[str, Any]] = program.get("blocks", [])
    if not blocks:
        return {}

    _OP_FNS: dict[str, Callable[[float, float], bool]] = {
        ">":  lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "=":  lambda a, b: abs(a - b) < 0.05,
        "<":  lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }

    # Robot position in sim coords (robot_x, robot_y); starts at (0, 0).
    # UI X = robot_y, UI Y = -robot_x (same convention as the real executor).
    pos = [0.0, 0.0]

    def _oob_error(rx: float, ry: float) -> str | None:
        if abs(rx) > 2.0 or abs(ry) > 2.0:
            return (
                "WOAH WOAH WOAH. That move goes off the edge of my world! "
                "I only exist between -2 and 2 on both axes — "
                "please keep me on the grid or I will cease to exist!!"
            )
        return None

    def _run(blks: list[dict[str, Any]], depth: int = 0) -> dict[str, Any] | None:
        if depth > 20:
            return None
        for blk in blks:
            btype    = blk["type"]
            block_id = blk.get("blockId")

            if btype == "move_base_to_target":
                rx = -float(blk.get("y", 0.0))
                ry =  float(blk.get("x", 0.0))
                err = _oob_error(rx, ry)
                if err:
                    return {"error": err, "error_block_id": block_id}
                pos[0], pos[1] = rx, ry

            elif btype == "move_base_by":
                # Real executor clips; abstract sim mirrors that.
                pos[0] = max(-2.0, min(2.0, pos[0] - float(blk.get("dy", 0.0))))
                pos[1] = max(-2.0, min(2.0, pos[1] + float(blk.get("dx", 0.0))))

            elif btype == "repeat_while":
                var       = blk.get("var", "X")
                op        = blk.get("op", ">")
                threshold = float(blk.get("threshold", 0.0))
                body      = blk.get("body", [])
                _default: Callable[[float, float], bool] = lambda a, b: False
                op_fn = _OP_FNS.get(op, _default)
                for _ in range(100):
                    cur: float
                    if var == "X":
                        cur = pos[1]        # UI X = robot_y
                    elif var == "Y":
                        cur = -pos[0]       # UI Y = -robot_x
                    else:
                        try:
                            cur = float(var)
                        except (ValueError, TypeError):
                            cur = 0.0
                    if not op_fn(cur, threshold):
                        break
                    result = _run(body, depth + 1)
                    if result:
                        return result
                else:
                    return {"infinite_loop": True, "error_block_id": block_id}

        return None

    return _run(blocks) or {}


def render_initial_frame(seed: int = 0) -> NDArray[np.uint8]:
    """Reset the environment and return the first rendered frame."""
    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
        config=_ENV_CONFIG,
    )
    try:
        env.reset(seed=seed)
        frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
        return frame
    finally:
        env.close()  # type: ignore[no-untyped-call]


def execute_program(
    program: dict[str, Any],
    seed: int = 0,
    trail_out: list[TrailSegment] | None = None,
    pen_events_out: list[PenEvent] | None = None,
    frame_labels_out: list[FrameLabel] | None = None,
    infinite_loop_out: list[bool] | None = None,
    stop_event: threading.Event | None = None,
) -> Iterator[NDArray[np.uint8]]:
    """Execute a Blockly program and yield rendered frames.

    Yields an initial frame after reset, then intermediate frames during skill
    execution.

    If *trail_out* is provided it is populated (in-place) with the line
    segments drawn by the pen so the caller can forward them to the frontend.
    """
    blocks: list[dict[str, Any]] = program.get("blocks", [])
    if not blocks:
        return

    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
        config=_ENV_CONFIG,
    )
    try:
        obs, _ = env.reset(seed=seed)
        state = env.observation_space.devectorize(obs)  # type: ignore[attr-defined]

        sim = ObjectCentricBaseMotion3DEnv(allow_state_access=True)

        # Resolve pybullet client for 3-D debug lines.
        client_id = _get_physics_client_id(env)

        # Pen starts UP — students must use set_pen_color or pen_down first.
        pen = _PenState()

        # Seed the previous position from the initial state.
        assert isinstance(state, BaseMotion3DObjectCentricState)
        pen.prev_xy = [state.base_pose.x, state.base_pose.y]

        frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
        if frame_labels_out is not None:
            frame_labels_out.append(None)
        yield frame

        state_box = [state]

        _OP_FNS = {
            ">":  lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "=":  lambda a, b: abs(a - b) < 0.05,
            "<":  lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
        }

        def run_blocks(  # pylint: disable=too-many-branches
            blks: list[dict[str, Any]], depth: int = 0,
        ) -> Iterator[NDArray[np.uint8]]:
            if depth > 20:
                return
            for blk in blks:
                if stop_event is not None and stop_event.is_set():
                    return
                btype = blk["type"]

                if btype == "set_pen_color":
                    pen.color_rgb = (
                        int(blk.get("r", 255)),
                        int(blk.get("g", 0)),
                        int(blk.get("b", 0)),
                    )

                elif btype == "pen_down":
                    was_down = pen.down
                    pen.down = True
                    if not was_down:
                        pen.record_event("down")

                elif btype == "pen_up":
                    was_down = pen.down
                    pen.down = False
                    if was_down:
                        pen.record_event("up")

                elif btype == "move_base_to_target":
                    target_x = -float(blk.get("y", 0.0))
                    target_y = float(blk.get("x", 0.0))
                    ui_x = float(blk.get("x", 0.0))
                    ui_y = float(blk.get("y", 0.0))
                    lbl: FrameLabel = {
                        "text": f"Move to ({ui_x:.1f}, {ui_y:.1f})",
                        "r": 116, "g": 91, "b": 166,
                    }
                    new_state, frames = _run_move_base_to(
                        env, state_box[0], sim, target_x, target_y,
                        pen=pen, client_id=client_id, stop_event=stop_event,
                    )
                    state_box[0] = new_state
                    for f in frames:
                        if frame_labels_out is not None:
                            frame_labels_out.append(lbl)
                        yield f

                elif btype == "move_base_by":
                    s = state_box[0]
                    assert isinstance(s, BaseMotion3DObjectCentricState)
                    robot_dx = -float(blk.get("dy", 0.0))
                    robot_dy = float(blk.get("dx", 0.0))
                    target_x = float(np.clip(s.base_pose.x + robot_dx, -2.0, 2.0))
                    target_y = float(np.clip(s.base_pose.y + robot_dy, -2.0, 2.0))
                    ui_dx = float(blk.get("dx", 0.0))
                    ui_dy = float(blk.get("dy", 0.0))
                    by_lbl: FrameLabel = {
                        "text": f"Move by ({ui_dx:+.1f}, {ui_dy:+.1f})",
                        "r": 168, "g": 143, "b": 224,
                    }
                    new_state, frames = _run_move_base_to(
                        env, state_box[0], sim, target_x, target_y,
                        pen=pen, client_id=client_id, stop_event=stop_event,
                    )
                    state_box[0] = new_state
                    for f in frames:
                        if frame_labels_out is not None:
                            frame_labels_out.append(by_lbl)
                        yield f

                elif btype == "repeat_while":
                    var = blk.get("var", "X")
                    op  = blk.get("op", ">")
                    threshold = float(blk.get("threshold", 0.0))
                    body = blk.get("body", [])
                    _default: Callable[[float, float], bool] = lambda a, b: False
                    op_fn = _OP_FNS.get(op, _default)
                    for _ in range(100):
                        s = state_box[0]
                        assert isinstance(s, BaseMotion3DObjectCentricState)
                        # UI X = robot Y; UI Y = -robot X; else: resolved param
                        if var == "X":
                            cur = float(s.base_pose.y)
                        elif var == "Y":
                            cur = -float(s.base_pose.x)
                        else:
                            try:
                                cur = float(var)
                            except (ValueError, TypeError):
                                cur = 0.0
                        if not op_fn(cur, threshold):
                            break
                        yield from run_blocks(body, depth + 1)
                    else:
                        if infinite_loop_out is not None:
                            infinite_loop_out[0] = True

        yield from run_blocks(blocks)

        # If program ends with pen still down, record implicit pen-up so ○ appears.
        if pen.down:
            pen.record_event("up")

        # Copy accumulated trail and events to the caller's lists.
        if trail_out is not None:
            trail_out.extend(pen.trail)
        if pen_events_out is not None:
            pen_events_out.extend(pen.events)
    finally:
        env.close()  # type: ignore[no-untyped-call]


def _run_move_base_to(
    env: Any,
    state: Any,
    sim: ObjectCentricBaseMotion3DEnv,
    target_x: float,
    target_y: float,
    pen: _PenState,
    client_id: int | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[Any, list[NDArray[np.uint8]]]:
    """Move the robot base to (target_x, target_y).

    Draws trail segments when the pen is down.
    """
    assert isinstance(state, BaseMotion3DObjectCentricState)
    sim.set_state(state)  # type: ignore[no-untyped-call]

    goal_pose = SE2Pose(target_x, target_y, 0.0)
    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        goal_pose,
        collision_bodies=set(),
        seed=0,
    )
    if base_plan is None:
        raise RuntimeError(f"Motion planning to ({target_x}, {target_y}) failed")

    plan = base_plan[1:]
    frames: list[NDArray[np.uint8]] = []

    for step_i, waypoint in enumerate(plan):
        if stop_event is not None and stop_event.is_set():
            break
        current_base_pose = state.base_pose
        delta = waypoint - current_base_pose
        action_lst = [delta.x, delta.y, delta.rot] + [0.0] * 8
        action = np.array(action_lst, dtype=np.float32)

        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)  # type: ignore[attr-defined]

        cur_xy = [state.base_pose.x, state.base_pose.y]

        if abs(cur_xy[0]) > 2.0 or abs(cur_xy[1]) > 2.0:
            raise RuntimeError(
                "WOAH WOAH WOAH. That move goes off the edge of my world! "
                "I only exist between -2 and 2 on both axes — "
                "please keep me on the grid or I will cease to exist!!"
            )

        if pen.down and pen.prev_xy is not None:
            r, g, b = pen.color_rgb

            # Record for the top-down canvas (plain floats for JSON).
            pen.trail.append({
                "x1": float(pen.prev_xy[0]), "y1": float(pen.prev_xy[1]),
                "x2": float(cur_xy[0]),      "y2": float(cur_xy[1]),
                "r": r, "g": g, "b": b,
            })

            # Place a thin flat box on the floor so it shows in renders.
            if client_id is not None:
                _add_trail_box(
                    pen.prev_xy[0], pen.prev_xy[1],
                    cur_xy[0], cur_xy[1],
                    pen.color_01, client_id,
                )

        pen.prev_xy = cur_xy

        if step_i % FRAME_SKIP == 0 or step_i == len(plan) - 1:
            frame: NDArray[np.uint8] = env.render()  # type: ignore[assignment]
            frames.append(frame)

    return state, frames
