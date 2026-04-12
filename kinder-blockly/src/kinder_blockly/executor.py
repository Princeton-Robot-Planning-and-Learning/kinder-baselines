"""Execute Blockly programs in KinDER environments."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import math

import kinder
import numpy as np
import pybullet as p
from kinder.envs.kinematic3d.base_motion3d import (
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

# Trail visual geometry settings.
TRAIL_HEIGHT = 0.005  # z-centre of the flat box (just above the floor)
TRAIL_HALF_WIDTH = 0.012  # half-width of the line
TRAIL_HALF_THICKNESS = 0.001  # half-height — paper-thin


# Trail segment returned to the frontend for the top-down canvas.
TrailSegment = dict[str, float]  # keys: x1 y1 x2 y2 r g b


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


def render_initial_frame(seed: int = 0) -> NDArray[np.uint8]:
    """Reset the environment and return the first rendered frame."""
    env = kinder.make(
        "kinder/BaseMotion3D-v0",
        render_mode="rgb_array",
        use_gui=False,
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
        yield frame

        for block in blocks:
            block_type = block["type"]

            if block_type == "set_pen_color":
                pen.color_rgb = (
                    int(block.get("r", 255)),
                    int(block.get("g", 0)),
                    int(block.get("b", 0)),
                )
                pen.down = True

            elif block_type == "pen_down":
                pen.down = True

            elif block_type == "pen_up":
                pen.down = False

            elif block_type == "move_base_to_target":
                target_x = float(block.get("x", 0.0))
                target_y = float(block.get("y", 0.0))
                state, frames = _run_move_base_to(
                    env, state, sim, target_x, target_y,
                    pen=pen, client_id=client_id,
                )
                yield from frames

        # Copy accumulated trail to the caller's list.
        if trail_out is not None:
            trail_out.extend(pen.trail)
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
        current_base_pose = state.base_pose
        delta = waypoint - current_base_pose
        action_lst = [delta.x, delta.y, delta.rot] + [0.0] * 8
        action = np.array(action_lst, dtype=np.float32)

        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)  # type: ignore[attr-defined]

        cur_xy = [state.base_pose.x, state.base_pose.y]

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
