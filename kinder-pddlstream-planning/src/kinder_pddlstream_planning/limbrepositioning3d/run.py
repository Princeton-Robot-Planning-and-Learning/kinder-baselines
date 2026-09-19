"""Solve KinDER's LimbRepositioning3D environments with PDDLStream.

Every plan is a move_base, grasp, move_limb skeleton filled in by stream.py.

Only `move_limb` goes through forward dynamics, since the action space is arm torques.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
from kinder.envs.dynamic3d.limbrepositioning3d import (
    ALL_VARIANTS,
    LimbRepositioning3DEnvConfig,
    ObjectCentricLimbRepositioning3DEnv,
    create_variant_config,
)
from kinder.envs.dynamic3d.limbs import (
    DEFAULT_RANGE_OF_MOTION,
    MOTION_NAMES,
    BodyMass,
)
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read
from pybullet_helpers.geometry import SE2Pose

from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    ArmTrajectory,
    LimbConf,
    LimbGrasp,
    LimbPath,
    LimbStreamContext,
    MPCConfig,
    TorqueTrajectory,
    check_human_joint_limits,
    check_human_torque_limits,
    check_robot_torque_limits,
    plan_base_motion,
    plan_grasp_motion,
    plan_limb_motion,
    plan_limb_motion_along,
    sample_base_pose,
    sample_grasp,
    sample_limb_path,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    DEFAULT_GRAVITY,
    DEFAULT_LIMB_JOINT_DAMPING,
    DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT,
    ROBOT_TORQUE_LIMITS,
    CoupledState,
    StreamLog,
    advance_corrected,
    apply_limb_joint_damping,
    engage_grasp,
    extend_with_fingers,
    is_grasping,
    release_grasp,
)
from kinder_pddlstream_planning.rendering import (
    DEFAULT_GIF_DIR,
    gif_output_path,
    render_frame,
    save_gif,
)

_HERE = Path(__file__).parent
DOMAIN_PDDL = read(str(_HERE / "domain.pddl"))
STREAM_PDDL = read(str(_HERE / "stream.pddl"))
OBJECT_FIRST_STREAM_PDDL = read(str(_HERE / "stream_object_first.pddl"))

# How far behind its grasp placement the base starts, in meters.
DEFAULT_START_STANDOFF = 1.0

LIMB_NAME = "limb"

# The limb's muscle tone model: "none" for a limp limb, "spring" for a spring-damper one.
DEFAULT_MUSCLE_TONE = "spring"

# The limb's joint limits model: "none" or "box".
DEFAULT_JOINT_LIMITS_MODEL = "box"


@dataclasses.dataclass
class RunResult:
    """What one (variant, settings, seed) run produced, for a sweep's CSV."""

    variant: str
    seed: int
    plan_found: bool = False
    goal_reached: bool = False
    plan_time: float = 0.0
    total_time: float = 0.0
    failure_reason: str = ""


# Variants whose shipped placement the defaults cannot solve
_VARIANT_OVERRIDES: dict[str, dict[str, Any]] = {
    "wheelchair-left-leg": {"check_base_collisions": False},
    "wheelchair-right-leg": {"check_base_collisions": False},
    "bed-left-leg": {"check_base_collisions": False},
    "bed-right-leg": {"check_base_collisions": False},
}


def variant_kwargs(variant: str, **kwargs: Any) -> dict[str, Any]:
    """Overlay the overrides `variant` needs onto the sweep's shared settings.

    The overrides win, since a sweep passes one set of settings to all sixteen.
    """
    return {**kwargs, **_VARIANT_OVERRIDES.get(variant, {})}


def start_base_pose(
    grasp_base_pose: SE2Pose, standoff: float = DEFAULT_START_STANDOFF
) -> SE2Pose:
    """A start pose `standoff` meters behind `grasp_base_pose`, same heading."""
    return SE2Pose(
        grasp_base_pose.x - standoff * np.cos(grasp_base_pose.rot),
        grasp_base_pose.y - standoff * np.sin(grasp_base_pose.rot),
        grasp_base_pose.rot,
    )


def create_env(
    variant: str,
    standoff: float = DEFAULT_START_STANDOFF,
    use_gui: bool = False,
    config: LimbRepositioning3DEnvConfig | None = None,
    robot_base_z: float | None = None,
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY,
    muscle_tone: str = DEFAULT_MUSCLE_TONE,
    joint_limits_model: str = DEFAULT_JOINT_LIMITS_MODEL,
    range_of_motion_scale: float = 1.0,
    goal_scale: float = 1.0,
    body_mass: BodyMass | None = None,
    robot_torque_limits: tuple[float, ...] = ROBOT_TORQUE_LIMITS,
    limb_joint_damping: float = DEFAULT_LIMB_JOINT_DAMPING,
) -> ObjectCentricLimbRepositioning3DEnv:
    """Build the environment with its base backed off to the start pose.

    The variant's own placement becomes the goal of `move_base`.

    The constructor's out-of-reach weld is released before the base drives away.

    The environment ships with gravity off, muscle tone off, and a +-1 N*m action
    space. The baseline turns gravity on, so the action space is widened to the Kinova's
    real limits: the shipped one cannot hold the arm up, let alone the limb.

    Segment masses are the environment's own: `body_mass` overrides the scene's
    `BodyMass`, and leaving it None keeps whatever the variant ships with.
    """
    if config is None:
        config = create_variant_config(variant)
    if robot_base_z is None:
        robot_base_z = config.robot_base_z
    goal = config.scene.limb_goal_joint_positions
    if goal_scale != 1.0:
        init = config.scene.limb_init_joint_positions
        goal = tuple(i + goal_scale * (g - i) for i, g in zip(init, goal))
    range_of_motion = config.scene.limb_range_of_motion
    if range_of_motion_scale != 1.0:
        range_of_motion = DEFAULT_RANGE_OF_MOTION.scaled(
            {name: range_of_motion_scale for name in MOTION_NAMES}
        )
    config = dataclasses.replace(
        config,
        robot_base_home_pose=start_base_pose(config.robot_base_home_pose, standoff),
        robot_base_z=robot_base_z,
        gravity=gravity,
        torque_lower_limits=tuple(-t for t in robot_torque_limits),
        torque_upper_limits=tuple(robot_torque_limits),
        scene=dataclasses.replace(
            config.scene,
            limb_muscle_tone_model_name=muscle_tone,
            limb_joint_limits_model_name=joint_limits_model,
            limb_range_of_motion=range_of_motion,
            limb_goal_joint_positions=goal,
            limb_body_mass=body_mass or config.scene.limb_body_mass,
        ),
    )
    sim = ObjectCentricLimbRepositioning3DEnv(
        variant=variant, config=config, use_gui=use_gui
    )
    apply_limb_joint_damping(sim, damping=limb_joint_damping)
    return sim


def build_stream_context(
    sim: ObjectCentricLimbRepositioning3DEnv,
    standoff: float = DEFAULT_START_STANDOFF,
    motion_seed: int = 0,
    mpc: MPCConfig | None = None,
    check_base_collisions: bool = True,
    check_robot_collisions: bool = True,
    check_base_furniture_collisions: bool = True,
    object_first: bool = False,
    limb_path_steps: int | None = None,
    filter_saturated_bases: bool = True,
    human_torque_limit: float | None = None,
    robot_induced_torque_limit: float = DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT,
) -> LimbStreamContext:
    """Precompute the context every stream shares.

    Split out of `create_problem` so tests can drive streams directly.
    """
    grasp_base_pose = create_variant_config(sim.variant).robot_base_home_pose
    return LimbStreamContext(
        sim=sim,
        start_base_pose=start_base_pose(grasp_base_pose, standoff),
        grasp_base_pose=grasp_base_pose,
        retract_joints=extend_with_fingers(list(sim.config.robot_initial_joints)),
        limb_name=LIMB_NAME,
        motion_seed=motion_seed,
        mpc=mpc or MPCConfig(),
        check_base_collisions=check_base_collisions,
        check_robot_collisions=check_robot_collisions,
        check_base_furniture_collisions=check_base_furniture_collisions,
        object_first=object_first,
        limb_path_steps=limb_path_steps,
        filter_saturated_bases=filter_saturated_bases,
        human_torque_limit=human_torque_limit,
        robot_induced_torque_limit=robot_induced_torque_limit,
    )


def _logged_gen_fn(ctx: LimbStreamContext, name: str, fn: Callable[..., Any]) -> Any:
    """Charge every `next()` PDDLStream pulls from `fn` to `name`, and log it."""

    def wrapped(*args: Any) -> Any:
        iterator = ctx.profile.wrap(name, iter(fn(*args)))
        while True:
            start = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                ctx.log.record(name, args, None, time.perf_counter() - start)
                return
            # A None item is a call that produced nothing but may be retried.
            ctx.log.record(name, args, item or (), time.perf_counter() - start)
            yield item

    return wrapped


def _logged_test(ctx: LimbStreamContext, name: str, fn: Callable[..., Any]) -> Any:
    """Charge and log each evaluation of the test `fn`."""

    def wrapped(*args: Any) -> Any:
        start = time.perf_counter()
        passed = False
        try:
            passed = bool(fn(*args))
            return passed
        finally:
            # A rejected check is time the search spent proving a trajectory unusable.
            seconds = time.perf_counter() - start
            ctx.profile.add(name, seconds, produced=passed)
            ctx.log.record(name, args, (), seconds, passed=passed)

    return wrapped


def _describe(obj: Any) -> Any:
    """A JSON-writable view of a stream object."""
    if isinstance(obj, LimbGrasp):
        return {"type": "grasp", "slide": obj.slide, "roll": obj.roll}
    if isinstance(obj, SE2Pose):
        return {"type": "base", "x": obj.x, "y": obj.y, "rot": obj.rot}
    if isinstance(obj, LimbConf):
        return {"type": "limb_conf", "positions": list(obj.positions)}
    if isinstance(obj, CoupledState):
        return {
            "type": "state",
            "base": [obj.base_pose.x, obj.base_pose.y, obj.base_pose.rot],
            "robot_positions": list(obj.robot_positions),
            "limb_positions": list(obj.limb_positions),
        }
    if isinstance(obj, ArmTrajectory):
        return {"type": "arm_trajectory", "waypoints": len(obj.joint_plan)}
    if isinstance(obj, LimbPath):
        return {"type": "limb_path", "steps": len(obj.waypoints) - 1}
    if isinstance(obj, TorqueTrajectory):
        return {"type": "torque_trajectory", "steps": len(obj.robot_torques)}
    if isinstance(obj, list):
        return {"type": "base_trajectory", "waypoints": len(obj)}
    return str(obj)


def write_stream_log(
    path: Path,
    log: StreamLog,
    result: RunResult,
    plan: list[tuple[str, tuple]] | None,
    profile: dict[str, Any],
) -> None:
    """Dump every stream call, the objects they exchanged, and the plan that used them."""
    plan_refs = (
        None
        if plan is None
        else [[name, [log.ref(arg) for arg in args]] for name, args in plan]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "result": dataclasses.asdict(result),
                "plan": plan_refs,
                "objects": [_describe(obj) for obj in log.objects],
                "calls": log.records,
                "profile": profile,
            }
        )
    )


def create_problem(ctx: LimbStreamContext) -> PDDLProblem:
    """Build a PDDLProblem for driving the limb to its goal configuration."""
    scene = ctx.sim.config.scene
    limb_init = LimbConf(tuple(scene.limb_init_joint_positions))
    limb_goal = LimbConf(tuple(scene.limb_goal_joint_positions))

    init: list[tuple] = [
        ("Limb", ctx.limb_name),
        ("BConf", ctx.start_base_pose),
        ("AtBConf", ctx.start_base_pose),
        ("CanMove",),
        ("HandEmpty",),
        ("Conf", limb_init),
        ("InitConf", ctx.limb_name, limb_init),
        ("Conf", limb_goal),
        ("GoalConf", ctx.limb_name, limb_goal),
    ]
    goal = ("AtConf", limb_goal)

    gen_fns: dict[str, Callable[..., Any]] = {
        "sample-grasp": sample_grasp,
        "sample-base-pose": sample_base_pose,
        "plan-grasp-motion": plan_grasp_motion,
        "plan-base-motion": plan_base_motion,
        "plan-limb-motion": plan_limb_motion,
    }
    if ctx.object_first:
        gen_fns["sample-limb-path"] = sample_limb_path
        gen_fns["plan-limb-motion"] = plan_limb_motion_along
    tests: dict[str, Callable[..., Any]] = {
        "check-human-joint-limits": check_human_joint_limits,
        "check-human-torque-limits": check_human_torque_limits,
        "check-robot-torque-limits": check_robot_torque_limits,
    }
    stream_map = {
        name: from_gen_fn(_logged_gen_fn(ctx, name, partial(fn, ctx)))
        for name, fn in gen_fns.items()
    } | {
        name: from_test(_logged_test(ctx, name, partial(fn, ctx)))
        for name, fn in tests.items()
    }

    stream_pddl = OBJECT_FIRST_STREAM_PDDL if ctx.object_first else STREAM_PDDL
    return PDDLProblem(DOMAIN_PDDL, {}, stream_pddl, stream_map, init, goal)


def plan_limbrepositioning3d(
    ctx: LimbStreamContext,
    max_time: float = 600.0,
    verbose: bool = False,
) -> list[tuple[str, tuple]] | None:
    """Solve for a move_base/grasp/move_limb plan, or None if none is found."""
    solution = solve(
        create_problem(ctx),
        algorithm="adaptive",
        unit_costs=True,
        max_time=max_time,
        verbose=verbose,
    )
    if verbose:
        print_solution(solution)
    plan, _, _ = solution
    return plan


def reset_to_start(ctx: LimbStreamContext) -> None:
    """Put the environment back where the plan assumes it starts.

    `reset()` alone restores neither the base pose nor the constructor's weld.
    """
    sim = ctx.sim
    if not is_grasping(sim):
        engage_grasp(sim)
    sim.reset()
    release_grasp(sim)
    sim.robot.set_base(ctx.start_base_pose)
    sim.robot.arm.set_joints(
        list(ctx.retract_joints), joint_velocities=[0.0] * len(ctx.retract_joints)
    )
    scene = sim.config.scene
    sim.limb.set_joints(
        list(scene.limb_init_joint_positions),
        joint_velocities=[0.0] * len(scene.limb_init_joint_positions),
    )
    ctx.refresh_collision_baseline()


def execute_plan(
    ctx: LimbStreamContext,
    plan: list[tuple[str, tuple]],
    frames: list | None = None,
    gif_every: int = 20,
) -> bool:
    """Execute a move_base/grasp/move_limb plan in the environment."""
    sim = ctx.sim

    def capture() -> None:
        if frames is not None:
            frames.append(render_frame(sim))

    for name, args in plan:
        if name == "move_base":
            _, base_plan, _ = args
            release_grasp(sim)
            for target_base in base_plan[1:]:
                sim.robot.set_base(target_base)
                capture()
        elif name == "grasp":
            _, _, base_conf, arm_trajectory, _ = args
            assert isinstance(arm_trajectory, ArmTrajectory)
            sim.robot.set_base(base_conf)
            for waypoint in arm_trajectory.joint_plan:
                sim.robot.arm.set_joints(list(waypoint))
                capture()

            engage_grasp(sim)
            capture()
        elif name == "move_limb":
            _, _, _, _, _, _, torque_trajectory, _ = args
            assert isinstance(torque_trajectory, TorqueTrajectory)
            for i, torque in enumerate(torque_trajectory.robot_torques):
                advance_corrected(sim, torque)
                if i % gif_every == 0:
                    capture()
            capture()
        else:
            raise ValueError(f"Unknown action: {name}")
    return sim.goal_reached()


def build_failed_attempt_plan(ctx: LimbStreamContext) -> list[tuple[str, tuple]] | None:
    """Assemble a plan-shaped replay of the closest rollout planning managed.

    Returns None when no motion was ever attempted, as for `wheelchair-*-arm`.
    """
    attempt = ctx.best_attempt
    if attempt is None:
        return None
    state = attempt.state
    base_plan = next(
        plan_base_motion(ctx, ctx.start_base_pose, state.base_pose), (None,)
    )[0]
    if base_plan is None:
        base_plan = [ctx.start_base_pose, state.base_pose]
    grasp = ctx.sim.scene.grasp_transform
    approach = state.approach or ArmTrajectory([list(state.robot_positions)])
    goal_conf = LimbConf(tuple(ctx.sim.config.scene.limb_goal_joint_positions))
    return [
        ("move_base", (ctx.start_base_pose, base_plan, state.base_pose)),
        ("grasp", (ctx.limb_name, grasp, state.base_pose, approach, state)),
        (
            "move_limb",
            (
                ctx.limb_name,
                grasp,
                state.base_pose,
                state,
                LimbConf(tuple(state.limb_positions)),
                goal_conf,
                TorqueTrajectory(
                    attempt.robot_torques, plan_seconds=attempt.plan_seconds
                ),
                state,
            ),
        ),
    ]


def no_plan_reason(ctx: LimbStreamContext) -> str:
    """Name what stopped a search that returned no plan."""
    if ctx.best_attempt is not None:
        return ctx.best_attempt.reason
    if ctx.base_rejections:
        cause, count = ctx.base_rejections.most_common(1)[0]
        total = sum(ctx.base_rejections.values())
        return f"no usable base pose: {cause} ({count}/{total} candidates)"
    return "no base pose was ever generated"


def solve_and_execute(
    variant: str = "wheelchair-left-arm",
    seed: int = 0,
    max_time: float = 600.0,
    standoff: float = DEFAULT_START_STANDOFF,
    use_gui: bool = False,
    verbose: bool = False,
    gif_path: str | Path | None = None,
    mpc: MPCConfig | None = None,
    check_base_collisions: bool = True,
    check_robot_collisions: bool = True,
    check_base_furniture_collisions: bool = True,
    object_first: bool = False,
    limb_path_steps: int | None = None,
    filter_saturated_bases: bool = True,
    human_torque_limit: float | None = None,
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY,
    muscle_tone: str = DEFAULT_MUSCLE_TONE,
    joint_limits_model: str = DEFAULT_JOINT_LIMITS_MODEL,
    range_of_motion_scale: float = 1.0,
    goal_scale: float = 1.0,
    limb_joint_damping: float = DEFAULT_LIMB_JOINT_DAMPING,
    robot_base_z: float | None = None,
    result: RunResult | None = None,
    log_path: str | Path | None = None,
) -> bool:
    """Build the variant, plan with PDDLStream, and execute the plan.

    Returns whether the limb ends up within `goal_atol` of its goal. A `result` is
    filled in with what the run took and measured, and `log_path` gets every stream call.
    """
    if result is None:
        result = RunResult(variant=variant, seed=seed)
    ctx: LimbStreamContext | None = None
    plan: list[tuple[str, tuple]] | None = None
    sim = create_env(
        variant,
        standoff=standoff,
        use_gui=use_gui,
        robot_base_z=robot_base_z,
        gravity=gravity,
        muscle_tone=muscle_tone,
        joint_limits_model=joint_limits_model,
        range_of_motion_scale=range_of_motion_scale,
        goal_scale=goal_scale,
        limb_joint_damping=limb_joint_damping,
    )
    try:
        sim.reset(seed=seed)
        ctx = build_stream_context(
            sim,
            standoff=standoff,
            motion_seed=seed,
            mpc=mpc,
            check_base_collisions=check_base_collisions,
            check_robot_collisions=check_robot_collisions,
            check_base_furniture_collisions=check_base_furniture_collisions,
            object_first=object_first,
            limb_path_steps=limb_path_steps,
            filter_saturated_bases=filter_saturated_bases,
            human_torque_limit=human_torque_limit,
        )
        reset_to_start(ctx)
        plan_start = time.time()
        try:
            plan = plan_limbrepositioning3d(ctx, max_time=max_time, verbose=verbose)
            result.plan_found = plan is not None
            if plan is None:
                result.failure_reason = no_plan_reason(ctx)
        except BaseException as exc:  # pylint: disable=broad-except
            # BaseException, so a wall-clock interrupt is named in the profile too.
            result.failure_reason = _exception_reason(exc)
            raise
        finally:
            result.plan_time = time.time() - plan_start

        reset_to_start(ctx)
        frames = [render_frame(sim)] if gif_path is not None else None
        try:
            if plan is None:
                # Replay the closest attempt, so the GIF shows what was tried.
                attempt_plan = build_failed_attempt_plan(ctx)
                if attempt_plan is None:
                    print(
                        f"No plan found, and no motion was ever attempted "
                        f"({result.failure_reason}); saving the start state only."
                    )
                    return False
                assert ctx.best_attempt is not None
                print(
                    f"No plan found; replaying the closest attempt "
                    f"({ctx.best_attempt.reason}) for the GIF."
                )
                execute_plan(ctx, attempt_plan, frames=frames)
                return False
            result.goal_reached = execute_plan(ctx, plan, frames=frames)
            if not result.goal_reached:
                result.failure_reason = "the executed plan missed the goal"
            return result.goal_reached
        finally:
            if frames is not None:
                assert gif_path is not None
                save_gif(gif_path, frames)
    finally:
        if log_path is not None and ctx is not None:
            write_stream_log(
                Path(log_path), ctx.log, result, plan, ctx.profile.as_dict()
            )
        sim.close()


def _exception_reason(exc: BaseException) -> str:
    """Name a crash by where it was raised, since a bare `assert` carries no message."""
    frame = traceback.extract_tb(exc.__traceback__)[-1]
    where = f"{Path(frame.filename).name}:{frame.lineno}"
    return f"{type(exc).__name__} at {where}: {exc}".rstrip(": ")


def solve_all_variants(
    gif_dir: str | Path | None = None,
    seed: int = 0,
    **kwargs,
) -> dict[str, RunResult]:
    """Run every variant in turn, returning {variant: result}.

    Exceptions are recorded as failures, so one variant cannot end the sweep.
    """
    results: dict[str, RunResult] = {}
    for index, variant in enumerate(ALL_VARIANTS, start=1):
        gif_path = (
            gif_output_path("limbrepositioning3d", variant, gif_dir)
            if gif_dir is not None
            else None
        )
        print(f"\n=== [{index}/{len(ALL_VARIANTS)}] {variant} ===", flush=True)
        start = time.time()
        result = RunResult(variant=variant, seed=seed)
        try:
            solve_and_execute(
                variant=variant,
                gif_path=gif_path,
                seed=seed,
                result=result,
                **variant_kwargs(variant, **kwargs),
            )
        except Exception as exc:  # pylint: disable=broad-except
            traceback.print_exc()
            result.failure_reason = _exception_reason(exc)
        result.total_time = time.time() - start
        results[variant] = result
        print(
            f"{variant}: {'SUCCESS' if result.goal_reached else 'FAIL'} "
            f"in {result.total_time:.0f}s",
            flush=True,
        )

    print("\n=== summary ===")
    width = max(len(v) for v in results)
    for variant, result in results.items():
        print(
            f"  {variant:{width}s}  "
            f"{'SUCCESS' if result.goal_reached else 'FAIL   '}  "
            f"{result.total_time:6.0f}s"
        )
    num_reached = sum(result.goal_reached for result in results.values())
    print(f"  {num_reached}/{len(results)} reached the goal")
    if gif_dir is not None:
        print(f"  GIFs written to {Path(gif_dir).resolve()}")
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        type=str,
        default="wheelchair-left-arm",
        choices=ALL_VARIANTS,
        help="Which of the sixteen scene/limb combinations to solve.",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help=("Run every variant in turn."),
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save a GIF of the rollout (default: off).",
    )
    parser.add_argument(
        "--gif-dir",
        type=Path,
        default=DEFAULT_GIF_DIR,
        help="Directory to write the GIF into (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument(
        "--standoff",
        type=float,
        default=DEFAULT_START_STANDOFF,
        help="How far behind its grasp placement the base starts, in meters.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=MPCConfig.num_rollouts,
        help="Predictive-sampling rollouts per MPC step.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=MPCConfig.horizon,
        help="Control steps in the MPC horizon.",
    )
    parser.add_argument(
        "--commit-steps",
        type=int,
        default=MPCConfig.commit_steps,
        help="Control steps of the winning horizon applied before re-planning.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=MPCConfig.noise_scale,
        help="Standard deviation of the torque noise sampled around the nominal.",
    )
    parser.add_argument(
        "--check-base-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Check collision between robot base and furniture/human."),
    )
    parser.add_argument(
        "--check-robot-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Check collision between robot arm and furniture/human"),
    )
    parser.add_argument(
        "--check-base-furniture-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check collision between robot base and furniture, in parking and driving.",
    )
    parser.add_argument(
        "--object-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample a limb path before the grasp and base, and pull MPC along it.",
    )
    parser.add_argument(
        "--limb-path-steps",
        type=int,
        default=None,
        help="Duration of the first limb path, in MPC control steps (object first only).",
    )
    parser.add_argument(
        "--filter-saturated-bases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Defer, and past a cap drop, base poses the arm cannot hold the limb at.",
    )
    parser.add_argument(
        "--human-torque-limit",
        type=float,
        default=None,
        help=(
            "Total torque, in N*m, a joint of the person may bear. "
            "Defaults to a per-limb value."
        ),
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=DEFAULT_GRAVITY[2],
        help="World z gravity in m/s^2. Pass 0 for the environment's own default.",
    )
    parser.add_argument(
        "--muscle-tone",
        type=str,
        default=DEFAULT_MUSCLE_TONE,
        choices=["none", "spring"],
        help="The limb's muscle tone model: limp, or a spring-damper.",
    )
    parser.add_argument(
        "--joint-limits-model",
        type=str,
        default=DEFAULT_JOINT_LIMITS_MODEL,
        choices=["none", "box"],
        help=(
            "The limb's joint limits model: unbounded, per-joint boxes, or the "
            "learned reachable region. The learned one covers arms only."
        ),
    )
    parser.add_argument(
        "--range-of-motion-scale",
        type=float,
        default=1.0,
        help=(
            "Factor on every range-of-motion magnitude, for a stiffer or looser "
            "person. 1.0 keeps the scene's own."
        ),
    )
    parser.add_argument(
        "--goal-scale",
        type=float,
        default=1.0,
        help=(
            "Interpolate the goal toward the start configuration: 1.0 keeps the "
            "variant's own goal, 0.5 asks for half the joint-space move."
        ),
    )
    parser.add_argument(
        "--limb-joint-damping",
        type=float,
        default=DEFAULT_LIMB_JOINT_DAMPING,
        help=(
            "Viscous damping at each of the limb's own joints, in N*m*s/rad. "
            "Pass 0 for the environment's own frictionless joints."
        ),
    )
    parser.add_argument(
        "--robot-base-z",
        type=float,
        default=None,
        help=(
            "World z in meters to place the robot at, overriding the variant's own value"
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Write every stream call of the run to this JSON file (default: off).",
    )
    parser.add_argument(
        "--use-gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the PyBullet GUI (default: off).",
    )
    args = parser.parse_args()
    gif_dir = args.gif_dir if args.save_gif else None
    shared = {
        "seed": args.seed,
        "max_time": args.max_time,
        "standoff": args.standoff,
        "use_gui": args.use_gui,
        "check_base_collisions": args.check_base_collisions,
        "check_robot_collisions": args.check_robot_collisions,
        "check_base_furniture_collisions": args.check_base_furniture_collisions,
        "object_first": args.object_first,
        "limb_path_steps": args.limb_path_steps,
        "filter_saturated_bases": args.filter_saturated_bases,
        "human_torque_limit": args.human_torque_limit,
        "gravity": (0.0, 0.0, args.gravity),
        "muscle_tone": args.muscle_tone,
        "joint_limits_model": args.joint_limits_model,
        "range_of_motion_scale": args.range_of_motion_scale,
        "goal_scale": args.goal_scale,
        "limb_joint_damping": args.limb_joint_damping,
        "robot_base_z": args.robot_base_z,
        "mpc": MPCConfig(
            num_rollouts=args.num_rollouts,
            horizon=args.horizon,
            noise_scale=args.noise_scale,
            commit_steps=args.commit_steps,
        ),
    }
    if args.all_variants:
        # Per-variant PDDLStream output would bury the summary table.
        solve_all_variants(gif_dir=gif_dir, verbose=False, **shared)
        return
    result = RunResult(variant=args.variant, seed=args.seed)
    try:
        success = solve_and_execute(
            variant=args.variant,
            gif_path=(
                gif_output_path("limbrepositioning3d", args.variant, gif_dir)
                if gif_dir is not None
                else None
            ),
            verbose=True,
            result=result,
            log_path=args.log_path,
            **shared,
        )
        print(f"Reached goal: {success}")
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        result.failure_reason = _exception_reason(exc)
        print(f"Run failed: {result.failure_reason}")


if __name__ == "__main__":
    main()
