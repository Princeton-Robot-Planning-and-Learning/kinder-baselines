"""Solve KinDER's LimbRepositioning3D environments with PDDLStream.

Every plan is a move_base, grasp, move_limb skeleton filled in by stream.py.

Only `move_limb` goes through forward dynamics, since the action space is arm torques.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

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

from kinder_pddlstream_planning.limbrepositioning3d.plots import plot_rollout
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    ArmTrajectory,
    LimbConf,
    LimbStreamContext,
    MPCConfig,
    TorqueTrajectory,
    check_human_joint_limits,
    check_human_torque_limits,
    check_robot_torque_limits,
    plan_base_motion,
    plan_grasp_motion,
    plan_limb_motion,
    sample_base_pose,
    sample_grasp,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    DEFAULT_GRAVITY,
    DEFAULT_LIMB_JOINT_DAMPING,
    DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT,
    ROBOT_TORQUE_LIMITS,
    RolloutLog,
    _limb_error,
    advance_corrected,
    advance_logged,
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
    num_torque_steps: int = 0
    max_human_torque: float = 0.0
    max_robot_induced_torque: float = 0.0
    failure_reason: str = ""


# Variants whose shipped placement the defaults cannot solve
_VARIANT_OVERRIDES: dict[str, dict[str, Any]] = {
    "wheelchair-left-arm": {"robot_base_z": 0.4},
    "wheelchair-right-arm": {"robot_base_z": 0.55},
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
        human_torque_limit=human_torque_limit,
        robot_induced_torque_limit=robot_induced_torque_limit,
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

    stream_map = {
        "sample-grasp": from_gen_fn(partial(sample_grasp, ctx)),
        "sample-base-pose": from_gen_fn(partial(sample_base_pose, ctx)),
        "plan-grasp-motion": from_gen_fn(partial(plan_grasp_motion, ctx)),
        "plan-base-motion": from_gen_fn(partial(plan_base_motion, ctx)),
        "plan-limb-motion": from_gen_fn(partial(plan_limb_motion, ctx)),
        "check-human-joint-limits": from_test(partial(check_human_joint_limits, ctx)),
        "check-human-torque-limits": from_test(partial(check_human_torque_limits, ctx)),
        "check-robot-torque-limits": from_test(partial(check_robot_torque_limits, ctx)),
    }

    return PDDLProblem(DOMAIN_PDDL, {}, STREAM_PDDL, stream_map, init, goal)


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
    log: RolloutLog | None = None,
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
                if log is None:
                    advance_corrected(sim, torque)
                else:
                    advance_logged(sim, torque, log)
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
                TorqueTrajectory(attempt.robot_torques),
                state,
            ),
        ),
    ]


def record_trajectory_metrics(result: RunResult, plan: list[tuple[str, tuple]]) -> None:
    """Fill in what the plan's `move_limb` trajectory cost the person."""
    for name, args in plan:
        if name != "move_limb":
            continue
        trajectory = args[6]
        assert isinstance(trajectory, TorqueTrajectory)
        result.num_torque_steps = len(trajectory.robot_torques)
        if trajectory.human_torques:
            result.max_human_torque = float(np.abs(trajectory.human_torques).max())
        if trajectory.robot_induced_torques:
            result.max_robot_induced_torque = float(
                np.abs(trajectory.robot_induced_torques).max()
            )


def save_rollout(path: str | Path, ctx: LimbStreamContext, log: RolloutLog) -> None:
    """Write one run's per-step measurements, with the limits they are read against."""
    sim = ctx.sim
    goal = np.asarray(sim.config.scene.limb_goal_joint_positions, dtype=np.float64)
    total_limit, robot_limit = ctx.human_torque_limits
    robot_lower, robot_upper = ctx.robot_torque_limits
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        variant=sim.variant,
        dt=sim.config.dt,
        goal_atol=ctx.goal_atol,
        limb_goal=goal,
        limb_joint_names=[info.jointName for info in ctx.limb_joint_infos],
        limb_lower_limits=np.asarray(sim.limb.joint_lower_limits),
        limb_upper_limits=np.asarray(sim.limb.joint_upper_limits),
        human_torque_limit=total_limit,
        robot_induced_torque_limit=robot_limit,
        robot_torque_lower=robot_lower,
        robot_torque_upper=robot_upper,
        goal_error=[_limb_error(ctx, q, goal) for q in log.limb_positions],
        correction_torques=np.asarray(log.correction_torques),
        commanded_torques=np.asarray(log.commanded_torques),
        limb_positions=np.asarray(log.limb_positions),
        limb_velocities=np.asarray(log.limb_velocities),
        human_total=np.asarray(log.human_total),
        human_gravity=np.asarray(log.human_gravity),
        human_tone=np.asarray(log.human_tone),
        human_robot_induced=np.asarray(log.human_robot_induced),
        grasp_wrenches=np.asarray(log.grasp_wrenches),
    )
    print(f"Rollout written to {path.resolve()}")


def solve_and_execute(
    variant: str = "wheelchair-left-arm",
    seed: int = 0,
    max_time: float = 600.0,
    standoff: float = DEFAULT_START_STANDOFF,
    use_gui: bool = False,
    verbose: bool = False,
    gif_path: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    mpc: MPCConfig | None = None,
    check_base_collisions: bool = True,
    check_robot_collisions: bool = True,
    human_torque_limit: float | None = None,
    gravity: tuple[float, float, float] = DEFAULT_GRAVITY,
    muscle_tone: str = DEFAULT_MUSCLE_TONE,
    joint_limits_model: str = DEFAULT_JOINT_LIMITS_MODEL,
    range_of_motion_scale: float = 1.0,
    limb_joint_damping: float = DEFAULT_LIMB_JOINT_DAMPING,
    robot_base_z: float | None = None,
    result: RunResult | None = None,
) -> bool:
    """Build the variant, plan with PDDLStream, and execute the plan.

    Returns whether the limb ends up within `goal_atol` of its goal. A `result` is
    filled in with what the run took and measured.
    """
    if result is None:
        result = RunResult(variant=variant, seed=seed)
    sim = create_env(
        variant,
        standoff=standoff,
        use_gui=use_gui,
        robot_base_z=robot_base_z,
        gravity=gravity,
        muscle_tone=muscle_tone,
        joint_limits_model=joint_limits_model,
        range_of_motion_scale=range_of_motion_scale,
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
            human_torque_limit=human_torque_limit,
        )
        reset_to_start(ctx)
        plan_start = time.time()
        plan = plan_limbrepositioning3d(ctx, max_time=max_time, verbose=verbose)
        result.plan_time = time.time() - plan_start
        result.plan_found = plan is not None

        reset_to_start(ctx)
        frames = [render_frame(sim)] if gif_path is not None else None
        log = RolloutLog() if trajectory_path is not None else None
        try:
            if plan is None:
                # Replay the closest attempt, so the GIF shows what was tried.
                attempt_plan = build_failed_attempt_plan(ctx)
                result.failure_reason = (
                    ctx.best_attempt.reason
                    if ctx.best_attempt is not None
                    else "no reachable base pose"
                )
                if attempt_plan is None:
                    print(
                        "No plan found, and no motion was ever attempted (no reachable "
                        "base pose); saving the start state only."
                    )
                    return False
                assert ctx.best_attempt is not None
                print(
                    f"No plan found; replaying the closest attempt "
                    f"({ctx.best_attempt.reason}) for the GIF."
                )
                execute_plan(ctx, attempt_plan, frames=frames, log=log)
                return False
            record_trajectory_metrics(result, plan)
            result.goal_reached = execute_plan(ctx, plan, frames=frames, log=log)
            if not result.goal_reached:
                result.failure_reason = "the executed plan missed the goal"
            return result.goal_reached
        finally:
            if frames is not None:
                assert gif_path is not None
                save_gif(gif_path, frames)
            if log is not None and log.limb_positions:
                assert trajectory_path is not None
                save_rollout(trajectory_path, ctx, log)
                plot_rollout(trajectory_path)
    finally:
        sim.close()


def _exception_reason(exc: BaseException) -> str:
    """Name a crash by where it was raised, since a bare `assert` carries no message."""
    frame = traceback.extract_tb(exc.__traceback__)[-1]
    where = f"{Path(frame.filename).name}:{frame.lineno}"
    return f"{type(exc).__name__} at {where}: {exc}".rstrip(": ")


def solve_all_variants(
    gif_dir: str | Path | None = None,
    trajectory_dir: str | Path | None = None,
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
        trajectory_path = (
            Path(trajectory_dir) / f"{variant}.npz"
            if trajectory_dir is not None
            else None
        )
        print(f"\n=== [{index}/{len(ALL_VARIANTS)}] {variant} ===", flush=True)
        start = time.time()
        result = RunResult(variant=variant, seed=seed)
        try:
            solve_and_execute(
                variant=variant,
                gif_path=gif_path,
                trajectory_path=trajectory_path,
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


def write_results_csv(
    path: str | Path, results: list[RunResult], settings: dict[str, Any]
) -> None:
    """Append one row per result, so a sweep of settings accumulates in one file."""
    rows = [{**settings, **dataclasses.asdict(result)} for result in results]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Results appended to {path.resolve()}")


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
        "--use-gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the PyBullet GUI (default: off).",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=None,
        help=(
            "If set, write <variant>.npz of per-step measurements and <variant>.png "
            "of their plots into this directory, for failed runs as well."
        ),
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="If set, append one row per run to this CSV, creating it if it is new.",
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
        "human_torque_limit": args.human_torque_limit,
        "gravity": (0.0, 0.0, args.gravity),
        "muscle_tone": args.muscle_tone,
        "joint_limits_model": args.joint_limits_model,
        "range_of_motion_scale": args.range_of_motion_scale,
        "limb_joint_damping": args.limb_joint_damping,
        "robot_base_z": args.robot_base_z,
        "mpc": MPCConfig(
            num_rollouts=args.num_rollouts,
            horizon=args.horizon,
            noise_scale=args.noise_scale,
        ),
    }
    settings = {
        "gravity": args.gravity,
        "muscle_tone": args.muscle_tone,
        "joint_limits_model": args.joint_limits_model,
        "range_of_motion_scale": args.range_of_motion_scale,
        "limb_joint_damping": args.limb_joint_damping,
        "human_torque_limit": args.human_torque_limit,
        "max_time": args.max_time,
    }
    if args.all_variants:
        # Per-variant PDDLStream output would bury the summary table.
        results = solve_all_variants(
            gif_dir=gif_dir,
            trajectory_dir=args.trajectory_dir,
            verbose=False,
            **shared,
        )
        if args.results_csv is not None:
            write_results_csv(args.results_csv, list(results.values()), settings)
        return
    result = RunResult(variant=args.variant, seed=args.seed)
    trajectory_path = (
        Path(args.trajectory_dir) / f"{args.variant}.npz"
        if args.trajectory_dir is not None
        else None
    )
    start = time.time()
    try:
        success = solve_and_execute(
            variant=args.variant,
            gif_path=(
                gif_output_path("limbrepositioning3d", args.variant, gif_dir)
                if gif_dir is not None
                else None
            ),
            trajectory_path=trajectory_path,
            verbose=True,
            result=result,
            **shared,
        )
        print(f"Reached goal: {success}")
    except Exception as exc:  # pylint: disable=broad-except
        # Still record the row, so a crashed cell of a sweep is not a silent gap.
        traceback.print_exc()
        result.failure_reason = _exception_reason(exc)
    result.total_time = time.time() - start
    if args.results_csv is not None:
        write_results_csv(args.results_csv, [result], settings)


if __name__ == "__main__":
    main()
