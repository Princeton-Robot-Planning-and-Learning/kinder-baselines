"""Solve KinDER's LimbRepositioning3D environments with PDDLStream.

Every plan is a move_base, grasp, move_limb skeleton filled in by stream.py.

Only `move_limb` goes through forward dynamics, since the action space is arm torques.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from kinder.envs.kinematic3d.limbrepositioning3d import (
    ALL_VARIANTS,
    LimbRepositioning3DEnvConfig,
    ObjectCentricLimbRepositioning3DEnv,
    create_variant_config,
)
from pybullet_helpers.geometry import SE2Pose

from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    ArmTrajectory,
    LimbConf,
    LimbStreamContext,
    MPCConfig,
    TorqueTrajectory,
    advance,
    check_torque_limits,
    engage_grasp,
    extend_with_fingers,
    is_grasping,
    plan_base_motion,
    plan_grasp_motion,
    plan_limb_motion,
    release_grasp,
    sample_base_pose,
    sample_grasp,
)
from kinder_pddlstream_planning.pddlstream_utils import ensure_pddlstream_importable
from kinder_pddlstream_planning.rendering import render_frame, save_gif

ensure_pddlstream_importable()

# pylint: disable=wrong-import-position,wrong-import-order
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read

_HERE = Path(__file__).parent
DOMAIN_PDDL = read(str(_HERE / "domain.pddl"))
STREAM_PDDL = read(str(_HERE / "stream.pddl"))

# How far behind its grasp placement the base starts, in meters.
DEFAULT_START_STANDOFF = 1.0

LIMB_NAME = "limb"

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
) -> ObjectCentricLimbRepositioning3DEnv:
    """Build the environment with its base backed off to the start pose.

    The variant's own placement becomes the goal of `move_base`.

    The constructor's out-of-reach weld is released before the base drives away.
    """
    if config is None:
        config = create_variant_config(variant)
    if robot_base_z is None:
        robot_base_z = config.robot_base_z
    config = dataclasses.replace(
        config,
        robot_base_home_pose=start_base_pose(config.robot_base_home_pose, standoff),
        robot_base_z=robot_base_z,
    )
    return ObjectCentricLimbRepositioning3DEnv(
        variant=variant, config=config, use_gui=use_gui
    )


def build_stream_context(
    sim: ObjectCentricLimbRepositioning3DEnv,
    standoff: float = DEFAULT_START_STANDOFF,
    motion_seed: int = 0,
    mpc: MPCConfig | None = None,
    check_base_collisions: bool = True,
    check_robot_collisions: bool = True,
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
        "check-torque-limits": from_test(partial(check_torque_limits, ctx)),
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
            _, _, _, _, _, torque_trajectory, _ = args
            assert isinstance(torque_trajectory, TorqueTrajectory)
            for i, torque in enumerate(torque_trajectory.torques):
                advance(sim, torque)
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
                goal_conf,
                TorqueTrajectory(attempt.torques),
                state,
            ),
        ),
    ]


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
    robot_base_z: float | None = None,
) -> bool:
    """Build the variant, plan with PDDLStream, and execute the plan.

    Returns whether the limb ends up within `goal_atol` of its goal.
    """
    sim = create_env(
        variant, standoff=standoff, use_gui=use_gui, robot_base_z=robot_base_z
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
        )
        reset_to_start(ctx)
        plan = plan_limbrepositioning3d(ctx, max_time=max_time, verbose=verbose)

        reset_to_start(ctx)
        frames = [render_frame(sim)] if gif_path is not None else None
        try:
            if plan is None:
                # Replay the closest attempt, so the GIF shows what was tried.
                attempt_plan = build_failed_attempt_plan(ctx)
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
                execute_plan(ctx, attempt_plan, frames=frames)
                return False
            return execute_plan(ctx, plan, frames=frames)
        finally:
            if frames is not None:
                assert gif_path is not None
                save_gif(gif_path, frames)
    finally:
        sim.close()


def solve_all_variants(
    gif_dir: str | Path | None = None,
    **kwargs,
) -> dict[str, tuple[bool, float]]:
    """Run every variant in turn, returning {variant: (reached_goal, seconds)}.

    Exceptions are recorded as failures, so one variant cannot end the sweep.
    """
    results: dict[str, tuple[bool, float]] = {}
    for index, variant in enumerate(ALL_VARIANTS, start=1):
        gif_path = Path(gif_dir) / f"{variant}.gif" if gif_dir is not None else None
        print(f"\n=== [{index}/{len(ALL_VARIANTS)}] {variant} ===", flush=True)
        start = time.time()
        try:
            reached = solve_and_execute(
                variant=variant, gif_path=gif_path, **variant_kwargs(variant, **kwargs)
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{variant}: raised {type(exc).__name__}: {exc}")
            reached = False
        elapsed = time.time() - start
        results[variant] = (reached, elapsed)
        print(
            f"{variant}: {'SUCCESS' if reached else 'FAIL'} in {elapsed:.0f}s",
            flush=True,
        )

    print("\n=== summary ===")
    width = max(len(v) for v in results)
    for variant, (reached, elapsed) in results.items():
        print(
            f"  {variant:{width}s}  {'SUCCESS' if reached else 'FAIL   '}  "
            f"{elapsed:6.0f}s"
        )
    num_reached = sum(reached for reached, _ in results.values())
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
        "--gif-dir",
        type=str,
        default=None,
        help=(
            "With --all-variants, the directory to write <variant>.gif into. "
            "Ignored otherwise. Use --gif-path for a single run."
        ),
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
        "--gif-path",
        type=str,
        default=None,
        help="If set, save a GIF of the rollout to this path.",
    )
    args = parser.parse_args()
    shared = {
        "seed": args.seed,
        "max_time": args.max_time,
        "standoff": args.standoff,
        "use_gui": args.use_gui,
        "check_base_collisions": args.check_base_collisions,
        "check_robot_collisions": args.check_robot_collisions,
        "robot_base_z": args.robot_base_z,
        "mpc": MPCConfig(
            num_rollouts=args.num_rollouts,
            horizon=args.horizon,
            noise_scale=args.noise_scale,
        ),
    }
    if args.all_variants:
        # Per-variant PDDLStream output would bury the summary table.
        solve_all_variants(gif_dir=args.gif_dir, verbose=False, **shared)
        return
    success = solve_and_execute(
        variant=args.variant, gif_path=args.gif_path, verbose=True, **shared
    )
    print(f"Reached goal: {success}")


if __name__ == "__main__":
    main()
