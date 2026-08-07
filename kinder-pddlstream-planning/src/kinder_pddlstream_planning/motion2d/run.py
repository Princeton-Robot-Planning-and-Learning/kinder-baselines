"""Solve KinDER's Motion2D environment with PDDLStream.

Mirrors the `motion` example that ships with pddlstream.

Unlike that example, `connect` runs a full RRT-Connect query, not one straight line.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
from kinder.envs.kinematic2d.motion2d import (
    ObjectCentricMotion2DEnv,
    TargetRegionType,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    crv_pose_plan_to_action_plan,
    rectangle_object_to_geom,
)
from numpy.typing import NDArray
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read
from relational_structs import ObjectCentricState

from kinder_pddlstream_planning.motion2d.stream import (
    Conf,
    Motion2DStreamContext,
    conf_key,
    distance,
    sample_region,
    test_connected,
)
from kinder_pddlstream_planning.rendering import render_frame, save_gif

_HERE = Path(__file__).parent
DOMAIN_PDDL = read(str(_HERE / "domain.pddl"))
STREAM_PDDL = read(str(_HERE / "stream.pddl"))


def build_stream_context(
    sim: ObjectCentricMotion2DEnv,
    state: ObjectCentricState,
    num_attempts: int = 10,
    num_iters: int = 200,
    smooth_amt: int = 50,
) -> Motion2DStreamContext:
    """Precompute the robot and target geometry every stream shares.

    Split out of `create_problem` so tests can drive streams directly.
    """
    robot = state.get_objects(CRVRobotType)[0]
    target_region = state.get_objects(TargetRegionType)[0]
    action_space = sim.action_space
    assert isinstance(action_space, CRVRobotActionSpace)

    # The environment's termination check only tests whether the robot's center point
    # lies in the target rectangle, so no robot-radius inset is needed.
    target_geom = rectangle_object_to_geom(state, target_region, {})
    target_lower = np.array([target_geom.x, target_geom.y])
    target_upper = np.array(
        [target_geom.x + target_geom.width, target_geom.y + target_geom.height]
    )

    return Motion2DStreamContext(
        sim=sim,
        state=state,
        robot=robot,
        action_space=action_space,
        theta=state.get(robot, "theta"),
        target_lower=target_lower,
        target_upper=target_upper,
        num_attempts=num_attempts,
        num_iters=num_iters,
        smooth_amt=smooth_amt,
    )


def create_problem(
    sim: ObjectCentricMotion2DEnv,
    state: ObjectCentricState,
    **context_kwargs,
) -> tuple[PDDLProblem, Motion2DStreamContext]:
    """Build a PDDLProblem for moving the robot in `state` to the target region.

    Also returns the context, whose `path_cache` holds the path found for each pair.
    """
    ctx = build_stream_context(sim, state, **context_kwargs)

    q0: Conf = np.array([state.get(ctx.robot, "x"), state.get(ctx.robot, "y")])
    init = [("Conf", q0), ("AtConf", q0), ("Region", "target")]
    goal = ("In", "target")

    stream_map = {
        "sample-region": from_gen_fn(partial(sample_region, ctx)),
        "connect": from_test(partial(test_connected, ctx)),
        "distance": partial(distance, ctx),
    }

    problem = PDDLProblem(DOMAIN_PDDL, {}, STREAM_PDDL, stream_map, init, goal)
    return problem, ctx


def plan_motion2d(
    sim: ObjectCentricMotion2DEnv,
    state: ObjectCentricState,
    max_time: float = 30.0,
    verbose: bool = False,
    **problem_kwargs,
) -> list[SE2Pose] | None:
    """Solve for a fine-grained SE2Pose path from `state` to the target region.

    Consecutive poses are one low-level action apart.
    """
    problem, ctx = create_problem(sim, state, **problem_kwargs)
    solution = solve(
        problem,
        algorithm="adaptive",
        unit_costs=False,
        max_time=max_time,
        verbose=verbose,
    )
    if verbose:
        print_solution(solution)
    plan, _, _ = solution
    if plan is None:
        return None

    pose_plan: list[SE2Pose] = []
    for name, args in plan:
        assert name == "move"
        q1, q2 = args
        segment = ctx.path_cache[(conf_key(q1), conf_key(q2))]
        pose_plan.extend(segment if not pose_plan else segment[1:])
    return pose_plan


def waypoints_to_actions(
    sim: ObjectCentricMotion2DEnv, pose_plan: list[SE2Pose]
) -> list[NDArray[np.float32]]:
    """Convert an RRT-Connect SE2Pose path into low-level robot actions.

    The path is already interpolated, so this reads off the deltas between poses.
    """
    action_space = sim.action_space
    assert isinstance(action_space, CRVRobotActionSpace)
    return [
        np.asarray(action, dtype=np.float32)
        for action in crv_pose_plan_to_action_plan(pose_plan, action_space)
    ]


def solve_and_execute(
    num_passages: int = 2,
    seed: int = 0,
    max_time: float = 30.0,
    verbose: bool = False,
    gif_path: str | Path | None = None,
    **problem_kwargs,
) -> bool:
    """Reset Motion2D, plan with PDDLStream, and execute the plan.

    Returns whether the robot reached the target.
    """
    sim = ObjectCentricMotion2DEnv(num_passages=num_passages)
    state, _ = sim.reset(seed=seed)
    frames = [render_frame(sim)] if gif_path is not None else None
    try:
        pose_plan = plan_motion2d(
            sim, state, max_time=max_time, verbose=verbose, **problem_kwargs
        )
        if pose_plan is None:
            return False
        for action in waypoints_to_actions(sim, pose_plan):
            _, _, terminated, _, _ = sim.step(action)
            if frames is not None:
                frames.append(render_frame(sim))
            if terminated:
                return True
        return False
    finally:
        if frames is not None:
            assert gif_path is not None
            save_gif(gif_path, frames)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-passages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=100.0)
    parser.add_argument("--num-attempts", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument(
        "--gif-path",
        type=str,
        default=None,
        help="If set, save a GIF of the rollout to this path.",
    )
    args = parser.parse_args()
    success = solve_and_execute(
        num_passages=args.num_passages,
        seed=args.seed,
        max_time=args.max_time,
        num_attempts=args.num_attempts,
        num_iters=args.num_iters,
        gif_path=args.gif_path,
        verbose=True,
    )
    print(f"Reached target: {success}")


if __name__ == "__main__":
    main()
