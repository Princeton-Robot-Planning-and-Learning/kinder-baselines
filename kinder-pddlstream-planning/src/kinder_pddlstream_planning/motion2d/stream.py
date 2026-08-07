"""PDDLStream stream implementations for Motion2D.

See run.py for the domain design.

Each function takes the `Motion2DStreamContext` built by `create_problem`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from kinder.envs.kinematic2d.motion2d import ObjectCentricMotion2DEnv
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    run_motion_planning_for_crv_robot,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState

Conf = NDArray[np.float64]
ConfKey = tuple[float, float]


def conf_key(conf: Conf) -> ConfKey:
    """A hashable, float-rounded key for a conf, used to cache RRT-Connect paths."""
    return (round(float(conf[0]), 6), round(float(conf[1]), 6))


@dataclass
class Motion2DStreamContext:
    """Shared context for all streams of one `create_problem` call."""

    sim: ObjectCentricMotion2DEnv
    state: ObjectCentricState
    robot: Object
    action_space: CRVRobotActionSpace
    # The robot's heading, held fixed because only its (x, y) position is planned over.
    theta: float
    # The corners of the target region that `sample_region` samples between.
    target_lower: Conf
    target_upper: Conf
    # RRT-Connect's own retry and iteration budget.
    num_attempts: int = 10
    num_iters: int = 200
    smooth_amt: int = 50
    # The pose path `test_connected` found for each pair of confs, so the caller can
    # recover the trajectory of each `move` action in the plan.
    path_cache: dict[tuple[ConfKey, ConfKey], list[SE2Pose]] = field(
        default_factory=dict, repr=False
    )


def sample_region(ctx: Motion2DStreamContext, region: str) -> Iterator[tuple[Conf]]:
    """Yield configurations drawn from `region`, which is always the target."""
    del region  # only 'target' is ever sampled
    while True:
        yield (ctx.sim.np_random.uniform(ctx.target_lower, ctx.target_upper),)


def test_connected(ctx: Motion2DStreamContext, q1: Conf, q2: Conf) -> bool:
    """Whether RRT-Connect finds a collision-free path from `q1` to `q2`."""
    key = (conf_key(q1), conf_key(q2))
    if key in ctx.path_cache:
        return True
    start_state = ctx.state.copy()
    start_state.set(ctx.robot, "x", float(q1[0]))
    start_state.set(ctx.robot, "y", float(q1[1]))
    start_state.set(ctx.robot, "theta", ctx.theta)
    start_state.data.update(ctx.sim.initial_constant_state.data)
    target_pose = SE2Pose(float(q2[0]), float(q2[1]), ctx.theta)
    seed = int(ctx.sim.np_random.integers(0, 2**31 - 1))
    pose_plan = run_motion_planning_for_crv_robot(
        start_state,
        ctx.robot,
        target_pose,
        ctx.action_space,
        seed=seed,
        num_attempts=ctx.num_attempts,
        num_iters=ctx.num_iters,
        smooth_amt=ctx.smooth_amt,
    )
    if pose_plan is None:
        return False
    ctx.path_cache[key] = pose_plan
    return True


def distance(ctx: Motion2DStreamContext, q1: Conf, q2: Conf) -> float:
    """A straight-line cost heuristic for moving between `q1` and `q2`."""
    del ctx
    # The real RRT path length is only known once `connect` has succeeded.
    return float(np.linalg.norm(q2 - q1))
