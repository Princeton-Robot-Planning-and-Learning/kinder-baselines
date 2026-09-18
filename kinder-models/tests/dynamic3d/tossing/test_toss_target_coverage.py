"""Navigation coverage for the farther receivers introduced by KINDER #191.

These tests establish reachable release poses, not successful ballistic throws.
"""

import kinder
import numpy as np
import pytest

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
)
from kinder_models.dynamic3d.utils import (
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
)


@pytest.mark.parametrize("seed", [10125, 10126, 10127])
def test_sample_domain_contains_reachable_far_side_release_pose(*, seed: int) -> None:
    """At least one sampled standoff must be reachable without crossing the wall."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o1-v0", allow_state_access=True)
    try:
        obs, _ = env.reset(seed=seed)
        state = env.observation_space.devectorize(obs)
        bin_object = state.get_object_from_name("bin_0")
        bin_pose = get_overhead_object_se2_pose(state, bin_object)
        low, high = MoveToTossLocationAndTossController.TARGET_DISTANCE_BOUNDS
        # An interior value is sampleable; a lone feasible endpoint has zero
        # probability under the uniform sampler and does not repair its support.
        distance = high - 0.05 * (high - low)
        target = get_target_robot_pose_from_parameters(bin_pose, distance, 0.0)
        plan = run_base_motion_planning(
            state,
            target,
            WORLD_X_BOUNDS,
            WORLD_Y_BOUNDS,
            seed=0,
            disable_collision_objects=["cube_0"],
        )
        assert plan is not None, (seed, bin_pose, distance, target)
        np.testing.assert_allclose(plan[-1].t, target.t, atol=1e-6)
    finally:
        env.close()
