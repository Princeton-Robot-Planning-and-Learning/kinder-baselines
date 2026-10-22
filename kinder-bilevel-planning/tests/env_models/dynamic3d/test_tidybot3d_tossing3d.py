"""Tests for tidybot3d_tossing3D.py."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
)

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_tidybot3d_tossing3d_bilevel_planning():
    """Plan and execute a pick and a throw in the Tossing3D environment."""

    num_objects = 1
    env = kinder.make(f"kinder/Tossing3D-o{num_objects}-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="TidyBot3D-tossing3d")

    seed = 125
    obs, info = env.reset(seed=seed)

    env_models = create_bilevel_planning_models(
        "tidybot3d_tossing3D",
        env.observation_space,
        env.action_space,
        num_objects=num_objects,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        # Receivers moved farther from the barrier (KINDER #191): 5 tosses rarely score.
        samples_per_step=40,
        # Farther receivers (KINDER #191), before #172's sim-only effort extension.
        planning_timeout=600.0,
        max_skill_horizon=400,
    )

    agent.reset(obs, info)
    for _ in range(4000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, reward, terminated or truncated, info)
        if (
            terminated
            or truncated
            or len(agent._current_plan) == 0  # pylint: disable=protected-access
        ):
            break
    else:
        assert False, "Did not terminate successfully"

    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    assert (
        sim._check_goals()  # pylint: disable=protected-access
    ), "Planned and executed, but the cube did not score"
    env.close()


def test_toss_sampler_uses_the_throw_profiles_only_for_bins_they_were_tuned_for():
    """#165's throw profiles were tuned for receivers at x in [2.6, 3.42]. A nearer bin
    (the barrier layout places them from x = 1.48) gets the uniform sampler instead: on
    one, all 24 profile tosses measured were thrown and 0/24 scored, overshooting by
    0.4-1.2 m."""
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=125)
    models = create_bilevel_planning_models(
        "tidybot3d_tossing3D", env.observation_space, env.action_space, num_objects=1
    )
    state = env.observation_space.devectorize(obs)
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    bin_obj = state.get_object_from_name("bin_0")
    lifted = {s.operator.name: s.controller for s in models.skills}
    toss = lifted["move_to_toss_location_and_toss"].ground((robot, cube, barrier))
    profiles = {
        (float(speed), float(ms))
        for speed, ms in MoveToTossLocationAndTossController.THROW_PROFILES
    }
    rng = np.random.default_rng(0)
    for bin_x, uses_profiles in ((2.9, True), (2.10, False)):
        moved = state.copy()
        moved.set(bin_obj, "x", bin_x)
        draws = [toss.sample_parameters(moved, rng) for _ in range(50)]
        drawn = {(round(float(np.rad2deg(p[2])), 6), float(p[3])) for p in draws}
        if uses_profiles:
            assert drawn <= profiles, bin_x
        else:
            assert not drawn <= profiles, bin_x
            for p in draws:
                # Never a launch pose past the barrier: the base stays on its side.
                assert bin_x - p[0] <= state.get(barrier, "x"), p
    env.close()
