"""Tests for vega_motion3d.py."""

import kinder
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

# VegaMotion3D needs prpl_kinematics, which is an optional extra of both kindergarden and
# this package. Skip the module rather than fail collection when it is not installed.
pytest.importorskip("kinder.envs.kinematic3d_v2.vega_motion3d")
pytest.importorskip("kinder_models.kinematic3d_v2.vega_motion3d.parameterized_skills")

kinder.register_all_environments()


@pytest.mark.parametrize("prefer_ompl", [True, False])
def test_vega_motion3d_bilevel_planning(prefer_ompl):
    """Tests for bilevel planning in the VegaMotion3D environment.

    Parameterized over both planners so the BiRRT fallback is covered even when ompl is
    installed.
    """
    env = kinder.make("kinder/VegaMotion3D-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"VegaMotion3D-bilevel-{prefer_ompl}"
        )

    env_models = create_bilevel_planning_models(
        "vega_motion3d",
        env.observation_space,
        env.action_space,
        prefer_ompl=prefer_ompl,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=60.0,
        max_skill_horizon=1000,
    )
    obs, info = env.reset(seed=123)
    total_reward = 0
    agent.reset(obs, info)

    for _ in range(1000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break

    else:
        assert False, "Did not terminate successfully"

    assert terminated, "Episode truncated rather than reaching the goal"

    env.close()
