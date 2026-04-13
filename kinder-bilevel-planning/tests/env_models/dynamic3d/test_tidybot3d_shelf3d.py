"""Tests for tidybot3d_shelf3D.py."""

import kinder
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_tidybot3d_cupboard_bilevel_planning():
    """Tests for bilevel planning in the Shelf3D real environment."""

    num_objects = 1
    env = kinder.make(f"kinder/Shelf3D-o{num_objects}-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="TidyBot3D-shelf3d")

    seed = 123
    obs, info = env.reset(seed=seed)
    total_reward = 0

    env_models = create_bilevel_planning_models(
        "tidybot3d_shelf3D",
        env.observation_space,
        env.action_space,
        num_objects=num_objects,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=120.0,
        max_skill_horizon=400,
    )

    agent.reset(obs, info)
    for _ in range(4000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if (
            terminated
            or truncated
            or len(agent._current_plan) == 0  # pylint: disable=protected-access
        ):
            break

    else:
        assert False, "Did not terminate successfully"

    env.close()
