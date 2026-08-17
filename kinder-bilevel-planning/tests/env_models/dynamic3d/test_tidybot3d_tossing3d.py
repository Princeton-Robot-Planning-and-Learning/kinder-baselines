import pytest
import kinder
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

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
        samples_per_step=5,
        planning_timeout=300.0,
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
