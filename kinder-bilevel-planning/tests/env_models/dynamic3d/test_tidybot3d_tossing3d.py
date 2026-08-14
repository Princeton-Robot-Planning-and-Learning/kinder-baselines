"""Tests for tidybot3d_tossing3D.py."""

from pathlib import Path

import kinder
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder_models.dynamic3d.tossing.state_abstractions import MovableInGoalRegion
from relational_structs import GroundAtom

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()

ENV_MODEL_NAME = "tidybot3d_tossing3D"

_TEST_TASKS = Path(__file__).parent.parent.parent / "test_tasks"


def test_tidybot3d_tossing_bilevel_planning():
    """Tests for bilevel planning in the Tossing3D environment."""
    num_objects = 1
    env = kinder.make(f"kinder/Tossing3D-o{num_objects}-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="TidyBot3D-tossing3d")

    seed = 125
    obs, info = env.reset(seed=seed)

    env_models = create_bilevel_planning_models(
        ENV_MODEL_NAME,
        env.observation_space,
        env.action_space,
        num_objects=num_objects,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        samples_per_step=5,
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

    # The plan is refined in a separate simulator, so executing it is what shows the
    # model is faithful rather than merely self-consistent.
    final_state = env_models.observation_to_state(obs)
    cube = final_state.get_object_from_name("cube_0")
    final_atoms = env_models.state_abstractor(final_state).atoms
    assert GroundAtom(MovableInGoalRegion, [cube]) in final_atoms

    env.close()


def test_tidybot3d_tossing_skills_ground_with_every_operator_parameter():
    """The toss and the move name objects their controllers never read.

    Both are given the operator's whole parameter tuple rather than the subset the
    controller uses, which is what lets GroundSkill pair the two.
    """
    env = kinder.make("kinder/Tossing3D-o1-v0")
    env_models = create_bilevel_planning_models(
        ENV_MODEL_NAME, env.observation_space, env.action_space, num_objects=1
    )

    operator_to_skill = {s.operator: s for s in env_models.skills}
    assert env_models.ground_operators is not None
    for ground_operator in env_models.ground_operators:
        assert ground_operator.parent is not None
        skill = operator_to_skill[ground_operator.parent]
        ground_skill = skill.ground(tuple(ground_operator.parameters))
        assert tuple(ground_skill.controller.objects) == tuple(
            ground_operator.parameters
        )

    env.close()


def test_tidybot3d_tossing_abstracts_against_the_scene_it_was_given():
    """A model built from an explicit scene reads that scene's goal region."""
    env = kinder.make("kinder/Tossing3D-o1-v0")
    obs, _ = env.reset(seed=125)

    variant_models = create_bilevel_planning_models(
        ENV_MODEL_NAME,
        env.observation_space,
        env.action_space,
        num_objects=1,
        task_config_path=str(_TEST_TASKS / "tidybot-tossing3d_near_goal-o1.json"),
    )
    default_models = create_bilevel_planning_models(
        ENV_MODEL_NAME, env.observation_space, env.action_space, num_objects=1
    )

    # The variant's goal region covers where the cube starts, so an untouched reset
    # satisfies the goal under it and cannot under the installed scene.
    state = variant_models.observation_to_state(obs)
    cube = state.get_object_from_name("cube_0")
    in_goal_region = GroundAtom(MovableInGoalRegion, [cube])
    assert in_goal_region in variant_models.state_abstractor(state).atoms
    assert in_goal_region not in default_models.state_abstractor(state).atoms

    env.close()


def test_tidybot3d_tossing_rejects_the_two_cube_variant():
    """The single-cube limit is reported here, not from inside the abstractor."""
    env = kinder.make("kinder/Tossing3D-o1-v0")

    with pytest.raises(NotImplementedError):
        create_bilevel_planning_models(
            ENV_MODEL_NAME,
            env.observation_space,
            env.action_space,
            num_objects=2,
        )

    env.close()
