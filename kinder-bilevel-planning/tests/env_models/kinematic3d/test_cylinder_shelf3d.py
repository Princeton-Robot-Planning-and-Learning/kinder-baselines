"""Tests for cylinder_shelf3d.py."""

import kinder
import numpy as np
import pytest
from bilevel_planning.structs import RelationalAbstractState
from kinder_models.structs import SkillCall
from relational_structs import GroundAtom

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def _make_agent(env, magic_skills):
    env_models = create_bilevel_planning_models(
        "cylinder_shelf3d",
        env.observation_space,
        env.action_space,
        magic_skills=magic_skills,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=10,
        samples_per_step=3,
        planning_timeout=600.0,
        max_skill_horizon=400,
    )
    return env_models, agent


def test_cylinder_shelf3d_magic_pick_planning_and_execution():
    """With Pick magic, the plan holds exactly one SkillCall for the pick, the
    predicted state satisfies the pick's effects, and executing the plan with the
    SkillCall carried out as a teleport reaches the goal."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", allow_state_access=True)
    env_models, agent = _make_agent(env, magic_skills=("Pick",))
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)

    trajectory = agent.plan()
    calls = [(x, u) for x, u in trajectory if isinstance(u, SkillCall)]
    assert len(calls) == 1
    pre_state, call = calls[0]
    assert call.skill_name == "Pick"
    assert [o.name for o in call.objects] == ["robot", "cylinder0"]
    assert not any(isinstance(u, SkillCall) for _, u in trajectory[-1:])

    # The predicted state carries the pick's abstract effects.
    robot = pre_state.get_object_from_name("robot")
    cylinder = pre_state.get_object_from_name("cylinder0")
    holding = next(p for p in env_models.predicates if p.name == "Holding")
    abstract = env_models.state_abstractor(call.predicted_state)
    assert isinstance(abstract, RelationalAbstractState)
    assert GroundAtom(holding, [robot, cylinder]) in abstract.atoms
    assert env_models.transition_fn(pre_state, call) == call.predicted_state

    # Execute: regular actions step the env; the SkillCall teleports it.
    terminated = False
    for _, action in trajectory:
        if isinstance(action, SkillCall):
            env.unwrapped.set_state(
                env.observation_space.vectorize(action.predicted_state)
            )
            continue
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated, "Executing the magic plan did not reach the goal"
    env.close()


def test_cylinder_shelf3d_magic_skills_validation():
    """Unknown skill names are rejected at model construction."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0")
    with pytest.raises(ValueError, match="Unknown magic skill"):
        create_bilevel_planning_models(
            "cylinder_shelf3d",
            env.observation_space,
            env.action_space,
            magic_skills=("Fly",),
        )
    env.close()


def test_cylinder_shelf3d_without_magic_has_no_skill_calls():
    """Without magic skills the plan is a plain low-level action sequence."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0")
    _, agent = _make_agent(env, magic_skills=())
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    trajectory = agent.plan()
    assert trajectory
    assert all(isinstance(u, np.ndarray) for _, u in trajectory)
    env.close()
