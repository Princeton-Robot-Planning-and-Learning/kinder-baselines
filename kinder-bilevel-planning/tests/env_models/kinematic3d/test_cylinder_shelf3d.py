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


def _make_agent(env, magic_skills, **kwargs):
    env_models = create_bilevel_planning_models(
        "cylinder_shelf3d",
        env.observation_space,
        env.action_space,
        magic_skills=magic_skills,
        **kwargs,
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


def _predicate(env_models, name):
    return next(p for p in env_models.predicates if p.name == name)


def test_cylinder_shelf3d_magic_grasp_planning_and_execution():
    """With Grasp magic, the plan reaches the pre-grasp pose with planned motion, holds
    exactly one SkillCall for the grasp whose predicted state satisfies Holding, and
    executing the plan with the SkillCall carried out as a teleport reaches the goal."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", allow_state_access=True)
    env_models, agent = _make_agent(env, magic_skills=("Grasp",))
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)

    trajectory = agent.plan()
    calls = [
        (i, x, u) for i, (x, u) in enumerate(trajectory) if isinstance(u, SkillCall)
    ]
    assert len(calls) == 1
    index, pre_state, call = calls[0]
    assert call.skill_name == "Grasp"
    assert [o.name for o in call.objects] == ["robot", "cylinder0"]
    # The approach to the pre-grasp pose is planned motion, not magic.
    assert index > 0
    assert all(isinstance(u, np.ndarray) for _, u in trajectory[:index])

    robot = pre_state.get_object_from_name("robot")
    cylinder = pre_state.get_object_from_name("cylinder0")
    at_pre_grasp = GroundAtom(_predicate(env_models, "AtPreGrasp"), [robot, cylinder])
    holding = GroundAtom(_predicate(env_models, "Holding"), [robot, cylinder])
    before = env_models.state_abstractor(pre_state)
    after = env_models.state_abstractor(call.predicted_state)
    assert isinstance(before, RelationalAbstractState)
    assert at_pre_grasp in before.atoms and holding not in before.atoms
    assert holding in after.atoms and at_pre_grasp not in after.atoms
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
            magic_skills=("Pick",),
        )
    env.close()


def test_cylinder_shelf3d_without_magic_has_no_skill_calls():
    """Without magic skills the plan is a plain low-level action sequence that reaches
    the goal."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0")
    _, agent = _make_agent(env, magic_skills=())
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    trajectory = agent.plan()
    assert trajectory
    assert all(isinstance(u, np.ndarray) for _, u in trajectory)
    terminated = False
    for _, action in trajectory:
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated
    env.close()


def test_cylinder_shelf3d_fixed_place_params():
    """With place_params the plan sets the cylinder down at the given offset from the
    shelf centre."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", allow_state_access=True)
    offset = (-0.10, -0.05)
    _, agent = _make_agent(env, magic_skills=("Grasp",), place_params=[offset])
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)

    terminated = False
    for _, action in agent.plan():
        if isinstance(action, SkillCall):
            obs = env.observation_space.vectorize(action.predicted_state)
            env.unwrapped.set_state(obs)
            continue
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated, "Executing the plan did not reach the goal"

    state = env.observation_space.devectorize(obs)
    shelf_pose = state.get_object_pose("shelf")
    cylinder = state.get_object_from_name("cylinder0")
    placed_xy = (state.get(cylinder, "pose_x"), state.get(cylinder, "pose_y"))
    expected_xy = (
        shelf_pose.position[0] + offset[0],
        shelf_pose.position[1] - 0.05 + offset[1],
    )
    assert np.allclose(placed_xy, expected_xy, atol=0.02), (placed_xy, expected_xy)
    env.close()


def test_cylinder_shelf3d_place_params_validation():
    """One place parameter entry per cylinder, or a ValueError."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0")
    with pytest.raises(ValueError, match="place_params has 2 entries"):
        create_bilevel_planning_models(
            "cylinder_shelf3d",
            env.observation_space,
            env.action_space,
            place_params=[(0.0, 0.0), None],
        )
    env.close()


def test_cylinder_shelf3d_two_cylinders_first_skeleton():
    """MoveToPreGrasp requires and deletes NotAtPreGrasp, so the abstract planner
    cannot chain two MoveToPreGrasps and its first skeleton for two cylinders is
    refinable: planning succeeds with a single abstract plan, one sample per step
    (the placements are fully fixed), and one MoveToPreGrasp + Grasp per
    cylinder."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o2-v0", allow_state_access=True)
    env_models = create_bilevel_planning_models(
        "cylinder_shelf3d",
        env.observation_space,
        env.action_space,
        num_objects=2,
        magic_skills=("MoveToPreGrasp", "Grasp"),
        place_params=[(-0.10, -0.05, 0.86), (0.10, -0.05, 0.86)],
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=600.0,
        max_skill_horizon=400,
    )
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    calls = [u for _, u in agent.plan() if isinstance(u, SkillCall)]
    assert [c.skill_name for c in calls] == ["MoveToPreGrasp", "Grasp"] * 2
    assert {c.objects[1].name for c in calls} == {"cylinder0", "cylinder1"}
    env.close()


def test_cylinder_shelf3d_plans_with_base_clearance():
    """base_clearance reaches the skills' base motion planner and planning still
    succeeds with a few centimetres of it."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", allow_state_access=True)
    _, agent = _make_agent(env, magic_skills=("Grasp",), base_clearance=0.05)
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    assert len(agent.plan()) > 0
    env.close()
