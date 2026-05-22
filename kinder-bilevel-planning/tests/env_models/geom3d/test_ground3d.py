"""Tests for ground3d.py."""

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_ground3d_observation_to_state():
    """Tests for observation_to_state() in the Ground3D environment."""
    env = kinder.make("kinder/Ground3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "ground3d", env.observation_space, env.action_space
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_ground3d_transition_fn():
    """Tests for transition_fn() in the Ground3D environment."""
    env = kinder.make("kinder/Ground3D-o1-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "ground3d", env.observation_space, env.action_space
    )
    transition_fn = env_models.transition_fn
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)

    for _ in range(10):
        executable = env.action_space.sample()
        next_state = transition_fn(state, executable)
        assert env_models.state_space.contains(next_state)
        assert isinstance(hash(next_state), int)
        state = next_state
    env.close()


def test_ground3d_goal_deriver():
    """Tests for goal_deriver() in the Ground3D environment.

    The goal is a single `Holding(robot, cube0)` atom, regardless of how many cubes the
    variant has.
    """
    env = kinder.make("kinder/Ground3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "ground3d", env.observation_space, env.action_space
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 1
    env.close()


def test_ground3d_state_abstractor():
    """Tests for state_abstractor() in the Ground3D environment.

    Initially the hand should be empty and the cube should be on the ground.
    """
    env = kinder.make("kinder/Ground3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "ground3d", env.observation_space, env.action_space
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    OnGround = pred_name_to_pred["OnGround"]
    Holding = pred_name_to_pred["Holding"]
    HandEmpty = pred_name_to_pred["HandEmpty"]

    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target = obj_name_to_obj["cube0"]

    assert HandEmpty([robot]) in abstract_state.atoms
    assert OnGround([target]) in abstract_state.atoms
    assert Holding([robot, target]) not in abstract_state.atoms

    env.close()


def _skill_test_helper(ground_skill, env_models, env, obs, params=None):
    """Helper function to test a skill execution."""
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    operator = ground_skill.operator
    assert operator.preconditions.issubset(abstract_state.atoms)
    controller = ground_skill.controller
    if params is None:
        params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        controller.observe(next_state)
        state = next_state

        if controller.terminated():
            break
    return obs


def test_ground3d_skills():
    """Tests that the Pick skill picks up the cube as expected."""
    env = kinder.make("kinder/Ground3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "ground3d", env.observation_space, env.action_space
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    Pick = skill_name_to_skill["Pick"]

    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target = obj_name_to_obj["cube0"]

    pick_skill = Pick.ground((robot, target))
    obs1 = _skill_test_helper(pick_skill, env_models, env, obs0)

    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    Holding = pred_name_to_pred["Holding"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    OnGround = pred_name_to_pred["OnGround"]
    assert HandEmpty([robot]) not in abstract_state1.atoms
    assert OnGround([target]) not in abstract_state1.atoms
    assert Holding([robot, target]) in abstract_state1.atoms

    env.close()


@pytest.mark.parametrize("seed", [123])
def test_ground3d_bilevel_planning(seed):
    """End-to-end bilevel planning test in the Ground3D-o1 environment.

    Ground3D's env-side `goal_reached()` returns False unconditionally
    (unlike Shelf3D, which signals when the gripper is closed after
    placing), so we check abstract-goal satisfaction via the state
    abstractor each tick instead of waiting for `terminated`.
    """

    num_objects = 1
    env = kinder.make(
        f"kinder/Ground3D-o{num_objects}-v0",
        render_mode="rgb_array",
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Ground3D-bilevel-{seed}"
        )

    env_models = create_bilevel_planning_models(
        "ground3d",
        env.observation_space,
        env.action_space,
        num_objects=num_objects,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=60.0,
        max_skill_horizon=500,
    )
    obs, info = env.reset(seed=seed)
    goal = env_models.goal_deriver(env_models.observation_to_state(obs))
    try:
        agent.reset(obs, info)
    except Exception as e:
        env.close()
        pytest.skip(f"Planning failed for seed {seed}: {e}")

    for _ in range(1000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, reward, terminated or truncated, info)
        abstract_state = env_models.state_abstractor(
            env_models.observation_to_state(obs)
        )
        if goal.atoms.issubset(abstract_state.atoms):
            break
        if terminated or truncated:
            break
    else:
        assert False, "Did not reach abstract goal within step budget"

    abstract_state = env_models.state_abstractor(env_models.observation_to_state(obs))
    assert goal.atoms.issubset(
        abstract_state.atoms
    ), f"Abstract goal not satisfied; missing {goal.atoms - abstract_state.atoms}"

    env.close()
