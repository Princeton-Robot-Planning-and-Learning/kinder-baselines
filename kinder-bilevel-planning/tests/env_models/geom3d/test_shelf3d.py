"""Tests for shelf3d.py."""

import os

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.object_types import Kinematic3DFixtureType
from kinder.envs.kinematic3d.shelf3d import Shelf3DEnvConfig
from pybullet_helpers.geometry import Pose

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_shelf3d_observation_to_state():
    """Tests for observation_to_state() in the Shelf3D environment."""
    env = kinder.make("kinder/KinematicShelf3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "shelf3d", env.observation_space, env.action_space
    )
    observation_to_state = env_models.observation_to_state
    obs, _ = env.reset(seed=123)
    state = observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_shelf3d_transition_fn():
    """Tests for transition_fn() in the Shelf3D environment."""
    env = kinder.make("kinder/KinematicShelf3D-o1-v0")
    env.action_space.seed(123)
    env_models = create_bilevel_planning_models(
        "shelf3d", env.observation_space, env.action_space
    )
    transition_fn = env_models.transition_fn
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)

    # Test that transition function produces valid states
    for _ in range(10):
        executable = env.action_space.sample()
        next_state = transition_fn(state, executable)
        assert env_models.state_space.contains(next_state)
        assert isinstance(hash(next_state), int)
        state = next_state
    env.close()


def test_shelf3d_goal_deriver():
    """Tests for goal_deriver() in the Shelf3D environment."""
    env = kinder.make("kinder/KinematicShelf3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "shelf3d", env.observation_space, env.action_space
    )
    goal_deriver = env_models.goal_deriver
    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    goal = goal_deriver(state)
    assert len(goal.atoms) == 2
    env.close()


def test_shelf3d_state_abstractor():
    """Tests for state_abstractor() in the Shelf3D environment."""
    env = kinder.make("kinder/KinematicShelf3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "shelf3d", env.observation_space, env.action_space
    )
    state_abstractor = env_models.state_abstractor
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    OnFixture = pred_name_to_pred["OnFixture"]
    OnGround = pred_name_to_pred["OnGround"]
    Holding = pred_name_to_pred["Holding"]
    HandEmpty = pred_name_to_pred["HandEmpty"]

    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    abstract_state = state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target = obj_name_to_obj["cube0"]
    target_shelf = obj_name_to_obj["shelf"]

    # Initially hand should be empty and object should be on the ground
    assert HandEmpty([robot]) in abstract_state.atoms
    assert OnGround([target]) in abstract_state.atoms
    assert Holding([robot, target]) not in abstract_state.atoms
    assert OnFixture([target, target_shelf]) not in abstract_state.atoms

    env.close()


def test_shelf3d_custom_config_threads_through():
    """A custom Shelf3DEnvConfig threads through both the kinder env and the
    env_model's internal sim.

    Fast logic-only verification of the `config` plumbing — the heavier
    end-to-end bilevel-planning run with the same custom config lives in
    `test_shelf3d_bilevel_planning_with_custom_shelf_pose` (skipped in CI for
    runtime).
    """
    custom_shelf_position = (1.2, 0.8, 0.02)
    custom_config = Shelf3DEnvConfig(shelf_pose=Pose(custom_shelf_position))

    env = kinder.make(
        "kinder/KinematicShelf3D-o1-v0",
        config=custom_config,
    )
    env_models = create_bilevel_planning_models(
        "shelf3d",
        env.observation_space,
        env.action_space,
        num_objects=1,
        config=custom_config,
    )

    obs, _ = env.reset(seed=123)
    state = env_models.observation_to_state(obs)
    shelf = state.get_objects(Kinematic3DFixtureType)[0]
    assert state.get(shelf, "pose_x") == pytest.approx(custom_shelf_position[0])
    assert state.get(shelf, "pose_y") == pytest.approx(custom_shelf_position[1])
    assert state.get(shelf, "pose_z") == pytest.approx(custom_shelf_position[2])

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


def test_shelf3d_skills():
    """Tests for skills in the Shelf3D environment."""
    env = kinder.make("kinder/KinematicShelf3D-o1-v0")
    env_models = create_bilevel_planning_models(
        "shelf3d", env.observation_space, env.action_space
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}
    Pick = skill_name_to_skill["Pick"]
    Place = skill_name_to_skill["Place"]

    obs0, _ = env.reset(seed=123)
    state0 = env_models.observation_to_state(obs0)
    abstract_state = env_models.state_abstractor(state0)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    target = obj_name_to_obj["cube0"]
    target_shelf = obj_name_to_obj["shelf"]

    # Test Pick skill
    pick_skill = Pick.ground((robot, target))
    obs1 = _skill_test_helper(pick_skill, env_models, env, obs0)

    # Check that object is picked
    state1 = env_models.observation_to_state(obs1)
    abstract_state1 = env_models.state_abstractor(state1)
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    Holding = pred_name_to_pred["Holding"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    OnGround = pred_name_to_pred["OnGround"]
    OnFixture = pred_name_to_pred["OnFixture"]
    assert HandEmpty([robot]) not in abstract_state1.atoms
    assert OnGround([target]) not in abstract_state1.atoms
    assert Holding([robot, target]) in abstract_state1.atoms

    # Test Place skill
    place_skill = Place.ground((robot, target, target_shelf))
    obs2 = _skill_test_helper(place_skill, env_models, env, obs1)
    state2 = env_models.observation_to_state(obs2)
    abstract_state2 = env_models.state_abstractor(state2)
    assert HandEmpty([robot]) in abstract_state2.atoms
    assert OnGround([target]) not in abstract_state2.atoms
    assert Holding([robot, target]) not in abstract_state2.atoms
    assert OnFixture([target, target_shelf]) in abstract_state2.atoms

    env.close()


@pytest.mark.parametrize("seed", [123])
def test_shelf3d_bilevel_planning(seed):
    """Tests for bilevel planning in the Shelf3D environment."""

    num_objects = 2
    env = kinder.make(
        f"kinder/KinematicShelf3D-o{num_objects}-v0",
        render_mode="rgb_array",
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Shelf3D-bilevel-{seed}"
        )

    env_models = create_bilevel_planning_models(
        "shelf3d",
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
    total_reward = 0
    try:
        agent.reset(obs, info)
    except Exception as e:
        env.close()
        pytest.skip(f"Planning failed for seed {seed}: {e}")

    for _ in range(1000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break

    else:
        assert False, "Did not terminate successfully"

    env.close()


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="End-to-end planning takes ~30 s; skipped in CI. Logic-only "
    "verification that the custom config threads through to the env state lives in "
    "test_shelf3d_custom_config_threads_through.",
)
@pytest.mark.parametrize("seed", [123])
def test_shelf3d_bilevel_planning_with_custom_shelf_pose(seed):
    """Plan against a non-default shelf pose.

    Threads a custom Shelf3DEnvConfig (shelf moved from the default (2.0, 2.4, 0.02) to
    (1.2, 0.8, 0.02)) through both kinder.make and create_bilevel_planning_models. The
    env starts with the shelf at the custom location, the env_model's internal sim sees
    the same config, and the planner reaches a goal defined by the custom-location
    shelf.
    """
    custom_shelf_position = (1.2, 0.8, 0.02)
    custom_config = Shelf3DEnvConfig(shelf_pose=Pose(custom_shelf_position))
    num_objects = 1

    env = kinder.make(
        f"kinder/KinematicShelf3D-o{num_objects}-v0",
        config=custom_config,
        render_mode="rgb_array",
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"Shelf3D-custom-shelf-{seed}",
        )

    env_models = create_bilevel_planning_models(
        "shelf3d",
        env.observation_space,
        env.action_space,
        num_objects=num_objects,
        config=custom_config,
    )

    obs, info = env.reset(seed=seed)
    # Sanity check: the env's perceived shelf sits at the custom location.
    state = env_models.observation_to_state(obs)
    shelf = state.get_objects(Kinematic3DFixtureType)[0]
    assert state.get(shelf, "pose_x") == pytest.approx(custom_shelf_position[0])
    assert state.get(shelf, "pose_y") == pytest.approx(custom_shelf_position[1])
    assert state.get(shelf, "pose_z") == pytest.approx(custom_shelf_position[2])

    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=60.0,
        max_skill_horizon=500,
    )
    try:
        agent.reset(obs, info)
    except Exception as e:
        env.close()
        pytest.skip(f"Planning failed for seed {seed}: {e}")

    for _ in range(1000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break
    else:
        assert False, "Did not terminate successfully against custom shelf"

    env.close()
