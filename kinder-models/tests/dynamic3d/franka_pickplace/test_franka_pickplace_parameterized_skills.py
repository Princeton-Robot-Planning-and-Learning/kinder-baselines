"""Tests for the Franka FR3 pick-and-place parameterized skills."""

import pickle
import time
from pathlib import Path

import kinder
import numpy as np
from conftest import MAKE_VIDEOS, SAVE_DEMOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoFR3RobotObjectType
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.franka_pickplace.parameterized_skills import (
    create_lifted_controllers,
)

kinder.register_all_environments()


def _save_demo(
    env_id: str,
    seed: int,
    observations: list,
    actions: list,
    rewards: list,
    terminated: bool,
    truncated: bool,
    demos_dir: str = "./demos",
) -> Path:
    """Save a demo pickle in the standard kindergarden demo format."""
    env_short_name = env_id.removeprefix("kinder/").removesuffix("-v0")
    timestamp = int(time.time())
    save_dir = Path(demos_dir) / env_short_name / str(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{timestamp}.p"
    demo = {
        "env_id": env_id,
        "timestamp": timestamp,
        "seed": seed,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
    }
    with open(save_path, "wb") as f:
        pickle.dump(demo, f)
    print(f"\nSaved demo to {save_path}")
    return save_path


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoFR3RobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_franka_pick_place_skill():
    """Pick and place one cube into the goal region on the desk."""
    env_id = "kinder/FrankaPickPlace3D-o1-v0"
    seed = 123
    env = kinder.make(env_id, render_mode="rgb_array")
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix="FrankaPickPlace3D-o1-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=seed)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    demo_observations = [obs]
    demo_actions: list = []
    demo_rewards: list = []
    demo_terminated = False
    demo_truncated = False

    controllers = create_lifted_controllers(env.action_space)

    # Create the pick ground controller.
    lifted_controller = controllers["pick"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    controller = lifted_controller.ground((robot, cube))
    params = controller.sample_parameters(state, np.random.default_rng(seed))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        demo_observations.append(obs)
        demo_actions.append(action)
        demo_rewards.append(reward)
        demo_terminated = terminated
        demo_truncated = truncated
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Pick controller did not terminate"

    # The cube must have been lifted off the desk.
    cube = state.get_object_from_name("cube1")
    assert state.get(cube, "z") > 0.85, "Cube was not lifted"

    # Create the place ground controller.
    lifted_controller = controllers["place"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    desk = state.get_object_from_name("desk_1")
    controller = lifted_controller.ground((robot, cube, desk))
    params = controller.sample_parameters(state, np.random.default_rng(seed))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
        action = controller.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        demo_observations.append(obs)
        demo_actions.append(action)
        demo_rewards.append(reward)
        demo_terminated = terminated
        demo_truncated = truncated
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Place controller did not terminate"

    # The goal (cube in the goal region) must have been reached.
    assert demo_terminated, "Environment did not terminate with the goal reached"

    env.close()

    if SAVE_DEMOS:
        _save_demo(
            env_id=env_id,
            seed=seed,
            observations=demo_observations,
            actions=demo_actions,
            rewards=demo_rewards,
            terminated=demo_terminated,
            truncated=demo_truncated,
        )
