"""Tests for DynPushPullHook2D parameterized skills."""

import kinder
import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic2d.dynpushpullhook2d.parameterized_skills import (
    create_lifted_controllers,
)

kinder.register_all_environments()


def test_grasp_hook_controller():
    """Test grasp-hook controller in DynPushPullHook2D environment."""

    # Create the environment.
    num_obstructions = 0
    env = kinder.make(
        f"kinder/DynPushPullHook2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"DynPushPullHook2D-o{num_obstructions}-grasp-hook",
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["grasp_hook"]
    robot = state.get_object_from_name("robot")
    hook = state.get_object_from_name("hook")
    object_parameters = (robot, hook)
    controller = lifted_controller.ground(object_parameters)

    # Sample parameters (arm_length).
    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(500):
        try:
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        except TrajectorySamplingFailure:
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_grasp_hook_controller_with_obstructions():
    """Test grasp-hook controller with obstructions present."""

    # Create the environment with obstructions.
    num_obstructions = 1
    env = kinder.make(
        f"kinder/DynPushPullHook2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"DynPushPullHook2D-o{num_obstructions}-grasp-hook",
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=42)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["grasp_hook"]
    robot = state.get_object_from_name("robot")
    hook = state.get_object_from_name("hook")
    object_parameters = (robot, hook)
    controller = lifted_controller.ground(object_parameters)

    # Sample parameters (arm_length).
    rng = np.random.default_rng(42)
    params = controller.sample_parameters(state, rng)

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(500):
        try:
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                break
        except TrajectorySamplingFailure:
            break
    else:
        assert False, "Controller did not terminate"

    env.close()
