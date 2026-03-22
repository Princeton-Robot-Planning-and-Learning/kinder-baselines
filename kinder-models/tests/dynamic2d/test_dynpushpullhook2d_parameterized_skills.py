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
# from imageio.v2 import imwrite

# imwrite("test.png", env.render())


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
    obs, _ = env.reset(seed=0)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space, env.unwrapped._object_centric_env.initial_constant_state)
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

    assert state.get(hook, "held"), "Hook should be held at the end of the controller execution."
    env.close()


def test_hook_controller():
    """Test hook controller: grasp hook then pull target block down."""

    # Create the environment (no obstructions for clean test).
    num_obstructions = 0
    env = kinder.make(
        f"kinder/DynPushPullHook2D-o{num_obstructions}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"DynPushPullHook2D-o{num_obstructions}-hook",
        )

    # Reset the environment and get the initial state.
    init_obs, _ = env.reset(seed=0)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(init_obs)

    controllers = create_lifted_controllers(env.action_space, env.unwrapped._object_centric_env.initial_constant_state)
    robot = state.get_object_from_name("robot")
    hook = state.get_object_from_name("hook")
    target_block = state.get_object_from_name("target_block")
    new_block_x = state.get(target_block, "x") + 2.3
    new_block_y = state.get(target_block, "y") - 0.5

    new_hook_x = state.get(hook, "x") - 0.2
    new_state = state.copy()
    new_state.set(target_block, "x", new_block_x)
    new_state.set(target_block, "y", new_block_y)
    new_state.set(hook, "x", new_hook_x)

    obs, _ = env.reset(options={"init_state": new_state})
    rng = np.random.default_rng(123)

    # Phase 1: Grasp the hook.
    grasp_ctrl = controllers["grasp_hook"].ground((robot, hook))
    params = grasp_ctrl.sample_parameters(new_state, rng)
    grasp_ctrl.reset(new_state, params)
    for _ in range(500):
        try:
            action = grasp_ctrl.step()
            obs, _, _, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            grasp_ctrl.observe(next_state)
            state = next_state
            if grasp_ctrl.terminated():
                break
        except TrajectorySamplingFailure:
            break
    else:
        assert False, "Grasp controller did not terminate"
    assert state.get(hook, "held"), "Hook should be held before hooking"

    # Phase 2: Use the hook to pull the target block.
    hook_ctrl = controllers["hook"].ground((robot, hook, target_block))
    params = hook_ctrl.sample_parameters(state, rng)
    hook_ctrl.reset(state, params)
    for _ in range(2000):
        try:
            action = hook_ctrl.step()
            obs, _, terminated, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            hook_ctrl.observe(next_state)
            state = next_state
            if hook_ctrl.terminated():
                break
        except TrajectorySamplingFailure:
            break
    else:
        assert False, "Hook controller did not terminate"

    env.close()

