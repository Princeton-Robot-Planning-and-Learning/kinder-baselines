"""Tests for sweep3D parameterized skills."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.sweep3D.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.utils import PyBulletSim

kinder.register_all_environments()


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_open_drawer():
    """Test open drawer."""

    # Create the environment.
    num_cubes = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(11))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["open_drawer"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    # params = controller.sample_parameters(state, np.random.default_rng(123))
    params = np.array([0.7, -np.pi])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_wiper():
    """Test pick wiper."""

    # Create the environment.
    num_cubes = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(11))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["pick_wiper"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    # params = controller.sample_parameters(state, np.random.default_rng(123))
    params = np.array([0.7, -np.pi])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_open_drawer_pick_sweep_wiper():
    """Test open drawer, pick and sweep wiper."""

    # Create the environment.
    num_cubes = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(11))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["open_drawer"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.7, -np.pi])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the pick ground controller.
    lifted_controller = controllers["pick_wiper"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.7, -np.pi])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the place ground controller.
    lifted_controller = controllers["sweep"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.55, -np.pi])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()