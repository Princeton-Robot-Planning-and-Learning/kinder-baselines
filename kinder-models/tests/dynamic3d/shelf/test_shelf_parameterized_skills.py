"""Tests for ground parameterized skills."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoObjectTypeFeatures,
    MujocoTidyBotRobotObjectType,
)
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict

from kinder_models.dynamic3d.shelf.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.utils import PyBulletSim

kinder.register_all_environments()


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def _create_robot_state(
    arm_joints: list[float],
    gripper: float,
    base_x: float,
    base_y: float,
    base_theta: float,
) -> ObjectCentricState:
    """Create an ObjectCentricState with the given robot and placeholder cube."""
    robot = Object("robot_0", MujocoTidyBotRobotObjectType)
    cube = Object("cube1", MujocoMovableObjectType)
    state_dict: dict[Object, dict[str, float]] = {
        robot: {
            "pos_base_x": base_x,
            "pos_base_y": base_y,
            "pos_base_rot": base_theta,
            **{f"pos_arm_joint{i+1}": v for i, v in enumerate(arm_joints)},
            "pos_gripper": gripper,
            "vel_base_x": 0.0,
            "vel_base_y": 0.0,
            "vel_base_rot": 0.0,
            **{f"vel_arm_joint{i+1}": 0.0 for i in range(7)},
            "vel_gripper": 0.0,
        },
        cube: {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "bb_x": 0.03,
            "bb_y": 0.03,
            "bb_z": 0.03,
        },
    }
    return create_state_from_dict(state_dict, MujocoObjectTypeFeatures)


def test_pick_place_skill():
    """Test pick and place skill in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(f"kinder/Shelf3D-o{num_cubes}-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["pick_shelf"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.55, 0.0])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
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
    lifted_controller = controllers["place_shelf"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([1.02, 0.0, -1.5707964])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
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


def test_pick_place_two_cubes_skill():
    """Test pick and place skill in ground environment with 1 cube."""

    # Create the environment.
    num_cubes = 2
    env = kinder.make(f"kinder/Shelf3D-o{num_cubes}-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)
    # create the pick ground controller.
    lifted_controller = controllers["pick_shelf"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    # target_distance = 0.6
    # target_rotation = 0.0
    # params = np.array([target_distance, target_rotation])
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
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
    lifted_controller = controllers["place_shelf"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    # target_distance = 0.85
    # offset = 0.0
    # target_rotation = -np.pi / 2
    # params = np.array([target_distance, offset, target_rotation])
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(800):
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
    lifted_controller = controllers["pick_shelf"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube2")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    # target_distance = 0.6
    # target_rotation = 0.0
    # params = np.array([target_distance, target_rotation])
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
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
    lifted_controller = controllers["place_shelf"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube2")
    cupboard = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube, cupboard)
    controller = lifted_controller.ground(object_parameters)
    # target_distance = 0.92
    # offset = 0.0
    # target_rotation = -np.pi / 2
    # params = np.array([target_distance, offset, target_rotation])
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(400):
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
