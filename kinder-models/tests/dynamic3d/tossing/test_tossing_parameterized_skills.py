"""Tests for ground parameterized skills."""

import gc
from pathlib import Path

import kinder
import numpy as np
import pybullet as p
from bilevel_planning.structs import GroundParameterizedController
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.envs import TidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoObjectTypeFeatures,
    MujocoTidyBotRobotObjectType,
)
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict
from spatialmath import SE2

from kinder_models.dynamic3d.shelf import parameterized_skills as shelf_skills
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
    PickCubeController,
    create_lifted_controllers,
    get_target_robot_pose_from_parameters,
)
from kinder_models.dynamic3d.tossing.toss_swing import (
    TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
    TOSS_MAX_VELOCITY,
    TOSS_RELEASE_ARM_CONFIGURATION,
    TOSS_SLICES_PER_CONTROL_STEP,
    TOSS_WINDUP_ARM_CONFIGURATION,
    toss_profile_limits,
)
from kinder_models.dynamic3d.utils import (
    _CONTROL_TIMESTEP,
    GRASP_CLOSE_THRESHOLD,
    MOVE_TO_TARGET_DISTANCE_BOUNDS,
    _trapezoidal_motion_profile,
)

kinder.register_all_environments()

_TEST_TASKS = Path(__file__).parent.parent.parent / "test_tasks"

# Size of PyBullet's fixed table of physics client slots. This is a scan bound, not a
# limit the test imposes: ids are handed out by first-free-slot scan, so a client can
# land on any id, including one left over from an earlier test in the same process.
# Scanning fewer slots would silently undercount and let a leak pass. The full scan
# costs about 0.2 ms. pybullet does not expose the value, so it is repeated here.
_MAX_PYBULLET_CLIENTS = 1024


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def _count_connected_pybullet_clients() -> int:
    """Helper to count the PyBullet physics clients that are currently connected."""
    num_connected = 0
    for client_id in range(_MAX_PYBULLET_CLIENTS):
        connection_info = p.getConnectionInfo(physicsClientId=client_id)
        if connection_info["isConnected"]:
            num_connected += 1
    return num_connected


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
            **{f"pos_arm_joint{i + 1}": v for i, v in enumerate(arm_joints)},
            "pos_gripper": gripper,
            # The eight Robotiq 2F-85 joints. `pos_gripper` is the commanded ctrl
            # value; these are where the fingers actually are, and zero is the open
            # hand this fixture describes. create_state_from_dict reads only the
            # features the robot type lists, so these keys are ignored by a
            # kindergarden whose type does not carry them yet.
            **{f"pos_gripper_joint{i + 1}": 0.0 for i in range(8)},
            "vel_base_x": 0.0,
            "vel_base_y": 0.0,
            "vel_base_rot": 0.0,
            **{f"vel_arm_joint{i + 1}": 0.0 for i in range(7)},
            "vel_gripper": 0.0,
            **{f"vel_gripper_joint{i + 1}": 0.0 for i in range(8)},
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


def test_get_target_robot_pose_from_parameters():
    """Tests for get_target_robot_pose_from_parameters()."""
    target = SE2(1.0, 0.0, 0.0)
    robot_pose = get_target_robot_pose_from_parameters(
        target, target_distance=1.0, target_rot=0.0
    )

    # Robot should be 1m behind the target, facing it
    assert np.isclose(robot_pose.x, 0.0)
    assert np.isclose(robot_pose.y, 0.0)
    assert np.isclose(robot_pose.theta(), 0.0)

    # With a rotation offset of 90 degrees (pi/2)
    robot_pose2 = get_target_robot_pose_from_parameters(
        target, target_distance=1.0, target_rot=np.pi / 2
    )
    assert np.isclose(robot_pose2.x, 1.0)
    assert np.isclose(robot_pose2.y, -1.0)
    assert np.isclose(robot_pose2.theta(), np.pi / 2)

    # Uncomment to debug.
    # import imageio.v2 as iio
    # from matplotlib import pyplot as plt
    # from prpl_utils.utils import fig2data

    # from kinder_models.dynamic3d.utils import get_overhead_object_se2_pose, \
    #     plot_overhead_scene

    # env = kinder.make("kinder/TidyBot3D-ground-o1-v0", render_mode="rgb_array")
    # assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    # obs, _ = env.reset(seed=123)
    # state = env.observation_space.devectorize(obs)
    # fig, ax = plot_overhead_scene(state, min_x=-1.5, max_x=1.5, min_y=-1.5, max_y=1.5)

    # target_distance = 0.75
    # target_object = state.get_object_from_name("cube1")
    # for target_rot in np.linspace(-np.pi, np.pi, num=24):
    #     target_object_pose = get_overhead_object_se2_pose(state, target_object)
    #     robot_pose = get_target_robot_pose_from_parameters(
    #         target_object_pose, target_distance, target_rot
    #     )
    #     th = robot_pose.theta()
    #     ax.arrow(
    #         robot_pose.x, robot_pose.y, 0.1 * np.cos(th), 0.1 * np.sin(th), width=0.01
    #     )

    # ax.set_title("Examples for get_target_robot_pose_from_parameters().")
    # plt.tight_layout()
    # plt.axis("equal")
    # img = fig2data(fig)
    # outfile = "get_target_robot_pose_from_parameters.png"
    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")


def test_move_to_target_controller_one_cube():
    """Test move-to-target controller in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / f"tidybot-ground-o{num_cubes}.json"),
        render_mode="rgb_array",
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

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


def test_move_to_target_arm_configuration():
    """Test move-arm-to-conf controller in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / f"tidybot-ground-o{num_cubes}.json"),
        render_mode="rgb_array",
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=124)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.zeros(7)
    params = target_conf

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


def test_repeated_grounding_does_not_leak_pybullet_clients():
    """Test that repeatedly grounding a controller does not leak PyBullet clients.

    The motion-planning controllers create a PyBulletSim the first time they are reset,
    and LiftedParameterizedController.ground() returns a new controller every call, so a
    planner or data-collection loop creates one PyBulletSim per skill execution. Those
    clients must be released when the controller is dropped.
    """
    # Create the environment.
    num_cubes = 1
    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / f"tidybot-ground-o{num_cubes}.json"),
        render_mode="rgb_array",
    )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=124)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)

    # Alternate between two configurations so that every execution moves the arm.
    target_confs = [
        np.zeros(7),
        np.deg2rad([0.0, -20.0, 180.0, -146.0, 0.0, -50.0, 90.0]),  # retract
    ]
    num_executions = 4
    steps_per_execution = []

    # Ground a fresh controller for each execution and drop it afterwards.
    clients_before = _count_connected_pybullet_clients()
    for execution_idx in range(num_executions):
        controller = lifted_controller.ground(object_parameters)
        params = target_confs[execution_idx % len(target_confs)]

        # Reset and execute the controller until it terminates.
        controller.reset(state, params)
        for num_steps in range(1, 201):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            next_state = env.observation_space.devectorize(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated():
                steps_per_execution.append(num_steps)
                break
        else:
            assert False, "Controller did not terminate"

        # Drop the last reference to the controller, which drops its PyBulletSim,
        # which runs the sim's weakref.finalize and disconnects the client.
        del controller
        # gc.collect() runs a full collection pass immediately, rather than whenever
        # CPython would next get round to it. It is belt-and-braces here: the sim is
        # freed by reference counting alone, which is immediate and does not need the
        # collector (verified by running this loop with gc disabled). It would only
        # start to matter if a future change put the sim in a reference cycle, which
        # refcounting cannot break. Calling it keeps the count below deterministic
        # either way.
        gc.collect()

    # Check that the executions above did real work, rather than terminating
    # immediately and making the client count below vacuously correct.
    assert len(steps_per_execution) == num_executions
    assert min(steps_per_execution) > 1

    clients_after = _count_connected_pybullet_clients()
    assert clients_after == clients_before, (
        f"Leaked {clients_after - clients_before} PyBullet clients over "
        f"{num_executions} skill executions"
    )

    env.close()


def test_move_to_target_arm_end_effector():
    """Test move-arm-to-end-effector controller in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / f"tidybot-ground-o{num_cubes}.json"),
        render_mode="rgb_array",
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=124)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    relative_target_end_effector_pose = np.array(
        [
            0.5,
            0,
            -0.1,
            1,
            0,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = relative_target_end_effector_pose

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


def test_close_gripper_controller():
    """Test close-gripper controller in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / f"tidybot-ground-o{num_cubes}.json"),
        render_mode="rgb_array",
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

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

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi / 2
    params = np.array([target_distance, target_rotation])

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

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["open_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
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


def test_pick_place_ground():
    """Test pick and place in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / f"tidybot-ground-o{num_cubes}.json"),
        render_mode="rgb_array",
        allow_state_access=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    _, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    arm_joints = np.deg2rad([0, -20, 180, -146, 0, -50, 90]).tolist()
    temp_state = _create_robot_state(arm_joints, 0.0, 0.8, 0.0, 0.0)
    env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
    state = (
        env.unwrapped._object_centric_env._get_object_centric_state()  # pylint: disable=protected-access
    )

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi
    params = np.array([target_distance, target_rotation])

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

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.39,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

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

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

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

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi / 2
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params, disable_collision_objects=["cube1"])
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

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.40,
            0.0,
            -0.3,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

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

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["open_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
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


def test_pick_place_shelf():
    """Test fake interface in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/Shelf3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    _, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    arm_joints = np.deg2rad([0, -20, 180, -146, 0, -50, 90]).tolist()
    temp_state = _create_robot_state(arm_joints, 0.0, -0.7, 0.0, 0.0)
    env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
    state = (
        env.unwrapped._object_centric_env._get_object_centric_state()  # pylint: disable=protected-access
    )

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = 0
    params = np.array([target_distance, target_rotation])

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

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.40,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

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

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

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

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cupboard_1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.9
    target_rotation = -np.pi / 2
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params, disable_collision_objects=["cube1"])
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

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.7,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

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

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["open_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
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


def test_velocity_tracking_mode():
    """Test pick and place in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / f"tidybot-ground-o{num_cubes}.json"),
        render_mode="rgb_array",
        allow_state_access=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    _, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    arm_joints = np.deg2rad([0, -20, 180, -146, 0, -50, 90]).tolist()
    temp_state = _create_robot_state(arm_joints, 0.0, 0.8, 0.0, 0.0)
    env.unwrapped._object_centric_env.set_state(temp_state)  # type: ignore # pylint: disable=protected-access
    state = (
        env.unwrapped._object_centric_env._get_object_centric_state()  # pylint: disable=protected-access
    )

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube1")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = np.pi
    params = np.array([target_distance, target_rotation])

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

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.39,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
        action = controller.step()
        action_18 = np.zeros(18)
        action_18[:10] = action[:10]
        action_18[10] = action[10]
        action_18[11:18] = 0.2 * action[3:10] * _ * np.ones(7)
        obs, _, _, _, _ = env.step(action_18)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_toss():
    """Test pick and place in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        scene_bg=False,
    )
    if MAKE_VIDEOS:
        env.unwrapped._object_centric_env.set_render_camera("task_view")  # type: ignore # pylint: disable=protected-access
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    state = env.observation_space.devectorize(obs)

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube_0")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 0.5
    target_rotation = 0
    params = np.array([target_distance, target_rotation])

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

    # create the move-arm controller.
    lifted_controller = controllers["move_arm_to_end_effector"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_end_effector_pose = np.array(
        [
            0.39,
            0.0,
            -0.35,
            0.707,
            0.707,
            0,
            0,
            0.0,
        ]
    )  # x, y, z, rw, rx, ry, rz, yaw for relative rotation of target object
    params = target_end_effector_pose

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

    # Create the controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["close_gripper"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)

    # Reset and execute the controller until it terminates.
    controller.reset(state)
    for _ in range(20):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # retract configuration
    params = target_conf

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

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("bin_0")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 1.35
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params, disable_collision_objects=["cube_0"])
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

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, 50, 180, -110, 0, -100, 90])  # pre toss
    params = target_conf

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

    # move the arm to the target configuration
    lifted_controller = controllers["toss"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, 20, 180, -35, 0, 25, 90])  # toss
    params = target_conf

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
    cube_position = [state.get(cube, "x"), state.get(cube, "y"), state.get(cube, "z")]
    cube_orientation = [
        state.get(cube, "qx"),
        state.get(cube, "qy"),
        state.get(cube, "qz"),
        state.get(cube, "qw"),
    ]
    robot_base_position = [
        state.get(robot, "pos_base_x"),
        state.get(robot, "pos_base_y"),
    ]
    distance = np.linalg.norm(
        np.array(cube_position[:2]) - np.array(robot_base_position[:2])
    )
    print("cube_position", cube_position)
    print("cube_orientation", cube_orientation)
    print("robot base position", robot_base_position)
    print("distance", distance)

    env.close()


def test_pick_ground_toss():
    """Test pick and place in ground environment with 1 cube."""
    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        scene_bg=False,
    )
    if MAKE_VIDEOS:
        env.unwrapped._object_centric_env.set_render_camera("task_view")  # type: ignore # pylint: disable=protected-access
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-ground-o{num_cubes}"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    state = env.observation_space.devectorize(obs)

    # Create the move-base controller.
    controllers = shelf_skills.create_lifted_controllers(env.action_space)

    # create the pick ground controller.
    lifted_controller = controllers["pick_shelf"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube_0")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.45, np.pi/4])

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

    # Create the move-base controller.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("bin_0")
    object_parameters = (robot, cube)
    controller = lifted_controller.ground(object_parameters)
    target_distance = 1.35
    target_rotation = 0.0
    params = np.array([target_distance, target_rotation])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params, disable_collision_objects=["cube_0"])
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

    # move the arm to the target configuration
    lifted_controller = controllers["move_arm_to_conf"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, 50, 180, -110, 0, -100, 90])  # pre toss
    params = target_conf

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

    # move the arm to the target configuration
    lifted_controller = controllers["toss"]
    robot = _get_robot_from_state(state)
    object_parameters = (robot,)
    controller = lifted_controller.ground(object_parameters)
    target_conf = np.deg2rad([0, 20, 180, -35, 0, 25, 90])  # toss
    params = target_conf

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
    cube_position = [state.get(cube, "x"), state.get(cube, "y"), state.get(cube, "z")]
    cube_orientation = [
        state.get(cube, "qx"),
        state.get(cube, "qy"),
        state.get(cube, "qz"),
        state.get(cube, "qw"),
    ]
    robot_base_position = [
        state.get(robot, "pos_base_x"),
        state.get(robot, "pos_base_y"),
    ]
    distance = np.linalg.norm(
        np.array(cube_position[:2]) - np.array(robot_base_position[:2])
    )
    print("cube_position", cube_position)
    print("cube_orientation", cube_orientation)
    print("robot base position", robot_base_position)
    print("distance", distance)

    env.close()


def test_toss_release_speed_raises_the_speed_the_profile_commands_at_release():
    """The point of the parameter: a higher setting must actually release faster.

    Compared below the ceiling, since TOSS_MAX_VELOCITY is both the default and the
    clamp. Asserted against the profile rather than a thrown cube: whether the arm
    tracks it is measured elsewhere.
    """
    total_dist = float(
        np.linalg.norm(TOSS_RELEASE_ARM_CONFIGURATION - TOSS_WINDUP_ARM_CONFIGURATION)
    )
    release_fraction = 0.46

    def commanded_release_speed(release_speed):
        max_vel, max_accel, max_decel = toss_profile_limits(release_speed)
        trajectory = _trapezoidal_motion_profile(
            total_dist,
            max_vel=max_vel,
            max_accel=max_accel,
            max_decel=max_decel,
            step_size=_CONTROL_TIMESTEP,
        )
        final = trajectory[-1]
        idx = int(np.argmax(trajectory / final >= release_fraction))
        return (trajectory[idx] - trajectory[idx - 1]) / _CONTROL_TIMESTEP

    default = commanded_release_speed(TOSS_MAX_VELOCITY)
    slower = commanded_release_speed(0.4 * TOSS_MAX_VELOCITY)
    assert default > 1.5 * slower


def _default_speed_trajectory():
    """The commanded distance profile of the shipped windup->release swing."""
    s_total = float(
        np.linalg.norm(TOSS_RELEASE_ARM_CONFIGURATION - TOSS_WINDUP_ARM_CONFIGURATION)
    )
    max_vel, max_accel, max_decel = toss_profile_limits(TOSS_MAX_VELOCITY)
    trajectory = _trapezoidal_motion_profile(
        s_total,
        max_vel=max_vel,
        max_accel=max_accel,
        max_decel=max_decel,
        step_size=_CONTROL_TIMESTEP,
    )
    return trajectory, s_total


def test_the_default_release_ms_falls_at_fraction_046_of_the_swing():
    """720 ms is where fraction 0.46 of the swing falls at the default speed.

    720 and not the nominal 723 because reset() profiles the motion-planned path, which
    moves the crossing 3 ms; a live rollout lands the cube 52 mm further with 723. The
    tolerance here holds on either path, so only that rollout distinguishes them.
    """
    assert TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS == 720
    trajectory, s_total = _default_speed_trajectory()
    step, slice_ = divmod(
        TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS, TOSS_SLICES_PER_CONTROL_STEP
    )
    # Linear interpolation between the two samples the release falls between, which is
    # how the commanded distance actually varies inside one held control period.
    below, above = float(trajectory[step]), float(trajectory[step + 1])
    covered = below + (above - below) * (slice_ / TOSS_SLICES_PER_CONTROL_STEP)
    assert abs(covered / s_total - 0.46) < 0.005


def test_a_release_ms_past_the_swing_never_opens_the_gripper():
    """The degenerate corner is reachable on purpose, not clamped away.

    A release past the swing's end leaves the cube still held when the controller
    terminates -- a real region of the parameter space a sweep must be able to reach.
    """
    trajectory, _ = _default_speed_trajectory()
    duration_ms = (len(trajectory) - 1) * TOSS_SLICES_PER_CONTROL_STEP
    assert duration_ms == 1700
    late_step, _ = divmod(2400, TOSS_SLICES_PER_CONTROL_STEP)
    assert late_step >= len(trajectory)


def test_toss_schedules_its_release_at_the_requested_millisecond():
    """End to end: exactly one action of a real toss is a control schedule.

    That schedule has to reach the simulator and land on the millisecond asked for
    rather than the next control-step boundary. Every other action is a plain (18,).
    """
    requested_ms = 812  # deliberately not a multiple of 100
    expected_step, expected_slice = divmod(requested_ms, TOSS_SLICES_PER_CONTROL_STEP)

    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array", scene_bg=False)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    shelf = shelf_skills.create_lifted_controllers(env.action_space)
    tossing = create_lifted_controllers(env.action_space)

    def _run(controller, params, **reset_kwargs):
        """Drive one controller to termination, returning the actions it emitted."""
        nonlocal state
        controller.reset(state, params, **reset_kwargs)
        emitted = []
        for _ in range(400):
            action = controller.step()
            emitted.append(np.array(action, copy=True))
            observation, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(observation)
            controller.observe(state)
            if controller.terminated():
                return emitted
        assert False, "Controller did not terminate"

    # The cube has to be *in* the gripper, or the release is a no-op: an empty hand
    # commands 0.0 both sides of the release.
    robot = _get_robot_from_state(state)
    pick = shelf["pick_shelf"].ground((robot, state.get_object_from_name("cube_0")))
    _run(pick, pick.sample_parameters(state, np.random.default_rng(123)))

    robot = _get_robot_from_state(state)
    move = tossing["move_to_target"].ground(
        (robot, state.get_object_from_name("bin_0"))
    )
    _run(move, np.array([1.35, 0.0]), disable_collision_objects=["cube_0"])

    robot = _get_robot_from_state(state)
    _run(tossing["move_arm_to_conf"].ground((robot,)), TOSS_WINDUP_ARM_CONFIGURATION)

    robot = _get_robot_from_state(state)
    toss = tossing["toss"].ground((robot,))
    actions = _run(
        toss, TOSS_RELEASE_ARM_CONFIGURATION, gripper_release_ms=requested_ms
    )

    scheduled = [i for i, action in enumerate(actions) if action.ndim == 2]
    assert scheduled == [expected_step]
    schedule = actions[expected_step]
    # A schedule covers the whole control period, so release is located by index.
    assert schedule.shape == (TOSS_SLICES_PER_CONTROL_STEP, 18)
    assert np.all(schedule[:expected_slice, 10] == schedule[0, 10])
    assert schedule[0, 10] > 0.0
    assert np.all(schedule[expected_slice:, 10] == 0.0)

    # Only the gripper column varies; the arm is commanded identically across slices.
    columns = [c for c in range(18) if c != 10]
    assert np.all(schedule[:, columns] == schedule[0, columns])

    # Everything before the release still holds the cube, everything after is open.
    assert all(action[10] > 0.0 for action in actions[:expected_step])
    assert all(action[10] == 0.0 for action in actions[expected_step + 1 :])

    env.close()


def test_pick_cube_takes_no_continuous_parameters():
    """PickCube is fully hardcoded, so sample_parameters is a 0-dim noop."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    controller = controllers["pick_cube"].ground((robot, cube, barrier))
    for seed in range(5):
        params = controller.sample_parameters(state, np.random.default_rng(seed))
        assert len(params) == 0
    env.close()


def test_pick_cube_stands_head_on_within_the_shelf_picks_reach():
    """No rotation offset from the cube's own facing, at a distance the shelf pick used
    to sample -- not a range, since the pick zone has nothing to route around."""
    distance = PickCubeController.TARGET_DISTANCE
    rot = PickCubeController.TARGET_ROTATION
    assert rot == 0.0
    assert (
        MOVE_TO_TARGET_DISTANCE_BOUNDS[0]
        <= distance
        <= MOVE_TO_TARGET_DISTANCE_BOUNDS[1]
    )


def test_pick_cube_resets_and_steps():
    """The controller is implemented: reset plans, and step drives the first phase."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    controller = controllers["pick_cube"].ground((robot, cube, barrier))
    params = controller.sample_parameters(state, np.random.default_rng(0))
    controller.reset(state, params)
    assert not controller.terminated()
    action = controller.step()
    assert np.asarray(action).shape == env.action_space.shape
    env.close()


def _pick_cube_arm_joint_names():
    return [f"pos_arm_joint{i}" for i in range(1, 8)]


def _pick_cube_arm_conf(state, robot):
    return np.array([state.get(robot, name) for name in _pick_cube_arm_joint_names()])


def _ground_pick_cube_on_seed_125():
    """A reset pick_cube controller on the canonical seed, plus its env and state."""
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=1)
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    controller = controllers["pick_cube"].ground((robot, cube, barrier))
    params = controller.sample_parameters(state, np.random.default_rng(0))
    controller.reset(state, params)
    return env, state, robot, controller


def test_pick_cube_hands_each_arm_plan_off_where_the_previous_one_ended():
    """Consecutive arm plans must be chained by the previous plan's own endpoint.

    Joints 1/3/5/7 are continuous, so a motion plan routes them the short way and lands
    on whichever 2*pi representative that reaches -- not necessarily the one inverse
    kinematics returned. Starting the next plan from the raw IK solution therefore puts
    its waypoints a full turn from where the arm physically is, and the arm unwinds to
    get there.
    """
    env, _, _, controller = _ground_pick_cube_on_seed_125()
    Phase = PickCubeController.PickCubeControllerPhase
    ordered = [
        Phase.MOVE_ARM_TO_HOVER_OVER_CUBE,
        Phase.MOVE_ARM_DOWN_AROUND_CUBE,
        Phase.LIFT_CUBE_TO_HOME,
    ]
    for earlier, later in zip(ordered[:-1], ordered[1:], strict=True):
        earlier_plan = controller.plans[earlier]
        later_plan = controller.plans[later]
        assert earlier_plan is not None and later_plan is not None
        handoff = np.abs(
            np.asarray(earlier_plan[-1])[:7] - np.asarray(later_plan[0])[:7]
        )
        assert np.max(handoff) < 1e-6, (
            f"{later.name} starts {np.max(handoff):.4f} rad from where "
            f"{earlier.name} ends"
        )
    env.close()


def test_pick_cube_never_unwinds_a_joint_by_a_whole_turn():
    """Executed travel must stay near planned travel, joint by joint.

    Tracking a waypoint costs a little more than the plan says -- the command is
    proportional, so the arm settles into each waypoint rather than arriving on it --
    but a joint that travels a further 2*pi is not settling, it is taking the long way
    round to a pose it is already standing in. Bounding the excess by pi separates the
    two.
    """
    env, state, robot, controller = _ground_pick_cube_on_seed_125()
    Phase = PickCubeController.PickCubeControllerPhase
    planned = np.zeros(7)
    for phase in (
        Phase.MOVE_ARM_TO_HOVER_OVER_CUBE,
        Phase.MOVE_ARM_DOWN_AROUND_CUBE,
        Phase.LIFT_CUBE_TO_HOME,
    ):
        plan = controller.plans[phase]
        assert plan is not None
        waypoints = np.array([np.asarray(w)[:7] for w in plan])
        planned += np.abs(np.diff(waypoints, axis=0)).sum(axis=0)

    confs = [_pick_cube_arm_conf(state, robot)]
    for _ in range(400):
        obs, _, _, _, _ = env.step(controller.step())
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
        confs.append(_pick_cube_arm_conf(state, robot))
        if controller.terminated():
            break
    else:
        assert False, "pick_cube did not terminate"

    executed = np.abs(np.diff(np.array(confs), axis=0)).sum(axis=0)
    excess = executed - planned
    worst = int(np.argmax(excess))
    assert np.max(excess) < np.pi, (
        f"joint {worst + 1} travelled {executed[worst]:.4f} rad against "
        f"{planned[worst]:.4f} rad planned"
    )
    env.close()


def test_move_to_toss_location_and_toss_samples_four_parameters():
    """Standoff, rotation, release speed and release millisecond, all in bounds."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    controller = controllers["move_to_toss_location_and_toss"].ground(
        (robot, cube, barrier)
    )
    draws = np.array(
        [
            controller.sample_parameters(state, np.random.default_rng(seed))
            for seed in range(50)
        ]
    )
    assert draws.shape == (50, 4)
    for column, (low, high) in enumerate(
        [
            MoveToTossLocationAndTossController.TARGET_DISTANCE_BOUNDS,
            MoveToTossLocationAndTossController.TARGET_ROTATION_BOUNDS,
            MoveToTossLocationAndTossController.SPEED_BOUNDS,
            MoveToTossLocationAndTossController.RELEASE_MS_BOUNDS,
        ]
    ):
        assert draws[:, column].min() >= low
        assert draws[:, column].max() <= high
        # A sampler, not a constant, in every component.
        assert draws[:, column].min() < draws[:, column].max()
    env.close()


def test_open_gripper_commands_open_until_the_gripper_reads_open():
    """The controller predates this branch and had no test of its own."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    controller = controllers["open_gripper"].ground((robot,))

    shut = state.copy()
    shut.set(robot, "pos_gripper", 1.0)
    controller.reset(shut)
    assert not controller.terminated()
    action = controller.step()
    assert action.shape == (11,)
    assert action[-1] == 0

    opened = state.copy()
    opened.set(robot, "pos_gripper", 0.0)
    controller.observe(opened)
    assert controller.terminated()
    env.close()


def test_no_op_controller_terminates_immediately_and_holds_the_gripper_command():
    """A no-op dispatches one all-zero-motion action and is already terminated --
    except the gripper index, which must re-issue the robot's own current command
    rather than zero, since zero means "open" (OpenGripperController.step()) and
    would drop anything currently held."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    controller = controllers["no_op"].ground((robot,))

    open_gripper = state.copy()
    open_gripper.set(robot, "pos_gripper", 0.0)
    controller.reset(open_gripper)
    assert controller.terminated()
    action = controller.step()
    assert action.shape == (11,)
    assert np.all(action == 0)

    closed_gripper = state.copy()
    closed_gripper.set(robot, "pos_gripper", 1.0)
    controller.reset(closed_gripper)
    assert controller.terminated()
    action = controller.step()
    assert action[:-1].tolist() == [0.0] * (action.shape[0] - 1)
    assert action[-1] == GRASP_CLOSE_THRESHOLD
    env.close()


def test_move_to_target_from_other_target_drives_to_the_target_it_names():
    """Three object parameters, but only the target governs where the base goes: the
    third exists so an operator can say where the robot came from."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    target_bin = state.get_object_from_name("bin_0")
    controller = controllers["move_to_target_from_other_target"].ground(
        (robot, cube, target_bin)
    )
    controller.reset(state, np.array([0.5, 0.0]))
    for _ in range(2000):
        if controller.terminated():
            break
        obs_, _, _, _, _ = env.step(controller.step())
        state = env.observation_space.devectorize(obs_)
        controller.observe(state)
    assert controller.terminated()
    achieved = np.hypot(
        state.get(cube, "x") - state.get(robot, "pos_base_x"),
        state.get(cube, "y") - state.get(robot, "pos_base_y"),
    )
    assert abs(achieved - 0.5) < 0.1
    env.close()


def test_move_to_toss_location_and_toss_holds_no_sub_controllers():
    """One flat controller over phase flags, as pick_shelf is, rather than a controller
    that drives controllers it owns."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    controller = controllers["move_to_toss_location_and_toss"].ground(
        (robot, cube, barrier)
    )
    nested = [
        name
        for name, value in vars(controller).items()
        if isinstance(value, GroundParameterizedController)
    ]
    assert not nested, f"holds sub-controllers: {nested}"
    env.close()


def test_move_to_toss_location_and_toss_plans_every_phase_in_reset():
    """As pick_shelf does: a planning failure surfaces from reset rather than from the
    middle of a throw, where a refiner would see a crash instead of a rejected
    sample."""
    num_cubes = 1
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=num_cubes
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    controllers = create_lifted_controllers(env.action_space)
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    controller = controllers["move_to_toss_location_and_toss"].ground(
        (robot, cube, barrier)
    )
    controller.reset(state, np.array([1.30, 0.0, TOSS_MAX_VELOCITY, 720.0]))
    # Every phase is planned before the first action is asked for.
    # pylint: disable-next=protected-access
    assert controller._current_base_motion_plan is not None
    assert len(controller._windup_trajectory) > 0  # pylint: disable=protected-access
    assert controller._swing is not None  # pylint: disable=protected-access
    env.close()
