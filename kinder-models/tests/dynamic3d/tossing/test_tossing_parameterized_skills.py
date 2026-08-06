"""Tests for ground parameterized skills."""

import gc
from pathlib import Path

import kinder
import numpy as np
import pybullet as p
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.envs import TidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoObjectTypeFeatures,
    MujocoTidyBotRobotObjectType,
)
from prpl_utils.utils import get_signed_angle_distance
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict
from spatialmath import SE2

from kinder_models.dynamic3d.shelf.parameterized_skills import (
    create_lifted_controllers as shelf_create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    create_lifted_controllers,
    get_target_robot_pose_from_parameters,
)
from kinder_models.dynamic3d.utils import (
    MOVE_TO_TARGET_DISTANCE_BOUNDS,
    MOVE_TO_TARGET_ROT_BOUNDS,
    WAYPOINT_TOL,
    PyBulletSim,
    get_overhead_object_se2_pose,
    run_base_motion_planning,
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
    controllers = shelf_create_lifted_controllers(env.action_space)

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


def test_move_to_throw_pose_controller(monkeypatch):
    """Test the throw-pose controller in the tossing environment with 1 cube."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        scene_bg=False,
    )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Ground the controller on (robot, target, held).
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_throw_pose"]
    assert len(lifted_controller.variables) == 3
    robot = _get_robot_from_state(state)
    target = state.get_object_from_name("bin_0")
    held = state.get_object_from_name("cube_0")
    controller = lifted_controller.ground((robot, target, held))

    # MoveToTargetGroundController returns the constant [0.5, 0.0], so a planner can
    # only ever try one base pose. This controller samples. Note that being inside the
    # bounds is not on its own evidence of that: the constant it replaces is inside
    # them too, so what distinguishes the two is that both components vary.
    rng = np.random.default_rng(123)
    draws = np.array([controller.sample_parameters(state, rng) for _ in range(20)])
    assert np.all(draws[:, 0] >= MOVE_TO_TARGET_DISTANCE_BOUNDS[0])
    assert np.all(draws[:, 0] <= MOVE_TO_TARGET_DISTANCE_BOUNDS[1])
    assert np.all(draws[:, 1] >= MOVE_TO_TARGET_ROT_BOUNDS[0])
    assert np.all(draws[:, 1] <= MOVE_TO_TARGET_ROT_BOUNDS[1])
    assert draws[:, 0].min() < draws[:, 0].max()
    assert draws[:, 1].min() < draws[:, 1].max()
    params = draws[0]

    # Record what the controller asks the base planner to ignore. Asserting on the
    # resulting plan would prove nothing: base collision checking is currently
    # commented out in run_base_motion_planning (dynamic3d/utils.py), so every plan
    # succeeds whether or not the held object is excluded. What this pins is that the
    # controller passes the held object down, which is what will matter when that
    # checking is switched back on.
    recorded_disabled: list[list[str] | None] = []

    def _recording_run_base_motion_planning(**kwargs):
        recorded_disabled.append(kwargs.get("disable_collision_objects"))
        return run_base_motion_planning(**kwargs)

    monkeypatch.setattr(
        "kinder_models.dynamic3d.tossing.parameterized_skills"
        ".run_base_motion_planning",
        _recording_run_base_motion_planning,
    )

    # Reset and execute the controller until it terminates. The held object is not
    # passed as a collision object, so this must plan without one being given.
    controller.reset(state, params)
    assert recorded_disabled == [["cube_0"]]
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

    # The base ended up the sampled distance from the target, facing it.
    robot = _get_robot_from_state(state)
    target_pose = get_overhead_object_se2_pose(state, target)
    expected_pose = get_target_robot_pose_from_parameters(
        target_pose, params[0], params[1]
    )
    assert np.isclose(
        state.get(robot, "pos_base_x"), expected_pose.x, atol=WAYPOINT_TOL
    )
    assert np.isclose(
        state.get(robot, "pos_base_y"), expected_pose.y, atol=WAYPOINT_TOL
    )
    # The heading matters as much as the position: the sampled rotation is half of what
    # this controller newly varies, and a throw is released along the base's heading.
    assert np.isclose(
        get_signed_angle_distance(
            state.get(robot, "pos_base_rot"), expected_pose.theta()
        ),
        0.0,
        atol=WAYPOINT_TOL,
    )

    env.close()


def test_toss_from_windup_matches_split_controllers():
    """Test that the composed toss emits the same actions as the two-controller sequence
    it replaces."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        scene_bg=False,
    )
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    # The two demonstrated arm configurations, exactly as test_pick_toss uses them.
    windup_conf = np.deg2rad([0, 50, 180, -110, 0, -100, 90])  # pre toss
    toss_conf = np.deg2rad([0, 20, 180, -35, 0, 25, 90])  # toss

    def _run_sequence(
        steps: list[tuple[str, np.ndarray]],
        controllers: dict,
    ) -> list[list[np.ndarray]]:
        """Run controllers back to back from a fresh reset, recording every action.

        The actions are returned grouped by controller, so that a caller can tell how
        much work each one did rather than only how much they did between them.
        """
        obs, _ = env.reset(seed=125)
        state = env.observation_space.devectorize(obs)
        actions_per_controller: list[list[np.ndarray]] = []
        for controller_name, params in steps:
            robot = _get_robot_from_state(state)
            controller = controllers[controller_name].ground((robot,))
            controller.reset(state, params)
            actions: list[np.ndarray] = []
            for _ in range(400):
                action = controller.step()
                actions.append(np.array(action, dtype=np.float32, copy=True))
                obs, _, _, _, _ = env.step(action)
                state = env.observation_space.devectorize(obs)
                controller.observe(state)
                if controller.terminated():
                    break
            else:
                assert False, "Controller did not terminate"
            actions_per_controller.append(actions)
        return actions_per_controller

    controllers = create_lifted_controllers(env.action_space)
    split_phases = _run_sequence(
        [("move_arm_to_conf", windup_conf), ("toss", toss_conf)], controllers
    )
    composed_phases = _run_sequence(
        [("toss_from_windup", np.array([windup_conf, toss_conf]))], controllers
    )

    # Both halves did real work, so the comparison below is not vacuous. Measured
    # locally: 16 windup actions then 18 toss actions, 34 in total. The exact counts
    # are not asserted because they follow from a motion plan, but a phase that
    # collapsed to a step or two would no longer be a windup or a toss.
    assert len(split_phases) == 2
    assert min(len(phase) for phase in split_phases) >= 5

    split_actions = [action for phase in split_phases for action in phase]
    assert len(composed_phases) == 1
    composed_actions = composed_phases[0]
    assert len(composed_actions) == len(split_actions)
    for composed_action, split_action in zip(
        composed_actions, split_actions, strict=True
    ):
        assert np.array_equal(composed_action, split_action)

    # Handing the factory a PyBullet sim to share must not change what comes out of
    # it. Without this, the sim is the client every sub-controller plans in, so a
    # mistake there would silently change the plans rather than fail loudly.
    obs, _ = env.reset(seed=125)
    initial_state = env.observation_space.devectorize(obs)
    shared_sim = PyBulletSim(initial_state)
    clients_before = _count_connected_pybullet_clients()
    shared_controllers = create_lifted_controllers(
        env.action_space, pybullet_sim=shared_sim
    )
    shared_phases = _run_sequence(
        [("toss_from_windup", np.array([windup_conf, toss_conf]))], shared_controllers
    )
    assert len(shared_phases) == 1
    shared_actions = shared_phases[0]
    assert len(shared_actions) == len(split_actions)
    for shared_action, split_action in zip(shared_actions, split_actions, strict=True):
        assert np.array_equal(shared_action, split_action)

    # The shared sim is still the caller's to reuse: no sub-controller disconnected it
    # out from under the caller, and nothing was leaked. Note the count check alone is
    # weak evidence of sharing, since PyBulletSim's finalizer releases a private client
    # too once the controller is dropped; what shows the shared sim was actually planned
    # in is that the actions above came out identical.
    assert p.getConnectionInfo(physicsClientId=shared_sim.physics_client_id)[
        "isConnected"
    ]
    assert _count_connected_pybullet_clients() == clients_before

    env.close()


def test_toss_from_windup_samples_the_demonstrated_confs():
    """Test that the composed toss samples the two demonstrated confs,
    deterministically."""

    # Create the environment.
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0",
        render_mode="rgb_array",
        scene_bg=False,
    )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    controllers = create_lifted_controllers(env.action_space)
    robot = _get_robot_from_state(state)
    controller = controllers["toss_from_windup"].ground((robot,))

    params = controller.sample_parameters(state, np.random.default_rng(123))
    other_params = controller.sample_parameters(state, np.random.default_rng(456))
    assert np.array_equal(params, other_params)
    assert np.allclose(params[0], np.deg2rad([0, 50, 180, -110, 0, -100, 90]))
    assert np.allclose(params[1], np.deg2rad([0, 20, 180, -35, 0, 25, 90]))

    env.close()
