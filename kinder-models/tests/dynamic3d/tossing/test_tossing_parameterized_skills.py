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
from prpl_utils.utils import get_signed_angle_distance
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace
from relational_structs.utils import create_state_from_dict
from spatialmath import SE2

import kinder_models.dynamic3d.tossing.parameterized_skills
from kinder_models.dynamic3d.shelf import parameterized_skills as shelf_skills
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    CUBE_ROTATION_SYMMETRIES,
    PICK_STANDOFF_LADDER,
    THROW_POSE_TOLERANCE,
    TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
    TOSS_MAX_ACCELERATION,
    TOSS_MAX_DECELERATION,
    TOSS_MAX_VELOCITY,
    TOSS_RELEASE_ARM_CONFIGURATION,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SLICES_PER_CONTROL_STEP,
    TOSS_SPEED_BOUNDS,
    TOSS_TARGET_DISTANCE_BOUNDS,
    TOSS_TARGET_ROTATION_BOUNDS,
    TOSS_WINDUP_ARM_CONFIGURATION,
    canonical_upright_rotation,
    create_lifted_controllers,
    get_target_robot_pose_from_parameters,
    plan_toss_swing,
    toss_profile_limits,
    toss_swing_action,
    upright_grasp_rotations,
)
from kinder_models.dynamic3d.utils import (
    _CONTROL_TIMESTEP,
    MINIMUM_HOLDING_HEIGHT,
    MOVE_TO_TARGET_DISTANCE_BOUNDS,
    MOVE_TO_TARGET_ROT_BOUNDS,
    WAYPOINT_TOLERANCE,
    PyBulletSim,
    _trapezoidal_motion_profile,
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
            **{f"pos_arm_joint{i + 1}": v for i, v in enumerate(arm_joints)},
            "pos_gripper": gripper,
            "vel_base_x": 0.0,
            "vel_base_y": 0.0,
            "vel_base_rot": 0.0,
            **{f"vel_arm_joint{i + 1}": 0.0 for i in range(7)},
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


def test_move_to_throw_pose_controller():
    """Test the throw-pose controller in the tossing environment with 1 cube."""

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
            env, "unit_test_videos", name_prefix=f"TidyBot3D-throw-pose-o{num_cubes}"
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

    # MoveToTargetGroundController returns the constant [0.5, 0.0]; this controller
    # samples, and both components must vary.
    rng = np.random.default_rng(123)
    draws = np.array([controller.sample_parameters(state, rng) for _ in range(20)])
    assert draws[:, 0].min() < draws[:, 0].max()
    assert draws[:, 1].min() < draws[:, 1].max()
    params = draws[0]

    # Record what the controller asks the base planner to ignore: the argument passed
    # down is the thing under test, where a plan could succeed for unrelated reasons.
    recorded_disabled: list[list[str] | None] = []

    def _recording_run_base_motion_planning(**kwargs):
        """Record the call, then delegate to the real planner."""
        recorded_disabled.append(kwargs.get("disable_collision_objects"))
        return run_base_motion_planning(**kwargs)

    # Swapped explicitly rather than with pytest's monkeypatch fixture: no test in
    # kinder-models uses a fixture.
    skills_module = kinder_models.dynamic3d.tossing.parameterized_skills
    original_run_base_motion_planning = skills_module.run_base_motion_planning
    skills_module.run_base_motion_planning = _recording_run_base_motion_planning

    # The held object is not passed, so the controller must supply the exclusion.
    try:
        controller.reset(state, params)
    finally:
        skills_module.run_base_motion_planning = original_run_base_motion_planning
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
        state.get(robot, "pos_base_x"), expected_pose.x, atol=WAYPOINT_TOLERANCE
    )
    assert np.isclose(
        state.get(robot, "pos_base_y"), expected_pose.y, atol=WAYPOINT_TOLERANCE
    )
    # The heading matters as much as the position: a throw is released along it.
    assert np.isclose(
        get_signed_angle_distance(
            state.get(robot, "pos_base_rot"), expected_pose.theta()
        ),
        0.0,
        atol=WAYPOINT_TOLERANCE,
    )

    env.close()


def test_toss_from_windup_matches_split_controllers():
    """Test the composed toss emits the actions of move_arm_to_conf then toss."""

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
            env, "unit_test_videos", name_prefix=f"TidyBot3D-toss-windup-o{num_cubes}"
        )
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)

    # The two demonstrated arm configurations, exactly as test_pick_toss uses them.
    windup_conf = np.deg2rad([0, 50, 180, -110, 0, -100, 90])  # pre toss
    toss_conf = np.deg2rad([0, 20, 180, -35, 0, 25, 90])  # toss

    def _run_sequence(
        steps: list[tuple[str, np.ndarray]],
        controllers: dict,
    ) -> list[list[np.ndarray]]:
        """Run controllers back to back from a fresh reset, grouped by controller."""
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

    # Both halves did real work, so the comparison below is not vacuous. Not asserted
    # exactly, since the counts follow from a motion plan.
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

    # Handing the factory a PyBullet sim to share must not change what comes out of it:
    # that sim is the client every sub-controller plans in.
    obs, _ = env.reset(seed=125)
    initial_state = env.observation_space.devectorize(obs)
    shared_sim = PyBulletSim(initial_state)
    shared_controllers = create_lifted_controllers(
        env.action_space, pybullet_sim=shared_sim
    )
    # Counted after the shared sim exists but before any controller has run, so the
    # baseline is not inflated by whatever the earlier composed run held.
    gc.collect()
    clients_before = _count_connected_pybullet_clients()
    shared_phases = _run_sequence(
        [("toss_from_windup", np.array([windup_conf, toss_conf]))], shared_controllers
    )
    assert len(shared_phases) == 1
    shared_actions = shared_phases[0]
    assert len(shared_actions) == len(split_actions)
    for shared_action, split_action in zip(shared_actions, split_actions, strict=True):
        assert np.array_equal(shared_action, split_action)

    # The shared sim is still the caller's to reuse, and nothing was leaked. The count
    # alone is weak evidence of sharing; the identical actions above are the strong one.
    assert p.getConnectionInfo(physicsClientId=shared_sim.physics_client_id)[
        "isConnected"
    ]
    gc.collect()
    assert _count_connected_pybullet_clients() == clients_before

    env.close()


def test_toss_from_windup_samples_the_demonstrated_confs():
    """Test the composed toss samples the two demonstrated confs, ignoring the rng."""

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


def test_move_to_throw_pose_samples_a_throwable_standoff():
    """The sampled standoff must be one a throw can actually be thrown from.

    Asserted against the interval this sampler really draws from, not a wider envelope,
    so the bound is tight enough to fail if the sampler drifts.
    """
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0", render_mode="rgb_array", scene_bg=False
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_throw_pose"]
    robot = _get_robot_from_state(state)
    target = state.get_object_from_name("bin_0")
    held = state.get_object_from_name("cube_0")
    controller = lifted_controller.ground((robot, target, held))

    rng = np.random.default_rng(123)
    draws = np.array([controller.sample_parameters(state, rng) for _ in range(50)])

    # Measured on the pose the parameters imply, not the sampled distance: the two agree
    # only because bin_init_region pins the bin's yaw at 0.
    target_pose = get_overhead_object_se2_pose(state, target)
    standoff = np.array(
        [
            target_pose.x
            - get_target_robot_pose_from_parameters(target_pose, distance, rot).x
            for distance, rot in draws
        ]
    )
    # Rotation shrinks the x-component, so the lower bound is the furthest the widest
    # rotation can pull the nearest draw in; the upper bound is reached at rot = 0.
    low, high = TOSS_TARGET_DISTANCE_BOUNDS
    low *= np.cos(max(abs(bound) for bound in TOSS_TARGET_ROTATION_BOUNDS))
    assert np.all(standoff >= low), standoff.min()
    assert np.all(standoff <= high), standoff.max()
    # Still a sampler, not a constant.
    assert draws[:, 0].min() < draws[:, 0].max()

    env.close()


def test_move_to_throw_pose_samples_a_pose_on_the_bin_axis():
    """Every sampled pose must be on the bin axis, in y as well as in x.

    A standoff test alone is satisfied by a whole ring of positions. Asserted on the
    offset the parameters imply, not the bounds, so it survives a retune.
    """
    num_cubes = 1
    env = kinder.make(
        f"kinder/Tossing3D-o{num_cubes}-v0", render_mode="rgb_array", scene_bg=False
    )
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_throw_pose"]
    robot = _get_robot_from_state(state)
    target = state.get_object_from_name("bin_0")
    held = state.get_object_from_name("cube_0")
    controller = lifted_controller.ground((robot, target, held))

    rng = np.random.default_rng(123)
    draws = np.array([controller.sample_parameters(state, rng) for _ in range(50)])

    # Measured on the pose the parameters imply: the two agree only because
    # bin_init_region pins the bin's yaw at 0.
    target_pose = get_overhead_object_se2_pose(state, target)
    off_axis = np.array(
        [
            abs(
                target_pose.y
                - get_target_robot_pose_from_parameters(target_pose, distance, rot).y
            )
            for distance, rot in draws
        ]
    )
    assert np.all(off_axis <= THROW_POSE_TOLERANCE), off_axis.max()
    # Still a sampler, not a constant, in both components.
    assert draws[:, 0].min() < draws[:, 0].max()
    assert draws[:, 1].min() < draws[:, 1].max()

    env.close()


def test_toss_release_speed_default_rebuilds_the_unscaled_profile():
    """The default must rebuild the profile the literal 140/300/200 deg/s limits give.

    Asserts equality of the sampled trajectory rather than of the three limits, so a
    refactor of how the limits reach the profile still has to keep the motion.
    """
    total_dist = float(
        np.linalg.norm(TOSS_RELEASE_ARM_CONFIGURATION - TOSS_WINDUP_ARM_CONFIGURATION)
    )
    expected = _trapezoidal_motion_profile(
        total_dist,
        max_vel=np.deg2rad(140),
        max_accel=np.deg2rad(300),
        max_decel=np.deg2rad(200),
        step_size=_CONTROL_TIMESTEP,
    )
    max_vel, max_accel, max_decel = toss_profile_limits()
    actual = _trapezoidal_motion_profile(
        total_dist,
        max_vel=max_vel,
        max_accel=max_accel,
        max_decel=max_decel,
        step_size=_CONTROL_TIMESTEP,
    )
    assert np.array_equal(actual, expected)


def test_toss_release_speed_scales_every_limit_by_the_same_factor():
    """A release speed is an effort scale on the whole profile, not on max_vel alone.

    Scaling max_vel alone moves the release out of the cruise phase, where the
    acceleration limits set the speed instead, so it stops tracking max_vel.
    """
    # The invariant is the profile's shape. Asserted a few ULP wide rather than bitwise:
    # recovering a ratio divides a factor back out, and a/(b*c)*c need not round back.
    shape = np.array([TOSS_MAX_ACCELERATION, TOSS_MAX_DECELERATION]) / TOSS_MAX_VELOCITY
    for scale in (0.25, 0.5, 1.0, 1.7, 2.0, 3.0):
        max_vel, max_accel, max_decel = toss_profile_limits(scale * TOSS_MAX_VELOCITY)
        assert np.allclose([max_accel, max_decel] / max_vel, shape, rtol=1e-15), scale
        assert max_vel == min(scale, 1.0) * TOSS_MAX_VELOCITY

    # Proportional through the origin, not merely affine, or "twice the speed" would
    # mean other than twice the effort. Below the clamp only; clamping is not
    # homogeneous.
    rng = np.random.default_rng(0)
    for _ in range(200):
        factor = rng.uniform(0.05, 1.0)
        speed = rng.uniform(0.05, 1.0) * TOSS_MAX_VELOCITY
        scaled = np.array(toss_profile_limits(factor * speed))
        assert np.allclose(
            scaled, factor * np.array(toss_profile_limits(speed)), rtol=1e-12
        )
    assert np.array_equal(np.array(toss_profile_limits(0.0)), np.zeros(3))


def test_toss_release_speed_clamps_the_effort_to_zero_and_one():
    """The arm's own ceiling, and no reverse swing below it."""
    at_ceiling = toss_profile_limits(TOSS_MAX_VELOCITY)
    assert toss_profile_limits(2.0 * TOSS_MAX_VELOCITY) == at_ceiling
    assert toss_profile_limits(-TOSS_MAX_VELOCITY) == toss_profile_limits(0.0)
    assert np.array_equal(np.array(toss_profile_limits(-1.0)), np.zeros(3))


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


def test_gripper_release_ms_splits_into_a_control_step_and_a_slice():
    """The parameter is absolute wall-clock milliseconds from the start of the swing.

    reset() only decomposes it; nothing rounds it to a control-step boundary.
    """
    assert TOSS_SLICES_PER_CONTROL_STEP == 100
    for ms, expected in [(0, (0, 0)), (723, (7, 23)), (100, (1, 0)), (2399, (23, 99))]:
        assert divmod(ms, TOSS_SLICES_PER_CONTROL_STEP) == expected


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
    """The sampler has nothing to draw, so a refiner has nothing to backtrack over."""
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
    controller = controllers["pick_cube"].ground((robot, cube))
    for seed in range(5):
        params = controller.sample_parameters(state, np.random.default_rng(seed))
        assert np.asarray(params).shape == (0,)
    env.close()


def test_pick_cube_walks_the_ladder_from_the_nominal_pose():
    """The first candidate is the nominal one, and every candidate is in the range the
    shelf pick used to sample."""
    assert PICK_STANDOFF_LADDER[0] == (0.55, 0.0)
    distances = [d for d, _ in PICK_STANDOFF_LADDER]
    rots = [r for _, r in PICK_STANDOFF_LADDER]
    assert min(distances) >= MOVE_TO_TARGET_DISTANCE_BOUNDS[0]
    assert max(distances) <= MOVE_TO_TARGET_DISTANCE_BOUNDS[1]
    assert min(rots) >= MOVE_TO_TARGET_ROT_BOUNDS[0]
    assert max(rots) <= MOVE_TO_TARGET_ROT_BOUNDS[1]


def test_pick_cube_lifts_the_cube_off_the_ground():
    """End to end, with no parameters supplied by the caller."""
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
    controller = controllers["pick_cube"].ground((robot, cube))
    controller.reset(
        state, controller.sample_parameters(state, np.random.default_rng(0))
    )
    for _ in range(2000):
        if controller.terminated():
            break
        obs, _, _, _, _ = env.step(controller.step())
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
    assert controller.terminated()
    assert state.get(cube, "z") > MINIMUM_HOLDING_HEIGHT
    env.close()


def test_pick_cube_releases_when_the_grasp_closed_on_nothing():
    """Otherwise the hand is neither empty nor holding, and no operator applies."""
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
    controller = controllers["pick_cube"].ground((robot, cube))
    controller.reset(
        state, controller.sample_parameters(state, np.random.default_rng(0))
    )
    # The arm finished its retract with the cube still on the floor and the gripper
    # commanded shut: a grasp that closed on nothing.
    missed = state.copy()
    missed.set(robot, "pos_gripper", 1.0)
    controller.observe(missed)
    controller._lifted = True  # pylint: disable=protected-access
    assert not controller.terminated()
    assert controller.step()[-1] == 0
    opened = missed.copy()
    opened.set(robot, "pos_gripper", 0.0)
    controller.observe(opened)
    assert controller.terminated()
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
    target_bin = state.get_object_from_name("bin_0")
    controller = controllers["move_to_toss_location_and_toss"].ground(
        (robot, target_bin, cube)
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
            TOSS_TARGET_DISTANCE_BOUNDS,
            TOSS_TARGET_ROTATION_BOUNDS,
            TOSS_SPEED_BOUNDS,
            TOSS_RELEASE_MS_BOUNDS,
        ]
    ):
        assert draws[:, column].min() >= low
        assert draws[:, column].max() <= high
        # A sampler, not a constant, in every component.
        assert draws[:, column].min() < draws[:, column].max()
    env.close()


def test_the_release_speeds_the_sampler_draws_are_never_clamped():
    """TOSS_SPEED_BOUNDS' top edge is the clamp point, so it must pass through."""
    for speed in np.linspace(*TOSS_SPEED_BOUNDS, 25):
        assert np.isclose(toss_profile_limits(speed)[0], speed)


def test_pick_cube_then_move_and_toss_scores():
    """The whole domain, end to end: two skills and no third."""
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

    def run(controller, params):
        nonlocal state
        controller.reset(state, params)
        for _ in range(3000):
            if controller.terminated():
                return
            obs_, _, _, _, _ = env.step(controller.step())
            state = env.observation_space.devectorize(obs_)
            controller.observe(state)
        raise AssertionError("controller did not terminate")

    pick = controllers["pick_cube"].ground((robot, cube))
    run(pick, pick.sample_parameters(state, np.random.default_rng(0)))
    assert state.get(cube, "z") > MINIMUM_HOLDING_HEIGHT

    toss = controllers["move_to_toss_location_and_toss"].ground(
        (robot, target_bin, cube)
    )
    run(toss, np.array([1.30, 0.0, TOSS_MAX_VELOCITY, 720.0]))
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    assert sim._check_goals()  # pylint: disable=protected-access
    env.close()


def _straight_swing(release_ms=TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS):
    """A planned swing between the two demonstrated confs, without a simulator."""
    start = list(TOSS_WINDUP_ARM_CONFIGURATION) + [0.0] * 6
    end = list(TOSS_RELEASE_ARM_CONFIGURATION) + [0.0] * 6
    return plan_toss_swing([end], start, TOSS_MAX_VELOCITY, release_ms)


def test_plan_toss_swing_splits_the_release_millisecond_into_step_and_slice():
    """The millisecond names a slice inside a control step, not a step boundary."""
    swing = _straight_swing(release_ms=725)
    step, slice_ = divmod(725, TOSS_SLICES_PER_CONTROL_STEP)
    assert swing.release_step == step
    assert swing.release_slice == slice_


def test_plan_toss_swing_direction_is_a_unit_vector_along_the_swing():
    """The profile carries the distance; the direction carries only the heading."""
    swing = _straight_swing()
    assert np.isclose(np.linalg.norm(swing.direction), 1.0)


def test_plan_toss_swing_direction_is_zero_for_a_swing_that_does_not_move():
    """Otherwise the unit vector is a division by zero."""
    conf = list(TOSS_WINDUP_ARM_CONFIGURATION) + [0.0] * 6
    swing = plan_toss_swing(
        [conf], conf, TOSS_MAX_VELOCITY, TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS
    )
    assert np.allclose(swing.direction, 0.0)


def test_toss_swing_action_holds_the_gripper_shut_before_the_release_step():
    swing = _straight_swing()
    action = toss_swing_action(swing, 0, [0.0] * 13, 1.0, False)
    assert action.shape == (18,)
    assert action[10] == 1.0


def test_toss_swing_action_opens_the_gripper_after_the_release_step():
    swing = _straight_swing()
    action = toss_swing_action(swing, swing.release_step + 1, [0.0] * 13, 1.0, True)
    assert action.shape == (18,)
    assert action[10] == 0.0


def test_toss_swing_action_emits_a_schedule_on_the_step_the_release_falls_inside():
    """A (slices, 18) schedule, so the millisecond means the millisecond."""
    swing = _straight_swing(release_ms=725)
    assert swing.release_slice != 0
    schedule = toss_swing_action(swing, swing.release_step, [0.0] * 13, 1.0, False)
    assert schedule.shape == (TOSS_SLICES_PER_CONTROL_STEP, 18)
    assert np.all(schedule[: swing.release_slice, 10] == 1.0)
    assert np.all(schedule[swing.release_slice :, 10] == 0.0)


def test_toss_swing_action_stays_flat_when_the_release_lands_on_a_step_boundary():
    """No schedule is needed when no millisecond inside the step is special."""
    swing = _straight_swing(release_ms=TOSS_SLICES_PER_CONTROL_STEP * 7)
    assert swing.release_slice == 0
    action = toss_swing_action(swing, swing.release_step, [0.0] * 13, 1.0, False)
    assert action.shape == (18,)
    assert action[10] == 0.0


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
    target_bin = state.get_object_from_name("bin_0")
    controller = controllers["move_to_toss_location_and_toss"].ground(
        (robot, target_bin, cube)
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
    middle of a throw, where a refiner would see a crash instead of a rejected sample.
    """
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
    controller = controllers["move_to_toss_location_and_toss"].ground(
        (robot, target_bin, cube)
    )
    controller.reset(state, np.array([1.30, 0.0, TOSS_MAX_VELOCITY, 720.0]))
    # Every phase is planned before the first action is asked for.
    assert controller._current_base_motion_plan is not None
    assert len(controller._windup_trajectory) > 0
    assert controller._swing is not None
    env.close()


def test_there_are_twenty_four_cube_rotation_symmetries():
    """A cube has 6 faces it can rest on, each at 4 yaws."""
    assert len(CUBE_ROTATION_SYMMETRIES) == 24
    seen = {tuple(np.round(q, 6)) for q in CUBE_ROTATION_SYMMETRIES}
    assert len(seen) == 24


def test_every_cube_symmetry_is_a_rotation():
    """Unit quaternions, so each maps the cube onto itself without scaling it."""
    for q in CUBE_ROTATION_SYMMETRIES:
        assert np.isclose(np.linalg.norm(q), 1.0)


def test_canonical_upright_rotation_flattens_every_face_down_rest():
    """Each of the six face-down rests is the same cube, so each must canonicalise to a
    pure yaw -- which is what makes a top-down grasp derivable from it."""

    def quat(axis, deg):
        a = np.deg2rad(deg) / 2
        v = np.array(axis, dtype=float)
        x, y, z = np.sin(a) * v
        return (float(x), float(y), float(z), float(np.cos(a)))

    for axis, deg in [
        ([0, 0, 1], 0),
        ([0, 0, 1], 90),
        ([1, 0, 0], 90),
        ([1, 0, 0], 180),
        ([1, 0, 0], -90),
        ([0, 1, 0], 90),
        ([0, 1, 0], -90),
        ([0, 1, 0], 180),
    ]:
        x, y, z, w = canonical_upright_rotation(quat(axis, deg))
        assert abs(x) < 1e-6, (axis, deg, x)
        assert abs(y) < 1e-6, (axis, deg, y)


def test_canonical_upright_rotation_keeps_the_yaw():
    """Yaw is the only real information in a cube's resting pose, so it must survive."""
    for deg in (0.0, 30.0, 90.0, 200.0):
        a = np.deg2rad(deg) / 2
        q = (0.0, 0.0, float(np.sin(a)), float(np.cos(a)))
        x, y, z, w = canonical_upright_rotation(q)
        got = np.rad2deg(2 * np.arctan2(z, w)) % 90.0
        assert np.isclose(got, deg % 90.0, atol=1e-4) or np.isclose(
            got, deg % 90.0 - 90.0, atol=1e-4
        ), (deg, got)


def test_pick_cube_plans_a_grasp_for_a_cube_resting_on_its_side():
    """pick_shelf derives the grasp from the raw rotation and finds no IK solution for
    any of the five non-original face-down rests. Canonicalising first, all of them
    plan."""
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
    controller = controllers["pick_cube"].ground((robot, cube))

    def rest(axis, deg):
        a = np.deg2rad(deg) / 2
        vec = np.array(axis, dtype=float)
        return tuple(float(v) for v in np.sin(a) * vec) + (float(np.cos(a)),)

    for axis, deg in [
        ([1, 0, 0], 90),
        ([1, 0, 0], 180),
        ([1, 0, 0], -90),
        ([0, 1, 0], 90),
        ([0, 1, 0], -90),
    ]:
        tipped = state.copy()
        for feature, value in zip(("qx", "qy", "qz", "qw"), rest(axis, deg)):
            tipped.set(cube, feature, value)
        # Raises if no candidate in the ladder plans.
        controller.reset(
            tipped, controller.sample_parameters(tipped, np.random.default_rng(0))
        )
    env.close()


def test_upright_grasp_rotations_offers_all_four_equivalent_yaws():
    """Resting on a face, a cube is four-fold symmetric about the vertical, so all four
    yaws are the same grasp -- and the arm cannot reach every one of them."""
    a = np.deg2rad(90) / 2
    pitched = (0.0, float(np.sin(a)), 0.0, float(np.cos(a)))
    rotations = upright_grasp_rotations(pitched)
    assert len(rotations) == 4
    for x, y, _, _ in rotations:
        assert abs(x) < 1e-6 and abs(y) < 1e-6
    yaws = sorted(
        round(np.rad2deg(2 * np.arctan2(z, w)) % 360.0, 3) for _, _, z, w in rotations
    )
    gaps = {round((b - a_) % 360.0, 3) for a_, b in zip(yaws, yaws[1:])}
    assert gaps == {90.0}, yaws
