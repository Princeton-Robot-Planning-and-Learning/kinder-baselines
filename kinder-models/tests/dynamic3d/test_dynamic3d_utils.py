"""Test utils for dynamic3d models."""

from pathlib import Path

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import TidyBot3DEnv
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from matplotlib import pyplot as plt
from pybullet_helpers.joint import get_joint_infos
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace
from spatialmath import SE2
from tomsgeoms2d.structs import Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

from kinder_models.dynamic3d.utils import (
    CIRCULAR_ARM_JOINT_INDICES,
    PyBulletSim,
    get_bounding_box,
    get_overhead_kinematic2ds,
    get_overhead_object_se2_pose,
    get_overhead_robot_se2_pose,
    plot_overhead_scene,
    run_base_motion_planning,
    wrap_arm_joint_difference,
)

kinder.register_all_environments()

_TEST_TASKS = Path(__file__).parent.parent / "test_tasks"


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_get_overhead_object_se2_pose():
    """Tests for get_overhead_object_se2_pose()."""

    # Get a real object-centric state.
    env = TidyBot3DEnv(task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state1 = env.observation_space.devectorize(obs)
    cube = state1.get_object_from_name("cube1")

    # Extract the initial SE2 pose.
    pose1 = get_overhead_object_se2_pose(state1, cube)

    # Moving the object z shouldn't change anything.
    state2 = state1.copy()
    state2.set(cube, "z", 1000)
    pose2 = get_overhead_object_se2_pose(state2, cube)
    assert np.allclose(pose1.A, pose2.A, atol=1e-5)

    # Move the object x should have an effect.
    state3 = state1.copy()
    state3.set(cube, "x", state1.get(cube, "x") + 1.0)
    pose3 = get_overhead_object_se2_pose(state3, cube)
    assert np.isclose(pose1.x + 1, pose3.x)
    assert np.isclose(pose1.y, pose3.y)
    assert np.isclose(pose1.theta(), pose3.theta())


def test_get_overhead_robot_se2_pose():
    """Tests for get_overhead_robot_se2_pose()."""

    # Get a real object-centric state.
    env = TidyBot3DEnv(task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state1 = env.observation_space.devectorize(obs)
    robot = _get_robot_from_state(state1)

    # Extract the initial SE2 pose.
    pose1 = get_overhead_robot_se2_pose(state1, robot)

    # Move the object x should have an effect.
    state2 = state1.copy()
    state2.set(robot, "pos_base_x", state1.get(robot, "pos_base_x") + 1.0)
    pose2 = get_overhead_robot_se2_pose(state2, robot)
    assert np.isclose(pose1.x + 1, pose2.x)
    assert np.isclose(pose1.y, pose2.y)
    assert np.isclose(pose1.theta(), pose2.theta())


def test_get_overhead_kinematic2ds():
    """Tests for get_overhead_kinematic2ds()."""
    env = TidyBot3DEnv(task_config_path=str(_TEST_TASKS / "tidybot-ground-o1.json"))
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)
    geoms = get_overhead_kinematic2ds(state)
    assert len(geoms) == 2
    robot = _get_robot_from_state(state)
    robot_geom = geoms[robot.name]
    assert isinstance(robot_geom, Rectangle)
    cube_geom = geoms["cube1"]
    assert isinstance(cube_geom, Rectangle)


def test_plot_overhead_scene():
    """Tests for plot_overhead_scene()."""

    env = TidyBot3DEnv(
        task_config_path=str(_TEST_TASKS / "tidybot-ground-o3.json"),
        render_mode="rgb_array",
    )
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)
    fig, ax = plot_overhead_scene(state, min_x=-1.5, max_x=1.5, min_y=-1.5, max_y=1.5)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Uncomment to debug.
    # from prpl_utils.utils import fig2data
    # import imageio.v2 as iio
    # ax.set_title("Overhead Scene Example")
    # plt.tight_layout()
    # img = fig2data(fig)
    # outfile = "out_plot_overhead_scene.png"
    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")
    # img = env.render()
    # outfile = "actual_scene.png"
    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")


def test_run_base_motion_planning():
    """Tests for run_base_motion_planning()."""

    env = kinder.make("kinder/Shelf3D-o1-v0", render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)

    target_base_pose = SE2(-1, 1, 0.0)
    x_bounds = (-1.5, 1.5)
    y_bounds = (-1.5, 1.5)
    seed = 123
    base_motion_plan = run_base_motion_planning(
        state,
        target_base_pose,
        x_bounds,
        y_bounds,
        seed,
        extend_xy_magnitude=0.5,
        extend_rot_magnitude=np.pi / 2,
    )
    assert base_motion_plan is not None

    fig, ax = plot_overhead_scene(
        state,
        min_x=x_bounds[0],
        max_x=x_bounds[1],
        min_y=y_bounds[0],
        max_y=y_bounds[1],
    )
    assert isinstance(fig, plt.Figure)
    robot = _get_robot_from_state(state)
    robot_width, robot_height, _ = get_bounding_box(state, robot)
    for pose in base_motion_plan:
        robot_geom = Rectangle.from_center(
            pose.x,
            pose.y,
            robot_width,
            robot_height,
            rotation_about_center=pose.theta(),
        )
        robot_geom.plot(ax, fc="none", ec="gray", linestyle="dashed")

    # Uncomment to debug.
    # from prpl_utils.utils import fig2data, get_signed_angle_distance

    # ax.set_title("Motion Planning Example")
    # plt.tight_layout()
    # img = fig2data(fig)
    # outfile = "base_motion_planning.png"
    # import imageio.v2 as iio

    # iio.imsave(outfile, img)
    # print(f"Wrote out to {outfile}")
    # print("Number of steps:", len(base_motion_plan))

    # import time
    # imgs = []
    # for t in range(1, len(base_motion_plan)):
    #     pose = base_motion_plan[t]
    #     max_control_steps = 10
    #     tolerance = 1e-2
    #     control_period = 0.1  # 10hz
    #     for control_step in range(max_control_steps):
    #         previous_pose = SE2(
    #             state.get(robot, "pos_base_x"),
    #             state.get(robot, "pos_base_y"),
    #             state.get(robot, "pos_base_rot"),
    #         )
    #         dx = pose.x - previous_pose.x
    #         dy = pose.y - previous_pose.y
    #         drot = get_signed_angle_distance(pose.theta(), previous_pose.theta())
    #         action = np.zeros(11, dtype=np.float32)
    #         action[0] = dx
    #         action[1] = dy
    #         action[2] = drot
    #         # assert env.action_space.contains(action)

    #         obs, _, _, _, _ = env.step(action)
    #         state = env.observation_space.devectorize(obs)
    #         print("Expected x, y, rot:", pose.x, pose.y, pose.theta())
    #         print(
    #             "Actual x, y, rot:",
    #             state.get(robot, "pos_base_x"),
    #             state.get(robot, "pos_base_y"),
    #             state.get(robot, "pos_base_rot"),
    #         )
    #         time.sleep(
    #             control_period
    #         )  # sleep for 100ms to allow the action to be executed
    #         if (
    #             np.isclose(state.get(robot, "pos_base_x"), pose.x, atol=tolerance)
    #             and np.isclose(state.get(robot, "pos_base_y"), pose.y, atol=tolerance)
    #             and np.isclose(
    #                 state.get(robot, "pos_base_rot"), pose.theta(), atol=tolerance
    #             )
    #         ):
    #             print(
    #                 f"Reached target pose {pose.x}, {pose.y}, {pose.theta()} "
    #                 f"in {control_step + 1} steps"
    #             )
    #             break
    #         img = env.render()
    #         imgs.append(img)
    # outfile = "base_motion_planning.mp4"
    # iio.mimsave(outfile, imgs)
    # print(f"Wrote out to {outfile}")


def test_run_base_motion_planning_avoids_obstacles():
    """Tests that run_base_motion_planning() avoids scene obstacles."""

    env = kinder.make("kinder/Shelf3D-o1-v0", render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)
    robot = _get_robot_from_state(state)

    # cube1 sits directly between the robot's start pose and this target, so a
    # straight-line path would drive right through it.
    target_base_pose = SE2(1.0, 0.17, 0.0)
    x_bounds = (-1.5, 1.5)
    y_bounds = (-1.5, 1.5)
    seed = 123
    base_motion_plan = run_base_motion_planning(
        state,
        target_base_pose,
        x_bounds,
        y_bounds,
        seed,
        extend_xy_magnitude=0.05,
        extend_rot_magnitude=np.pi / 2,
    )

    assert base_motion_plan is not None

    geoms = get_overhead_kinematic2ds(state)
    cube_geom = geoms["cube1"]
    robot_width, robot_height, _ = get_bounding_box(state, robot)
    for pose in base_motion_plan:
        robot_geom = Rectangle.from_center(
            pose.x,
            pose.y,
            robot_width,
            robot_height,
            rotation_about_center=pose.theta(),
        )
        assert not geom2ds_intersect(robot_geom, cube_geom)


def test_wrap_arm_joint_difference_leaves_an_unambiguous_difference_alone():
    """Anything inside a half turn comes back bit-identical, limited joints
    included.

    The guard exists to correct a wrong 2*pi representative, not to re-derive a
    difference that was already fine, so an ordinary difference must survive it
    untouched -- otherwise every motion already measured would shift in its last bits.
    """
    difference = np.array([0.3, -3.0, 0.05, 2.6, -1.4, 2.2, 3.1])
    wrapped = wrap_arm_joint_difference(difference)
    assert np.array_equal(wrapped, difference)


def test_wrap_arm_joint_difference_routes_a_continuous_joint_the_short_way():
    """A whole turn on a continuous joint is no motion at all."""
    difference = np.zeros(7)
    for index in CIRCULAR_ARM_JOINT_INDICES:
        difference[index] = 2 * np.pi
    wrapped = wrap_arm_joint_difference(difference)
    assert np.allclose(wrapped, 0.0, atol=1e-9)


def test_wrap_arm_joint_difference_leaves_a_limited_joint_a_whole_turn_alone():
    """Only the continuous joints are ambiguous.

    A limited joint cannot reach the same pose two ways, so a large
    difference there is real travel and wrapping it would silently ask
    for a different pose.
    """
    difference = np.zeros(7)
    limited = [i for i in range(7) if i not in CIRCULAR_ARM_JOINT_INDICES]
    for index in limited:
        difference[index] = 2 * np.pi
    wrapped = wrap_arm_joint_difference(difference)
    assert np.array_equal(wrapped, difference)


def test_the_circular_arm_joints_are_the_ones_the_robot_reports():
    """CIRCULAR_ARM_JOINT_INDICES is a constant because two callers have no
    robot to ask.

    This is what stops it going quietly wrong: PyBullet's own joint info is the
    authority, and a different arm fails here rather than inside a rollout.
    """
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array", num_objects=1)
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    sim = PyBulletSim(state)
    joint_infos = get_joint_infos(
        sim.robot.robot_id, sim.robot.arm_joints, sim.robot.physics_client_id
    )
    reported = tuple(
        index for index, info in enumerate(joint_infos[:7]) if info.is_circular
    )
    assert reported == CIRCULAR_ARM_JOINT_INDICES
    sim.close()
    env.close()
