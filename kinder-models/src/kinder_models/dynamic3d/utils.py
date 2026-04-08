"""Utils for tidybot environments."""

import math
from typing import Iterable

import numpy as np
import pybullet as p
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers
from matplotlib import pyplot as plt
from prpl_utils.motion_planning import BiRRT
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.motion_planning import create_joint_distance_fn
from pybullet_helpers.robots import SingleArmPyBulletRobot, create_pybullet_robot
from pybullet_helpers.utils import create_pybullet_block, create_pybullet_shelf
from relational_structs import (
    Object,
    ObjectCentricState,
)
from spatialmath import SE2, UnitQuaternion
from tomsgeoms2d.structs import Geom2D, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

# Control period in seconds (10 Hz).
_CONTROL_DT = 0.1

# Robot geometry.
ROBOT_ARM_POSE_TO_BASE = Pose((0.12, 0.0, 0.4))

# Arm joint velocity and acceleration limits (rad/s, rad/s²).
_ARM_MAX_VEL = np.deg2rad(np.array([80.0, 80.0, 80.0, 80.0, 70.0, 70.0, 70.0]))
_ARM_MAX_ACCEL = np.deg2rad(np.array([297.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]))

# Base motion limits.
MAX_BASE_MOVEMENT_MAGNITUDE = 1e-1

# Gripper thresholds.
GRIPPER_OPEN_THRESHOLD = 0.01
GRASP_CLOSE_THRESHOLD = 1.0  # for stable grasp
GRIPPER_CLOSED_THRESHOLD = 0.02

# Waypoint tolerance for arm configuration convergence.
WAYPOINT_TOL = 4 * 1e-2

# Base navigation sampling bounds.
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.5, 0.6)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi / 4, np.pi / 4)
OPEN_DRAWER_DISTANCE_BOUNDS = (0.8, 0.8)
OPEN_DRAWER_ROT_BOUNDS = (-np.pi, -np.pi)
PICK_WIPER_DISTANCE_BOUNDS = (0.7, 0.7)
PICK_WIPER_ROT_BOUNDS = (-np.pi, -np.pi)
SWEEP_DISTANCE_BOUNDS = (0.55, 0.55)
SWEEP_ROT_BOUNDS = (-np.pi, -np.pi)
WORLD_X_BOUNDS = (-2.5, 2.5)
WORLD_Y_BOUNDS = (-2.5, 2.5)

# End-effector transforms for each skill.
GRASP_TRANSFORM_TO_OBJECT = Pose((-0.005, 0, 0.01), (0.707, 0.707, 0, 0))
WIPER_TRANSFORM_TO_OBJECT = Pose.from_rpy(
    (0.02, 0, 0.03), (-np.pi - np.pi / 16, 0, -np.pi / 2)
)  # Pose((0.035, 0, 0.04), (-0.707, 0.707, 0, 0))
WIPER_SWEEP_TRANSFORM = Pose.from_rpy(
    (-0.05, -0.1, 0.025), (-np.pi + np.pi / 16, 0, -np.pi / 2)
)  # Pose((-0.05, 0, 0.04), (-0.707, 0.707, 0, 0))
WIPER_SWEEP_TRANSFORM_END = Pose.from_rpy(
    (0.15, 0.05, 0.025), (-np.pi + np.pi / 16, 0, -np.pi / 2)
)  # Pose((0.15, 0, 0.04), (-0.707, 0.707, 0, 0))
WIPER_SWEEP_TRANSFORM_END_2 = Pose.from_rpy(
    (0.28, 0.15, 0.025), (-np.pi + np.pi / 16, 0, -np.pi / 2)
)  # Pose((0.15, 0, 0.04), (-0.707, 0.707, 0, 0))
DRAWER_TRANSFORM_TO_OBJECT = Pose.from_rpy(
    (0.07, 0.3, -0.12), (-np.pi - np.pi / 8, 0, -np.pi / 2)
)
DRAWER_TRANSFORM_TO_OBJECT_END = Pose.from_rpy(
    (0.18, 0.3, -0.12), (-np.pi - np.pi / 8, 0, -np.pi / 2)
)

# Cupboard-specific geometry and sampling bounds.
BASE_DISTANCE_TO_CUPBOARD = 0.95
ARM_MOVEMENT_CUPBOARD = Pose((0.8, 0.0, 0.28), (0.5, 0.5, 0.5, 0.5))
PLACE_SAMPLER_COLLISION_THRESHOLD = 0.05
PLACE_SAMPLER_X_OFFSET_BOUNDS = (0.05, 0.1)
PLACE_SAMPLER_Y_OFFSET_BOUNDS = (-0.05, 0.05)
MAX_SAMPLER_ATTEMPTS = 100
BASE_TO_CUPBOARD_ROTATION = -np.pi / 2


def get_overhead_object_se2_pose(state: ObjectCentricState, obj: Object) -> SE2:
    """Get the top-down SE2 pose for an object in a dynamic3D state."""
    assert obj.is_instance(MujocoObjectType)
    x = state.get(obj, "x")
    y = state.get(obj, "y")
    q = UnitQuaternion(
        s=state.get(obj, "qw"),
        v=(
            state.get(obj, "qx"),
            state.get(obj, "qy"),
            state.get(obj, "qz"),
        ),
    )
    rpy = q.rpy()
    yaw = rpy[2]
    return SE2(x, y, yaw)


def get_overhead_robot_se2_pose(state: ObjectCentricState, obj: Object) -> SE2:
    """Get the top-down SE2 pose for an object in a dynamic3D state."""
    assert obj.is_instance(MujocoTidyBotRobotObjectType)
    x = state.get(obj, "pos_base_x")
    y = state.get(obj, "pos_base_y")
    yaw = state.get(obj, "pos_base_rot")
    return SE2(x, y, yaw)


def get_bounding_box(
    state: ObjectCentricState, obj: Object
) -> tuple[float, float, float]:
    """Returns (x extent, y extent, z extent) for the given object.

    We may want to later add something to the state that allows these values to be
    extracted automatically.
    """
    if obj.is_instance(MujocoTidyBotRobotObjectType):
        # NOTE: hardcoded for now.
        return (0.5, 0.5, 1.0)
    if obj.is_instance(MujocoFixtureObjectType):
        # NOTE: hardcoded for now.
        return (0.61, 0.26, 1.0)
    if obj.is_instance(MujocoObjectType):
        return (
            state.get(obj, "bb_x"),
            state.get(obj, "bb_y"),
            state.get(obj, "bb_z"),
        )
    raise NotImplementedError


def get_overhead_kinematic2ds(state: ObjectCentricState) -> dict[str, Geom2D]:
    """Get a mapping from object name to Geom2D from an overhead perspective."""
    geoms: dict[str, Geom2D] = {}
    for obj in state:
        print(obj.name)
        if obj.is_instance(MujocoTidyBotRobotObjectType):
            pose = get_overhead_robot_se2_pose(state, obj)
        elif obj.is_instance(MujocoObjectType):
            pose = get_overhead_object_se2_pose(state, obj)
        else:
            raise NotImplementedError
        width, height, _ = get_bounding_box(state, obj)
        geom = Rectangle.from_center(
            pose.x, pose.y, width, height, rotation_about_center=pose.theta()
        )
        geoms[obj.name] = geom
    return geoms


def plot_overhead_scene(
    state: ObjectCentricState,
    min_x: float = -2.5,
    max_x: float = 2.5,
    min_y: float = -2.5,
    max_y: float = 2.5,
    fontsize: int = 6,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a matplotlib figure with a top-down scene rendering."""

    fig, ax = plt.subplots()

    fontdict = {
        "fontsize": fontsize,
        "color": "black",
        "ha": "center",
        "va": "center",
        "fontweight": "medium",
        "bbox": {"facecolor": "white", "alpha": 0.25, "edgecolor": "none", "pad": 2},
    }

    geoms = get_overhead_kinematic2ds(state)
    for obj_name, geom in geoms.items():
        geom.plot(ax, facecolor="white", edgecolor="black")
        assert isinstance(geom, Rectangle)
        x, y = geom.center
        dx = geom.width / 1.5 * np.cos(geom.theta)
        dy = geom.height / 1.5 * np.sin(geom.theta)
        arrow_width = max(max_x - min_x, max_y - min_y) / 250.0
        ax.arrow(x, y, dx, dy, color="gray", width=arrow_width)
        ax.text(x, y, obj_name, fontdict=fontdict)

    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))

    return fig, ax


def run_base_motion_planning(
    state: ObjectCentricState,
    target_base_pose: SE2,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    seed: int,
    extend_xy_magnitude: float = 0.025,
    extend_rot_magnitude: float = np.pi / 8,
    num_attempts: int = 10,
    num_iters: int = 100,
    smooth_amt: int = 50,
    disable_collision_objects: list[str] | None = None,
) -> list[SE2] | None:
    """Run motion planning for the robot base."""

    rng = np.random.default_rng(seed)

    # Construct geoms.
    (robot,) = state.get_objects(MujocoTidyBotRobotObjectType)
    robot_width, robot_height, _ = get_bounding_box(state, robot)
    obstacles = state.get_objects(MujocoObjectType)
    if disable_collision_objects is not None:
        obstacles = [o for o in obstacles if o.name not in disable_collision_objects]
    obstacle_geoms: set[Geom2D] = set()
    # uncomment to fully consider the collisions.
    # geoms = get_overhead_kinematic2ds(state)
    # obstacle_geoms = {geoms[o.name] for o in obstacles}

    # Set up the RRT methods.
    def sample_fn(_: SE2) -> SE2:
        """Sample a robot pose."""
        x = rng.uniform(*x_bounds)
        y = rng.uniform(*y_bounds)
        theta = rng.uniform(-np.pi, np.pi)
        return SE2(x, y, theta)

    def extend_fn(pt1: SE2, pt2: SE2) -> Iterable[SE2]:
        """Interpolate between the two poses."""
        # Make sure that we obey the bounds on actions.
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta(), pt1.theta())
        x_num_steps = int(abs(dx) / extend_xy_magnitude) + 1
        assert x_num_steps > 0
        y_num_steps = int(abs(dy) / extend_xy_magnitude) + 1
        assert y_num_steps > 0
        theta_num_steps = int(abs(dtheta) / extend_rot_magnitude) + 1
        assert theta_num_steps > 0
        num_steps = max(x_num_steps, y_num_steps, theta_num_steps)
        x = pt1.x
        y = pt1.y
        theta = pt1.theta()
        yield SE2(x, y, theta)
        for _ in range(num_steps):
            x += dx / num_steps
            y += dy / num_steps
            theta = wrap_angle(theta + dtheta / num_steps)
            yield SE2(x, y, theta)

    def collision_fn(pt: SE2) -> bool:
        """Check for collisions if the robot were at this pose."""
        # Get the new robot geom.
        new_state = state.copy()
        new_state.set(robot, "pos_base_x", pt.x)
        new_state.set(robot, "pos_base_y", pt.y)
        new_state.set(robot, "pos_base_rot", pt.theta())
        pose = get_overhead_robot_se2_pose(new_state, robot)
        robot_geom = Rectangle.from_center(
            pose.x,
            pose.y,
            robot_width,
            robot_height,
            rotation_about_center=pose.theta(),
        )
        for obstacle_geom in obstacle_geoms:
            if geom2ds_intersect(robot_geom, obstacle_geom):
                return True
        return False

    def distance_fn(pt1: SE2, pt2: SE2) -> float:
        """Return a distance between the two points."""
        dx = pt2.x - pt1.x
        dy = pt2.y - pt1.y
        dtheta = get_signed_angle_distance(pt2.theta(), pt1.theta())
        return np.sqrt(dx**2 + dy**2) + abs(dtheta)

    birrt = BiRRT(
        sample_fn,
        extend_fn,
        collision_fn,
        distance_fn,
        rng,
        num_attempts,
        num_iters,
        smooth_amt,
    )

    initial_pose = get_overhead_robot_se2_pose(state, robot)
    path = birrt.query(initial_pose, target_base_pose)
    return path


# Based on https://github.com/jimmyyhwu/tidybot/blob/main/robot/kinova.py#L310
def _trapezoidal_motion_profile(
    total_dist: float,
    max_vel: float,
    max_accel: float,
    max_decel: float,
    step_size: float,
) -> np.ndarray:
    """Compute a trapezoidal motion profile.

    Returns array of positions along the path sampled at step_size.
    """
    assert total_dist >= 0

    # Duration of each phase.
    if total_dist < 0.5 * max_vel**2 / max_accel + 0.5 * max_vel**2 / max_decel:
        accel_duration = (
            2 * total_dist / (max_accel + max_accel**2 / max_decel)
        ) ** 0.5
        decel_duration = (max_accel / max_decel) * accel_duration
        const_duration = 0.0
    else:
        accel_duration = max_vel / max_accel
        decel_duration = max_vel / max_decel
        const_duration = (
            total_dist / max_vel - 0.5 * max_vel / max_accel - 0.5 * max_vel / max_decel
        )
    total_duration = accel_duration + const_duration + decel_duration

    t = np.arange(0, total_duration + step_size, step_size)
    pos = np.zeros_like(t)

    accel_idx = math.ceil(accel_duration / step_size)
    pos[:accel_idx] = 0.5 * max_accel * t[:accel_idx] ** 2

    decel_idx = math.ceil((accel_duration + const_duration) / step_size)
    pos[accel_idx:decel_idx] = 0.5 * max_accel * accel_duration**2 + max_vel * (
        t[accel_idx:decel_idx] - accel_duration
    )

    tmp = t[decel_idx:] - (accel_duration + const_duration)
    pos[decel_idx:] = 0.5 * max_accel * accel_duration**2 + max_vel * const_duration
    pos[decel_idx:] += (max_decel * decel_duration) * tmp - 0.5 * max_decel * tmp**2

    return pos


def _compute_per_joint_profile(
    start_conf: np.ndarray,
    end_conf: np.ndarray,
    max_vel: np.ndarray,
    max_accel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute trapezoidal profile along a straight-line joint path.

    Parameterizes the path as q(s) = start + direction * s, where s goes from 0 to
    total_dist. The scalar max velocity and acceleration along the path are determined
    by the most constrained joint.

    Returns (trajectory_positions, direction) where trajectory_positions is a 1-D array
    of s values at each control step.
    """
    dq = end_conf - start_conf
    s_total = float(np.linalg.norm(dq))
    if s_total < 1e-8:
        return np.array([0.0]), np.zeros(len(end_conf))

    direction = dq / s_total
    abs_dir = np.abs(direction)

    # The effective scalar limits along the path: for each joint i,
    # |dir_i| * ds/dt <= max_vel_i  =>  ds/dt <= max_vel_i / |dir_i|
    # Take the minimum across joints that actually move.
    moving = abs_dir > 1e-6
    effective_max_vel = float(np.min(max_vel[moving] / abs_dir[moving]))
    effective_max_accel = float(np.min(max_accel[moving] / abs_dir[moving]))

    trajectory = _trapezoidal_motion_profile(
        s_total,
        max_vel=effective_max_vel,
        max_accel=effective_max_accel,
        max_decel=effective_max_accel,
        step_size=_CONTROL_DT,
    )
    return trajectory, direction


def get_target_robot_pose_from_parameters(
    target_object_pose: SE2, target_distance: float, target_rot: float
) -> SE2:
    """Determine the pose for the robot given the state and parameters.

    The robot will be facing the target_object_pose position while being target_distance
    away, and rotated w.r.t. the target_object_pose rotation by target_rot.
    """
    # Absolute angle of the line from the robot to the target.
    ang = target_object_pose.theta() + target_rot

    # Place the robot `target_distance` away from the target along -ang
    tx, ty = target_object_pose.t  # target translation (x, y).
    rx = tx - target_distance * np.cos(ang)
    ry = ty - target_distance * np.sin(ang)

    # Robot faces the target: heading points along +ang (toward the target).
    return SE2(rx, ry, ang)


class PyBulletSim:
    """An interface to PyBullet for the TidyBot3D ground environment."""

    def __init__(
        self, initial_state: ObjectCentricState, rendering: bool = False
    ) -> None:
        """NOTE: for now, this is extremely specific to the Ground environment where
        there is exactly one cube. We will generalize this later."""

        # Hardcode the transform from the base pose to the arm pose.
        self._base_to_arm_pose = ROBOT_ARM_POSE_TO_BASE

        # Create the PyBullet simulator.
        if rendering:
            self._physics_client_id = create_gui_connection(
                camera_pitch=-90, background_rgb=(1.0, 1.0, 1.0)
            )
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        # Create the robot, assuming that it is a kinova gen3.
        self._robot = create_pybullet_robot(
            "kinova-gen3",
            self._physics_client_id,
            fixed_base=False,
            control_mode="reset",
        )

        self.base_link_to_held_obj: Pose | None = None

        # Create all the cubes.
        self._cubes: dict[str, int] = {}
        for cube_name in initial_state.get_object_names():
            if "cube" in cube_name:
                cube_obj = initial_state.get_object_from_name(cube_name)
                cube_half_extents = (
                    initial_state.get(cube_obj, "bb_x") / 2,
                    initial_state.get(cube_obj, "bb_y") / 2,
                    initial_state.get(cube_obj, "bb_z") / 2,
                )
                self._cubes[cube_name] = create_pybullet_block(
                    color=(1.0, 0.0, 0.0, 1.0),  # doesn't matter,
                    half_extents=cube_half_extents,
                    physics_client_id=self._physics_client_id,
                )

        self._cupboard1_shelf_id = None
        if "cupboard_1" in initial_state.get_object_names():
            self._cupboard1_shelf_id, self._cupboard1_surface_ids = (
                create_pybullet_shelf(
                    color=(0.5, 0.5, 0.5, 1.0),
                    shelf_width=0.60198,
                    shelf_depth=0.254,
                    shelf_height=0.0127,
                    spacing=0.254,
                    support_width=0.0127,
                    num_layers=4,
                    physics_client_id=self._physics_client_id,
                )
            )

        # Used for checking if two confs are close.
        self._joint_distance_fn = create_joint_distance_fn(self._robot)

    @property
    def physics_client_id(self) -> int:
        """The physics client ID."""
        return self._physics_client_id

    @property
    def robot(self) -> SingleArmPyBulletRobot:
        """The robot pybullet."""
        return self._robot

    def get_robot_joints(self) -> JointPositions:
        """Get the current robot joints from the simulator."""
        return self._robot.get_joint_positions()

    def set_state(
        self, x: ObjectCentricState, held_object: Object | None = None
    ) -> None:
        """Update the internal state of the simulator from an object-centric state."""
        # Update the robot state.
        robots = x.get_objects(MujocoTidyBotRobotObjectType)
        assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
        robot_obj = list(robots)[0]
        # Update the arm base.
        base_pose = Pose.from_rpy(
            (x.get(robot_obj, "pos_base_x"), x.get(robot_obj, "pos_base_y"), 0.0),
            (0, 0, x.get(robot_obj, "pos_base_rot")),
        )
        arm_pose = multiply_poses(base_pose, self._base_to_arm_pose)
        self._robot.set_base(arm_pose)
        # Update the arm conf.
        arm_conf = [
            x.get(robot_obj, "pos_arm_joint1"),
            x.get(robot_obj, "pos_arm_joint2"),
            x.get(robot_obj, "pos_arm_joint3"),
            x.get(robot_obj, "pos_arm_joint4"),
            x.get(robot_obj, "pos_arm_joint5"),
            x.get(robot_obj, "pos_arm_joint6"),
            x.get(robot_obj, "pos_arm_joint7"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        self._robot.set_joints(arm_conf)

        # Update the cube state.
        for cube_name in x.get_object_names():
            if "cube" in cube_name:
                cube_obj = x.get_object_from_name(cube_name)
                cube_pose = Pose(
                    (x.get(cube_obj, "x"), x.get(cube_obj, "y"), x.get(cube_obj, "z")),
                    (
                        x.get(cube_obj, "qx"),
                        x.get(cube_obj, "qy"),
                        x.get(cube_obj, "qz"),
                        x.get(cube_obj, "qw"),
                    ),
                )
                set_pose(self._cubes[cube_name], cube_pose, self._physics_client_id)

        if "cupboard_1" in x.get_object_names():
            cupboard1_obj = x.get_object_from_name("cupboard_1")
            cupboard1_shelf_pose = Pose(
                (
                    x.get(cupboard1_obj, "x"),
                    x.get(cupboard1_obj, "y"),
                    x.get(cupboard1_obj, "z"),
                ),
                (
                    x.get(cupboard1_obj, "qx"),
                    x.get(cupboard1_obj, "qy"),
                    x.get(cupboard1_obj, "qz"),
                    x.get(cupboard1_obj, "qw"),
                ),
            )
            assert self._cupboard1_shelf_id is not None
            set_pose(
                self._cupboard1_shelf_id,
                cupboard1_shelf_pose,
                self._physics_client_id,
            )

        if held_object:
            held_object_id = self._cubes[held_object.name]
            set_robot_joints_with_held_object(
                self._robot,
                self._physics_client_id,
                held_object_id,
                self.base_link_to_held_obj,
                extend_joints_to_include_fingers(arm_conf[:7]),
            )

    def get_ee_pose(self) -> Pose:
        """Get the end effector pose."""
        return self._robot.get_end_effector_pose()

    def get_collision_bodies(self, held_object: int | None = None) -> set[int]:
        """Get pybullet IDs for collision bodies."""
        collision_bodies: set[int] = set()
        collision_bodies.update(self._cubes.values())
        if self._cupboard1_shelf_id is not None:
            collision_bodies.add(self._cupboard1_shelf_id)
        if held_object is not None:
            collision_bodies.discard(held_object)
        return collision_bodies

    def get_joint_distance(self, conf1: JointPositions, conf2: JointPositions) -> float:
        """Get the distance between two arm confs."""
        return self._joint_distance_fn(conf1, conf2)

    def close(self) -> None:
        """Close the PyBullet simulator."""
        p.disconnect(self._physics_client_id)
