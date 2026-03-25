"""Utils for tidybot environments."""

import math
from typing import Iterable

import numpy as np
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from matplotlib import pyplot as plt
from prpl_utils.motion_planning import BiRRT
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs import (
    Object,
    ObjectCentricState,
)
from spatialmath import SE2, UnitQuaternion
from tomsgeoms2d.structs import Geom2D, Rectangle
from tomsgeoms2d.utils import geom2ds_intersect

# Control period in seconds (10 Hz).
_CONTROL_DT = 0.1


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

    disable_collision = True
    # Construct geoms.
    (robot,) = state.get_objects(MujocoTidyBotRobotObjectType)
    robot_width, robot_height, _ = get_bounding_box(state, robot)
    obstacles = state.get_objects(MujocoObjectType)
    if disable_collision_objects is not None:
        obstacles = [o for o in obstacles if o.name not in disable_collision_objects]
    if disable_collision:
        obstacle_geoms = set()
    else:
        geoms = get_overhead_kinematic2ds(state)
        obstacle_geoms = {geoms[o.name] for o in obstacles}

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
    return birrt.query(initial_pose, target_base_pose)


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
