"""Utils for kinematic3d environments."""

import numpy as np
from pybullet_helpers.geometry import SE2Pose


# Utility functions.
def get_target_robot_pose_from_parameters(
    target_object_pose: SE2Pose, target_distance: float, target_rot: float
) -> SE2Pose:
    """Determine the pose for the robot given the state and parameters.

    The robot will be facing the target_object_pose position while being target_distance
    away, and rotated w.r.t. the target_object_pose rotation by target_rot.
    """
    # Absolute angle of the line from the robot to the target.
    ang = target_object_pose.rot + target_rot

    # Place the robot `target_distance` away from the target along -ang
    tx, ty = target_object_pose.x, target_object_pose.y  # target translation (x, y).
    rx = tx - target_distance * np.cos(ang)
    ry = ty - target_distance * np.sin(ang)

    # Robot faces the target: heading points along +ang (toward the target).
    return SE2Pose(rx, ry, ang)


def step_toward_se2_waypoint(
    current_base_pose: SE2Pose,
    plan: list[SE2Pose],
    max_action_mag: float,
) -> tuple[list[float], bool]:
    """Compute a bounded base delta toward the next waypoint in the plan.

    The waypoint is popped only once it is reachable within max_action_mag on
    every coordinate, so a clipped or otherwise short step is retried instead of
    silently skipping ahead. Returns the [dx, dy, drot] delta and whether the
    plan is exhausted after this step.
    """
    target_base_pose = plan[0]
    delta = target_base_pose - current_base_pose
    coords = [delta.x, delta.y, delta.rot]
    if all(abs(c) <= max_action_mag for c in coords):
        plan.pop(0)
    else:
        coords = [float(np.clip(c, -max_action_mag, max_action_mag)) for c in coords]
    return coords, not plan
