"""How hard a toss is thrown, and the swing itself.

Plain functions over a motion plan rather than controller methods, so the release
schedule can be tested without a simulator and so every toss is timed the same way. The
limits are hand-tuned for throwing rather than derived from the arm's own, and are
demonstrated on the real TidyBot (`yixuanhuang98/tidybot_real`, `robot/kinova.py`).
"""

from typing import NamedTuple

import numpy as np
from kinder.envs.dynamic3d.mujoco_utils import CONTROL_SCHEDULE_TIMESTEP
from pybullet_helpers.inverse_kinematics import JointPositions
from relational_structs import Array

from kinder_models.dynamic3d.utils import _CONTROL_TIMESTEP, _trapezoidal_motion_profile

# Wind up and back, then swing forward and release.
TOSS_WINDUP_ARM_CONFIGURATION = np.deg2rad(
    [0.0, 50.0, 180.0, -110.0, 0.0, -100.0, 90.0]
)
TOSS_RELEASE_ARM_CONFIGURATION = np.deg2rad([0.0, 20.0, 180.0, -35.0, 0.0, 25.0, 90.0])

# Deliberately over-driving _ARM_MAX_VELOCITY: a toss throws hard on purpose.
TOSS_MAX_VELOCITY = np.deg2rad(140.0)
TOSS_MAX_ACCELERATION = np.deg2rad(300.0)
TOSS_MAX_DECELERATION = np.deg2rad(200.0)

# 1 ms, matching the real robot's 1 kHz servo loop.
TOSS_SLICES_PER_CONTROL_STEP = int(round(_CONTROL_TIMESTEP / CONTROL_SCHEDULE_TIMESTEP))

# Milliseconds from the start of the swing, as movej_primitive.execute() takes.
# Re-derive by running the swing, not by recomputing from the confs.
TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS = 720


class TossSwing(NamedTuple):
    """A planned swing: where the arm goes, and when the gripper opens."""

    trajectory: np.ndarray
    direction: np.ndarray
    start_joint_angles: np.ndarray
    release_step: int
    release_slice: int


def toss_swing_action(
    swing: TossSwing,
    step_idx: int,
    current_joint_angles: JointPositions,
    gripper_pose: float,
    has_released: bool,
) -> Array:
    """The swing's command for one control step, opening the gripper mid-step.

    The usual (18,) action, except on the step the release falls inside, which returns a
    (TOSS_SLICES_PER_CONTROL_STEP, 18) schedule so gripper_release_ms means the
    millisecond it names rather than the next step boundary.
    """
    action = np.zeros(18, dtype=np.float32)
    idx = min(step_idx, len(swing.trajectory) - 1)
    s = float(swing.trajectory[idx])
    if idx > 0:
        ds = (swing.trajectory[idx] - swing.trajectory[idx - 1]) / _CONTROL_TIMESTEP
    else:
        ds = 0.0
    kp = 2.0
    kv = 2.0
    target_joint_angles = swing.start_joint_angles + swing.direction * s
    action[3:10] = kp * (target_joint_angles - np.array(current_joint_angles[:7]))
    action[11:18] = swing.direction * (ds * kv)

    if has_released or step_idx > swing.release_step:
        action[10] = 0.0
        return action
    if step_idx != swing.release_step:
        action[10] = gripper_pose
        return action
    if swing.release_slice == 0:
        action[10] = 0.0
        return action
    schedule = np.repeat(action[None], TOSS_SLICES_PER_CONTROL_STEP, axis=0)
    schedule[: swing.release_slice, 10] = gripper_pose
    schedule[swing.release_slice :, 10] = 0.0
    return schedule


def plan_toss_swing(
    joint_plan: list[JointPositions],
    current_joint_angles: JointPositions,
    release_speed: float = TOSS_MAX_VELOCITY,
    gripper_release_ms: int = TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
) -> TossSwing:
    """Time a motion plan as a toss, and fix the millisecond the gripper opens on.

    gripper_release_ms is deliberately NOT clamped to the swing's duration: a value at
    or past the end means the gripper never opens and the cube is never thrown.
    """
    dq = np.subtract(joint_plan[-1], current_joint_angles)[:7]
    s_total = float(np.linalg.norm(dq))
    # Not the real robot's controller: the parameter space matches, the trajectory does
    # not. Do not align this with the _compute_per_joint_profile siblings.
    direction = dq / s_total if s_total > 1e-4 else np.zeros(7)
    max_vel, max_accel, max_decel = toss_profile_limits(release_speed)
    trajectory = _trapezoidal_motion_profile(
        s_total,
        max_vel=max_vel,
        max_accel=max_accel,
        max_decel=max_decel,
        step_size=_CONTROL_TIMESTEP,
    )
    release_step, release_slice = divmod(
        int(gripper_release_ms), TOSS_SLICES_PER_CONTROL_STEP
    )
    return TossSwing(
        trajectory,
        direction,
        np.array(current_joint_angles[:7]),
        release_step,
        release_slice,
    )


def toss_profile_limits(
    release_speed: float = TOSS_MAX_VELOCITY,
) -> tuple[float, float, float]:
    """The (max_vel, max_accel, max_decel) triple a toss at release_speed is timed by.

    One factor on all three, so this is an effort and not a speed cap: raising max_vel
    alone turns the profile triangular and moves the release into the acceleration
    phase. Clamped at 1, the real arm's own ceiling.
    """
    effort = min(max(release_speed / TOSS_MAX_VELOCITY, 0.0), 1.0)
    return (
        TOSS_MAX_VELOCITY * effort,
        TOSS_MAX_ACCELERATION * effort,
        TOSS_MAX_DECELERATION * effort,
    )
