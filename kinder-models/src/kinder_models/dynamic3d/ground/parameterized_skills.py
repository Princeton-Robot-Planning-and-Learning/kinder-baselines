"""Parameterized skills for the TidyBot3D ground environment."""

import math
from typing import Any, Union

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from gymnasium.spaces import Box
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import (
    TidyBot3DRobotActionSpace,
)
from prpl_utils.utils import get_signed_angle_distance
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    JointPositions,
    inverse_kinematics,
)
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
)
from relational_structs import (
    Array,
    ObjectCentricState,
    Variable,
)
from spatialmath import SE2

from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCEL,
    _ARM_MAX_VEL,
    _CONTROL_DT,
    ARM_MOVEMENT_CUPBOARD,
    BASE_DISTANCE_TO_CUPBOARD,
    BASE_TO_CUPBOARD_ROTATION,
    GRASP_CLOSE_THRESHOLD,
    GRASP_TRANSFORM_TO_OBJECT,
    GRIPPER_CLOSED_THRESHOLD,
    GRIPPER_OPEN_THRESHOLD,
    MAX_SAMPLER_ATTEMPTS,
    MOVE_TO_TARGET_DISTANCE_BOUNDS,
    MOVE_TO_TARGET_ROT_BOUNDS,
    PLACE_SAMPLER_COLLISION_THRESHOLD,
    PLACE_SAMPLER_X_OFFSET_BOUNDS,
    PLACE_SAMPLER_Y_OFFSET_BOUNDS,
    ROBOT_ARM_POSE_TO_BASE,
    WAYPOINT_TOL,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    _compute_per_joint_profile,
    _trapezoidal_motion_profile,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
)

# Constants.
_CONTROL_DT = 0.1  # Control period in seconds (10 Hz).
_ARM_MAX_VEL = np.deg2rad(np.array([80.0, 80.0, 80.0, 80.0, 70.0, 70.0, 70.0]))
_ARM_MAX_ACCEL = np.deg2rad(np.array([297.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]))
MAX_BASE_MOVEMENT_MAGNITUDE = 1e-1
GRIPPER_OPEN_THRESHOLD = 0.01
GRASP_CLOSE_THRESHOLD = 1.0  # for stable grasp
GRIPPER_CLOSED_THRESHOLD = 0.02
WAYPOINT_TOL = 4 * 1e-2
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.45, 0.6)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi, np.pi)
WORLD_X_BOUNDS = (-2.5, 2.5)  # we should move these later
WORLD_Y_BOUNDS = (-2.5, 2.5)  # we should move these later
ROBOT_ARM_POSE_TO_BASE = Pose((0.12, 0.0, 0.4))
GRASP_TRANSFORM_TO_OBJECT = Pose((-0.005, 0, 0.01), (0.707, 0.707, 0, 0))
WIPER_TRANSFORM_TO_OBJECT = Pose.from_rpy(
    (0.06, 0, 0.03), (-np.pi - np.pi / 16, 0, -np.pi / 2)
)  # Pose((0.035, 0, 0.04), (-0.707, 0.707, 0, 0))
WIPER_SWEEP_TRANSFORM = Pose.from_rpy(
    (-0.04, 0, 0.045), (-np.pi, 0, -np.pi / 2)
)  # Pose((-0.05, 0, 0.04), (-0.707, 0.707, 0, 0))
WIPER_SWEEP_TRANSFORM_END = Pose.from_rpy(
    (0.25, 0, 0.095), (-np.pi + np.pi / 4, 0, -np.pi / 2)
)  # Pose((0.15, 0, 0.04), (-0.707, 0.707, 0, 0))
DRAWER_TRANSFORM_TO_OBJECT = Pose.from_rpy(
    (0.07, 0.3, -0.12), (-np.pi - np.pi / 16, 0, -np.pi / 2)
)
DRAWER_TRANSFORM_TO_OBJECT_END = Pose.from_rpy(
    (0.18, 0.3, -0.12), (-np.pi - np.pi / 16, 0, -np.pi / 2)
)
BASE_DISTANCE_TO_CUPBOARD = 0.95
ARM_MOVEMENT_CUPBOARD = Pose((0.8, 0.0, 0.25), (0.5, 0.5, 0.5, 0.5))
PLACE_SAMPLER_COLLISION_THRESHOLD = 0.05
PLACE_SAMPLER_X_OFFSET_BOUNDS = (-0.10, 0)
PLACE_SAMPLER_Y_OFFSET_BOUNDS = (-0.15, 0.15)
MAX_SAMPLER_ATTEMPTS = 100
BASE_TO_CUPBOARD_ROTATION = -np.pi / 2


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


# Utility functions.
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


class MoveToTargetGroundController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Controller for motion planning to reach a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.

    The continuous parameters are:
        target_distance: float
        target_rot: float (radians)

    The controller uses motion planning to move the robot base to reach the target. The
    target base pose is computed as follows: starting with the target object pose, get
    the target _robot_ pose by applying the target distance and target rot from the
    continuous parameters. Note that the robot will always be facing directly towards
    the target object.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: Union[tuple[float, ...], float, None] = None
        self._current_base_motion_plan: list[SE2] | None = None

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator, rotate: bool = False
    ) -> Any:
        if rotate:
            if self.objects[2].name == "cube1":
                distance = 0.85
            elif self.objects[2].name == "cube2":
                distance = 0.92
            else:
                raise ValueError(f"Unknown target object: {self.objects[2].name}")
            rot = -np.pi / 2
        else:
            distance = 0.5  # for stable grasp
            rot = 0.0
        return np.array([distance, rot])

    def reset(
        self,
        x: ObjectCentricState,
        params: Union[tuple[float, ...], float],
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
        disable_collision_objects: list[str] | None = None,
    ) -> None:
        self._last_state = x
        self._current_params = params
        # Derive the target pose for the robot.
        target_distance, target_rot = self._current_params
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, target_distance, target_rot
        )
        # Run motion planning.
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # use a constant seed to effectively make this "deterministic"
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
            disable_collision_objects=disable_collision_objects,
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

    def terminated(self) -> bool:
        assert self._current_base_motion_plan is not None
        return self._robot_is_close_to_pose(self._current_base_motion_plan[-1])

    def step(self) -> Array:
        assert self._current_base_motion_plan is not None
        while len(self._current_base_motion_plan) > 1:
            peek_pose = self._current_base_motion_plan[0]
            # Close enough, pop and continue.
            if self._robot_is_close_to_pose(peek_pose):
                self._current_base_motion_plan.pop(0)
            # Not close enough, stop popping.
            break
        robot_pose = self._get_current_robot_pose()
        next_pose = self._current_base_motion_plan[0]
        dx = next_pose.x - robot_pose.x
        dy = next_pose.y - robot_pose.y
        drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
        action = np.zeros(11, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        action[2] = drot
        action[-1] = self._get_current_robot_gripper_pose()
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_pose(self, pose: SE2, atol: float = WAYPOINT_TOL) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.isclose(robot_pose.x, pose.x, atol=atol)
            and np.isclose(robot_pose.y, pose.y, atol=atol)
            and np.isclose(
                get_signed_angle_distance(robot_pose.theta(), pose.theta()),
                0.0,
                atol=atol,
            )
        )


class MoveArmToConfController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning the arm to reach a target conf.

    The object parameters are:
        robot: The robot itself.

    The continuous parameters are:
        joint1_target: float
        joint2_target: float
        ...
        joint7_target: float

    The controller uses motion planning in pybullet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: Union[tuple[float, ...], float, None] = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None
        self._trajectory: np.ndarray = np.array([])
        self._traj_dir: np.ndarray = np.zeros(7)
        self._start_joint_angles: np.ndarray = np.zeros(7)
        self._step_idx: int = 0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target arm conf themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Union[tuple[float, ...], float]) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        self._current_params = params
        target_joints = list(self._current_params) + ([0.0] * 6)
        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(x)
        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )
        assert plan is not None, "Motion planning failed"
        self._current_arm_joint_plan = plan
        # Compute trapezoidal velocity profile along the path.
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        final = np.array(self._current_arm_joint_plan[-1][:7])
        self._trajectory, self._traj_dir = _compute_per_joint_profile(
            curr,
            final,
            _ARM_MAX_VEL,
            _ARM_MAX_ACCEL,
        )
        self._start_joint_angles = curr.copy()
        self._step_idx = 0

    def terminated(self) -> bool:
        return self._step_idx >= len(self._trajectory)

    def step(self) -> Array:
        gripper_pose = self._get_current_robot_gripper_pose()
        action = np.zeros(18, dtype=np.float32)

        idx = min(self._step_idx, len(self._trajectory) - 1)
        s = float(self._trajectory[idx])

        # Velocity via finite difference.
        if idx > 0:
            ds = (self._trajectory[idx] - self._trajectory[idx - 1]) / _CONTROL_DT
        else:
            ds = 0.0

        kp = 2.0
        kv = 2.0
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        target = self._start_joint_angles + self._traj_dir * s
        action[3:10] = kp * (target - curr)
        action[11:18] = self._traj_dir * (ds * kv)
        action[10] = gripper_pose

        self._step_idx += 1
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0


class TossController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning the arm to reach a target conf.

    The object parameters are:
        robot: The robot itself.

    The continuous parameters are:
        joint1_target: float
        joint2_target: float
        ...
        joint7_target: float

    The controller uses motion planning in pybullet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: Union[tuple[float, ...], float, None] = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None
        # Fraction of toss path at which to release gripper.
        self._release_fraction: float = 0.46
        self._step_idx: int = 0
        self._toss_dir: np.ndarray = np.zeros(7)
        self._trajectory: np.ndarray = np.array([])
        self._has_released: bool = False
        self._start_joint_angles: np.ndarray = np.zeros(7)

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target arm conf themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Union[tuple[float, ...], float]) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        self._current_params = params
        target_joints = list(self._current_params) + ([0.0] * 6)
        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(x)
        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )
        assert plan is not None, "Motion planning failed"
        self._current_arm_joint_plan = plan
        # Compute trapezoidal velocity profile along the path.
        curr_joint_angles = self._get_current_robot_arm_conf()
        final_joint_angles = self._current_arm_joint_plan[-1]
        dq = np.subtract(final_joint_angles, curr_joint_angles)[:7]
        s_total = float(np.linalg.norm(dq))
        if s_total > 1e-4:
            self._toss_dir = dq / s_total
        else:
            self._toss_dir = np.zeros(7)
        self._trajectory = _trapezoidal_motion_profile(
            s_total,
            max_vel=np.deg2rad(140),
            max_accel=np.deg2rad(300),
            max_decel=np.deg2rad(200),
            step_size=_CONTROL_DT,
        )
        self._start_joint_angles = np.array(curr_joint_angles[:7])
        self._has_released = False
        self._step_idx = 0

    def terminated(self) -> bool:
        # Terminate when we've gone through the entire profile.
        return self._step_idx >= len(self._trajectory)

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        gripper_pose = self._get_current_robot_gripper_pose()
        action = np.zeros(18, dtype=np.float32)

        # Look up target distance along path from precomputed trapezoidal profile.
        idx = min(self._step_idx, len(self._trajectory) - 1)
        s = float(self._trajectory[idx])
        # Compute velocity via finite difference of the profile.
        if idx > 0:
            ds = (self._trajectory[idx] - self._trajectory[idx - 1]) / _CONTROL_DT
        else:
            ds = 0.0

        # Position target with feedforward gain to compensate for tracking lag.
        kp = 2.0
        kv = 2.0
        curr_joint_angles = self._get_current_robot_arm_conf()
        target_joint_angles = self._start_joint_angles + self._toss_dir * s
        action[3:10] = kp * (target_joint_angles - np.array(curr_joint_angles[:7]))

        # Velocity feedforward along the toss direction.
        action[11:18] = self._toss_dir * (ds * kv)

        # Determine release point based on fraction of total distance.
        s_total = self._trajectory[-1] if len(self._trajectory) > 0 else 0.0
        fraction_covered = s / s_total if s_total > 0 else 1.0
        should_release = (
            self._has_released or fraction_covered >= self._release_fraction
        )

        if should_release:
            action[10] = 0.0
            self._has_released = True
        else:
            action[10] = gripper_pose

        self._step_idx += 1
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(self, conf: JointPositions) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < 6 * 1e-2


class MoveArmToEndEffectorController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Controller for motion planning the arm to reach a target end effector pose.

    The object parameters are:
        robot: The robot itself.

    The continuous parameters are:
        end_effector_pose: np.ndarray (x, y, z, rw, rx, ry, rz)

    The controller uses motion planning in pybullet.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: Union[tuple[float, ...], float, None] = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None
        self._trajectory: np.ndarray = np.array([])
        self._traj_dir: np.ndarray = np.zeros(7)
        self._start_joint_angles: np.ndarray = np.zeros(7)
        self._step_idx: int = 0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Union[tuple[float, ...], float]) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        self._current_params = params

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(x)

        current_arm_base_pose = self._pybullet_sim.robot.get_base_pose()

        target_end_effector_pose_temp = multiply_poses(
            current_arm_base_pose,
            Pose(
                (
                    self._current_params[0],
                    self._current_params[1],
                    self._current_params[2],
                ),
                (
                    self._current_params[3],
                    self._current_params[4],
                    self._current_params[5],
                    self._current_params[6],
                ),
            ),
        )

        rotation = Pose.from_rpy((0, 0, 0), (0, 0, self._current_params[7]))
        target_end_effector_pose = multiply_poses(
            target_end_effector_pose_temp,
            rotation,
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        self._current_arm_joint_plan = plan
        # Compute trapezoidal velocity profile along the path.
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        final = np.array(self._current_arm_joint_plan[-1][:7])
        self._trajectory, self._traj_dir = _compute_per_joint_profile(
            curr,
            final,
            _ARM_MAX_VEL,
            _ARM_MAX_ACCEL,
        )
        self._start_joint_angles = curr.copy()
        self._step_idx = 0

    def terminated(self) -> bool:
        return self._step_idx >= len(self._trajectory)

    def step(self) -> Array:
        gripper_pose = self._get_current_robot_gripper_pose()
        action = np.zeros(18, dtype=np.float32)

        idx = min(self._step_idx, len(self._trajectory) - 1)
        s = float(self._trajectory[idx])

        # Velocity via finite difference.
        if idx > 0:
            ds = (self._trajectory[idx] - self._trajectory[idx - 1]) / _CONTROL_DT
        else:
            ds = 0.0

        kp = 2.0
        kv = 2.0
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        target = self._start_joint_angles + self._traj_dir * s
        action[3:10] = kp * (target - curr)
        action[11:18] = self._traj_dir * (ds * kv)
        action[10] = gripper_pose

        self._step_idx += 1
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0


class CloseGripperController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for closing the gripper.

    The object parameters are:
        robot: The robot itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self.last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Union[tuple[float, ...], float, None] = None) -> None:
        # Update the current state and parameters.
        self._last_state = x

    def terminated(self) -> bool:
        return self._robot_gripper_is_closed(atol=0.02)

    def step(self) -> Array:
        self.last_gripper_state = self._get_current_gripper_pose()
        action = np.zeros(11, dtype=np.float32)
        action[-1] = 1
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_gripper_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return state.get(robot, "pos_gripper")

    def _robot_gripper_is_closed(self, atol: float = GRIPPER_CLOSED_THRESHOLD) -> bool:
        current_gripper_pose = self._get_current_gripper_pose()
        return current_gripper_pose > 0.2 and np.isclose(
            current_gripper_pose, self.last_gripper_state, atol=atol
        )


class OpenGripperController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for opening the gripper.

    The object parameters are:
        robot: The robot itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self.last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target end effector pose themselves.
        raise NotImplementedError

    def reset(self, x: ObjectCentricState, params: Union[tuple[float, ...], float, None] = None) -> None:
        # Update the current state and parameters.
        self._last_state = x

    def terminated(self) -> bool:
        return self._robot_gripper_is_open()

    def step(self) -> Array:
        self.last_gripper_state = self._get_current_gripper_pose()
        action = np.zeros(11, dtype=np.float32)
        action[-1] = 0
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_gripper_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return state.get(robot, "pos_gripper")

    def _robot_gripper_is_open(self, atol: float = GRIPPER_OPEN_THRESHOLD) -> bool:
        current_gripper_pose = self._get_current_gripper_pose()
        return current_gripper_pose < atol


class PickGroundController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning to pick up a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: Union[tuple[float, ...], float, None] = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_base_motion_plan: list[SE2] | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._navigated: bool = False
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration
        # Trapezoidal velocity profiles (approach and retract phases).
        self._approach_trajectory: np.ndarray = np.array([])
        self._approach_traj_dir: np.ndarray = np.zeros(7)
        self._approach_start_joints: np.ndarray = np.zeros(7)
        self._approach_step_idx: int = 0
        self._retract_trajectory: np.ndarray = np.array([])
        self._retract_traj_dir: np.ndarray = np.zeros(7)
        self._retract_start_joints: np.ndarray = np.zeros(7)
        self._retract_step_idx: int = 0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)

        for _ in range(MAX_SAMPLER_ATTEMPTS):
            distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
            rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
            target_base_pose = get_target_robot_pose_from_parameters(
                target_object_pose, distance, rot
            )
            collision = False
            for other_object in x.get_objects(MujocoMovableObjectType):
                if (
                    "cube" in other_object.name
                    and other_object.name != target_object.name
                ):
                    other_object_pose = get_overhead_object_se2_pose(x, other_object)
                    collision_distance = float(
                        np.linalg.norm(
                            [
                                target_base_pose.x - other_object_pose.x,
                                target_base_pose.y - other_object_pose.y,
                            ]
                        )
                    )
                    if collision_distance < 0.6:
                        collision = True
                        break
            if not collision:
                return np.array([distance, rot])

        raise ValueError("No valid parameters found")

    def reset(
        self,
        x: ObjectCentricState,
        params: Union[tuple[float, ...], float],
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        self._current_params = params
        # Derive the target pose for the robot.
        target_distance, target_rot = self._current_params
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, target_distance, target_rot
        )
        # Run motion planning.
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # use a constant seed to effectively make this "deterministic"
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        plan_x = x.copy()
        robot = self.objects[0]  # Robot is first parameter
        target_base_pose = self._current_base_motion_plan[-1]
        if not self._navigated:
            plan_x.set(robot, "pos_base_x", target_base_pose.x)
            plan_x.set(robot, "pos_base_y", target_base_pose.y)
            plan_x.set(robot, "pos_base_rot", target_base_pose.theta())

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(plan_x)

        target_object = self.objects[1]

        target_grap_pose_world = Pose(
            (
                plan_x.get(target_object, "x"),
                plan_x.get(target_object, "y"),
                plan_x.get(target_object, "z"),
            ),
            (
                plan_x.get(target_object, "qx"),
                plan_x.get(target_object, "qy"),
                plan_x.get(target_object, "qz"),
                plan_x.get(target_object, "qw"),
            ),
        )

        target_end_effector_pose = multiply_poses(
            target_grap_pose_world,
            GRASP_TRANSFORM_TO_OBJECT,
        )

        self._pybullet_sim.base_link_to_held_obj = multiply_poses(
            target_end_effector_pose.invert(),
            target_grap_pose_world,
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            target_joints,
            self.home_joints.tolist(),
            collision_bodies=self._pybullet_sim.get_collision_bodies(  # pylint: disable=protected-access
                held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                    target_object.name
                ]
            ),
            held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                target_object.name
            ],
            base_link_to_held_obj=self._pybullet_sim.base_link_to_held_obj,  # pylint: disable=protected-access
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        assert retract_plan is not None, "Motion planning failed"

        # Remap the plan to ensure we stay within action limits.
        plan = remap_joint_position_plan_to_constant_distance(
            plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        # Remap the plan to ensure we stay within action limits.
        retract_plan = remap_joint_position_plan_to_constant_distance(
            retract_plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        self._current_arm_joint_plan = plan
        self._current_retract_plan = retract_plan
        # Compute trapezoidal velocity profile for approach (current → grasp conf).
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        final = np.array(plan[-1][:7])
        self._approach_trajectory, self._approach_traj_dir = _compute_per_joint_profile(
            curr, final, _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._approach_start_joints = curr.copy()
        self._approach_step_idx = 0
        # Compute trapezoidal velocity profile for retract (grasp conf → home).
        self._retract_trajectory, self._retract_traj_dir = _compute_per_joint_profile(
            final, self.home_joints[:7], _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._retract_start_joints = final.copy()
        self._retract_step_idx = 0

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_retract_plan is not None
        )
        return self._lifted

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        assert self._current_base_motion_plan is not None
        # first substep
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
            robot_pose = self._get_current_robot_pose()
            next_pose = self._current_base_motion_plan[0]
            dx = next_pose.x - robot_pose.x
            dy = next_pose.y - robot_pose.y
            drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
            action = np.zeros(11, dtype=np.float32)
            action[0] = dx
            action[1] = dy
            action[2] = drot
            action[-1] = self._get_current_robot_gripper_pose()
            return action
        if self._navigated and not self._pre_grasp and not self._closed_gripper:
            if self._approach_step_idx >= len(self._approach_trajectory):
                self._pre_grasp = True
            idx = min(self._approach_step_idx, len(self._approach_trajectory) - 1)
            s = float(self._approach_trajectory[idx])
            kp = 2.0
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._approach_start_joints + self._approach_traj_dir * s
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = kp * (target - curr)
            action[-1] = self._get_current_robot_gripper_pose()
            self._approach_step_idx += 1
            return action
        if self._pre_grasp and not self._closed_gripper:
            if self._get_current_robot_gripper_pose() > 0.2 and np.isclose(
                self._get_current_robot_gripper_pose(),
                self._last_gripper_state,
                atol=0.02,
            ):
                self._closed_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 1
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._pre_grasp and self._closed_gripper:
            if self._retract_step_idx >= len(self._retract_trajectory):
                self._lifted = True
            idx = min(self._retract_step_idx, len(self._retract_trajectory) - 1)
            s = float(self._retract_trajectory[idx])
            kp = 2.0
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._retract_start_joints + self._retract_traj_dir * s
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = kp * (target - curr)
            action[-1] = self._get_current_robot_gripper_pose()
            self._retract_step_idx += 1
            return action
        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        # return x.get(robot_obj, "pos_gripper")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = WAYPOINT_TOL
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

    def _robot_is_close_to_pose(self, pose: SE2, atol: float = WAYPOINT_TOL) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.isclose(robot_pose.x, pose.x, atol=atol)
            and np.isclose(robot_pose.y, pose.y, atol=atol)
            and np.isclose(
                get_signed_angle_distance(robot_pose.theta(), pose.theta()),
                0.0,
                atol=atol,
            )
        )


class PlaceGroundController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning to place a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: Union[tuple[float, ...], float, None] = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_base_motion_plan: list[SE2] | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._navigated: bool = False
        self._pre_place: bool = False
        self._open_gripper: bool = False
        self._returned: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration
        # Trapezoidal velocity profiles (approach and retract phases).
        self._approach_trajectory: np.ndarray = np.array([])
        self._approach_traj_dir: np.ndarray = np.zeros(7)
        self._approach_start_joints: np.ndarray = np.zeros(7)
        self._approach_step_idx: int = 0
        self._retract_trajectory: np.ndarray = np.array([])
        self._retract_traj_dir: np.ndarray = np.zeros(7)
        self._retract_start_joints: np.ndarray = np.zeros(7)
        self._retract_step_idx: int = 0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        cupboard_obj = x.get_object_from_name("cupboard_1")
        cupboard_pose = get_overhead_object_se2_pose(x, cupboard_obj)
        rot = BASE_TO_CUPBOARD_ROTATION
        # sample placements
        for _ in range(MAX_SAMPLER_ATTEMPTS):
            pose_x_offset = rng.uniform(*PLACE_SAMPLER_X_OFFSET_BOUNDS)
            pose_y_offset = rng.uniform(*PLACE_SAMPLER_Y_OFFSET_BOUNDS)
            collision = False
            for other_obj in x.get_objects(MujocoMovableObjectType):
                if other_obj.name == self.objects[1].name:
                    continue
                other_object_pose = get_overhead_object_se2_pose(x, other_obj)
                if (
                    np.linalg.norm(
                        np.array(
                            [
                                pose_x_offset
                                + cupboard_pose.x
                                + (
                                    ARM_MOVEMENT_CUPBOARD.position[0]
                                    + ROBOT_ARM_POSE_TO_BASE.position[0]
                                    - BASE_DISTANCE_TO_CUPBOARD
                                ),  # the offset of the cupboard from the cubes.
                                pose_y_offset + cupboard_pose.y,
                            ]
                        )
                        - np.array([other_object_pose.x, other_object_pose.y])
                    )
                    < PLACE_SAMPLER_COLLISION_THRESHOLD
                ):
                    collision = True
                    break
            if not collision:
                return np.array(
                    [BASE_DISTANCE_TO_CUPBOARD + pose_x_offset, pose_y_offset, rot]
                )
        raise ValueError("No valid parameters found")

    def reset(
        self,
        x: ObjectCentricState,
        params: Union[tuple[float, ...], float],
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        self._current_params = params
        # Derive the target pose for the robot.
        target_distance, target_offset, target_rot = self._current_params
        target_object = self.objects[2]
        target_object_pose_temp = get_overhead_object_se2_pose(x, target_object)
        target_object_pose = SE2(
            target_object_pose_temp.x,
            target_object_pose_temp.y + target_offset,
            target_object_pose_temp.theta(),
        )
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, target_distance, target_rot
        )
        # Run motion planning.
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # use a constant seed to effectively make this "deterministic"
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
            disable_collision_objects=[self.objects[1].name],
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        plan_x = x.copy()
        robot = self.objects[0]  # Robot is first parameter
        target_base_pose = self._current_base_motion_plan[-1]
        plan_x.set(robot, "pos_base_x", target_base_pose.x)
        plan_x.set(robot, "pos_base_y", target_base_pose.y)
        plan_x.set(robot, "pos_base_rot", target_base_pose.theta())

        target_object_place = self.objects[1]

        assert target_object_place is not None
        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(plan_x, target_object_place)

        current_arm_base_pose = self._pybullet_sim.robot.get_base_pose()

        target_end_effector_pose = ARM_MOVEMENT_CUPBOARD

        target_end_effector_pose = multiply_poses(
            current_arm_base_pose, target_end_effector_pose
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(
                held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                    target_object_place.name
                ]
            ),
            seed=0,  # use a constant seed to make this effectively deterministic
            held_object=self._pybullet_sim._cubes[  # pylint: disable=protected-access
                target_object_place.name
            ],
            base_link_to_held_obj=self._pybullet_sim.base_link_to_held_obj,
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            target_joints,
            self.home_joints.tolist(),
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        assert retract_plan is not None, "Motion planning failed"

        # Remap the plan to ensure we stay within action limits.
        plan = remap_joint_position_plan_to_constant_distance(
            plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        # Remap the plan to ensure we stay within action limits.
        retract_plan = remap_joint_position_plan_to_constant_distance(
            retract_plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        self._current_arm_joint_plan = plan
        self._current_retract_plan = retract_plan
        # Compute trapezoidal velocity profile for approach (current → place conf).
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        final = np.array(plan[-1][:7])
        self._approach_trajectory, self._approach_traj_dir = _compute_per_joint_profile(
            curr, final, _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._approach_start_joints = curr.copy()
        self._approach_step_idx = 0
        # Compute trapezoidal velocity profile for retract (place conf → home).
        self._retract_trajectory, self._retract_traj_dir = _compute_per_joint_profile(
            final, self.home_joints[:7], _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._retract_start_joints = final.copy()
        self._retract_step_idx = 0

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_retract_plan is not None
        )
        return self._returned

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        assert self._current_base_motion_plan is not None
        # first substep
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
            robot_pose = self._get_current_robot_pose()
            next_pose = self._current_base_motion_plan[0]
            dx = next_pose.x - robot_pose.x
            dy = next_pose.y - robot_pose.y
            drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
            action = np.zeros(11, dtype=np.float32)
            action[0] = dx
            action[1] = dy
            action[2] = drot
            action[-1] = self._get_current_robot_gripper_pose()
            return action
        if self._navigated and not self._pre_place and not self._open_gripper:
            if self._approach_step_idx >= len(self._approach_trajectory):
                self._pre_place = True
            idx = min(self._approach_step_idx, len(self._approach_trajectory) - 1)
            s = float(self._approach_trajectory[idx])
            kp = 2.0
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._approach_start_joints + self._approach_traj_dir * s
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = kp * (target - curr)
            action[-1] = self._get_current_robot_gripper_pose()
            self._approach_step_idx += 1
            return action
        if self._pre_place and not self._open_gripper:
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._open_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 0
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._pre_place and self._open_gripper:
            if self._retract_step_idx >= len(self._retract_trajectory):
                self._returned = True
            idx = min(self._retract_step_idx, len(self._retract_trajectory) - 1)
            s = float(self._retract_trajectory[idx])
            kp = 2.0
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._retract_start_joints + self._retract_traj_dir * s
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = kp * (target - curr)
            action[-1] = self._get_current_robot_gripper_pose()
            self._retract_step_idx += 1
            return action
        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        # return x.get(robot_obj, "pos_gripper")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = WAYPOINT_TOL
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

    def _robot_is_close_to_pose(self, pose: SE2, atol: float = WAYPOINT_TOL) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.isclose(robot_pose.x, pose.x, atol=atol)
            and np.isclose(robot_pose.y, pose.y, atol=atol)
            and np.isclose(
                get_signed_angle_distance(robot_pose.theta(), pose.theta()),
                0.0,
                atol=atol,
            )
        )


class OpenDrawerSweepController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Controller for motion planning to pick up a wiper.

    The object parameters are:
        robot: The robot itself.
        object: The target drawer.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_open_plan: list[JointPositions] | None = None
        self._current_base_motion_plan: list[SE2] | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._navigated: bool = False
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._open_gripper: bool = False
        self._lifted: bool = False
        self._returned: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)

        for _ in range(MAX_SAMPLER_ATTEMPTS):
            distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
            rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
            target_base_pose = get_target_robot_pose_from_parameters(
                target_object_pose, distance, rot
            )
            collision = False
            for other_object in x.get_objects(MujocoMovableObjectType):
                if (
                    "cube" in other_object.name
                    and other_object.name != target_object.name
                ):
                    other_object_pose = get_overhead_object_se2_pose(x, other_object)
                    collision_distance = float(
                        np.linalg.norm(
                            [
                                target_base_pose.x - other_object_pose.x,
                                target_base_pose.y - other_object_pose.y,
                            ]
                        )
                    )
                    if collision_distance < 0.6:
                        collision = True
                        break
            if not collision:
                return np.array([distance, rot])

        raise ValueError("No valid parameters found")

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()
        # Derive the target pose for the robot.
        target_distance, target_rot = self._current_params
        target_object = self.objects[1]
        target_object_pose_ori = get_overhead_object_se2_pose(x, target_object)
        target_object_pose = SE2(
            target_object_pose_ori.x,
            target_object_pose_ori.y + 0.3,
            target_object_pose_ori.theta(),
        )
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, target_distance, target_rot
        )
        # Run motion planning.
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # use a constant seed to effectively make this "deterministic"
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        plan_x = x.copy()
        robot = self.objects[0]  # Robot is first parameter
        target_base_pose = self._current_base_motion_plan[-1]
        if not self._navigated:
            plan_x.set(robot, "pos_base_x", target_base_pose.x)
            plan_x.set(robot, "pos_base_y", target_base_pose.y)
            plan_x.set(robot, "pos_base_rot", target_base_pose.theta())

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(plan_x)

        target_object = self.objects[1]

        target_grap_pose_world = Pose(
            (
                plan_x.get(target_object, "x"),
                plan_x.get(target_object, "y"),
                plan_x.get(target_object, "z"),
            ),
            (
                plan_x.get(target_object, "qx"),
                plan_x.get(target_object, "qy"),
                plan_x.get(target_object, "qz"),
                plan_x.get(target_object, "qw"),
            ),
        )

        target_end_effector_pose = multiply_poses(
            target_grap_pose_world,
            DRAWER_TRANSFORM_TO_OBJECT,
        )

        target_end_effector_pose_end = multiply_poses(
            target_grap_pose_world,
            DRAWER_TRANSFORM_TO_OBJECT_END,
        )

        self._pybullet_sim.base_link_to_held_obj = multiply_poses(
            target_end_effector_pose.invert(),
            target_grap_pose_world,
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        target_joints_end = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose_end,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        open_plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints_end,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            target_joints_end,
            self.home_joints.tolist(),
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        assert open_plan is not None, "Motion planning failed"
        assert retract_plan is not None, "Motion planning failed"

        # Remap the plan to ensure we stay within action limits.
        plan = remap_joint_position_plan_to_constant_distance(
            plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        # Remap the plan to ensure we stay within action limits.
        open_plan = remap_joint_position_plan_to_constant_distance(
            open_plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        # Remap the plan to ensure we stay within action limits.
        retract_plan = remap_joint_position_plan_to_constant_distance(
            retract_plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        self._current_arm_joint_plan = plan
        self._current_open_plan = open_plan
        self._current_retract_plan = retract_plan

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_open_plan is not None
            and self._current_retract_plan is not None
        )
        return self._returned

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        assert self._current_base_motion_plan is not None
        # first substep
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
            robot_pose = self._get_current_robot_pose()
            next_pose = self._current_base_motion_plan[0]
            dx = next_pose.x - robot_pose.x
            dy = next_pose.y - robot_pose.y
            drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
            action = np.zeros(11, dtype=np.float32)
            action[0] = dx
            action[1] = dy
            action[2] = drot
            action[-1] = self._get_current_robot_gripper_pose()
            return action
        if self._navigated and not self._pre_grasp and not self._closed_gripper:
            while len(self._current_arm_joint_plan) > 1:
                peek_conf = self._current_arm_joint_plan[0]
                # Close enough, pop and continue.
                if len(self._current_arm_joint_plan) == 2:
                    if self._robot_is_close_to_conf(peek_conf, atol=0.02):
                        self._current_arm_joint_plan.pop(0)
                else:
                    if self._robot_is_close_to_conf(peek_conf):
                        self._current_arm_joint_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_arm_joint_plan[-1]):
                self._pre_grasp = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_arm_joint_plan[0]
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.get_arm_joint_infos()[:7]  # type: ignore  # pylint: disable=protected-access
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        if self._pre_grasp and not self._closed_gripper:
            if self._get_current_robot_gripper_pose() > 0.2 and np.isclose(
                self._get_current_robot_gripper_pose(),
                self._last_gripper_state,
                atol=0.02,
            ):
                self._closed_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 1
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._pre_grasp and self._closed_gripper and not self._lifted:
            while len(self._current_open_plan) > 1:  # type: ignore
                peek_conf = self._current_open_plan[0]  # type: ignore
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf, atol=0.08):
                    self._current_open_plan.pop(0)  # type: ignore
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_open_plan[-1]):  # type: ignore # pylint: disable=line-too-long
                self._lifted = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_open_plan[0]  # type: ignore
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.get_arm_joint_infos()[:7]  # type: ignore  # pylint: disable=protected-access
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        if self._lifted and not self._open_gripper:
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._open_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 0
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._open_gripper:
            while len(self._current_retract_plan) > 1:  # type: ignore
                peek_conf = self._current_retract_plan[0]  # type: ignore
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf, atol=0.08):
                    self._current_retract_plan.pop(0)  # type: ignore
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_retract_plan[-1]):  # type: ignore # pylint: disable=line-too-long
                self._returned = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_retract_plan[0]  # type: ignore
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.get_arm_joint_infos()[:7]  # type: ignore  # pylint: disable=protected-access
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        # return x.get(robot_obj, "pos_gripper")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = WAYPOINT_TOL
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

    def _robot_is_close_to_pose(self, pose: SE2, atol: float = WAYPOINT_TOL) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.isclose(robot_pose.x, pose.x, atol=atol)
            and np.isclose(robot_pose.y, pose.y, atol=atol)
            and np.isclose(
                get_signed_angle_distance(robot_pose.theta(), pose.theta()),
                0.0,
                atol=atol,
            )
        )


class PickWiperOriController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning to pick up a wiper.

    The object parameters are:
        robot: The robot itself.
        object: The target object.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_base_motion_plan: list[SE2] | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._navigated: bool = False
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)

        for _ in range(MAX_SAMPLER_ATTEMPTS):
            distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
            rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
            target_base_pose = get_target_robot_pose_from_parameters(
                target_object_pose, distance, rot
            )
            collision = False
            for other_object in x.get_objects(MujocoMovableObjectType):
                if (
                    "cube" in other_object.name
                    and other_object.name != target_object.name
                ):
                    other_object_pose = get_overhead_object_se2_pose(x, other_object)
                    collision_distance = float(
                        np.linalg.norm(
                            [
                                target_base_pose.x - other_object_pose.x,
                                target_base_pose.y - other_object_pose.y,
                            ]
                        )
                    )
                    if collision_distance < 0.6:
                        collision = True
                        break
            if not collision:
                return np.array([distance, rot])

        raise ValueError("No valid parameters found")

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()
        # Derive the target pose for the robot.
        target_distance, target_rot = self._current_params
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, target_distance, target_rot
        )
        # Run motion planning.
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # use a constant seed to effectively make this "deterministic"
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        plan_x = x.copy()
        robot = self.objects[0]  # Robot is first parameter
        target_base_pose = self._current_base_motion_plan[-1]
        if not self._navigated:
            plan_x.set(robot, "pos_base_x", target_base_pose.x)
            plan_x.set(robot, "pos_base_y", target_base_pose.y)
            plan_x.set(robot, "pos_base_rot", target_base_pose.theta())

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(plan_x)

        target_object = self.objects[1]

        target_grap_pose_world = Pose(
            (
                plan_x.get(target_object, "x"),
                plan_x.get(target_object, "y"),
                plan_x.get(target_object, "z"),
            ),
            (
                plan_x.get(target_object, "qx"),
                plan_x.get(target_object, "qy"),
                plan_x.get(target_object, "qz"),
                plan_x.get(target_object, "qw"),
            ),
        )

        target_end_effector_pose = multiply_poses(
            target_grap_pose_world,
            WIPER_TRANSFORM_TO_OBJECT,
        )

        self._pybullet_sim.base_link_to_held_obj = multiply_poses(
            target_end_effector_pose.invert(),
            target_grap_pose_world,
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            target_joints,
            self.home_joints.tolist(),
            collision_bodies=self._pybullet_sim.get_collision_bodies(),  # pylint: disable=protected-access
            base_link_to_held_obj=self._pybullet_sim.base_link_to_held_obj,  # pylint: disable=protected-access
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        assert retract_plan is not None, "Motion planning failed"

        # Remap the plan to ensure we stay within action limits.
        plan = remap_joint_position_plan_to_constant_distance(
            plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        # Remap the plan to ensure we stay within action limits.
        retract_plan = remap_joint_position_plan_to_constant_distance(
            retract_plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        self._current_arm_joint_plan = plan
        self._current_retract_plan = retract_plan

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_retract_plan is not None
        )
        return self._lifted

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        assert self._current_base_motion_plan is not None
        # first substep
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
            robot_pose = self._get_current_robot_pose()
            next_pose = self._current_base_motion_plan[0]
            dx = next_pose.x - robot_pose.x
            dy = next_pose.y - robot_pose.y
            drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
            action = np.zeros(11, dtype=np.float32)
            action[0] = dx
            action[1] = dy
            action[2] = drot
            action[-1] = self._get_current_robot_gripper_pose()
            return action
        if self._navigated and not self._pre_grasp and not self._closed_gripper:
            while len(self._current_arm_joint_plan) > 1:
                peek_conf = self._current_arm_joint_plan[0]
                # Close enough, pop and continue.
                if len(self._current_arm_joint_plan) == 2:
                    if self._robot_is_close_to_conf(peek_conf, atol=0.02):
                        self._current_arm_joint_plan.pop(0)
                else:
                    if self._robot_is_close_to_conf(peek_conf):
                        self._current_arm_joint_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_arm_joint_plan[-1]):
                self._pre_grasp = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_arm_joint_plan[0]
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.get_arm_joint_infos()[:7]  # type: ignore  # pylint: disable=protected-access
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        if self._pre_grasp and not self._closed_gripper:
            if self._get_current_robot_gripper_pose() > 0.2 and np.isclose(
                self._get_current_robot_gripper_pose(),
                self._last_gripper_state,
                atol=0.02,
            ):
                self._closed_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 1
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._pre_grasp and self._closed_gripper:
            while len(self._current_retract_plan) > 1:  # type: ignore
                peek_conf = self._current_retract_plan[0]  # type: ignore
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf, atol=0.08):
                    self._current_retract_plan.pop(0)  # type: ignore
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_retract_plan[-1]):  # type: ignore # pylint: disable=line-too-long
                self._lifted = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_retract_plan[0]  # type: ignore
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.get_arm_joint_infos()[:7]  # type: ignore  # pylint: disable=protected-access
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        # return x.get(robot_obj, "pos_gripper")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = WAYPOINT_TOL
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

    def _robot_is_close_to_pose(self, pose: SE2, atol: float = WAYPOINT_TOL) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.isclose(robot_pose.x, pose.x, atol=atol)
            and np.isclose(robot_pose.y, pose.y, atol=atol)
            and np.isclose(
                get_signed_angle_distance(robot_pose.theta(), pose.theta()),
                0.0,
                atol=atol,
            )
        )


class SweepOriController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for sweeping target objects.

    The object parameters are:
        robot: The robot itself.
        wiper: The wiper to sweep.
        target_objects: The target objects to sweep.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_base_motion_plan: list[SE2] | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._navigated: bool = False
        self._pre_place: bool = False
        self._open_gripper: bool = False
        self._returned: bool = False
        self._last_gripper_state: float = 0.0
        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        cupboard_obj = x.get_object_from_name("cupboard_1")
        cupboard_pose = get_overhead_object_se2_pose(x, cupboard_obj)
        rot = BASE_TO_CUPBOARD_ROTATION
        # sample placements
        for _ in range(MAX_SAMPLER_ATTEMPTS):
            pose_x_offset = rng.uniform(*PLACE_SAMPLER_X_OFFSET_BOUNDS)
            pose_y_offset = rng.uniform(*PLACE_SAMPLER_Y_OFFSET_BOUNDS)
            collision = False
            for other_obj in x.get_objects(MujocoMovableObjectType):
                if other_obj.name == self.objects[1].name:
                    continue
                other_object_pose = get_overhead_object_se2_pose(x, other_obj)
                if (
                    np.linalg.norm(
                        np.array(
                            [
                                pose_x_offset
                                + cupboard_pose.x
                                + (
                                    ARM_MOVEMENT_CUPBOARD.position[0]
                                    + ROBOT_ARM_POSE_TO_BASE.position[0]
                                    - BASE_DISTANCE_TO_CUPBOARD
                                ),  # the offset of the cupboard from the cubes.
                                pose_y_offset + cupboard_pose.y,
                            ]
                        )
                        - np.array([other_object_pose.x, other_object_pose.y])
                    )
                    < PLACE_SAMPLER_COLLISION_THRESHOLD
                ):
                    collision = True
                    break
            if not collision:
                return np.array(
                    [BASE_DISTANCE_TO_CUPBOARD + pose_x_offset, pose_y_offset, rot]
                )
        raise ValueError("No valid parameters found")

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x

        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()
        # Derive the target pose for the robot.
        target_distance, target_rot = self._current_params
        target_object = self.objects[2]
        target_object_pose_ori = get_overhead_object_se2_pose(x, target_object)
        target_object_pose = SE2(
            target_object_pose_ori.x,
            target_object_pose_ori.y,
            0.0,
        )
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, target_distance, target_rot
        )
        # Run motion planning.
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # use a constant seed to effectively make this "deterministic"
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
        )
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        plan_x = x.copy()
        robot = self.objects[0]  # Robot is first parameter
        target_base_pose = self._current_base_motion_plan[-1]
        if not self._navigated:
            plan_x.set(robot, "pos_base_x", target_base_pose.x)
            plan_x.set(robot, "pos_base_y", target_base_pose.y)
            plan_x.set(robot, "pos_base_rot", target_base_pose.theta())

        # Reset PyBullet given the current state.
        self._pybullet_sim.set_state(plan_x)

        target_object = self.objects[2]

        target_place_pose_world = Pose(
            (
                plan_x.get(target_object, "x"),
                plan_x.get(target_object, "y"),
                plan_x.get(target_object, "z"),
            ),
        )

        target_end_effector_pose = multiply_poses(
            target_place_pose_world,
            WIPER_SWEEP_TRANSFORM,
        )

        target_end_effector_pose_end = multiply_poses(
            target_place_pose_world,
            WIPER_SWEEP_TRANSFORM_END,
        )

        self._pybullet_sim.base_link_to_held_obj = multiply_poses(
            target_end_effector_pose.invert(),
            target_place_pose_world,
        )

        target_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose,
            set_joints=False,
        )

        target_joints_end = inverse_kinematics(
            self._pybullet_sim.robot,
            target_end_effector_pose_end,
            set_joints=False,
        )

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        retract_plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints_end,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        assert plan is not None, "Motion planning failed"
        assert retract_plan is not None, "Motion planning failed"

        # Remap the plan to ensure we stay within action limits.
        plan = remap_joint_position_plan_to_constant_distance(
            plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        # Remap the plan to ensure we stay within action limits.
        retract_plan = remap_joint_position_plan_to_constant_distance(
            retract_plan,
            self._pybullet_sim.robot,
            max_distance=0.4,
        )

        self._current_arm_joint_plan = plan
        self._current_retract_plan = retract_plan

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_retract_plan is not None
        )
        return self._returned

    def step(self) -> Array:
        assert self._current_arm_joint_plan is not None
        assert self._current_base_motion_plan is not None
        # first substep
        if not self._navigated:
            while len(self._current_base_motion_plan) > 1:
                peek_pose = self._current_base_motion_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_pose(peek_pose):
                    self._current_base_motion_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
                self._navigated = True
            robot_pose = self._get_current_robot_pose()
            next_pose = self._current_base_motion_plan[0]
            dx = next_pose.x - robot_pose.x
            dy = next_pose.y - robot_pose.y
            drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
            action = np.zeros(11, dtype=np.float32)
            action[0] = dx
            action[1] = dy
            action[2] = drot
            action[-1] = self._get_current_robot_gripper_pose()
            return action
        if self._navigated and not self._pre_place:
            while len(self._current_arm_joint_plan) > 1:
                peek_conf = self._current_arm_joint_plan[0]
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf):
                    self._current_arm_joint_plan.pop(0)
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_arm_joint_plan[-1]):
                self._pre_place = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_arm_joint_plan[0]
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.get_arm_joint_infos()[:7]  # type: ignore  # pylint: disable=protected-access
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        if self._pre_place:
            while len(self._current_retract_plan) > 1:  # type: ignore
                peek_conf = self._current_retract_plan[0]  # type: ignore
                # Close enough, pop and continue.
                if self._robot_is_close_to_conf(peek_conf, atol=0.08):
                    self._current_retract_plan.pop(0)  # type: ignore
                # Not close enough, stop popping.
                break
            if self._robot_is_close_to_conf(self._current_retract_plan[-1]):  # type: ignore # pylint: disable=line-too-long
                self._returned = True
            robot_conf = self._get_current_robot_arm_conf()
            gripper_pose = self._get_current_robot_gripper_pose()
            next_conf = self._current_retract_plan[0]  # type: ignore
            action = np.zeros(11, dtype=np.float32)
            joint_infos = self._pybullet_sim.robot.get_arm_joint_infos()[:7]  # type: ignore  # pylint: disable=protected-access
            free_joints_infos = [
                joint_info for joint_info in joint_infos if joint_info.qIndex > -1
            ]
            action[3:10] = get_jointwise_difference(
                free_joints_infos[:7], next_conf[:7], robot_conf[:7]
            )
            action[-1] = gripper_pose
            return action
        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return SE2(
            state.get(robot, "pos_base_x"),
            state.get(robot, "pos_base_y"),
            state.get(robot, "pos_base_rot"),
        )

    def _get_current_robot_arm_conf(self) -> JointPositions:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        return [
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

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._last_state
        assert x is not None
        robot_obj = self.objects[0]  # Robot is first parameter
        # return x.get(robot_obj, "pos_gripper")
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = WAYPOINT_TOL
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

    def _robot_is_close_to_pose(self, pose: SE2, atol: float = WAYPOINT_TOL) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.isclose(robot_pose.x, pose.x, atol=atol)
            and np.isclose(robot_pose.y, pose.y, atol=atol)
            and np.isclose(
                get_signed_angle_distance(robot_pose.theta(), pose.theta()),
                0.0,
                atol=atol,
            )
        )


def create_lifted_controllers(
    action_space: TidyBot3DRobotActionSpace,
    init_constant_state: ObjectCentricState | None = None,
    pybullet_sim: PyBulletSim | None = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for the TidyBot3D ground environment."""

    del action_space, init_constant_state  # not used

    # Controllers.

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)

    # Parameter space: [distance, rotation]
    # Based on sample_parameters: distance can be 0.5, 0.85, 0.92; rotation can be 0.0, -π/2
    move_to_target_params_space = Box(
        low=np.array([0.3, -np.pi], dtype=np.float32),
        high=np.array([1.5, np.pi], dtype=np.float32),
        dtype=np.float32,
    )

    LiftedMoveToTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetGroundController,
            params_space=move_to_target_params_space,
        )
    )

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)
    prev_target = Variable("?prev_target", MujocoObjectType)

    # Uses same parameter space as move_to_target
    LiftedMoveToTargetFromOtherTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, prev_target],
            MoveToTargetGroundController,
            params_space=move_to_target_params_space,
        )
    )

    # Move arm to conf controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    # Parameter space: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
    # Based on reset(): expects 7 joint angle parameters
    # Joint limits from Kinova Gen3 robot (gen3_2f85.xml):
    # joint_1, 3, 5, 7: unlimited rotation → use ±2π
    # joint_2: ±2.24, joint_4: ±2.57, joint_6: ±2.09
    arm_conf_params_space = Box(
        low=np.array(
            [-2 * np.pi, -2.24, -2 * np.pi, -2.57, -2 * np.pi, -2.09, -2 * np.pi],
            dtype=np.float32,
        ),
        high=np.array(
            [2 * np.pi, 2.24, 2 * np.pi, 2.57, 2 * np.pi, 2.09, 2 * np.pi],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )

    LiftedMoveArmToConfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            MoveArmToConfController,
            params_space=arm_conf_params_space,
        )
    )

    # Toss controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    # Uses same parameter space as move_arm_to_conf (7 joint angles)
    LiftedTossController: LiftedParameterizedController = LiftedParameterizedController(
        [robot],
        TossController,
        params_space=arm_conf_params_space,
    )

    # Move arm to end effector controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    # Parameter space: [x, y, z, qw, qx, qy, qz, rot]
    # Based on reset(): params are [x, y, z, qw, qx, qy, qz, rot] for end effector pose
    end_effector_params_space = Box(
        low=np.array(
            [
                WORLD_X_BOUNDS[0],  # x
                WORLD_Y_BOUNDS[0],  # y
                0.0,  # z (above ground)
                -1.0,  # qw
                -1.0,  # qx
                -1.0,  # qy
                -1.0,  # qz
                -np.pi,  # rot
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                WORLD_X_BOUNDS[1],  # x
                WORLD_Y_BOUNDS[1],  # y
                2.0,  # z
                1.0,  # qw
                1.0,  # qx
                1.0,  # qy
                1.0,  # qz
                np.pi,  # rot
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )

    LiftedMoveArmToEndEffectorController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            MoveArmToEndEffectorController,
            params_space=end_effector_params_space,
        )
    )

    # Close gripper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedCloseGripperController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            CloseGripperController,
        )
    )

    # Open gripper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedOpenGripperController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            OpenGripperController,
        )
    )

    # Create wrapper class that captures pybullet_sim
    class PickController(PickGroundController):
        """Pick controller with pre-configured PyBullet sim."""

        def __init__(self, objects):
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    class PlaceController(PlaceGroundController):
        """Place controller with pre-configured PyBullet sim."""

        def __init__(self, objects):
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    class OpenDrawerController(OpenDrawerSweepController):
        """Open drawer controller."""

        def __init__(self, objects):
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    class PickWiperController(PickWiperOriController):
        """Pick wiper controller."""

        def __init__(self, objects):
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    class SweepController(SweepOriController):
        """Sweep controller."""

        def __init__(self, objects):
            super().__init__(pybullet_sim=pybullet_sim, objects=objects)

    # Pick ground controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    # Parameter space: [distance, rotation]
    pick_ground_params_space = Box(
        low=np.array(
            [MOVE_TO_TARGET_DISTANCE_BOUNDS[0], MOVE_TO_TARGET_ROT_BOUNDS[0]],
            dtype=np.float32,
        ),
        high=np.array(
            [MOVE_TO_TARGET_DISTANCE_BOUNDS[1], MOVE_TO_TARGET_ROT_BOUNDS[1]],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )

    LiftedPickGroundController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            PickController,
            params_space=pick_ground_params_space,
        )
    )

    # Place controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    target_place = Variable("?target_place", MujocoFixtureObjectType)

    # Parameter space: [distance, y_offset, rotation]
    place_ground_params_space = Box(
        low=np.array(
            [
                BASE_DISTANCE_TO_CUPBOARD + PLACE_SAMPLER_X_OFFSET_BOUNDS[0],
                PLACE_SAMPLER_Y_OFFSET_BOUNDS[0],
                BASE_TO_CUPBOARD_ROTATION,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                BASE_DISTANCE_TO_CUPBOARD + PLACE_SAMPLER_X_OFFSET_BOUNDS[1],
                PLACE_SAMPLER_Y_OFFSET_BOUNDS[1],
                BASE_TO_CUPBOARD_ROTATION,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )

    LiftedPlaceGroundController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, target_place],
            PlaceController,
            params_space=place_ground_params_space,
        )
    )

    # Open drawer controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    LiftedOpenDrawerController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            OpenDrawerController,
        )
    )

    # Pick wiper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    LiftedPickWiperController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            PickWiperController,
        )
    )

    # Sweep controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    LiftedSweepController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, wiper, target],
            SweepController,
        )
    )

    return {
        "move_to_target": LiftedMoveToTargetController,
        "move_to_target_from_other_target": LiftedMoveToTargetFromOtherTargetController,
        "move_arm_to_conf": LiftedMoveArmToConfController,
        "toss": LiftedTossController,
        "move_arm_to_end_effector": LiftedMoveArmToEndEffectorController,
        "close_gripper": LiftedCloseGripperController,
        "open_gripper": LiftedOpenGripperController,
        "pick_ground": LiftedPickGroundController,
        "place_ground": LiftedPlaceGroundController,
        "open_drawer": LiftedOpenDrawerController,
        "pick_wiper": LiftedPickWiperController,
        "sweep": LiftedSweepController,
    }
