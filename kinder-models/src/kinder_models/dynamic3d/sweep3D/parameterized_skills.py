"""Parameterized skills for the TidyBot3D sweep3D environment."""

from typing import Any

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from kinder.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoTidyBotRobotObjectType,
    MujocoDrawerObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import (
    TidyBot3DRobotActionSpace,
)
from prpl_utils.utils import get_signed_angle_distance
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    smoothly_follow_end_effector_path,
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
    DRAWER_TRANSFORM_TO_OBJECT,
    DRAWER_TRANSFORM_TO_OBJECT_END,
    GRASP_CLOSE_THRESHOLD,
    GRIPPER_OPEN_THRESHOLD,
    WAYPOINT_TOL,
    WIPER_SWEEP_TRANSFORM,
    WIPER_SWEEP_TRANSFORM_END,
    WIPER_SWEEP_TRANSFORM_END_2,
    WIPER_TRANSFORM_TO_OBJECT,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    _compute_per_joint_profile,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
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
        # Trapezoidal velocity profiles (approach, open-drawer, and retract phases).
        self._approach_trajectory: np.ndarray = np.array([])
        self._approach_traj_dir: np.ndarray = np.zeros(7)
        self._approach_start_joints: np.ndarray = np.zeros(7)
        self._approach_step_idx: int = 0
        self._open_trajectory: np.ndarray = np.array([])
        self._open_traj_dir: np.ndarray = np.zeros(7)
        self._open_start_joints: np.ndarray = np.zeros(7)
        self._open_step_idx: int = 0
        self._retract_trajectory: np.ndarray = np.array([])
        self._retract_traj_dir: np.ndarray = np.zeros(7)
        self._retract_start_joints: np.ndarray = np.zeros(7)
        self._retract_step_idx: int = 0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
        # rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([0.7, -np.pi])

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
        # Compute trapezoidal velocity profile for approach (current -> grasp conf).
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        grasp_conf = np.array(plan[-1][:7])
        open_conf = np.array(open_plan[-1][:7])
        self._approach_trajectory, self._approach_traj_dir = _compute_per_joint_profile(
            curr, grasp_conf, _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._approach_start_joints = curr.copy()
        self._approach_step_idx = 0
        # Compute trapezoidal velocity profile for open-drawer (grasp conf -> open conf).
        self._open_trajectory, self._open_traj_dir = _compute_per_joint_profile(
            grasp_conf, open_conf, _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._open_start_joints = grasp_conf.copy()
        self._open_step_idx = 0
        # Compute trapezoidal velocity profile for retract (open conf -> home).
        self._retract_trajectory, self._retract_traj_dir = _compute_per_joint_profile(
            open_conf, self.home_joints[:7], _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._retract_start_joints = open_conf.copy()
        self._retract_step_idx = 0

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
        if self._pre_grasp and self._closed_gripper and not self._lifted:
            if self._open_step_idx >= len(self._open_trajectory):
                self._lifted = True
            idx = min(self._open_step_idx, len(self._open_trajectory) - 1)
            s = float(self._open_trajectory[idx])
            kp = 2.0
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._open_start_joints + self._open_traj_dir * s
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = kp * (target - curr)
            action[-1] = self._get_current_robot_gripper_pose()
            self._open_step_idx += 1
            return action
        if self._lifted and not self._open_gripper:
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._open_gripper = True
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 0
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._open_gripper:
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
        # distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
        # rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([0.7, -np.pi])

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
        # Compute trapezoidal velocity profile for approach (current -> grasp conf).
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        grasp_conf = np.array(plan[-1][:7])
        self._approach_trajectory, self._approach_traj_dir = _compute_per_joint_profile(
            curr, grasp_conf, _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._approach_start_joints = curr.copy()
        self._approach_step_idx = 0
        # Compute trapezoidal velocity profile for retract (grasp conf -> home).
        self._retract_trajectory, self._retract_traj_dir = _compute_per_joint_profile(
            grasp_conf, self.home_joints[:7], _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._retract_start_joints = grasp_conf.copy()
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
        # Trapezoidal velocity profiles (approach and sweep phases).
        self._approach_trajectory: np.ndarray = np.array([])
        self._approach_traj_dir: np.ndarray = np.zeros(7)
        self._approach_start_joints: np.ndarray = np.zeros(7)
        self.target_joints_end: list[float] = []
        self.target_joints_end_2: list[float] = []
        self._approach_step_idx: int = 0
        self._sweep_trajectory: np.ndarray = np.array([])
        self._sweep_traj_dir: np.ndarray = np.zeros(7)
        self._sweep_start_joints: np.ndarray = np.zeros(7)
        self._sweep_step_idx: int = 0
        self._sweep_step_idx_2: int = 0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
        # rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([0.55, -np.pi])

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
        target_object = self.objects[3]
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

        target_object = self.objects[3]

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

        target_end_effector_pose_end_2 = multiply_poses(
            target_place_pose_world,
            WIPER_SWEEP_TRANSFORM_END_2,
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

        # Run motion planning.
        plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            target_joints,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,  # use a constant seed to make this effectively deterministic
            physics_client_id=self._pybullet_sim.physics_client_id,
        )

        joint_distance_fn = create_joint_distance_fn(self._pybullet_sim.robot)
        # Run motion planning to the target joint positions.
        retract_plan = smoothly_follow_end_effector_path(
            self._pybullet_sim.robot,
            [target_end_effector_pose_end, target_end_effector_pose_end_2],
            initial_joints=target_joints,
            collision_ids={},  # type: ignore
            seed=0,  # for determinism
            joint_distance_fn=joint_distance_fn,
            max_smoothing_iters_per_step=1,
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
        # Compute trapezoidal velocity profile for approach
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        sweep_start_conf = np.array(plan[-1][:7])
        sweep_end_conf = np.array(retract_plan[-1][:7])
        self._approach_trajectory, self._approach_traj_dir = _compute_per_joint_profile(
            curr, sweep_start_conf, _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._approach_start_joints = curr.copy()
        self._approach_step_idx = 0
        # Compute trapezoidal velocity profile for sweep (sweep start -> sweep end conf).
        self._sweep_trajectory, self._sweep_traj_dir = _compute_per_joint_profile(
            sweep_start_conf, sweep_end_conf, _ARM_MAX_VEL, _ARM_MAX_ACCEL
        )
        self._sweep_start_joints = sweep_start_conf.copy()
        self._sweep_step_idx = 0

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
        if self._pre_place:
            if self._sweep_step_idx >= len(self._sweep_trajectory):
                self._returned = True
            idx = min(self._sweep_step_idx, len(self._sweep_trajectory) - 1)
            s = float(self._sweep_trajectory[idx])
            kp = 2.0
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._sweep_start_joints + self._sweep_traj_dir * s
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = kp * (target - curr)
            action[-1] = self._get_current_robot_gripper_pose()
            self._sweep_step_idx += 1
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

    # Open drawer controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    drawer = Variable("?drawer", MujocoDrawerObjectType)
    cube0 = Variable("?cube0", MujocoMovableObjectType)
    cube1 = Variable("?cube1", MujocoMovableObjectType)
    cube2 = Variable("?cube2", MujocoMovableObjectType)
    cube3 = Variable("?cube3", MujocoMovableObjectType)
    cube4 = Variable("?cube4", MujocoMovableObjectType)

    LiftedOpenDrawerController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4],
            OpenDrawerController,
        )
    )

    # Pick wiper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    drawer = Variable("?drawer", MujocoDrawerObjectType)
    cube0 = Variable("?cube0", MujocoMovableObjectType)
    cube1 = Variable("?cube1", MujocoMovableObjectType)
    cube2 = Variable("?cube2", MujocoMovableObjectType)
    cube3 = Variable("?cube3", MujocoMovableObjectType)
    cube4 = Variable("?cube4", MujocoMovableObjectType)

    LiftedPickWiperController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4],
            PickWiperController,
        )
    )

    # Pick wiper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    drawer = Variable("?drawer", MujocoDrawerObjectType)
    cube0 = Variable("?cube0", MujocoMovableObjectType)
    cube1 = Variable("?cube1", MujocoMovableObjectType)
    cube2 = Variable("?cube2", MujocoMovableObjectType)
    cube3 = Variable("?cube3", MujocoMovableObjectType)
    cube4 = Variable("?cube4", MujocoMovableObjectType)

    LiftedSweepController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4],
            SweepController,
        )
    )

    return {
        "open_drawer": LiftedOpenDrawerController,
        "pick_wiper": LiftedPickWiperController,
        "sweep": LiftedSweepController,
    }
