"""Parameterized skills for the TidyBot3D tossing environment."""

import enum
from typing import Any

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from kinder.envs.dynamic3d.object_types import (
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

from kinder_models.dynamic3d.cube_symmetry import upright_grasp_rotations
from kinder_models.dynamic3d.tossing.toss_swing import (
    TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
    TOSS_MAX_VELOCITY,
    TOSS_RELEASE_ARM_CONFIGURATION,
    TOSS_WINDUP_ARM_CONFIGURATION,
    TossSwing,
    plan_toss_swing,
    toss_swing_action,
)
from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    _CONTROL_TIMESTEP,
    GRASP_CLOSE_THRESHOLD,
    GRASP_TRANSFORM_TO_OBJECT,
    GRIPPER_CLOSED_THRESHOLD,
    GRIPPER_OPEN_COMMAND_TOLERANCE,
    MOVE_TO_TARGET_DISTANCE_BOUNDS,
    MOVE_TO_TARGET_ROT_BOUNDS,
    WAYPOINT_TOLERANCE,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    _compute_per_joint_profile,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
)

# Per-joint motion between consecutive control steps below which the arm counts as
# stopped. A stand-in for joint velocity, which the state does not expose for the arm.
# Measured floor: the proportional controller's tail creeps at ~2.6e-3 rad/step and never
# fully stops, so a threshold below that never fires and the phase hangs forever.
ARM_SETTLE_TOLERANCE = 5e-3

# How close the arm must be to a phase's final conf before the settling gate applies.
# Tighter than WAYPOINT_TOLERANCE (4e-2), which is the pass-through band for intermediate
# waypoints: handing off to the gripper at 4e-2 leaves the grasp ~2.8cm off the cube.
ARM_FINAL_CONF_TOLERANCE = 5e-3


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
        self._current_params: np.ndarray | None = None
        self._current_base_motion_plan: list[SE2] | None = None

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        distance = 0.5  # for stable grasp
        rot = 0.0
        return np.array([distance, rot])

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
        disable_collision_objects: list[str] | None = None,
    ) -> None:
        self._last_state = x
        # Convert params to ndarray for compatibility (accepts tuple or array)
        self._current_params = np.asarray(params, dtype=np.float32)
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

    def _robot_is_close_to_pose(
        self, pose: SE2, atol: float = WAYPOINT_TOLERANCE
    ) -> bool:
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
        self._current_params: np.ndarray | None = None
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

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        # Convert params to ndarray for compatibility (accepts tuple or array)
        self._current_params = np.asarray(params, dtype=np.float32)
        target_joints = self._current_params.tolist() + ([0.0] * 6)
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
            _ARM_MAX_VELOCITY,
            _ARM_MAX_ACCELERATION,
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
            ds = (self._trajectory[idx] - self._trajectory[idx - 1]) / _CONTROL_TIMESTEP
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
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._pybullet_sim: PyBulletSim | None = None
        self._step_idx: int = 0
        self._has_released: bool = False
        self._swing: TossSwing | None = None

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        # We can later implement sampling if it's helpful, but usually the user would
        # want to specify the target arm conf themselves.
        raise NotImplementedError

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        release_speed: float = TOSS_MAX_VELOCITY,
        gripper_release_ms: int = TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
    ) -> None:
        """Plan the swing, and fix the millisecond the gripper opens on.

        The two knobs the real robot's movej_primitive.execute() takes (used by
        the real robot's kinova controller).
        gripper_release_ms is deliberately NOT clamped to the swing's duration: a value
        at or past the end means the gripper never opens and the cube is never thrown.
        """
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        # Convert params to ndarray for compatibility (accepts tuple or array)
        self._current_params = np.asarray(params, dtype=np.float32)
        target_joints = self._current_params.tolist() + ([0.0] * 6)
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
        self._swing = plan_toss_swing(
            plan,
            self._get_current_robot_arm_conf(),
            release_speed,
            gripper_release_ms,
        )
        self._has_released = False
        self._step_idx = 0

    def terminated(self) -> bool:
        # Terminate when we've gone through the entire profile.
        assert self._swing is not None
        return self._step_idx >= len(self._swing.trajectory)

    def step(self) -> Array:
        """The swing's command for one control step, opening the gripper mid-step.

        The usual (18,) action, except on the step the release falls inside, which
        returns a (TOSS_SLICES_PER_CONTROL_STEP, 18) schedule so gripper_release_ms
        means the millisecond it names rather than the next step boundary.
        """
        assert self._swing is not None
        action = toss_swing_action(
            self._swing,
            self._step_idx,
            self._get_current_robot_arm_conf(),
            self._get_current_robot_gripper_pose(),
            self._has_released,
        )
        if self._step_idx == self._swing.release_step:
            self._has_released = True
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
        self._current_params: np.ndarray | None = None
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

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
        self._last_state = x
        # Convert params to ndarray for compatibility (accepts tuple or array)
        self._current_params = np.asarray(params, dtype=np.float32)

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
            _ARM_MAX_VELOCITY,
            _ARM_MAX_ACCELERATION,
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
            ds = (self._trajectory[idx] - self._trajectory[idx - 1]) / _CONTROL_TIMESTEP
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

    def reset(self, x: ObjectCentricState, params: Any | None = None) -> None:
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

    def _get_current_gripper_pose(self) -> float:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return state.get(robot, "pos_gripper")

    def _robot_gripper_is_closed(self, atol: float = GRIPPER_CLOSED_THRESHOLD) -> bool:
        current_gripper_pose = self._get_current_gripper_pose()
        return bool(
            current_gripper_pose > 0.2
            and np.isclose(current_gripper_pose, self.last_gripper_state, atol=atol)
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

    def reset(self, x: ObjectCentricState, params: Any | None = None) -> None:
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

    def _get_current_gripper_pose(self) -> float:
        assert self._last_state is not None
        state = self._last_state
        robot = self.objects[0]
        return state.get(robot, "pos_gripper")

    def _robot_gripper_is_open(
        self, atol: float = GRIPPER_OPEN_COMMAND_TOLERANCE
    ) -> bool:
        current_gripper_pose = self._get_current_gripper_pose()
        return current_gripper_pose < atol


class PickCubeController(GroundParameterizedController[ObjectCentricState, Array]):
    """Pick a cube up off the ground.

    Stub controller for picking a cube; raises NotImplementedError.
    """

    TARGET_DISTANCE = 0.55
    TARGET_ROTATION = 0.0

    class PickCubeControllerPhase(enum.Enum):
        """The ordered stages of a pick, stepped through in this order."""

        OPEN_GRIPPER = enum.auto()
        BASE_MOTION = enum.auto()
        MOVE_ARM_TO_HOVER_OVER_CUBE = enum.auto()
        MOVE_ARM_DOWN_AROUND_CUBE = enum.auto()
        CLOSE_GRIPPER_TO_GRASP_CUBE = enum.auto()
        LIFT_CUBE_TO_HOME = enum.auto()

    def __init__(
        self, *args: Any, pybullet_sim: PyBulletSim | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)

        self.MAX_SAMPLER_ATTEMPTS = 100

        self.current_phase = self.PickCubeControllerPhase.OPEN_GRIPPER

        self._pybullet_sim = pybullet_sim
        self._last_state: ObjectCentricState | None = None
        self._last_gripper_state: float = 0.0
        self._closed_gripper: bool = False
        self._lifted: bool = False
        # Previous tick's arm conf, for the settling check in _robot_arm_has_settled.
        self._prev_arm_conf: np.ndarray | None = None
        # Which waypoint of self.plans[phase] step() is currently driving toward. A fresh
        # controller is ground per refinement attempt, so this starts at 0 per attempt.
        self._plan_step_idx: dict[PickCubeController.PickCubeControllerPhase, int] = {
            self.PickCubeControllerPhase.MOVE_ARM_TO_HOVER_OVER_CUBE: 0,
            self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE: 0,
            self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME: 0,
        }

        self.home_joints = np.deg2rad(
            [0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0]
        )  # retract configuration

        self.plans: dict[
            PickCubeController.PickCubeControllerPhase,
            list[SE2] | list[JointPositions] | None,
        ] = {
            self.PickCubeControllerPhase.OPEN_GRIPPER: None,
            self.PickCubeControllerPhase.BASE_MOTION: None,
            self.PickCubeControllerPhase.MOVE_ARM_TO_HOVER_OVER_CUBE: None,
            self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE: None,
            self.PickCubeControllerPhase.CLOSE_GRIPPER_TO_GRASP_CUBE: None,
            self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME: None,
        }

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        target_object = self.objects[1]
        target_object_pose = get_overhead_object_se2_pose(x, target_object)

        for _ in range(self.MAX_SAMPLER_ATTEMPTS):
            distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)
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
        # This is an entirely hardcoded controller
        del params
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)

        # Reset to the first phase
        self.current_phase = self.PickCubeControllerPhase.OPEN_GRIPPER
        self._last_state = x
        self._last_gripper_state = 0.0
        self._closed_gripper = False
        self._lifted = False

        cube_to_pick_up = self.objects[1]
        # The cube is always the index 1 object at construction time, and the
        # barrier the index 2 object.

        # Pre-compute all motion planning
        # BASE_MOTION planning
        cube_pose = get_overhead_object_se2_pose(x, cube_to_pick_up)
        target_base_pose = get_target_robot_pose_from_parameters(
            cube_pose, self.TARGET_DISTANCE, self.TARGET_ROTATION
        )
        self.plans[self.PickCubeControllerPhase.BASE_MOTION] = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,  # To make this effectively deterministic
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
        )
        assert self.plans[self.PickCubeControllerPhase.BASE_MOTION] is not None

        # MOVE_ARM_TO_HOVER_OVER_CUBE planning
        # Get the last state from the BASE_MOTION plan so that next steps can build on it
        state_after_base_motion = x.copy()
        robot = self.objects[0]  # robot is the first parameter
        target_base_pose_plan = self.plans[self.PickCubeControllerPhase.BASE_MOTION]
        assert target_base_pose_plan is not None
        target_base_pose = target_base_pose_plan[-1]
        assert isinstance(target_base_pose, SE2)
        assert self.current_phase == self.PickCubeControllerPhase.OPEN_GRIPPER
        state_after_base_motion.set(robot, "pos_base_x", target_base_pose.x)
        state_after_base_motion.set(robot, "pos_base_y", target_base_pose.y)
        state_after_base_motion.set(robot, "pos_base_rot", target_base_pose.theta())

        # Set the simulation to the state after base motion so that it can be used
        # for grasp planning.
        self._pybullet_sim.set_state(state_after_base_motion)

        # We care about the cube's graspable orientation, not absolute orentiation.
        cube_raw_quat = (
            state_after_base_motion.get(cube_to_pick_up, "qx"),
            state_after_base_motion.get(cube_to_pick_up, "qy"),
            state_after_base_motion.get(cube_to_pick_up, "qz"),
            state_after_base_motion.get(cube_to_pick_up, "qw"),
        )
        cube_grasp_qx, cube_grasp_qy, cube_grasp_qz, cube_grasp_qw = (
            upright_grasp_rotations(cube_raw_quat)[0]
        )

        target_hover_end_effector_pose = Pose(
            (
                state_after_base_motion.get(cube_to_pick_up, "x"),
                state_after_base_motion.get(cube_to_pick_up, "y"),
                state_after_base_motion.get(cube_to_pick_up, "z")
                + state_after_base_motion.get(cube_to_pick_up, "bb_z") / 2
                # 12 cm above the cube's top surface. "z" is the cube's centre, so
                # we add half its height ("bb_z").
                + 0.12,
            ),
            (cube_grasp_qx, cube_grasp_qy, cube_grasp_qz, cube_grasp_qw),
        )
        target_hover_end_effector_pose = multiply_poses(
            target_hover_end_effector_pose, GRASP_TRANSFORM_TO_OBJECT
        )  # Offset by the intended grasp location
        target_hover_joints = inverse_kinematics(
            self._pybullet_sim.robot, target_hover_end_effector_pose, set_joints=False
        )
        self.plans[self.PickCubeControllerPhase.MOVE_ARM_TO_HOVER_OVER_CUBE] = (
            run_motion_planning(
                self._pybullet_sim.robot,
                self._pybullet_sim.get_robot_joints(),
                target_hover_joints,
                # Includes the cube to grasp, because we have not grasped it yet.
                collision_bodies=self._pybullet_sim.get_collision_bodies(),
                seed=0,
                physics_client_id=self._pybullet_sim.physics_client_id,
            )
        )

        # MOVE_ARM_DOWN_AROUND_CUBE planning
        target_around_cube_end_effector_pose = Pose(
            (
                state_after_base_motion.get(cube_to_pick_up, "x"),
                state_after_base_motion.get(cube_to_pick_up, "y"),
                state_after_base_motion.get(cube_to_pick_up, "z"),
            ),
            (cube_grasp_qx, cube_grasp_qy, cube_grasp_qz, cube_grasp_qw),
        )
        target_around_cube_end_effector_pose = multiply_poses(
            target_around_cube_end_effector_pose, GRASP_TRANSFORM_TO_OBJECT
        )  # Offset by the intended grasp location
        target_around_joints = inverse_kinematics(
            self._pybullet_sim.robot,
            target_around_cube_end_effector_pose,
            set_joints=False,
        )
        self.plans[self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE] = (
            run_motion_planning(
                self._pybullet_sim.robot,
                target_hover_joints,  # Going from the hover joints to the around joints
                target_around_joints,
                # The cube is still a collision body here.
                collision_bodies=self._pybullet_sim.get_collision_bodies(),
                seed=0,
                physics_client_id=self._pybullet_sim.physics_client_id,
            )
        )

        # CLOSE_GRIPPER_TO_GRASP_CUBE planning
        # Nothing, because the arm is opened/closed at inference time, not planning time

        # LIFT_CUBE_TO_HOME planning
        # pylint: disable=protected-access
        held_object = self._pybullet_sim._cubes[
            cube_to_pick_up.name
        ]  # For collision detection
        self._pybullet_sim.base_link_to_held_obj = (
            GRASP_TRANSFORM_TO_OBJECT.invert()
        )  # For Motion planning so it knows to avoid bonking the cube on things
        self.plans[self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME] = (
            run_motion_planning(
                self._pybullet_sim.robot,
                target_around_joints,
                self.home_joints.tolist(),
                collision_bodies=self._pybullet_sim.get_collision_bodies(
                    held_object=held_object
                ),
                held_object=held_object,
                base_link_to_held_obj=self._pybullet_sim.base_link_to_held_obj,
                seed=0,
                physics_client_id=self._pybullet_sim.physics_client_id,
            )
        )

        for name, plan in self.plans.items():
            if name in {
                self.PickCubeControllerPhase.OPEN_GRIPPER,
                self.PickCubeControllerPhase.BASE_MOTION,
                self.PickCubeControllerPhase.CLOSE_GRIPPER_TO_GRASP_CUBE,
            }:
                # No arm motion planning for these phases, so nothing to remap.
                continue
            # Ensure motion planning didn't fail
            assert plan is not None, f"Motion Planning Failed at {name}"

            # Map the plan onto real robot space, with constant distance
            self.plans[name] = remap_joint_position_plan_to_constant_distance(
                plan, self._pybullet_sim.robot, max_distance=0.2
            )

    def step(self) -> Array:
        if self.current_phase == self.PickCubeControllerPhase.OPEN_GRIPPER:
            return self._step_open_gripper()
        if self.current_phase == self.PickCubeControllerPhase.BASE_MOTION:
            return self._step_base_motion()
        if (
            self.current_phase
            == self.PickCubeControllerPhase.MOVE_ARM_TO_HOVER_OVER_CUBE
        ):
            return self._step_trajectory_phase(
                self.PickCubeControllerPhase.MOVE_ARM_TO_HOVER_OVER_CUBE,
                self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE,
            )
        if self.current_phase == self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE:
            return self._step_trajectory_phase(
                self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE,
                self.PickCubeControllerPhase.CLOSE_GRIPPER_TO_GRASP_CUBE,
            )
        if (
            self.current_phase
            == self.PickCubeControllerPhase.CLOSE_GRIPPER_TO_GRASP_CUBE
        ):
            return self._step_close_gripper()
        if self.current_phase == self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME:
            return self._step_trajectory_phase(
                self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME,
                None,
            )
        raise ValueError(f"Invalid phase: {self.current_phase}")

    def _step_open_gripper(self) -> Array:
        """Open the gripper before reaching for the cube.

        A no-op when the gripper already reads open: the phase falls straight through to
        the base motion in the same step rather than spending a control step commanding
        a state the gripper is already in.
        """
        if self._robot_gripper_is_open():
            self.current_phase = self.PickCubeControllerPhase.BASE_MOTION
            return self._step_base_motion()
        action = np.zeros(11, dtype=np.float32)
        action[-1] = 0
        return action

    def _step_base_motion(self) -> Array:
        base_plan = self.plans[self.PickCubeControllerPhase.BASE_MOTION]
        assert base_plan is not None and len(base_plan) > 0
        assert isinstance(base_plan[0], SE2)
        while len(base_plan) > 1:
            peek_pose = base_plan[0]
            assert isinstance(peek_pose, SE2)
            if self._robot_is_close_to_pose(peek_pose):
                base_plan.pop(0)
            else:
                break
        final_pose = base_plan[-1]
        assert isinstance(final_pose, SE2)
        if self._robot_is_close_to_pose(final_pose):
            self.current_phase = (
                self.PickCubeControllerPhase.MOVE_ARM_TO_HOVER_OVER_CUBE
            )
        robot_pose = self._get_current_robot_pose()
        next_pose = base_plan[0]
        assert isinstance(next_pose, SE2)
        dx = next_pose.x - robot_pose.x
        dy = next_pose.y - robot_pose.y
        drot = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
        action = np.zeros(11, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        action[2] = drot
        action[-1] = self._get_current_robot_gripper_pose()
        return action

    def _step_trajectory_phase(
        self,
        phase: "PickCubeController.PickCubeControllerPhase",
        next_phase: "PickCubeController.PickCubeControllerPhase | None",
    ) -> Array:
        plan = self.plans[phase]
        assert plan is not None and len(plan) > 0, f"No plan computed for {phase}"
        idx = self._plan_step_idx[phase]
        # Drive through the waypoints of the actual (collision-checked) motion-planned
        # path, advancing once close to the current one, rather than interpolating a
        # straight line between only the first and last confs.
        while idx < len(plan) - 1 and self._robot_is_close_to_conf(plan[idx]):
            idx += 1
        self._plan_step_idx[phase] = idx
        target_waypoint = plan[idx]
        kp = 2.0
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        target = np.array(target_waypoint[:7])
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = kp * (target - curr)
        action[-1] = self._get_current_robot_gripper_pose()
        if idx >= len(plan) - 1 and self._robot_is_close_to_conf(target_waypoint):
            if next_phase is not None:
                self.current_phase = next_phase
            else:
                self._lifted = True
        return action

    def _step_close_gripper(self) -> Array:
        if self._get_current_robot_gripper_pose() > 0.2 and np.isclose(
            self._get_current_robot_gripper_pose(),
            self._last_gripper_state,
            atol=0.02,
        ):
            self._closed_gripper = True
            self.current_phase = self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME
        action = np.zeros(11, dtype=np.float32)
        action[-1] = 1
        self._last_gripper_state = self._get_current_robot_gripper_pose()
        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def terminated(self) -> bool:
        return self._lifted

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
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_pose(
        self, pose: SE2, atol: float = WAYPOINT_TOLERANCE
    ) -> bool:
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

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = WAYPOINT_TOLERANCE
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

    def _robot_gripper_is_open(
        self, atol: float = GRIPPER_OPEN_COMMAND_TOLERANCE
    ) -> bool:
        """Whether the gripper is commanded open.

        Reads pos_gripper directly rather than through _get_current_robot_gripper_pose,
        which quantises to 0.0 / GRASP_CLOSE_THRESHOLD and so cannot distinguish
        "open" from "nearly open".
        """
        x = self._last_state
        assert x is not None
        return bool(x.get(self.objects[0], "pos_gripper") < atol)

    def _record_arm_conf(self, conf: np.ndarray) -> None:
        """Remember this tick's arm conf, so the next tick can measure motion."""
        self._prev_arm_conf = np.array(conf)

    def _robot_arm_has_settled(self, atol: float = ARM_SETTLE_TOLERANCE) -> bool:
        """Whether the arm has stopped moving since the previous control step.

        The second of the two gates on a phase handoff. Proximity says the arm is inside
        the tolerance band; this says it is no longer travelling through it, so the next
        phase (notably closing the gripper) does not act on a moving target.
        """
        if self._prev_arm_conf is None:
            return False
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        return bool(np.max(np.abs(curr - self._prev_arm_conf)) < atol)


class MoveToTossLocationAndTossController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Drive to a pose to throw from and throw, as one skill.

    The object parameters are:
        robot: The robot itself.
        target: The object to throw at.
        held: The movable the robot is holding.

    The continuous parameters are:
        distance_to_target: float (metres)
        rotation_to_target: float (radians)
        tossing_speed: float (radians/second)
        tossing_ms: float (milliseconds into the swing that the gripper opens)

    Composed rather than split so that no predicate has to name the pose between them.
    The cost is that a standoff which cannot score is only discovered by throwing from
    it, where a separate move could have been rejected first.

    One flat controller over an explicit phase, as pick_shelf is: base motion, then
    the windup, then the swing. The swing is planned at the windup's end rather than
    in reset, because it has to start from the arm conf actually reached.
    """

    class MoveToTossLocationAndTossControllerPhase(enum.Enum):
        """Which leg of drive-then-throw step() is currently stepping."""

        BASE_MOTION = enum.auto()
        WINDUP = enum.auto()
        SWING = enum.auto()

    # Where a throw is possible; the upper part does not score.
    TARGET_DISTANCE_BOUNDS = (1.25, 1.45)

    # Widest rotation that stays within half of WAYPOINT_TOLERANCE at max standoff.
    MAX_TARGET_ROTATION = float(
        np.arcsin(0.5 * WAYPOINT_TOLERANCE / TARGET_DISTANCE_BOUNDS[1])
    )
    TARGET_ROTATION_BOUNDS = (-MAX_TARGET_ROTATION, MAX_TARGET_ROTATION)

    # TossController's two dials, opened up as sampled parameters. Narrowed from the
    # originally-shipped (60, TOSS_MAX_VELOCITY) / (600, 840): measured directly
    # (toss_param_probe4.py, isolated toss draws from a real post-pick state, 480
    # draws across 16 seeds) that every scoring draw fell in speed_deg [117.5, 140.0]
    # and release_ms [710.4, 836.1] -- the wide bounds spent the large majority of
    # the sampler's budget on combinations that can never score. A few degrees/ms of
    # margin below the measured minimums, since 480 draws is not exhaustive.
    SPEED_BOUNDS = (np.deg2rad(115.0), TOSS_MAX_VELOCITY)
    RELEASE_MS_BOUNDS = (700.0, 840.0)

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # State and simulator; pybullet_sim is injectable so groundings can share one.
        self._last_state: ObjectCentricState | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim

        # The two sampled tossing dials.
        self._release_speed: float = TOSS_MAX_VELOCITY
        self._gripper_release_ms: int = TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS
        self._phase = self.MoveToTossLocationAndTossControllerPhase.BASE_MOTION

        # Base motion: the drive to the toss pose.
        self._current_base_motion_plan: list[SE2] | None = None

        # Windup: the arm swinging back before the throw.
        self._windup_trajectory: np.ndarray = np.array([])
        self._windup_dir: np.ndarray = np.zeros(7)
        self._windup_start_joint_angles: np.ndarray = np.zeros(7)
        self._windup_step_idx: int = 0

        # Swing: the throw itself, planned once the windup ends.
        self._swing: TossSwing | None = None
        self._swing_step_idx: int = 0
        self._has_released: bool = False

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x  # not used
        return np.array(
            [
                rng.uniform(*self.TARGET_DISTANCE_BOUNDS),
                rng.uniform(*self.TARGET_ROTATION_BOUNDS),
                rng.uniform(*self.SPEED_BOUNDS),
                rng.uniform(*self.RELEASE_MS_BOUNDS),
            ]
        )

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
        disable_collision_objects: list[str] | None = None,
    ) -> None:
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        current_params = np.asarray(params, dtype=np.float32)
        assert current_params.shape == (4,)
        self._last_state = x
        self._release_speed = float(current_params[2])
        self._gripper_release_ms = int(round(float(current_params[3])))
        self._phase = self.MoveToTossLocationAndTossControllerPhase.BASE_MOTION
        self._windup_step_idx = 0
        self._swing = None
        self._swing_step_idx = 0
        self._has_released = False

        self._current_base_motion_plan = self._plan_base_motion(
            x,
            current_params,
            extend_xy_magnitude,
            extend_rot_magnitude,
            disable_collision_objects,
        )
        final_base_pose = self._current_base_motion_plan[-1]
        self._plan_arm_toss(x, final_base_pose)

    def _plan_base_motion(
        self,
        x: ObjectCentricState,
        current_params: np.ndarray,
        extend_xy_magnitude: float,
        extend_rot_magnitude: float,
        disable_collision_objects: list[str] | None,
    ) -> list[SE2]:
        # The robot's own cargo would otherwise reject every base plan.
        if disable_collision_objects is None:
            disable_collision_objects = [self.objects[1].name]
        target_object = x.get_object_from_name("bin_0")
        target_object_pose = get_overhead_object_se2_pose(x, target_object)
        target_base_pose = get_target_robot_pose_from_parameters(
            target_object_pose, current_params[0], current_params[1]
        )
        base_motion_plan = run_base_motion_planning(
            state=x,
            target_base_pose=target_base_pose,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
            disable_collision_objects=disable_collision_objects,
        )
        if base_motion_plan is None:
            raise TrajectorySamplingFailure("Base motion planning failed")
        return base_motion_plan

    def _plan_arm_toss(self, x: ObjectCentricState, final_base_pose: SE2) -> None:
        """Plan the windup and the swing, from where the base motion will end.

        As pick_shelf does: planning the whole arm motion here, rather than on
        arrival, surfaces a planning failure from reset instead of from mid-throw.
        """
        assert self._pybullet_sim is not None
        plan_x = x.copy()
        robot = self.objects[0]
        plan_x.set(robot, "pos_base_x", final_base_pose.x)
        plan_x.set(robot, "pos_base_y", final_base_pose.y)
        plan_x.set(robot, "pos_base_rot", final_base_pose.theta())
        self._pybullet_sim.set_state(plan_x)
        windup_plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            list(TOSS_WINDUP_ARM_CONFIGURATION) + [0.0] * 6,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,
            physics_client_id=self._pybullet_sim.physics_client_id,
        )
        if windup_plan is None:
            raise TrajectorySamplingFailure("Motion planning failed")
        windup_start = np.array(self._get_current_robot_arm_conf()[:7])
        self._windup_trajectory, self._windup_dir = _compute_per_joint_profile(
            windup_start,
            np.array(windup_plan[-1][:7]),
            _ARM_MAX_VELOCITY,
            _ARM_MAX_ACCELERATION,
        )
        self._windup_start_joint_angles = windup_start

        for joint, angle in enumerate(windup_plan[-1][:7], start=1):
            plan_x.set(robot, f"pos_arm_joint{joint}", angle)
        self._pybullet_sim.set_state(plan_x)
        swing_plan = run_motion_planning(
            self._pybullet_sim.robot,
            self._pybullet_sim.get_robot_joints(),
            list(TOSS_RELEASE_ARM_CONFIGURATION) + [0.0] * 6,
            collision_bodies=self._pybullet_sim.get_collision_bodies(),
            seed=0,
            physics_client_id=self._pybullet_sim.physics_client_id,
        )
        if swing_plan is None:
            raise TrajectorySamplingFailure("Motion planning failed")
        self._swing = plan_toss_swing(
            swing_plan,
            windup_plan[-1],
            self._release_speed,
            self._gripper_release_ms,
        )

    def terminated(self) -> bool:
        return self._swing is not None and self._swing_step_idx >= len(
            self._swing.trajectory
        )

    def step(self) -> Array:
        assert self._current_base_motion_plan is not None
        if self._phase is self.MoveToTossLocationAndTossControllerPhase.BASE_MOTION:
            return self._action_base_motion()
        if self._phase is self.MoveToTossLocationAndTossControllerPhase.WINDUP:
            return self._action_windup()
        return self._action_swing()

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _action_base_motion(self) -> Array:
        assert self._current_base_motion_plan is not None
        while len(self._current_base_motion_plan) > 1:
            peek_pose = self._current_base_motion_plan[0]
            if self._robot_is_close_to_pose(peek_pose):
                self._current_base_motion_plan.pop(0)
            break
        if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
            self._phase = self.MoveToTossLocationAndTossControllerPhase.WINDUP
        robot_pose = self._get_current_robot_pose()
        next_pose = self._current_base_motion_plan[0]
        action = np.zeros(11, dtype=np.float32)
        action[0] = next_pose.x - robot_pose.x
        action[1] = next_pose.y - robot_pose.y
        action[2] = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
        action[-1] = self._get_current_robot_gripper_pose()
        return action

    def _action_windup(self) -> Array:
        action = np.zeros(18, dtype=np.float32)
        idx = min(self._windup_step_idx, len(self._windup_trajectory) - 1)
        s = float(self._windup_trajectory[idx])
        if idx > 0:
            ds = (
                self._windup_trajectory[idx] - self._windup_trajectory[idx - 1]
            ) / _CONTROL_TIMESTEP
        else:
            ds = 0.0
        kp = 2.0
        kv = 2.0
        curr = np.array(self._get_current_robot_arm_conf()[:7])
        target = self._windup_start_joint_angles + self._windup_dir * s
        action[3:10] = kp * (target - curr)
        action[11:18] = self._windup_dir * (ds * kv)
        action[10] = self._get_current_robot_gripper_pose()
        self._windup_step_idx += 1
        if self._windup_step_idx >= len(self._windup_trajectory):
            self._phase = self.MoveToTossLocationAndTossControllerPhase.SWING
        return action

    def _action_swing(self) -> Array:
        assert self._swing is not None
        action = toss_swing_action(
            self._swing,
            self._swing_step_idx,
            self._get_current_robot_arm_conf(),
            self._get_current_robot_gripper_pose(),
            self._has_released,
        )
        if self._swing_step_idx == self._swing.release_step:
            self._has_released = True
        self._swing_step_idx += 1
        return action

    def _get_current_robot_pose(self) -> SE2:
        assert self._last_state is not None
        robot = self.objects[0]
        return SE2(
            self._last_state.get(robot, "pos_base_x"),
            self._last_state.get(robot, "pos_base_y"),
            self._last_state.get(robot, "pos_base_rot"),
        )

    def _robot_is_close_to_pose(self, pose: SE2) -> bool:
        robot_pose = self._get_current_robot_pose()
        return bool(
            np.hypot(pose.x - robot_pose.x, pose.y - robot_pose.y) < WAYPOINT_TOLERANCE
            and abs(get_signed_angle_distance(pose.theta(), robot_pose.theta()))
            < WAYPOINT_TOLERANCE
        )

    def _get_current_robot_arm_conf(self) -> JointPositions:
        assert self._last_state is not None
        robot = self.objects[0]
        return [
            self._last_state.get(robot, f"pos_arm_joint{i}") for i in range(1, 8)
        ] + [0.0] * 6

    def _get_current_robot_gripper_pose(self) -> float:
        assert self._last_state is not None
        return float(self._last_state.get(self.objects[0], "pos_gripper"))


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

    LiftedMoveToTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetGroundController,
        )
    )

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)
    prev_target = Variable("?prev_target", MujocoObjectType)

    LiftedMoveToTargetFromOtherTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, prev_target],
            MoveToTargetGroundController,
        )
    )

    # Move arm to conf controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedMoveArmToConfController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            MoveArmToConfController,
        )
    )

    # Toss controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedTossController: LiftedParameterizedController = LiftedParameterizedController(
        [robot],
        TossController,
    )

    # Move arm to end effector controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedMoveArmToEndEffectorController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            MoveArmToEndEffectorController,
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

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    cube = Variable("?cube", MujocoMovableObjectType)
    # Unused by the controller; present so an operator can say the cube is still on
    # this side of it, as move_to_target_from_other_target carries ?prev_target.
    barrier = Variable("?barrier", MujocoMovableObjectType)

    LiftedPickCubeController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, cube, barrier],
            PickCubeController,
        )
    )

    class MoveToTossLocationAndToss(MoveToTossLocationAndTossController):
        """Composed move-and-toss with pre-configured PyBullet sim."""

        def __init__(self, objects):
            super().__init__(objects, pybullet_sim=pybullet_sim)

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    held = Variable("?held", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoMovableObjectType)

    LiftedMoveToTossLocationAndTossController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, held, barrier],
            MoveToTossLocationAndToss,
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
        "pick_cube": LiftedPickCubeController,
        "move_to_toss_location_and_toss": LiftedMoveToTossLocationAndTossController,
    }
