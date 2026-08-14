"""Parameterized skills for the TidyBot3D tossing environment."""

from typing import Any

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from kinder.envs.dynamic3d.mujoco_utils import (
    CONTROL_SCHEDULE_TIMESTEP,
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
    run_motion_planning,
)
from relational_structs import (
    Array,
    ObjectCentricState,
    Variable,
)
from spatialmath import SE2

from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    _CONTROL_TIMESTEP,
    GRASP_CLOSE_THRESHOLD,
    GRIPPER_CLOSED_THRESHOLD,
    GRIPPER_OPEN_COMMAND_TOLERANCE,
    WAYPOINT_TOLERANCE,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    _compute_per_joint_profile,
    _trapezoidal_motion_profile,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
)

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


# Where a throw is possible; the upper part does not score.
TOSS_TARGET_DISTANCE_BOUNDS = (1.25, 1.45)

# Wider than WAYPOINT_TOLERANCE: the sampler already spends half of it off-axis.
THROW_POSE_TOLERANCE = 2 * WAYPOINT_TOLERANCE

# The widest rotation that still spends only half of WAYPOINT_TOLERANCE off the bin
# axis at the furthest standoff. Derived, so retuning either cannot invalidate it.
TOSS_MAX_TARGET_ROTATION = float(
    np.arcsin(0.5 * WAYPOINT_TOLERANCE / TOSS_TARGET_DISTANCE_BOUNDS[1])
)
TOSS_TARGET_ROTATION_BOUNDS = (-TOSS_MAX_TARGET_ROTATION, TOSS_MAX_TARGET_ROTATION)


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


class MoveToThrowPoseController(MoveToTargetGroundController):
    """Controller for motion planning to a base pose to throw a held object from.

    The object parameters are:
        robot: The robot itself.
        object: The target object to throw at.
        held: The movable object the robot is currently holding.

    The continuous parameters are the same as MoveToTargetGroundController's:
        target_distance: float
        target_rot: float (radians)

    Excludes the held object from base collision checking by default, or the robot's own
    cargo rejects every plan. sweep3D's wipe controller excludes its wiper likewise.
    """

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        distance = rng.uniform(*TOSS_TARGET_DISTANCE_BOUNDS)  # type: ignore
        rot = rng.uniform(*TOSS_TARGET_ROTATION_BOUNDS)
        return np.array([distance, rot])

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
        disable_collision_objects: list[str] | None = None,
    ) -> None:
        if disable_collision_objects is None:
            disable_collision_objects = [self.objects[2].name]
        super().reset(
            x,
            params,
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
            disable_collision_objects=disable_collision_objects,
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

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        # None builds one on first reset; a caller sharing a sim across groundings
        # passes it here, since a planner grounds a fresh controller per attempt.
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
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

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        # See MoveArmToConfController.__init__ for why this is injectable.
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._step_idx: int = 0
        self._toss_dir: np.ndarray = np.zeros(7)
        self._trajectory: np.ndarray = np.array([])
        self._has_released: bool = False
        self._start_joint_angles: np.ndarray = np.zeros(7)
        # The control step the gripper opens on, and the millisecond within it.
        self._release_step: int = 0
        self._release_slice: int = 0

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

        The two knobs the real robot's movej_primitive.execute() takes.
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
        # Compute trapezoidal velocity profile along the path.
        curr_joint_angles = self._get_current_robot_arm_conf()
        final_joint_angles = self._current_arm_joint_plan[-1]
        dq = np.subtract(final_joint_angles, curr_joint_angles)[:7]
        # Not the real robot's controller: the parameter space matches, the trajectory
        # does not. Do not align this with the _compute_per_joint_profile siblings.
        s_total = float(np.linalg.norm(dq))
        if s_total > 1e-4:
            self._toss_dir = dq / s_total
        else:
            self._toss_dir = np.zeros(7)
        max_vel, max_accel, max_decel = toss_profile_limits(release_speed)
        self._trajectory = _trapezoidal_motion_profile(
            s_total,
            max_vel=max_vel,
            max_accel=max_accel,
            max_decel=max_decel,
            step_size=_CONTROL_TIMESTEP,
        )
        self._start_joint_angles = np.array(curr_joint_angles[:7])
        self._release_step, self._release_slice = divmod(
            int(gripper_release_ms), TOSS_SLICES_PER_CONTROL_STEP
        )
        self._has_released = False
        self._step_idx = 0

    def terminated(self) -> bool:
        # Terminate when we've gone through the entire profile.
        return self._step_idx >= len(self._trajectory)

    def step(self) -> Array:
        """The swing's command for one control step, opening the gripper mid-step.

        The usual (18,) action, except on the step the release falls inside, which
        returns a (TOSS_SLICES_PER_CONTROL_STEP, 18) schedule so gripper_release_ms
        means the millisecond it names rather than the next step boundary.
        """
        assert self._current_arm_joint_plan is not None
        gripper_pose = self._get_current_robot_gripper_pose()
        action = np.zeros(18, dtype=np.float32)

        # Look up target distance along path from precomputed trapezoidal profile.
        idx = min(self._step_idx, len(self._trajectory) - 1)
        s = float(self._trajectory[idx])
        # Compute velocity via finite difference of the profile.
        if idx > 0:
            ds = (self._trajectory[idx] - self._trajectory[idx - 1]) / _CONTROL_TIMESTEP
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

        # Open the gripper on the step reset() computed, at the millisecond it computed.
        released_before_now = self._has_released or self._step_idx > self._release_step
        opens_this_step = self._step_idx == self._release_step
        self._step_idx += 1

        if released_before_now:
            action[10] = 0.0
            return action
        if not opens_this_step:
            action[10] = gripper_pose
            return action

        self._has_released = True
        if self._release_slice == 0:
            action[10] = 0.0
            return action
        schedule = np.repeat(action[None], TOSS_SLICES_PER_CONTROL_STEP, axis=0)
        schedule[: self._release_slice, 10] = gripper_pose
        schedule[self._release_slice :, 10] = 0.0
        return schedule

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


class TossFromWindupController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Controller for winding the arm up and then tossing the held object.

    A composition of the two existing controllers -- MoveArmToConfController to the
    windup conf, then TossController to the release conf -- the way shelf's pick_shelf
    composes approach, grasp and retract into one task-level skill.

    The object parameters are:
        robot: The robot itself.

    The continuous parameters are a (2, 7) array of arm configurations:
        params[0]: the windup conf, reached with MoveArmToConfController.
        params[1]: the release conf, swung to and released at by TossController.

    Run as one controller rather than two skills, since splitting them would need a
    predicate over the windup conf to chain the operators.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._windup_controller = MoveArmToConfController(
            self.objects, pybullet_sim=pybullet_sim
        )
        self._toss_controller = TossController(self.objects, pybullet_sim=pybullet_sim)
        self._last_state: ObjectCentricState | None = None
        self._toss_params: np.ndarray | None = None
        self._tossing: bool = False

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x, rng  # not used
        return np.array([TOSS_WINDUP_ARM_CONFIGURATION, TOSS_RELEASE_ARM_CONFIGURATION])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        current_params = np.asarray(params, dtype=np.float32)
        assert current_params.shape == (2, 7)
        self._last_state = x
        self._toss_params = current_params[1]
        self._tossing = False
        # The toss is planned when the windup ends, from the state actually reached.
        self._windup_controller.reset(x, current_params[0])

    def terminated(self) -> bool:
        return self._tossing and self._toss_controller.terminated()

    def step(self) -> Array:
        if not self._tossing and self._windup_controller.terminated():
            assert self._last_state is not None
            assert self._toss_params is not None
            self._toss_controller.reset(self._last_state, self._toss_params)
            self._tossing = True
        if self._tossing:
            return self._toss_controller.step()
        return self._windup_controller.step()

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x
        self._windup_controller.observe(x)
        self._toss_controller.observe(x)


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


def create_lifted_controllers(
    action_space: TidyBot3DRobotActionSpace,
    init_constant_state: ObjectCentricState | None = None,
    pybullet_sim: PyBulletSim | None = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for the TidyBot3D ground environment."""
    del action_space, init_constant_state  # not used

    class TossFromWindup(TossFromWindupController):
        """Toss-from-windup controller with pre-configured PyBullet sim."""

        def __init__(self, objects):
            super().__init__(objects, pybullet_sim=pybullet_sim)

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

    # Move to throw pose controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)
    # The held object is necessarily movable. Typing it MujocoObjectType would let a
    # planner ground it to a fixture, or to the same object as ?target.
    held = Variable("?held", MujocoMovableObjectType)

    LiftedMoveToThrowPoseController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, held],
            MoveToThrowPoseController,
        )
    )

    # Toss from windup controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)

    LiftedTossFromWindupController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot],
            TossFromWindup,
        )
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

    return {
        "move_to_target": LiftedMoveToTargetController,
        "move_to_target_from_other_target": LiftedMoveToTargetFromOtherTargetController,
        "move_arm_to_conf": LiftedMoveArmToConfController,
        "toss": LiftedTossController,
        "move_to_throw_pose": LiftedMoveToThrowPoseController,
        "toss_from_windup": LiftedTossFromWindupController,
        "move_arm_to_end_effector": LiftedMoveArmToEndEffectorController,
        "close_gripper": LiftedCloseGripperController,
        "open_gripper": LiftedOpenGripperController,
    }
