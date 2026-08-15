"""Parameterized skills for the TidyBot3D tossing environment."""

from typing import Any, NamedTuple

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

from kinder_models.dynamic3d.shelf.parameterized_skills import PickShelfController
from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    _CONTROL_TIMESTEP,
    GRASP_CLOSE_THRESHOLD,
    GRIPPER_CLOSED_THRESHOLD,
    GRIPPER_OPEN_COMMAND_TOLERANCE,
    MINIMUM_HOLDING_HEIGHT,
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

# The two dials TossController already takes, opened up as sampled parameters. The
# speed tops out at the profile's own clamp; the millisecond window is centred on the
# demonstrated default.
TOSS_SPEED_BOUNDS = (np.deg2rad(60.0), TOSS_MAX_VELOCITY)
TOSS_RELEASE_MS_BOUNDS = (600.0, 840.0)


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


class TossSwing(NamedTuple):
    """A planned swing: where the arm goes, and when the gripper opens."""

    trajectory: np.ndarray
    direction: np.ndarray
    start_joint_angles: np.ndarray
    release_step: int
    release_slice: int


def plan_toss_swing(
    joint_plan: list[JointPositions],
    current_joint_angles: JointPositions,
    release_speed: float,
    gripper_release_ms: int,
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


def toss_swing_action(
    swing: TossSwing,
    step_idx: int,
    current_joint_angles: JointPositions,
    gripper_pose: float,
    has_released: bool,
) -> Array:
    """The swing's command for one control step, opening the gripper mid-step.

    The usual (18,) action, except on the step the release falls inside, which returns
    a (TOSS_SLICES_PER_CONTROL_STEP, 18) schedule so gripper_release_ms means the
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
        self._release_speed: float = TOSS_MAX_VELOCITY
        self._gripper_release_ms: int = TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS
        self._tossing: bool = False

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x, rng  # not used
        return np.array([TOSS_WINDUP_ARM_CONFIGURATION, TOSS_RELEASE_ARM_CONFIGURATION])

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        release_speed: float = TOSS_MAX_VELOCITY,
        gripper_release_ms: int = TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
    ) -> None:
        current_params = np.asarray(params, dtype=np.float32)
        assert current_params.shape == (2, 7)
        self._last_state = x
        self._toss_params = current_params[1]
        self._release_speed = release_speed
        self._gripper_release_ms = gripper_release_ms
        self._tossing = False
        # The toss is planned when the windup ends, from the state actually reached.
        self._windup_controller.reset(x, current_params[0])

    def terminated(self) -> bool:
        return self._tossing and self._toss_controller.terminated()

    def step(self) -> Array:
        if not self._tossing and self._windup_controller.terminated():
            assert self._last_state is not None
            assert self._toss_params is not None
            self._toss_controller.reset(
                self._last_state,
                self._toss_params,
                release_speed=self._release_speed,
                gripper_release_ms=self._gripper_release_ms,
            )
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


def _quaternion_product(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _cube_rotation_symmetries() -> tuple[tuple[float, float, float, float], ...]:
    """The 24 rotations mapping a cube onto itself: 6 faces down, each at 4 yaws.

    Closed under composition from the three quarter-turns, which is the definition
    rather than a listing that could be mistyped.
    """
    half = np.sqrt(0.5)
    generators = [
        (half, 0.0, 0.0, half),
        (0.0, half, 0.0, half),
        (0.0, 0.0, half, half),
    ]

    def canonical(q: tuple[float, float, float, float]) -> tuple[float, ...]:
        # q and -q are the same rotation; pick one so the set dedupes.
        rounded = tuple(0.0 + round(v, 6) for v in q)
        for value in rounded:
            if value > 1e-9:
                return rounded
            if value < -1e-9:
                return tuple(-v + 0.0 for v in rounded)
        return rounded

    found = {canonical((0.0, 0.0, 0.0, 1.0)): (0.0, 0.0, 0.0, 1.0)}
    frontier = [(0.0, 0.0, 0.0, 1.0)]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            product = _quaternion_product(current, generator)
            key = canonical(product)
            if key not in found:
                found[key] = (
                    float(key[0]),
                    float(key[1]),
                    float(key[2]),
                    float(key[3]),
                )
                frontier.append(product)
    return tuple(found.values())


CUBE_ROTATION_SYMMETRIES = _cube_rotation_symmetries()


def upright_grasp_rotations(
    rotation: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    """Every upright rotation the cube could equally be grasped at, nearest yaw first.

    A cube resting on any face is the same cube, so its roll and pitch carry no
    information -- deriving a grasp from them asks the gripper to approach along
    whichever face happens to be up, from underneath the floor for a cube on its top.
    Resting on a face it is also four-fold symmetric about the vertical, so all four
    yaws are the same grasp and a caller can fall through when the arm cannot reach one.
    """
    best_tilt = np.inf
    yaw = 0.0
    for symmetry in CUBE_ROTATION_SYMMETRIES:
        x, y, z, w = _quaternion_product(rotation, symmetry)
        tilt = x * x + y * y
        if tilt < best_tilt - 1e-12:
            best_tilt = tilt
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return tuple(
        (0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2)))
        for angle in (yaw, yaw + np.pi / 2, yaw - np.pi / 2, yaw + np.pi)
    )


def canonical_upright_rotation(
    rotation: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """The nearest upright rotation, the first of upright_grasp_rotations."""
    return upright_grasp_rotations(rotation)[0]


# Pick standoffs to try in order, nominal first, spanning what the shelf pick used to
# sample. Fixed rather than drawn, so the skill takes no continuous parameters.
PICK_STANDOFF_LADDER = tuple(
    (distance, rot)
    for rot in (0.0, np.pi / 8, -np.pi / 8, np.pi / 4, -np.pi / 4)
    for distance in (0.55, 0.5, 0.6)
)


class PickCubeController(PickShelfController):
    """Pick a cube up off the ground, taking no continuous parameters.

    The object parameters are:
        robot: The robot itself.
        cube: The cube to pick up.

    Where to stand is derived rather than sampled: PICK_STANDOFF_LADDER is walked from
    the nominal pose outwards until one plans, so a caller cannot draw an unreachable
    pose and there is nothing for a refiner to backtrack over. A grasp that closes on
    nothing releases before terminating, leaving the hand empty rather than commanded
    shut on air.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._releasing: bool = False

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        return np.zeros(0, dtype=np.float32)

    def reset(
        self,
        x: ObjectCentricState,
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        del params
        self._releasing = False
        # Grasp the cube as if it were upright. Every face-down rest is the same cube,
        # so the raw rotation only tells the grasp which face happens to be up -- and
        # for a cube resting on its top that asks the gripper to come from below.
        cube = self.objects[1]
        rotations = upright_grasp_rotations(
            (
                x.get(cube, "qx"),
                x.get(cube, "qy"),
                x.get(cube, "qz"),
                x.get(cube, "qw"),
            )
        )
        for rotation in rotations:
            upright = x.copy()
            for feature, value in zip(("qx", "qy", "qz", "qw"), rotation):
                upright.set(cube, feature, value)
            for distance, rot in PICK_STANDOFF_LADDER:
                try:
                    super().reset(
                        upright,
                        np.array([distance, rot]),
                        extend_xy_magnitude=extend_xy_magnitude,
                        extend_rot_magnitude=extend_rot_magnitude,
                    )
                    return
                except (AssertionError, ValueError, RuntimeError):
                    continue
        raise ValueError(
            f"no reachable pick pose among {len(rotations)} grasp rotations "
            f"x {len(PICK_STANDOFF_LADDER)} standoffs"
        )

    def terminated(self) -> bool:
        if not super().terminated():
            return False
        if self._grasp_took():
            return True
        self._releasing = True
        return self._gripper_is_open()

    def step(self) -> Array:
        if self._releasing:
            action = np.zeros(11, dtype=np.float32)
            action[-1] = 0
            return action
        return super().step()

    def _grasp_took(self) -> bool:
        assert self._last_state is not None
        return bool(self._last_state.get(self.objects[1], "z") > MINIMUM_HOLDING_HEIGHT)

    def _gripper_is_open(self) -> bool:
        assert self._last_state is not None
        pos = self._last_state.get(self.objects[0], "pos_gripper")
        return bool(pos < GRIPPER_OPEN_COMMAND_TOLERANCE)


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

    One flat controller over phase flags, as pick_shelf is: base motion, then the
    windup, then the swing. The swing is planned at the windup's end rather than in
    reset, because it has to start from the arm conf actually reached.
    """

    def __init__(
        self, *args, pybullet_sim: PyBulletSim | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._pybullet_sim: PyBulletSim | None = pybullet_sim
        self._release_speed: float = TOSS_MAX_VELOCITY
        self._gripper_release_ms: int = TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS
        self._navigated: bool = False
        self._wound_up: bool = False
        self._current_base_motion_plan: list[SE2] | None = None
        self._windup_trajectory: np.ndarray = np.array([])
        self._windup_dir: np.ndarray = np.zeros(7)
        self._windup_start_joint_angles: np.ndarray = np.zeros(7)
        self._windup_step_idx: int = 0
        self._swing: TossSwing | None = None
        self._swing_step_idx: int = 0
        self._has_released: bool = False

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x  # not used
        return np.array(
            [
                rng.uniform(*TOSS_TARGET_DISTANCE_BOUNDS),
                rng.uniform(*TOSS_TARGET_ROTATION_BOUNDS),
                rng.uniform(*TOSS_SPEED_BOUNDS),
                rng.uniform(*TOSS_RELEASE_MS_BOUNDS),
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
        self._navigated = False
        self._wound_up = False
        self._windup_step_idx = 0
        self._swing = None
        self._swing_step_idx = 0
        self._has_released = False

        # The robot's own cargo would otherwise reject every base plan.
        if disable_collision_objects is None:
            disable_collision_objects = [self.objects[2].name]
        target_object_pose = get_overhead_object_se2_pose(x, self.objects[1])
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
        assert base_motion_plan is not None
        self._current_base_motion_plan = base_motion_plan

        # Plan the arm from where the base motion will end, as pick_shelf does. The
        # windup lands within 0.03 rad of its target, which leaves the swing's profile
        # the same length, so planning it here rather than on arrival costs nothing and
        # surfaces a planning failure from reset.
        plan_x = x.copy()
        robot = self.objects[0]
        final_base_pose = base_motion_plan[-1]
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
        assert windup_plan is not None, "Motion planning failed"
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
        assert swing_plan is not None, "Motion planning failed"
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
        if not self._navigated:
            return self._base_motion_step()
        if not self._wound_up:
            return self._windup_step()
        return self._swing_step()

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _base_motion_step(self) -> Array:
        assert self._current_base_motion_plan is not None
        while len(self._current_base_motion_plan) > 1:
            peek_pose = self._current_base_motion_plan[0]
            if self._robot_is_close_to_pose(peek_pose):
                self._current_base_motion_plan.pop(0)
            break
        if self._robot_is_close_to_pose(self._current_base_motion_plan[-1]):
            self._navigated = True
        robot_pose = self._get_current_robot_pose()
        next_pose = self._current_base_motion_plan[0]
        action = np.zeros(11, dtype=np.float32)
        action[0] = next_pose.x - robot_pose.x
        action[1] = next_pose.y - robot_pose.y
        action[2] = get_signed_angle_distance(next_pose.theta(), robot_pose.theta())
        action[-1] = self._get_current_robot_gripper_pose()
        return action

    def _windup_step(self) -> Array:
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
            self._wound_up = True
        return action

    def _swing_step(self) -> Array:
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

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    cube = Variable("?cube", MujocoMovableObjectType)

    LiftedPickCubeController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, cube],
            PickCubeController,
        )
    )

    class MoveToTossLocationAndToss(MoveToTossLocationAndTossController):
        """Composed move-and-toss with pre-configured PyBullet sim."""

        def __init__(self, objects):
            super().__init__(objects, pybullet_sim=pybullet_sim)

    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)
    held = Variable("?held", MujocoMovableObjectType)

    LiftedMoveToTossLocationAndTossController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, held],
            MoveToTossLocationAndToss,
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
        "pick_cube": LiftedPickCubeController,
        "move_to_toss_location_and_toss": LiftedMoveToTossLocationAndTossController,
    }
