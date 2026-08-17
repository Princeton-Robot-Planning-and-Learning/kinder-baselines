"""The TidyBot arm's pick: drive to the object, grasp it, retract to home.

Nothing here is specific to a scene or a task. What the controller does is base
navigation, IK and motion planning through :mod:`kinder_models.dynamic3d.utils`'s
``PyBulletSim``, a trapezoidal approach profile, a gripper close, and a trapezoidal
retract to the arm's home configuration -- TidyBot-arm logic, all of it, over the
shared constants in that same module.

It lived in ``shelf/parameterized_skills.py`` until now only because shelf was the
first domain that needed it, which left the tossing domain subclassing across a domain
boundary to reuse an arm. It sits beside ``utils.py`` and ``ik_solver.py`` instead, so
every domain reaches for it as a peer.

Domains specialise it by subclassing and overriding the class attributes below --
where the end effector aims, and how the approach settles before the gripper closes.
"""

from typing import Any

import numpy as np
from bilevel_planning.structs import GroundParameterizedController
from kinder.envs.dynamic3d.object_types import MujocoMovableObjectType
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
)
from spatialmath import SE2

from kinder_models.dynamic3d.utils import (
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    GRASP_CLOSE_THRESHOLD,
    GRASP_TRANSFORM_TO_OBJECT,
    MAX_SAMPLER_ATTEMPTS,
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


class TidyBotPickController(GroundParameterizedController[ObjectCentricState, Array]):
    """Controller for motion planning to pick up a target.

    The object parameters are:
        robot: The robot itself.
        object: The target object.
    """

    # Where the end effector aims, in the target's own frame. The default is the shared
    # module constant: a small approach offset and a height 10 mm above the object's
    # own origin, which suits an object whose origin sits at its base. A class attribute
    # rather than the bare constant so a subclass picking a differently-shaped object
    # can aim elsewhere without moving anyone else's grasp.
    GRASP_TRANSFORM = GRASP_TRANSFORM_TO_OBJECT

    # How long the approach may keep tracking after its trapezoidal profile runs out.
    # Zero -- the base behaviour -- makes the profile's last index the last step taken,
    # arrived or not, and closes the gripper immediately after it. That is enough
    # wherever the residual joint error at the end of the profile is small against the
    # grasp's own tolerance. A subclass reaching for a shape that has to be gripped
    # squarely can wait instead. A class attribute for the same reason GRASP_TRANSFORM
    # is one: the difference belongs to the subclass, not to this controller.
    APPROACH_SETTLE_STEPS = 0
    # A one-shot feedforward added to the arm command while settling, and how many
    # settling steps to wait before measuring it. The proportional command alone leaves
    # a standing joint error and has to: holding the arm against gravity needs a
    # non-zero command, and the only thing producing one is the error itself. Adding
    # `gain` times that error to the command shifts the steady state by gain/kp of it,
    # so a gain of kp nulls it. Zero -- the base behaviour -- adds nothing, and the
    # block is unreachable at zero settle steps anyway, so a subclass that wants the
    # cancellation has to raise APPROACH_SETTLE_STEPS as well.
    APPROACH_FEEDFORWARD_GAIN = 0.0
    APPROACH_FEEDFORWARD_DELAY = 0

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
        self._approach_command_offset: np.ndarray = np.zeros(7)
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
        params: Any,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
    ) -> None:
        # Initialize the PyBullet interface if this is the first time ever.
        if self._pybullet_sim is None:
            self._pybullet_sim = PyBulletSim(x)
        # Update the current state and parameters.
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

        target_grasp_pose_world = Pose(
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
            target_grasp_pose_world,
            self.GRASP_TRANSFORM,
        )

        self._pybullet_sim.base_link_to_held_obj = multiply_poses(
            target_end_effector_pose.invert(),
            target_grasp_pose_world,
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
            curr, final, _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        self._approach_start_joints = curr.copy()
        self._approach_step_idx = 0
        self._approach_command_offset = np.zeros(7)
        # Compute trapezoidal velocity profile for retract (grasp conf → home).
        self._retract_trajectory, self._retract_traj_dir = _compute_per_joint_profile(
            final, self.home_joints[:7], _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        self._retract_start_joints = final.copy()
        self._retract_step_idx = 0

    def terminated(self) -> bool:
        assert (
            self._current_arm_joint_plan is not None
            and self._current_retract_plan is not None
        )
        return self._lifted

    def _approach_has_settled(self) -> bool:
        """Whether the approach is finished and the gripper may close.

        The trapezoidal profile says when the *commanded* path ends; it says nothing
        about where the arm actually is, because each step commands a proportional
        correction (kp times the remaining joint error) rather than a position. So the
        arm reaches the last profile index still trailing the target, and holding that
        final command for APPROACH_SETTLE_STEPS more steps is what closes the gap.

        The residual plateaus rather than converging -- proportional velocity control
        against gravity needs a non-zero command to hold a pose, and the only thing
        producing one is the error itself. Measured over six seeds it settles at
        0.72-1.04 degrees, so this waits a fixed number of steps rather than for a
        tolerance it would never reach.
        """
        if self._approach_step_idx < len(self._approach_trajectory):
            return False
        overrun = self._approach_step_idx - len(self._approach_trajectory)
        return overrun >= self.APPROACH_SETTLE_STEPS

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
            if self._approach_has_settled():
                self._pre_grasp = True
            idx = min(self._approach_step_idx, len(self._approach_trajectory) - 1)
            s = float(self._approach_trajectory[idx])
            kp = 2.0
            curr = np.array(self._get_current_robot_arm_conf()[:7])
            target = self._approach_start_joints + self._approach_traj_dir * s
            if (
                self._approach_step_idx - len(self._approach_trajectory)
                == self.APPROACH_FEEDFORWARD_DELAY
            ):
                self._approach_command_offset = self.APPROACH_FEEDFORWARD_GAIN * (
                    target - curr
                )
            action = np.zeros(11, dtype=np.float32)
            action[3:10] = kp * (target - curr) + self._approach_command_offset
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
        if x.get(robot_obj, "pos_gripper") > 0.2:
            return GRASP_CLOSE_THRESHOLD
        return 0.0

    def _robot_is_close_to_conf(
        self, conf: JointPositions, atol: float = WAYPOINT_TOLERANCE
    ) -> bool:
        current_conf = self._get_current_robot_arm_conf()
        assert self._pybullet_sim is not None
        dist = self._pybullet_sim.get_joint_distance(current_conf, conf)
        return dist < atol

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
