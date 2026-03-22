"""Parameterized skills for the DynPushPullHook2D environment."""

from typing import Optional, Sequence, cast

import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.dynamic2d.dyn_pushpullhook2d import (
    HookType,
)
from kinder.envs.dynamic2d.object_types import KinRobotType
from kinder.envs.dynamic2d.utils import KinRobotActionSpace
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.utils import state_2d_has_collision
from numpy.typing import NDArray
from prpl_utils.utils import wrap_angle
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.objects import Object, Variable

from kinder_models.dynamic2d.utils import Dynamic2dRobotController


class GroundGraspHookController(Dynamic2dRobotController):
    """Controller for grasping the hook from the long-side bottom.

    The hook is an L-shaped object with two sides. In the hook's local frame
    (theta=0), the geometry is:

        (-l1, 0)────────────────────(0, 0)
          │     side1 (long bar)       │
        (-l1, -w)───(-w, -w)          │ side2
                      │                │ (short bar)
                   (-w, -l2)────(0, -l2)

    This controller approaches the hook from below the long side (side1)
    and grasps the bar of thickness ``w``.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: KinRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._hook = objects[1]

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        """Sample arm length parameter for the grasp."""
        max_arm_length = x.get(self._robot, "arm_length")
        min_arm_length = (
            x.get(self._robot, "base_radius")
            + x.get(self._robot, "gripper_base_width")
            + 1e-4
        )
        arm_length = rng.uniform(min_arm_length, max_arm_length)
        return float(arm_length)

    def _requires_multi_phase_gripper(self) -> bool:
        """Grasp uses two phases: move to hook, then close gripper."""
        return True

    def _get_gripper_actions(self, state: ObjectCentricState) -> tuple[float, float]:
        """Keep gripper open during movement, close to hook width after.

        Returns:
            (delta_during, delta_after) for finger_gap changes.
        """
        curr_finger_gap = state.get(self._robot, "finger_gap")
        finger_width = state.get(self._robot, "finger_width")
        hook_width = state.get(self._hook, "width")

        # Desired finger gap: slightly larger than hook bar width for grasping
        desired_finger_gap = max(0.0, hook_width + finger_width - 0.175)
        delta_needed = desired_finger_gap - curr_finger_gap

        return 0.0, delta_needed

    def _calculate_pre_grasp_robot_pose(
        self,
        state: ObjectCentricState,
        arm_length: float,
    ) -> SE2Pose:
        """Calculate the pre-grasp pose below the hook's long-side bottom."""
        hook_x = state.get(self._hook, "x")
        hook_y = state.get(self._hook, "y")
        hook_theta = wrap_angle(state.get(self._hook, "theta"))
        hook_l1 = state.get(self._hook, "length_side1")
        hook_w = state.get(self._hook, "width")

        finger_width = state.get(self._robot, "finger_width")
        gripper_base_width = state.get(self._robot, "gripper_base_width")

        custom_dx = - arm_length - gripper_base_width - finger_width - hook_l1
        custom_dy = - hook_w / 2

        target_se2_pose = SE2Pose(hook_x, hook_y, hook_theta) * SE2Pose(
            custom_dx, custom_dy, 0.0
        )
        return target_se2_pose

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate waypoints to grasp the hook from the long-side bottom."""
        desired_arm_length = cast(float, self._current_params)

        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = wrap_angle(state.get(self._robot, "theta"))
        robot_radius = state.get(self._robot, "base_radius")
        finger_width = state.get(self._robot, "finger_width")

        target_se2_pre_pose = self._calculate_pre_grasp_robot_pose(
            state, desired_arm_length
        )

        # Check if the target pose is collision-free.
        full_state = state.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            full_state.data.update(init_constant_state.data)

        full_state.set(self._robot, "x", target_se2_pre_pose.x)
        full_state.set(self._robot, "y", target_se2_pre_pose.y)
        full_state.set(self._robot, "theta", target_se2_pre_pose.theta)
        full_state.set(self._robot, "arm_joint", desired_arm_length)

        moving_objects = {self._robot}
        static_objects = set(full_state) - moving_objects

        if state_2d_has_collision(full_state, moving_objects, static_objects, {}):
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free pre-grasp pose for hook."
            )

        # Waypoints: retract arm -> navigate to pre-grasp -> move in for contact.
        final_waypoints: list[tuple[SE2Pose, float]] = [
            (SE2Pose(robot_x, robot_y, robot_theta), robot_radius)
        ]
        final_waypoints.append((target_se2_pre_pose, desired_arm_length))

        # Move closer along robot's forward direction to contact the bar.
        relative_move_in = SE2Pose(finger_width * 0.9, 0, 0)
        final_waypoints.append(
            (target_se2_pre_pose * relative_move_in, desired_arm_length)
        )

        return final_waypoints


def create_lifted_controllers(
    action_space: KinRobotActionSpace,
    init_constant_state: Optional[ObjectCentricState] = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for DynPushPullHook2D.

    Args:
        action_space: The action space for the KinRobot.
        init_constant_state: Optional initial constant state.

    Returns:
        Dictionary mapping controller names to LiftedParameterizedController instances.
    """
    arm_length_params_space = Box(
        low=np.array([0.0]),
        high=np.array([1.0]),
        dtype=np.float32,
    )

    class GraspHookController(GroundGraspHookController):
        """Lifted wrapper that binds action_space and init_constant_state."""

        def __init__(self, objects: Sequence[Object]) -> None:
            super().__init__(objects, action_space, init_constant_state)

    robot = Variable("?robot", KinRobotType)
    hook = Variable("?hook", HookType)

    grasp_hook_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, hook],
            GraspHookController,
            arm_length_params_space,
        )
    )

    return {
        "grasp_hook": grasp_hook_controller,
    }
