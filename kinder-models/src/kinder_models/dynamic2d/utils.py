"""Utilities for 2D dynamic robot manipulation tasks."""

import abc
from typing import Optional, Sequence, Union

import numpy as np
from bilevel_planning.structs import GroundParameterizedController
from kinder.envs.dynamic2d.dyn_obstruction2d import (
    DynObstruction2DEnvConfig,
)
from kinder.envs.dynamic2d.object_types import KinRobotType
from kinder.envs.dynamic2d.utils import (
    KinRobotActionSpace,
    run_motion_planning_for_kin_robot,
)
from kinder.envs.kinematic2d.structs import SE2Pose
from numpy.typing import NDArray
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.objects import Object


class Dynamic2dRobotController(GroundParameterizedController, abc.ABC):
    """General controller for 2D dynamic robot manipulation tasks using SE2
    waypoints."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: KinRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
        skip_collision_check: bool = False,
    ) -> None:
        self._robot = objects[0]
        assert self._robot.is_instance(KinRobotType)
        super().__init__(objects)
        self._current_params: Union[tuple[float, ...], float] = 0.0
        self._current_plan: Union[list[NDArray[np.float32]], None] = None
        self._current_state: Union[ObjectCentricState, None] = None
        self._action_space = action_space
        self._init_constant_state = init_constant_state
        self._skip_collision_check = skip_collision_check
        # Extract max deltas from action space bounds
        self._max_delta_x = action_space.high[0]
        self._max_delta_y = action_space.high[1]
        self._max_delta_theta = action_space.high[2]
        self._max_delta_arm = action_space.high[3]
        self._max_delta_gripper = action_space.high[4]

        env_config = DynObstruction2DEnvConfig()
        self.world_x_min = env_config.world_min_x + env_config.robot_base_radius
        self.world_x_max = env_config.world_max_x - env_config.robot_base_radius
        self.world_y_min = env_config.world_min_y + env_config.robot_base_radius
        self.world_y_max = env_config.world_max_y - env_config.robot_base_radius
        self.finger_gap_max = env_config.gripper_base_height

    @abc.abstractmethod
    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate a waypoint plan with SE2 pose and arm length values."""

    @abc.abstractmethod
    def _get_gripper_actions(self, state: ObjectCentricState) -> tuple[float, float]:
        """Get gripper actions (deltas) for during and after waypoint movement.

        Args:
            state: Current state to calculate gripper delta from.

        Returns:
            Tuple of (delta_during_plan, delta_after_plan) where values are:
            - Positive values mean opening the gripper (increasing finger_gap)
            - Negative values mean closing the gripper (decreasing finger_gap)
            - 0.0 means no change
            These are changes (deltas) in finger_gap, not absolute values.
        """

    def _requires_multi_phase_gripper(self) -> bool:
        """Check if this controller requires multi-phase gripper execution.

        Override this method to force multi-phase execution (e.g., for pick controllers
        that need to move to target first, then close gripper).

        Args:
            state: Current state.

        Returns:
            True if multi-phase execution is required, False otherwise.
        """
        return False

    def _interpolate_se2(
        self,
        start: SE2Pose,
        end: SE2Pose,
    ) -> list[SE2Pose]:
        """Linearly interpolate between two SE2 poses respecting action-space bounds."""
        dx = end.x - start.x
        dy = end.y - start.y
        dtheta = get_signed_angle_distance(end.theta, start.theta)

        abs_x = self._max_delta_x if dx > 0 else abs(self._max_delta_x)
        abs_y = self._max_delta_y if dy > 0 else abs(self._max_delta_y)
        abs_theta = self._max_delta_theta if dtheta > 0 else abs(self._max_delta_theta)

        x_steps = max(1, int(np.ceil(abs(dx) / abs_x)) if abs_x > 0 else 1)
        y_steps = max(1, int(np.ceil(abs(dy) / abs_y)) if abs_y > 0 else 1)
        theta_steps = max(
            1, int(np.ceil(abs(dtheta) / abs_theta)) if abs_theta > 0 else 1
        )
        num_steps = max(x_steps, y_steps, theta_steps)

        path: list[SE2Pose] = []
        for i in range(num_steps + 1):
            alpha = i / num_steps if num_steps > 0 else 1.0
            path.append(
                SE2Pose(
                    start.x + alpha * dx,
                    start.y + alpha * dy,
                    wrap_angle(start.theta + alpha * dtheta),
                )
            )
        return path

    def _waypoints_to_plan(
        self,
        state: ObjectCentricState,
        waypoints: list[tuple[SE2Pose, float]],
        gripper_during_plan: float,
    ) -> list[NDArray[np.float64]]:
        """Convert waypoints to an action plan.

        Uses ``run_motion_planning_for_kin_robot`` (BiRRT on SE2) for
        collision-free path segments and linearly interpolates arm_joint
        along the resulting path.  Falls back to direct interpolation
        when the planner fails or collision checking is disabled.
        """
        curr_x = state.get(self._robot, "x")
        curr_y = state.get(self._robot, "y")
        curr_theta = state.get(self._robot, "theta")
        curr_arm = state.get(self._robot, "arm_joint")
        current_pos: tuple[SE2Pose, float] = (
            SE2Pose(curr_x, curr_y, curr_theta),
            curr_arm,
        )
        waypoints = [current_pos] + waypoints

        # Build full state (with constant objects) for the motion planner.
        full_state = state.copy()
        if self._init_constant_state is not None:
            full_state.data.update(self._init_constant_state.data)

        plan: list[NDArray[np.float64]] = []
        for (start_pose, start_arm), (end_pose, end_arm) in zip(
            waypoints[:-1], waypoints[1:]
        ):
            if np.allclose(
                [start_pose.x, start_pose.y, start_pose.theta, start_arm],
                [end_pose.x, end_pose.y, end_pose.theta, end_arm],
            ):
                continue

            # Update full_state to the segment's starting configuration so
            # that the motion planner sees the correct robot position.
            full_state.set(self._robot, "x", start_pose.x)
            full_state.set(self._robot, "y", start_pose.y)
            full_state.set(self._robot, "theta", start_pose.theta)
            full_state.set(self._robot, "arm_joint", start_arm)

            # Plan a collision-free SE2 path (arm is interpolated separately).
            se2_path: list[SE2Pose] | None = None
            if not self._skip_collision_check:
                se2_path = run_motion_planning_for_kin_robot(
                    full_state,
                    self._robot,
                    end_pose,
                    self._action_space,
                )

            if se2_path is None:
                # Direct interpolation (fallback or skip_collision_check).
                se2_path = self._interpolate_se2(start_pose, end_pose)

            # Ensure the SE2 path has enough steps for the arm change.
            total_darm = abs(end_arm - start_arm)
            if total_darm > 1e-8:
                arm_steps_needed = int(np.ceil(total_darm / abs(self._max_delta_arm)))
                while len(se2_path) - 1 < arm_steps_needed:
                    se2_path.append(se2_path[-1])

            # Convert SE2 path to actions, linearly interpolating arm_joint.
            n = len(se2_path)
            for i in range(n - 1):
                p1, p2 = se2_path[i], se2_path[i + 1]
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dtheta = get_signed_angle_distance(p2.theta, p1.theta)

                alpha_prev = i / max(1, n - 1)
                alpha_next = (i + 1) / max(1, n - 1)
                darm = (start_arm + alpha_next * (end_arm - start_arm)) - (
                    start_arm + alpha_prev * (end_arm - start_arm)
                )
                darm = float(
                    np.clip(darm, -abs(self._max_delta_arm), abs(self._max_delta_arm))
                )

                action = np.array(
                    [dx, dy, dtheta, darm, gripper_during_plan],
                    dtype=np.float64,
                )
                plan.append(action)

        return plan

    def reset(
        self, x: ObjectCentricState, params: Union[tuple[float, ...], float]
    ) -> None:
        """Reset the controller with new state and parameters."""
        self._current_params = params
        self._current_plan = None
        # Normalize theta values in the initial state (same as observe())
        x_normalized = x.copy()
        for obj in x_normalized:
            try:
                theta = x_normalized.get(obj, "theta")
                x_normalized.set(obj, "theta", wrap_angle(theta))
            except KeyError:
                pass
        self._current_state = x_normalized

    def terminated(self) -> bool:
        """Check if the controller has finished executing its plan."""
        return self._current_plan is not None and len(self._current_plan) == 0

    def step(self) -> NDArray[np.float32]:
        """Execute the next action in the controller's plan."""
        assert self._current_state is not None
        if self._current_plan is None:
            self._current_plan = self._generate_plan(self._current_state)
        return self._current_plan.pop(0)

    def observe(self, x: ObjectCentricState) -> None:
        """Update the controller with a new observed state.

        IMPORTANT: Normalize all theta values to [-pi, pi] since the simulation
        may not wrap them after accumulating angular changes.
        """
        x_normalized = x.copy()
        for obj in x_normalized:
            try:
                theta = x_normalized.get(obj, "theta")
                x_normalized.set(obj, "theta", wrap_angle(theta))
            except KeyError:
                # Object doesn't have theta attribute, skip
                pass
        self._current_state = x_normalized

    def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
        waypoints = self._generate_waypoints(x)
        gripper_delta_during_plan, gripper_delta_after_plan = self._get_gripper_actions(
            x
        )

        max_gripper_delta = abs(self._max_delta_gripper)

        # Check if we need multi-phase execution
        # Either explicitly requested or
        # if gripper_after_plan is different than gripper_delta_during_plan
        requires_multi_phase = (
            self._requires_multi_phase_gripper()
            or abs(gripper_delta_after_plan - gripper_delta_during_plan) > 0
        )

        if requires_multi_phase:
            # Multi-phase: move to waypoint, then adjust gripper
            # Phase 1: Move to final waypoint with gripper_delta_during_plan
            # (typically 0.0 for pick)
            waypoint_plan = self._waypoints_to_plan(
                x, waypoints, gripper_delta_during_plan
            )

            # Phase 2: Adjust gripper gradually
            gripper_plan: list[NDArray[np.float32]] = []
            remaining_delta = gripper_delta_after_plan
            while abs(remaining_delta) > 1e-6:
                step_delta = np.clip(
                    remaining_delta, -max_gripper_delta, max_gripper_delta
                )
                gripper_plan.append(
                    np.array([0, 0, 0, 0, step_delta], dtype=np.float32)
                )
                remaining_delta -= step_delta

            # Add waiting step to allow physics to update
            gripper_plan.append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
            gripper_plan.append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
            return waypoint_plan + gripper_plan

        # Single phase: move with gripper action without final gripper adjustment
        waypoint_plan = self._waypoints_to_plan(x, waypoints, gripper_delta_during_plan)

        # Add waiting step to allow physics to update
        waypoint_plan.append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
        waypoint_plan.append(np.array([0, 0, 0, 0, 0], dtype=np.float32))
        return waypoint_plan
