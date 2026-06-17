"""Provided motion-planning plumbing for the lab -- you do NOT need to edit this.

A skill (controller) drives the robot to do one thing -- pick a block, place a
block -- by producing low-level actions. The hard part is getting there without
crashing into anything, so these controllers call a real **motion planner**
(BiRRT) to find a collision-free path of robot poses; the path is then turned
into actions for you.

To write a new place-style skill you only subclass ``MotionPlannedController``
and answer one question -- ``_target_pose_and_arm``: *where should the robot end
up* (and how far should its arm be extended) to do the placement? The base class
plans a collision-free route there and carries any held block along the way.
"""

import abc
from typing import Optional, Sequence, Union

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from kinder.envs.kinematic2d.obstruction2d import TargetSurfaceType
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    run_motion_planning_for_crv_robot,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState, Variable

__all__ = [
    "MotionPlannedController",
    "get_robot_transfer_position",
    "make_lifted_pick_controller",
    "make_lifted_controller",
]


# Vendored from kinder_models.kinematic2d so the lab needs no kinder-models
# install: the SE2-waypoint controller base class and the transfer-position
# helper. Kept verbatim from kinder_models.kinematic2d.{utils,
# envs.obstruction2d.parameterized_skills} except for this note.
class Kinematic2dRobotController(GroundParameterizedController, abc.ABC):
    """General controller for 2D robot manipulation tasks using SE2 waypoints."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
        safe_y: float = 0.8,
    ) -> None:
        self._robot = objects[0]
        assert self._robot.is_instance(CRVRobotType)
        super().__init__(objects)
        self._current_params: Union[tuple[float, ...], float] = 0.0
        self._current_plan: Union[list[NDArray[np.float32]], None] = None
        self._current_state: Union[ObjectCentricState, None] = None
        self._safe_y = safe_y
        self._init_constant_state = init_constant_state
        # Extract max deltas from action space bounds
        self._max_delta_x = action_space.high[0]
        self._max_delta_y = action_space.high[1]
        self._max_delta_theta = action_space.high[2]
        self._max_delta_arm = action_space.high[3]

    @abc.abstractmethod
    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate a waypoint plan with SE2 pose and arm length values."""

    @abc.abstractmethod
    def _get_vacuum_actions(self) -> tuple[float, float]:
        """Get vacuum actions for during and after waypoint movement."""

    def _waypoints_to_plan(
        self,
        state: ObjectCentricState,
        waypoints: list[tuple[SE2Pose, float]],
        vacuum_during_plan: float,
    ) -> list[NDArray[np.float32]]:
        curr_x = state.get(self._robot, "x")
        curr_y = state.get(self._robot, "y")
        curr_theta = state.get(self._robot, "theta")
        curr_arm = state.get(self._robot, "arm_joint")
        current_pos = (SE2Pose(curr_x, curr_y, curr_theta), curr_arm)
        waypoints = [current_pos] + waypoints
        plan: list[NDArray[np.float32]] = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            start_pose = np.array([start[0].x, start[0].y, start[0].theta, start[1]])
            end_pose = np.array([end[0].x, end[0].y, end[0].theta, end[1]])
            if np.allclose(start_pose, end_pose):
                continue
            total_dx = end[0].x - start[0].x
            total_dy = end[0].y - start[0].y
            total_dtheta = end[0].theta - start[0].theta
            # NOTE: Handle angle wrapping for shortest path
            if abs(total_dtheta) > np.pi:
                if total_dtheta > 0:
                    total_dtheta -= 2 * np.pi
                else:
                    total_dtheta += 2 * np.pi
            total_darm = end[1] - start[1]
            num_steps = int(
                max(
                    np.ceil(abs(total_dx) / self._max_delta_x),
                    np.ceil(abs(total_dy) / self._max_delta_y),
                    np.ceil(abs(total_dtheta) / self._max_delta_theta),
                    np.ceil(abs(total_darm) / self._max_delta_arm),
                )
            )
            dx = total_dx / num_steps
            dy = total_dy / num_steps
            dtheta = total_dtheta / num_steps
            darm = total_darm / num_steps
            action = np.array(
                [dx, dy, dtheta, darm, vacuum_during_plan], dtype=np.float32
            )
            for _ in range(num_steps):
                plan.append(action)

        return plan

    def reset(
        self, x: ObjectCentricState, params: Union[tuple[float, ...], float]
    ) -> None:
        """Reset the controller with new state and parameters."""
        self._current_params = params
        self._current_plan = None
        self._current_state = x

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
        """Update the controller with a new observed state."""
        self._current_state = x

    def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
        waypoints = self._generate_waypoints(x)
        vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
        waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
        plan_suffix: list[NDArray[np.float32]] = [
            # Change the vacuum.
            np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
        ]
        return waypoint_plan + plan_suffix


def get_robot_transfer_position(
    block: Object,
    state: ObjectCentricState,
    block_x: float,
    robot_arm_joint: float,
    relative_x_offset: float = 0,
) -> tuple[float, float]:
    """Get the x, y position the robot should be at to place or grasp the block."""
    robot = state.get_objects(CRVRobotType)[0]
    surface = state.get_objects(TargetSurfaceType)[0]
    ground = state.get(surface, "y") + state.get(surface, "height")
    padding = 1e-4
    x = block_x + relative_x_offset
    y = (
        ground
        + state.get(block, "height")
        + robot_arm_joint
        + state.get(robot, "gripper_width") / 2
        + padding
    )
    return (x, y)


class MotionPlannedController(Kinematic2dRobotController):
    """Drive the robot to a target pose via BiRRT, then act.

    Subclasses implement:
      * ``_target_pose_and_arm(state)`` -> (robot SE2 pose, arm length) to end at,
      * ``_retract_arm_in_transit()`` -> keep the arm in while moving? (True when
        nothing is held, so the robot stays compact; False while carrying so the
        held block is carried along and checked for collisions), and
      * ``_get_vacuum_actions()`` -> (vacuum while moving, vacuum at the end).
    """

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._action_space = action_space

    def _target_pose_and_arm(self, state: ObjectCentricState) -> tuple[SE2Pose, float]:
        raise NotImplementedError

    def _retract_arm_in_transit(self) -> bool:
        raise NotImplementedError

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot = self._robot
        robot_radius = state.get(robot, "base_radius")
        start_pose = SE2Pose(
            state.get(robot, "x"), state.get(robot, "y"), state.get(robot, "theta")
        )
        target_pose, target_arm = self._target_pose_and_arm(state)

        # Hold the arm fixed while transiting (the planner plans the base motion).
        transit_arm = robot_radius if self._retract_arm_in_transit() else target_arm
        mp_state = state.copy()
        mp_state.set(robot, "arm_joint", transit_arm)
        if self._init_constant_state is not None:
            mp_state.data.update(self._init_constant_state.data)
        assert isinstance(self._action_space, CRVRobotActionSpace)
        path = run_motion_planning_for_crv_robot(
            mp_state, robot, target_pose, self._action_space
        )
        if path is None:
            raise TrajectorySamplingFailure(
                "Motion planning failed to find a collision-free path."
            )
        waypoints: list[tuple[SE2Pose, float]] = [(start_pose, transit_arm)]
        for pose in path:
            waypoints.append((pose, transit_arm))
        # End at the target with the final arm length (extends to grasp/place).
        waypoints.append((target_pose, target_arm))
        return waypoints


class _GroundPickController(MotionPlannedController):
    """Pick a block: transit with the arm retracted, then extend to grasp."""

    def __init__(self, objects, action_space, init_constant_state=None) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        assert self._block.is_instance(RectangleType)

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        gripper_height = x.get(self._robot, "gripper_height")
        block_width = x.get(self._block, "width")
        return rng.uniform(-gripper_height / 2, block_width + gripper_height / 2)

    def _retract_arm_in_transit(self) -> bool:
        return True  # nothing held; stay compact while navigating

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 1.0  # off while moving, on to grasp at the end

    def _target_pose_and_arm(self, state):
        arm = state.get(self._robot, "arm_joint")
        offset = (
            self._current_params[0]
            if isinstance(self._current_params, (tuple, list))
            else self._current_params
        )
        block_x = state.get(self._block, "x")
        tx, ty = get_robot_transfer_position(
            self._block, state, block_x, arm, relative_x_offset=offset
        )
        return SE2Pose(tx, ty, state.get(self._robot, "theta")), arm


def make_lifted_controller(
    variables: Sequence[Variable],
    ground_controller_cls: type,
    action_space: CRVRobotActionSpace,
    init_constant_state: Optional[ObjectCentricState] = None,
) -> LiftedParameterizedController:
    """Wrap a ground-controller class into a lifted controller (params in [0, 1])."""
    params_space = Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)

    class _Bound(ground_controller_cls):  # type: ignore[misc, valid-type]
        def __init__(self, objects: Sequence[Object]) -> None:
            super().__init__(objects, action_space, init_constant_state)

    return LiftedParameterizedController(list(variables), _Bound, params_space)


def make_lifted_pick_controller(
    action_space: CRVRobotActionSpace,
    init_constant_state: Optional[ObjectCentricState] = None,
) -> LiftedParameterizedController:
    """The provided pick skill, as a lifted controller over [robot, block]."""
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    return make_lifted_controller(
        [robot, block], _GroundPickController, action_space, init_constant_state
    )
