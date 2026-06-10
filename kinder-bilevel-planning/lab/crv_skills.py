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

from typing import Optional, Sequence

import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    run_motion_planning_for_crv_robot,
)
from kinder_models.kinematic2d.envs.obstruction2d.parameterized_skills import (
    get_robot_transfer_position,
)
from kinder_models.kinematic2d.utils import Kinematic2dRobotController
from relational_structs import Object, ObjectCentricState, Variable

__all__ = [
    "MotionPlannedController",
    "get_robot_transfer_position",
    "make_lifted_pick_controller",
    "make_lifted_controller",
]


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
