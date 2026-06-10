"""Part 1 skill: place a held block on top of a support block ("stack").

You implement ONE small thing here: the resting height -- see TODO(3). Everything
else (the motion planner, carrying the held block, turning the path into actions)
is provided by ``crv_skills.MotionPlannedController``.
"""

# pylint: disable=fixme  # this file intentionally contains TODO markers
from typing import Optional

import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from crv_skills import (
    MotionPlannedController,
    make_lifted_controller,
    make_lifted_pick_controller,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace
from relational_structs import Object, ObjectCentricState, Variable


def support_top_y(state: ObjectCentricState, support: Object) -> float:
    """The y-coordinate of the TOP edge of the ``support`` block.

    A block at ``y`` with the given ``height`` occupies ``[y, y + height]`` (y is
    its bottom edge), so its top edge is ``y + height``.
    """
    # TODO(3): return the y of the support block's top edge, so the held block
    # comes to rest ON TOP of it. (Look at state.get(support, "y") / "height".)
    raise NotImplementedError("TODO(3): resting height -- top edge of the support")


def get_robot_stack_position(
    held: Object,
    support: Object,
    state: ObjectCentricState,
    robot_x: float,
    robot_arm_joint: float,
) -> tuple[float, float]:
    """Robot (x, y) at which releasing ``held`` leaves it resting on ``support``."""
    robot = state.get_objects(CRVRobotType)[0]
    ground = support_top_y(state, support)  # <-- TODO(3) lives in this helper
    padding = 1e-4
    y = (
        ground
        + state.get(held, "height")
        + robot_arm_joint
        + state.get(robot, "gripper_width") / 2
        + padding
    )
    return (robot_x, y)


class GroundPlaceOnBlockController(MotionPlannedController):
    """Stack a held block on a support block.

    Objects: ``[robot, held, support]``.
    """

    def __init__(self, objects, action_space, init_constant_state=None) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._support = objects[2]
        assert self._block.is_instance(RectangleType)
        assert self._support.is_instance(RectangleType)

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        # Choose where along the support's top to release, so the held block stays
        # on. (Provided.)
        support_x = x.get(self._support, "x")
        support_width = x.get(self._support, "width")
        block_x = x.get(self._block, "x")
        robot_x = x.get(self._robot, "x")
        offset_x = robot_x - block_x
        block_width = x.get(self._block, "width")
        lower_x = support_x + offset_x
        upper_x = lower_x + (support_width - block_width)
        if lower_x > upper_x:
            lower_x, upper_x = upper_x, lower_x
        return rng.uniform(lower_x, upper_x)

    def _retract_arm_in_transit(self) -> bool:
        return False  # carrying a block: keep the arm out so it's carried + checked

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 1.0, 0.0  # hold while moving, release at the end

    def _target_pose_and_arm(self, state):
        arm = state.get(self._robot, "arm_joint")
        placement_x = (
            self._current_params[0]
            if isinstance(self._current_params, (tuple, list))
            else self._current_params
        )
        tx, ty = get_robot_stack_position(
            self._block, self._support, state, placement_x, arm
        )
        return SE2Pose(tx, ty, state.get(self._robot, "theta")), arm


def create_stacking_controllers(
    action_space: CRVRobotActionSpace,
    init_constant_state: Optional[ObjectCentricState] = None,
) -> dict[str, LiftedParameterizedController]:
    """The provided ``pick`` controller plus the ``place_on_block`` controller."""
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    support = Variable("?support", RectangleType)
    return {
        "pick": make_lifted_pick_controller(action_space, init_constant_state),
        "place_on_block": make_lifted_controller(
            [robot, block, support],
            GroundPlaceOnBlockController,
            action_space,
            init_constant_state,
        ),
    }
