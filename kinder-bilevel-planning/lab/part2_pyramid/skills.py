"""Part 2 skills -- you write the place skill(s) the pyramid needs.

There is no single hole here: you design the skills. Use your Part 1
``place_on_block`` controller as the pattern. To make a place-style skill,
subclass ``crv_skills.MotionPlannedController`` and implement:

  * ``_target_pose_and_arm(state)`` -> the robot SE2 pose + arm length to end at,
  * ``_retract_arm_in_transit()`` -> False while carrying a block (so it's carried),
  * ``_get_vacuum_actions()`` -> (1.0, 0.0) to hold then release, and
  * ``sample_parameters(state, rng)`` -> where to release (a float).

``get_robot_transfer_position`` and the block geometry conventions (a block at
``y`` with ``height`` occupies ``[y, y + height]``; ``x`` is its left edge) are
the same as in Part 1.
"""

# pylint: disable=fixme,unused-import  # intentional TODOs; imports are for your skills
from typing import Optional

from bilevel_planning.structs import LiftedParameterizedController
from crv_skills import (  # noqa: F401
    MotionPlannedController,
    get_robot_transfer_position,
    make_lifted_controller,
    make_lifted_pick_controller,
)
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace
from relational_structs import ObjectCentricState

# TODO: define your place controller class(es) here, e.g.
#
#   class GroundPlaceNextToController(MotionPlannedController):
#       ...
#
# Each should subclass MotionPlannedController and implement the four methods
# listed in the module docstring.


def create_pyramid_controllers(
    action_space: CRVRobotActionSpace,
    init_constant_state: Optional[ObjectCentricState] = None,
) -> dict[str, LiftedParameterizedController]:
    """Return the lifted controllers your skills need (keyed by name).

    ``pick`` is provided. Add an entry for each place skill you write, using
    ``make_lifted_controller([...variables...], YourController, action_space,
    init_constant_state)``.
    """
    controllers = {
        "pick": make_lifted_pick_controller(action_space, init_constant_state),
    }
    # TODO: add your place controllers, e.g.
    #   controllers["place_next_to"] = make_lifted_controller(
    #       [robot, block, anchor], GroundPlaceNextToController, action_space,
    #       init_constant_state)
    return controllers
