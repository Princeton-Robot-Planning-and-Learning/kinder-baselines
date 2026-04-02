"""Perception-based bilevel planning models for the stick button 2D environment.

The state_abstractor queries a VLM with a rendered image instead of using programmatic
geometric/color checks.
"""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic2d.object_types import (
    CircleType,
    CRVRobotType,
    RectangleType,
)
from kinder.envs.kinematic2d.stickbutton2d import ObjectCentricStickButton2DEnv
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace
from kinder_models.kinematic2d.envs.stickbutton2d.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from prpl_llm_utils.models import PretrainedLargeModel
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

from kinder_perception_planning.vlm_utils import query_vlm_for_atom_vals

# Natural-language descriptions of each predicate for the VLM prompt.
_PREDICATE_DESCRIPTIONS: dict[str, str] = {
    "Grasped": (
        "The robot (the multi-part mechanism at the top) is currently "
        "holding/grasping the stick (a thin rectangle)."
    ),
    "HandEmpty": ("The robot is NOT holding anything — its gripper is empty."),
    "Pressed": (
        "The specified button (circle) has been pressed. A pressed button "
        "appears in a different color than an unpressed one."
    ),
    "RobotAboveButton": (
        "The robot's body is directly above and overlapping with the "
        "specified button (circle)."
    ),
    "StickAboveButton": (
        "The stick (thin rectangle) is directly above and overlapping "
        "with the specified button (circle)."
    ),
    "AboveNoButton": (
        "Neither the robot nor the stick is currently positioned above " "any button."
    ),
}


def create_perception_planning_models(
    observation_space: Space,
    action_space: Space,
    num_buttons: int,
    vlm: PretrainedLargeModel | None = None,
) -> SesameModels:
    """Create the env models for stick button 2D with VLM-based state abstraction."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricStickButton2DEnv(num_buttons=num_buttons)

    # Convert observations into states.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState, u: NDArray[np.float32]
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs

    # Types.
    types = {CRVRobotType, RectangleType, CircleType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Grasped = Predicate("Grasped", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    Pressed = Predicate("Pressed", [CircleType])
    RobotAboveButton = Predicate("RobotAboveButton", [CRVRobotType, CircleType])
    StickAboveButton = Predicate("StickAboveButton", [RectangleType, CircleType])
    AboveNoButton = Predicate("AboveNoButton", [])
    predicates = {
        Grasped,
        HandEmpty,
        Pressed,
        RobotAboveButton,
        StickAboveButton,
        AboveNoButton,
    }

    # State abstractor: queries the VLM with a rendered image.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state by querying the VLM with a rendered image."""
        robot = x.get_objects(CRVRobotType)[0]
        stick = x.get_objects(RectangleType)[0]
        buttons = x.get_objects(CircleType)
        objects = {robot, stick} | set(buttons)

        # Build all candidate ground atoms.
        candidate_atoms: list[GroundAtom] = []
        candidate_atoms.append(GroundAtom(Grasped, [robot, stick]))
        candidate_atoms.append(GroundAtom(HandEmpty, [robot]))
        for button in buttons:
            candidate_atoms.append(GroundAtom(Pressed, [button]))
            candidate_atoms.append(GroundAtom(RobotAboveButton, [robot, button]))
            candidate_atoms.append(GroundAtom(StickAboveButton, [stick, button]))
        candidate_atoms.append(GroundAtom(AboveNoButton, []))

        # Render the scene.
        sim.reset(options={"init_state": x.copy()})
        rendered = sim.render()
        assert rendered is not None

        # Query the VLM.
        assert vlm is not None, "VLM must be provided for perception planning"
        true_atoms = query_vlm_for_atom_vals(
            vlm, rendered, candidate_atoms, _PREDICATE_DESCRIPTIONS
        )

        return RelationalAbstractState(true_atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to press all buttons."""
        del x  # not needed
        atoms: set[GroundAtom] = set()
        for i in range(num_buttons):
            button = Object(f"button{i}", CircleType)
            atoms.add(GroundAtom(Pressed, [button]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    stick = Variable("?stick", RectangleType)
    button = Variable("?button", CircleType)
    from_button = Variable("?from_button", CircleType)

    RobotPressButtonFromNothingOperator = LiftedOperator(
        "RobotPressButtonFromNothing",
        [robot, button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        },
        delete_effects={LiftedAtom(AboveNoButton, [])},
    )

    RobotPressButtonFromButtonOperator = LiftedOperator(
        "RobotPressButtonFromButton",
        [robot, button, from_button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
        add_effects={
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        },
        delete_effects={LiftedAtom(RobotAboveButton, [robot, from_button])},
    )

    PickStickFromNothingOperator = LiftedOperator(
        "PickStickFromNothing",
        [robot, stick],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(Grasped, [robot, stick]),
        },
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PickStickFromButtonOperator = LiftedOperator(
        "PickStickFromButton",
        [robot, stick, from_button],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
        add_effects={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        },
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        },
    )

    StickPressButtonFromNothingOperator = LiftedOperator(
        "StickPressButtonFromNothing",
        [robot, stick, button],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        },
        add_effects={
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button]),
        },
        delete_effects={LiftedAtom(AboveNoButton, [])},
    )

    StickPressButtonFromButtonOperator = LiftedOperator(
        "StickPressButtonFromButton",
        [robot, stick, button, from_button],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(StickAboveButton, [stick, from_button]),
        },
        add_effects={
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button]),
        },
        delete_effects={LiftedAtom(StickAboveButton, [stick, from_button])},
    )

    PlaceStickOperator = LiftedOperator(
        "PlaceStick",
        [robot, stick],
        preconditions={
            LiftedAtom(Grasped, [robot, stick]),
        },
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(AboveNoButton, [])},
        delete_effects={LiftedAtom(Grasped, [robot, stick])},
    )

    # Get lifted controllers from kinder_models.
    lifted_controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state
    )
    RobotPressButtonFromNothingController = lifted_controllers[
        "robot_press_button_from_nothing"
    ]
    RobotPressButtonFromButtonController = lifted_controllers[
        "robot_press_button_from_button"
    ]
    PickStickFromNothingController = lifted_controllers["pick_stick_from_nothing"]
    PickStickFromButtonController = lifted_controllers["pick_stick_from_button"]
    StickPressButtonFromNothingController = lifted_controllers[
        "stick_press_button_from_nothing"
    ]
    StickPressButtonFromButtonController = lifted_controllers[
        "stick_press_button_from_button"
    ]
    RobotPlaceStickController = lifted_controllers["robot_place_stick"]

    # Finalize the skills.
    skills = {
        LiftedSkill(PickStickFromNothingOperator, PickStickFromNothingController),
        LiftedSkill(PickStickFromButtonOperator, PickStickFromButtonController),
        LiftedSkill(PlaceStickOperator, RobotPlaceStickController),
        LiftedSkill(
            RobotPressButtonFromNothingOperator, RobotPressButtonFromNothingController
        ),
        LiftedSkill(
            RobotPressButtonFromButtonOperator, RobotPressButtonFromButtonController
        ),
        LiftedSkill(
            StickPressButtonFromNothingOperator, StickPressButtonFromNothingController
        ),
        LiftedSkill(
            StickPressButtonFromButtonOperator, StickPressButtonFromButtonController
        ),
    }

    # Finalize the models.
    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
