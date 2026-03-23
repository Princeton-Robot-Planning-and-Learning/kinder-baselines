"""Perception-based bilevel planning models for the motion 2D environment.

The state_abstractor queries a VLM with a rendered image instead of using programmatic
geometric checks.
"""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic2d.motion2d import (
    ObjectCentricMotion2DEnv,
    RectangleType,
    TargetRegionType,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace
from kinder_models.kinematic2d.envs.motion2d.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from prpl_llm_utils.models import PretrainedLargeModel
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

from kinder_perception_planning.vlm_utils import query_vlm_for_atom_vals

# Natural-language descriptions of each predicate for the VLM prompt.
_PREDICATE_DESCRIPTIONS: dict[str, str] = {
    "AtTgt": (
        "The robot (small circle) is inside the target region "
        "(the green rectangular area)."
    ),
    "NotAtTgt": (
        "The robot (small circle) is NOT inside the target region "
        "(the green rectangular area)."
    ),
    "AtPassage": (
        "The robot is positioned in the gap (passage) between two "
        "vertically-aligned rectangular obstacles that share the same "
        "x-coordinate."
    ),
    "NotAtPassage": (
        "The robot is NOT positioned in the gap (passage) between the "
        "two specified rectangular obstacles."
    ),
    "NotAtAnyPassage": (
        "The robot is not positioned at any passage between any pair " "of obstacles."
    ),
}


def create_perception_planning_models(
    observation_space: Space,
    action_space: Space,
    num_passages: int = 2,
    vlm: PretrainedLargeModel | None = None,
) -> SesameModels:
    """Create the env models for motion 2D with VLM-based state abstraction."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricMotion2DEnv(num_passages=num_passages)

    # Convert observations into states.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {CRVRobotType, TargetRegionType, RectangleType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    AtTgt = Predicate("AtTgt", [CRVRobotType, TargetRegionType])
    NotAtTgt = Predicate("NotAtTgt", [CRVRobotType, TargetRegionType])
    AtPassage = Predicate("AtPassage", [CRVRobotType, RectangleType, RectangleType])
    NotAtPassage = Predicate(
        "NotAtPassage", [CRVRobotType, RectangleType, RectangleType]
    )
    NotAtAnyPassage = Predicate("NotAtAnyPassage", [CRVRobotType])
    predicates = {AtTgt, NotAtTgt, AtPassage, NotAtPassage, NotAtAnyPassage}

    # State abstractor: queries the VLM with a rendered image.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state by querying the VLM with a rendered image."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]
        obstacles = x.get_objects(RectangleType)
        objects = {robot, target_region} | set(obstacles)

        # Build all candidate ground atoms.
        candidate_atoms: list[GroundAtom] = []
        candidate_atoms.append(GroundAtom(AtTgt, [robot, target_region]))
        candidate_atoms.append(GroundAtom(NotAtTgt, [robot, target_region]))
        candidate_atoms.append(GroundAtom(NotAtAnyPassage, [robot]))
        for obs1 in obstacles:
            for obs2 in obstacles:
                if obs1 != obs2:
                    candidate_atoms.append(GroundAtom(AtPassage, [robot, obs1, obs2]))
                    candidate_atoms.append(
                        GroundAtom(NotAtPassage, [robot, obs1, obs2])
                    )

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
        """The goal is to have the robot at the target region."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]
        atoms = {GroundAtom(AtTgt, [robot, target_region])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target = Variable("?target", TargetRegionType)
    obstacle1 = Variable("?obstacle1", RectangleType)
    obstacle2 = Variable("?obstacle2", RectangleType)
    obstacle3 = Variable("?obstacle3", RectangleType)
    obstacle4 = Variable("?obstacle4", RectangleType)

    MoveToTgtFromNoPassageOperator = LiftedOperator(
        "MoveToTgtFromNoPassage",
        [robot, target],
        preconditions={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(NotAtAnyPassage, [robot]),
        },
        add_effects={LiftedAtom(AtTgt, [robot, target])},
        delete_effects={LiftedAtom(NotAtTgt, [robot, target])},
    )

    MoveToTgtFromPassageOperator = LiftedOperator(
        "MoveToTgtFromPassage",
        [robot, target, obstacle1, obstacle2],
        preconditions={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(AtPassage, [robot, obstacle1, obstacle2]),
        },
        add_effects={
            LiftedAtom(AtTgt, [robot, target]),
            LiftedAtom(NotAtAnyPassage, [robot]),
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
        },
        delete_effects={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(AtPassage, [robot, obstacle1, obstacle2]),
        },
    )

    MoveToPassageFromNoPassageOperator = LiftedOperator(
        "MoveToPassageFromNoPassage",
        [robot, obstacle1, obstacle2],
        preconditions={
            LiftedAtom(NotAtAnyPassage, [robot]),
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
        },
        add_effects={LiftedAtom(AtPassage, [robot, obstacle1, obstacle2])},
        delete_effects={
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
            LiftedAtom(NotAtAnyPassage, [robot]),
        },
    )

    MoveToPassageFromPassageOperator = LiftedOperator(
        "MoveToPassageFromPassage",
        [robot, obstacle1, obstacle2, obstacle3, obstacle4],
        preconditions={
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
            LiftedAtom(AtPassage, [robot, obstacle3, obstacle4]),
        },
        add_effects={LiftedAtom(NotAtPassage, [robot, obstacle3, obstacle4])},
        delete_effects={
            LiftedAtom(AtPassage, [robot, obstacle3, obstacle4]),
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
        },
    )

    # Get lifted controllers from kinder_models.
    lifted_controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state
    )
    MoveToTgtFromNoPassageController = lifted_controllers["move_to_tgt_from_no_passage"]
    MoveToTgtFromPassageController = lifted_controllers["move_to_tgt_from_passage"]
    MoveToPassageFromNoPassageController = lifted_controllers[
        "move_to_passage_from_no_passage"
    ]
    MoveToPassageFromPassageController = lifted_controllers[
        "move_to_passage_from_passage"
    ]

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToTgtFromNoPassageOperator, MoveToTgtFromNoPassageController),
        LiftedSkill(MoveToTgtFromPassageOperator, MoveToTgtFromPassageController),
        LiftedSkill(
            MoveToPassageFromNoPassageOperator, MoveToPassageFromNoPassageController
        ),
        LiftedSkill(
            MoveToPassageFromPassageOperator, MoveToPassageFromPassageController
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
