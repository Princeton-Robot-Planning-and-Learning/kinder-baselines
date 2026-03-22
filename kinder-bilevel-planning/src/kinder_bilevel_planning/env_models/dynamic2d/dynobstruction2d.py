"""Bilevel planning models for the dynamic obstruction 2D environment."""

from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.dynamic2d.dyn_obstruction2d import (
    ObjectCentricDynObstruction2DEnv,
    TargetBlockType,
    TargetSurfaceType,
)
from kinder.envs.dynamic2d.object_types import DynRectangleType, KinRobotType
from kinder.envs.dynamic2d.utils import (
    KinRobotActionSpace,
)
from kinder.envs.kinematic2d.utils import is_on
from kinder_models.dynamic2d.dynobstruction2d.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for dynamic obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, KinRobotActionSpace)

    sim = ObjectCentricDynObstruction2DEnv(num_obstructions=num_obstructions)

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray,
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {KinRobotType, DynRectangleType, TargetBlockType, TargetSurfaceType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Holding = Predicate("Holding", [KinRobotType, DynRectangleType])
    IsTargetBlock = Predicate("IsTargetBlock", [DynRectangleType])
    IsObstruction = Predicate("IsObstruction", [DynRectangleType])
    HandEmpty = Predicate("HandEmpty", [KinRobotType])
    OnTable = Predicate("OnTable", [DynRectangleType])
    OnTarget = Predicate("OnTarget", [DynRectangleType])
    predicates = {Holding, HandEmpty, OnTable, OnTarget, IsTargetBlock, IsObstruction}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(KinRobotType)[0]
        target_surface = x.get_objects(TargetSurfaceType)[0]
        blocks = x.get_objects(DynRectangleType)

        atoms: set[GroundAtom] = set()

        # Add holding / handempty atoms.
        held_blocks: set = set()
        for block in blocks:
            if x.get(block, "held"):
                atoms.add(GroundAtom(Holding, [robot, block]))
                held_blocks.add(block)
            if block.name == "target_block":
                atoms.add(GroundAtom(IsTargetBlock, [block]))
            else:
                atoms.add(GroundAtom(IsObstruction, [block]))
        if not held_blocks:
            atoms.add(GroundAtom(HandEmpty, [robot]))

        # Add "on" atoms.
        for block in blocks:
            if block in held_blocks:
                continue
            if is_on(x, block, target_surface, {}):
                atoms.add(GroundAtom(OnTarget, [block]))
            else:
                atoms.add(GroundAtom(OnTable, [block]))

        objects = {robot, target_surface} | set(blocks)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to place the target block on the target surface."""
        target_block = x.get_objects(TargetBlockType)[0]
        atoms = {GroundAtom(OnTarget, [target_block])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    # Variable names must match the lifted controller variables exactly.
    robot = Variable("?robot", KinRobotType)
    obstruction = Variable("?obstruction", DynRectangleType)
    target_block = Variable("?target_block", TargetBlockType)
    target_surface = Variable("?target_surface", TargetSurfaceType)

    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, target_block],
        preconditions={
            LiftedAtom(IsTargetBlock, [target_block]),
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [target_block]),
        },
        add_effects={LiftedAtom(Holding, [robot, target_block])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [target_block]),
        },
    )
    PickFromTargetOperator = LiftedOperator(
        "PickFromTarget",
        [robot, obstruction],
        preconditions={
            LiftedAtom(IsObstruction, [obstruction]),
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTarget, [obstruction]),
        },
        add_effects={LiftedAtom(Holding, [robot, obstruction])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTarget, [obstruction]),
        },
    )
    PlaceOnTableOperator = LiftedOperator(
        "PlaceOnTable",
        [robot, obstruction],
        preconditions={
            LiftedAtom(Holding, [robot, obstruction]),
            LiftedAtom(IsObstruction, [obstruction]),
        },
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [obstruction]),
        },
        delete_effects={LiftedAtom(Holding, [robot, obstruction])},
    )
    PlaceOnTargetOperator = LiftedOperator(
        "PlaceOnTarget",
        [robot, target_block, target_surface],
        preconditions={
            LiftedAtom(Holding, [robot, target_block]),
            LiftedAtom(IsTargetBlock, [target_block]),
        },
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTarget, [target_block]),
        },
        delete_effects={LiftedAtom(Holding, [robot, target_block])},
    )

    # Get lifted controllers from kinder_models
    lifted_controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state
    )
    PickTargetController = lifted_controllers["pick_tgt"]
    PickObstructionController = lifted_controllers["pick_obstruction"]
    PlaceOnTableController = lifted_controllers["place_obstruction"]
    PlaceOnTargetController = lifted_controllers["place_tgt_surface"]

    # Finalize the skills.
    skills = {
        LiftedSkill(PickFromTableOperator, PickTargetController),
        LiftedSkill(PickFromTargetOperator, PickObstructionController),
        LiftedSkill(PlaceOnTableOperator, PlaceOnTableController),
        LiftedSkill(PlaceOnTargetOperator, PlaceOnTargetController),
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
