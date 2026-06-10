"""Part 1 domain: stack the obstruction on top of the target block.

This file wires up the planning models. You fill TWO small holes:
  * TODO(1) -- the ``On`` predicate's classifier (when is a block on another block?)
  * TODO(2) -- the ``Stack`` operator's preconditions and effects.
The skill hole, TODO(3), is in ``skills.py``.

Everything else (the env, the state abstractor's structure, the other operators,
the goal) is provided.
"""

# pylint: disable=fixme  # this file intentionally contains TODO markers
import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from kinder.envs.kinematic2d.obstruction2d import (
    ObjectCentricObstruction2DEnv,
    TargetBlockType,
    TargetSurfaceType,
)
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    is_on,
)
from numpy.typing import NDArray
from part1_stacking.skills import create_stacking_controllers
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


def create_stacking_models(
    observation_space: Space,
    action_space: Space,
    num_obstructions: int,
    init_constant_state: ObjectCentricState | None = None,
) -> SesameModels:
    """Create the planning models for the stacking task."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricObstruction2DEnv(num_obstructions=num_obstructions)

    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        return observation_space.devectorize(o)

    def transition_fn(
        x: ObjectCentricState, u: NDArray[np.float32]
    ) -> ObjectCentricState:
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {CRVRobotType, RectangleType, TargetBlockType, TargetSurfaceType}
    state_space = ObjectCentricStateSpace(types)

    # Predicates. ``On`` is the new one for this part.
    Holding = Predicate("Holding", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTable = Predicate("OnTable", [RectangleType])
    OnTarget = Predicate("OnTarget", [RectangleType])
    On = Predicate("On", [RectangleType, RectangleType])
    predicates = {Holding, HandEmpty, OnTable, OnTarget, On}

    def find_support(
        x: ObjectCentricState, block: Object, candidates: set[Object]
    ) -> Object | None:
        """Return the block in ``candidates`` that ``block`` is resting on, else None.

        ``is_on(x, a, b, {})`` is True exactly when block ``a`` is resting on top of
        block ``b``. ``block`` is never resting on itself.
        """
        # TODO(1): look through ``candidates`` and return the one that ``block`` is
        # resting on (use ``is_on``), or None if it is on none of them.
        raise NotImplementedError("TODO(1): On predicate -- find the support block")

    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        robot = Object("robot", CRVRobotType)
        target = Object("target_block", TargetBlockType)
        target_surface = Object("target_surface", TargetSurfaceType)
        obstructions = {
            Object(f"obstruction{i}", RectangleType) for i in range(num_obstructions)
        }
        blocks = obstructions | {target}
        atoms: set[GroundAtom] = set()
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        for obj in suctioned_objs & blocks:
            atoms.add(GroundAtom(Holding, [robot, obj]))
        if not suctioned_objs:
            atoms.add(GroundAtom(HandEmpty, [robot]))
        # Each non-held block is OnTarget, On(another block), or OnTable.
        for block in blocks:
            if block in suctioned_objs:
                continue
            if is_on(x, block, target_surface, {}):
                atoms.add(GroundAtom(OnTarget, [block]))
                continue
            support = find_support(x, block, blocks - {block})
            if support is not None:
                atoms.add(GroundAtom(On, [block, support]))
            else:
                atoms.add(GroundAtom(OnTable, [block]))
        objects = {robot, target, target_surface} | obstructions
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal: stack obstruction0 on top of the target block."""
        del x
        obstruction = Object("obstruction0", RectangleType)
        target = Object("target_block", TargetBlockType)
        return RelationalAbstractGoal(
            {GroundAtom(On, [obstruction, target])}, state_abstractor
        )

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    support = Variable("?support", RectangleType)
    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
    )

    # TODO(2): fill in the Stack operator's preconditions and effects (the three
    # empty sets below). Stacking requires holding the block and the support being
    # on the table; afterward the block is On the support and the hand is empty (no
    # longer holding). Build LiftedAtoms from the predicates above, e.g.
    # ``LiftedAtom(Holding, [robot, block])``.
    StackOperator = LiftedOperator(
        "Stack",
        [robot, block, support],
        preconditions=set(),  # TODO(2)
        add_effects=set(),  # TODO(2)
        delete_effects=set(),  # TODO(2)
    )

    # Skills (provided controllers paired with the operators above).
    controllers = create_stacking_controllers(action_space, init_constant_state)
    skills = {
        LiftedSkill(PickFromTableOperator, controllers["pick"]),
        LiftedSkill(StackOperator, controllers["place_on_block"]),
    }

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
