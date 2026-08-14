"""State abstractions for the VegaPickPlace3D environment."""

from typing import Callable

from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from kinder.envs.kinematic3d_v2.object_types import (
    Kinematic3Dv2GraspArmRobotType,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_pickplace3d import (
    ARM_SIDES,
    CUBE_NODE,
    TARGET_NODE,
    ObjectCentricVegaPickPlace3DEnv,
    VegaPickPlace3DObjectCentricState,
)
from relational_structs import (
    GroundAtom,
    Object,
    ObjectCentricState,
    Predicate,
)

# Predicates. Holding and HandEmpty partition the arms in every state; NotHeld marks
# the cube while neither arm holds it. On holds when the environment's own goal is
# reached, i.e., the cube rests on the table inside the target patch.
Holding = Predicate("Holding", [Kinematic3Dv2GraspArmRobotType, Kinematic3Dv2PointType])
HandEmpty = Predicate("HandEmpty", [Kinematic3Dv2GraspArmRobotType])
NotHeld = Predicate("NotHeld", [Kinematic3Dv2PointType])
On = Predicate("On", [Kinematic3Dv2PointType, Kinematic3Dv2PointType])

PREDICATES = {Holding, HandEmpty, NotHeld, On}

StateAbstractor = Callable[[ObjectCentricState], RelationalAbstractState]
GoalDeriver = Callable[[ObjectCentricState], RelationalAbstractGoal]


def get_scene_objects(
    state: ObjectCentricState,
) -> tuple[dict[str, Object], Object, Object]:
    """The arm objects (keyed by side), the cube, and the target in the state."""
    arms = {side: state.get_object_from_name(f"{side}_arm") for side in ARM_SIDES}
    cube = state.get_object_from_name(CUBE_NODE)
    target = state.get_object_from_name(TARGET_NODE)
    return arms, cube, target


def create_state_abstractor(
    sim: ObjectCentricVegaPickPlace3DEnv,
) -> StateAbstractor:
    """Create a state abstractor backed by ``sim``.

    Whether the cube is on the target is a function of forward kinematics, not of the
    state features alone, so this defers to the environment's own goal check rather than
    recomputing the containment test and risking drift from it.
    """

    def state_abstractor(state: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        assert isinstance(state, VegaPickPlace3DObjectCentricState)
        arms, cube, target = get_scene_objects(state)
        atoms: set[GroundAtom] = set()
        for side in ARM_SIDES:
            if state.grasping(side):
                atoms.add(GroundAtom(Holding, [arms[side], cube]))
            else:
                atoms.add(GroundAtom(HandEmpty, [arms[side]]))
        if state.holder is None:
            atoms.add(GroundAtom(NotHeld, [cube]))
        sim.set_state(state)
        if sim.goal_reached():
            atoms.add(GroundAtom(On, [cube, target]))
        objects = set(arms.values()) | {cube, target}
        return RelationalAbstractState(atoms, objects)

    return state_abstractor


def create_goal_deriver(state_abstractor: StateAbstractor) -> GoalDeriver:
    """Create a goal deriver that pairs with ``state_abstractor``."""

    def goal_deriver(state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the cube resting on the target patch."""
        _, cube, target = get_scene_objects(state)
        atoms = {GroundAtom(On, [cube, target])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    return goal_deriver
