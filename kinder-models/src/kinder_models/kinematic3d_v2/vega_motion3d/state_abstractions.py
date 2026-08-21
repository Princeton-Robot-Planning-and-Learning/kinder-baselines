"""State abstractions for the VegaMotion3D environment."""

from typing import Callable

from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from kinder.envs.kinematic3d_v2.object_types import (
    Kinematic3Dv2ArmRobotType,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_motion3d import (
    ObjectCentricVegaMotion3DEnv,
    VegaMotion3DObjectCentricState,
)
from relational_structs import (
    GroundAtom,
    Object,
    ObjectCentricState,
    Predicate,
)

# Predicates.
AtTarget = Predicate("AtTarget", [Kinematic3Dv2ArmRobotType, Kinematic3Dv2PointType])

StateAbstractor = Callable[[ObjectCentricState], RelationalAbstractState]
GoalDeriver = Callable[[ObjectCentricState], RelationalAbstractGoal]


def _get_robot_and_target(state: ObjectCentricState) -> tuple[Object, Object]:
    """The unique robot and target in the state."""
    robots = state.get_objects(Kinematic3Dv2ArmRobotType)
    targets = state.get_objects(Kinematic3Dv2PointType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    assert len(targets) == 1, f"Expected 1 target, got {len(targets)}"
    return robots[0], targets[0]


def create_state_abstractor(sim: ObjectCentricVegaMotion3DEnv) -> StateAbstractor:
    """Create a state abstractor backed by ``sim``.

    Whether the end effector has reached the target is a function of forward kinematics,
    not of the state features alone, so this defers to the environment's own goal check
    rather than recomputing the threshold and risking drift from it.
    """

    def state_abstractor(state: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot, target = _get_robot_and_target(state)
        atoms: set[GroundAtom] = set()
        assert isinstance(state, VegaMotion3DObjectCentricState)
        sim.set_state(state)
        if sim.goal_reached():
            atoms.add(GroundAtom(AtTarget, [robot, target]))
        return RelationalAbstractState(atoms, {robot, target})

    return state_abstractor


def create_goal_deriver(state_abstractor: StateAbstractor) -> GoalDeriver:
    """Create a goal deriver that pairs with ``state_abstractor``."""

    def goal_deriver(state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the end effector at the target region."""
        robot, target = _get_robot_and_target(state)
        atoms = {GroundAtom(AtTarget, [robot, target])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    return goal_deriver
