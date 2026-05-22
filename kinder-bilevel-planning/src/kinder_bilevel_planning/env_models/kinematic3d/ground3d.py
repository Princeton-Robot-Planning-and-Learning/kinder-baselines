"""Bilevel planning models for the Ground3D environment.

Simplification of shelf3d.py for the "pick a cube off the floor" task:

* No `Kinematic3DFixtureType` — Ground3D has no shelf to place on.
* Predicates: `OnGround`, `Holding`, `HandEmpty` (drop `OnFixture`).
* Single `Pick` operator (drop `Place`, even though
  `kinder_models.kinematic3d.ground3d.parameterized_skills` exposes a
  place controller — Ground3D's goal is just to grasp).
* Goal: `Holding(robot, cube0)` — the planner picks the first declared
  cube. For Ground3D-o1 there's only one cube so this is unambiguous;
  for Ground3D-o{>1} any obstructing cubes still surface in the state
  abstractor but only `cube0` is the goal target.
"""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.ground3d import (
    Ground3DObjectCentricState,
    Kinematic3DRobotType,
    ObjectCentricGround3DEnv,
)
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from kinder_models.kinematic3d.ground3d.parameterized_skills import (
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

GRIPPER_OPEN_THRESHOLD = 0.01


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
) -> SesameModels:
    """Create the env models for Ground3D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)

    sim = ObjectCentricGround3DEnv(num_cubes=num_objects, allow_state_access=True)

    # Convert observations into states. The important thing is that states are hashable.
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
        assert isinstance(state, Ground3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {Kinematic3DCuboidType, Kinematic3DRobotType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    OnGround = Predicate("OnGround", [Kinematic3DCuboidType])
    Holding = Predicate("Holding", [Kinematic3DRobotType, Kinematic3DCuboidType])
    HandEmpty = Predicate("HandEmpty", [Kinematic3DRobotType])
    predicates = {OnGround, Holding, HandEmpty}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)

        atoms: set[GroundAtom] = set()

        assert isinstance(x, Ground3DObjectCentricState)
        sim.set_state(x)

        # OnGround.
        on_ground_tol = 0.01
        for target in target_objects:
            z = x.get(target, "pose_z")
            bb_z = x.get(target, "half_extent_z")
            if np.isclose(z, bb_z, atol=on_ground_tol):
                atoms.add(GroundAtom(OnGround, [target]))

        # HandEmpty.
        if x.grasped_object is None:
            if x.get(robot, "finger_state") < GRIPPER_OPEN_THRESHOLD:
                atoms.add(GroundAtom(HandEmpty, [robot]))

        # Holding.
        for target in target_objects:
            if (
                x.get(target, "pose_z") > 0.3
                and x.get(robot, "finger_state") > GRIPPER_OPEN_THRESHOLD
            ):
                if target.name == x.grasped_object:
                    atoms.add(GroundAtom(Holding, [robot, target]))

        objects = {robot} | set(target_objects)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to hold the first cube.

        Any obstructing cubes (Ground3D-o{>1}) appear in the state abstractor but are
        not themselves goal targets — the planner picks them up only if it needs to
        clear them out of the way of cube0.
        """
        target_objects = x.get_objects(Kinematic3DCuboidType)
        robot = x.get_objects(Kinematic3DRobotType)[0]
        atoms: set[GroundAtom] = {
            GroundAtom(Holding, [robot, target_objects[0]]),
        }
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot_var = Variable("?robot", Kinematic3DRobotType)
    target_var = Variable("?target", Kinematic3DCuboidType)

    PickOperator = LiftedOperator(
        "Pick",
        [robot_var, target_var],
        preconditions={
            LiftedAtom(HandEmpty, [robot_var]),
            LiftedAtom(OnGround, [target_var]),
        },
        add_effects={LiftedAtom(Holding, [robot_var, target_var])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot_var]),
            LiftedAtom(OnGround, [target_var]),
        },
    )

    # Get lifted controllers from kinder_models. The ground3d skills package
    # also exposes "place" but Ground3D doesn't need it — the goal is grasping.
    lifted_controllers = create_lifted_controllers(action_space, sim)
    PickController = lifted_controllers["pick"]

    skills = {
        LiftedSkill(PickOperator, PickController),
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
