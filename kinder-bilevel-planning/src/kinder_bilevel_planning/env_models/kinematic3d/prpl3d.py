"""Bilevel planning models for the PrplLab3D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
)
from kinder.envs.kinematic3d.prpl3d import (
    Kinematic3DRobotType,
    ObjectCentricPrplLab3DEnv,
    PrplLab3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from kinder_models.kinematic3d.prpl3d.parameterized_skills import (
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
ON_GROUND_TOL = 0.01


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
) -> SesameModels:
    """Create the env models for PrplLab3D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)

    sim = ObjectCentricPrplLab3DEnv(num_cubes=num_objects, allow_state_access=True)

    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        return observation_space.devectorize(o)

    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        state = x.copy()
        assert isinstance(state, PrplLab3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {Kinematic3DCuboidType, Kinematic3DFixtureType, Kinematic3DRobotType}
    state_space = ObjectCentricStateSpace(types)

    OnFixture = Predicate("OnFixture", [Kinematic3DCuboidType, Kinematic3DFixtureType])
    OnGround = Predicate("OnGround", [Kinematic3DCuboidType])
    Holding = Predicate("Holding", [Kinematic3DRobotType, Kinematic3DCuboidType])
    HandEmpty = Predicate("HandEmpty", [Kinematic3DRobotType])
    predicates = {OnFixture, OnGround, Holding, HandEmpty}

    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_fixtures = x.get_objects(Kinematic3DFixtureType)

        assert isinstance(x, PrplLab3DObjectCentricState)
        sim.set_state(x)

        atoms: set[GroundAtom] = set()

        # OnGround: cube resting on the floor.
        for target in target_objects:
            z = x.get(target, "pose_z")
            bb_z = x.get(target, "half_extent_z")
            if np.isclose(z, bb_z, atol=ON_GROUND_TOL):
                atoms.add(GroundAtom(OnGround, [target]))

        # HandEmpty: gripper is open and not holding anything.
        if x.grasped_object is None:
            if x.get(robot, "finger_state") < GRIPPER_OPEN_THRESHOLD:
                atoms.add(GroundAtom(HandEmpty, [robot]))

        # Holding: cube is actively grasped.
        for target in target_objects:
            if (
                x.get(target, "pose_z") > 0.3
                and x.get(robot, "finger_state") > GRIPPER_OPEN_THRESHOLD
                and target.name == x.grasped_object
            ):
                atoms.add(GroundAtom(Holding, [robot, target]))

        # OnFixture: cube is on the countertop (z > 0.5, not grasped).
        for target in target_objects:
            for fixture in target_fixtures:
                if target.name != x.grasped_object and x.get(target, "pose_z") > 0.5:
                    atoms.add(GroundAtom(OnFixture, [target, fixture]))

        objects = {robot} | set(target_objects) | set(target_fixtures)
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_fixture = x.get_objects(Kinematic3DFixtureType)[0]
        atoms: set[GroundAtom] = set()
        for target in target_objects:
            atoms.add(GroundAtom(OnFixture, [target, target_fixture]))
        atoms.add(GroundAtom(HandEmpty, [robot]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    PickOperator = LiftedOperator(
        "Pick",
        [robot, target],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnGround, [target])},
        add_effects={LiftedAtom(Holding, [robot, target])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnGround, [target])},
    )

    lifted_controllers = create_lifted_controllers(action_space, sim)
    PickController = lifted_controllers["pick"]

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    target_fixture = Variable("?target_fixture", Kinematic3DFixtureType)

    PlaceOperator = LiftedOperator(
        "Place",
        [robot, target, target_fixture],
        preconditions={LiftedAtom(Holding, [robot, target])},
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnFixture, [target, target_fixture]),
        },
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )

    PlaceController = lifted_controllers["place"]

    skills = {
        LiftedSkill(PickOperator, PickController),
        LiftedSkill(PlaceOperator, PlaceController),
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
