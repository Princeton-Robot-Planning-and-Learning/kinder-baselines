"""Bilevel planning models for the cylinder shelf 3D environment."""

from collections.abc import Collection, Sequence

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CylinderShelf3DEnvConfig,
    CylinderShelf3DObjectCentricState,
    Kinematic3DRobotType,
    ObjectCentricCylinderShelf3DEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from kinder_models.kinematic3d.cylinder_shelf3d.parameterized_skills import (
    create_lifted_controllers,
    is_at_pre_grasp,
)
from kinder_models.magic import make_magic_lifted_controller
from kinder_models.structs import SkillCall
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

# Operator name -> key in kinder_models' create_lifted_controllers.
_SKILL_CONTROLLER_KEYS = {
    "MoveToPreGrasp": "move_to_pre_grasp",
    "Grasp": "grasp",
    "Place": "place",
}


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
    config: CylinderShelf3DEnvConfig | None = None,
    magic_skills: Collection[str] = (),
    place_params: Sequence[Sequence[float] | None] | None = None,
) -> SesameModels:
    """Create the env models for cylinder shelf 3D.

    ``config`` overrides the env config of the internal sim. Defaults to
    ``CylinderShelf3DEnvConfig()`` if not provided. Pass a custom config to
    plan against e.g. a non-default shelf pose; the planner's transition
    function and state abstractor both see this config via the internal sim,
    so abstract-state checks (OnFixture) are computed against the configured
    shelf rather than the dataclass default.

    ``magic_skills`` names operators (any of "MoveToPreGrasp", "Grasp",
    "Place") whose low-level policy is not simulated during planning. Each
    becomes a one-step skill emitting a ``SkillCall`` whose predicted state
    comes from the controller's own outcome model, and the transition
    function maps that call straight to the predicted state. Plans then
    contain a ``SkillCall`` wherever the executor must carry the skill out
    by other means. Only skills whose controller implements
    ``OutcomePredictor`` can be made magic.

    ``place_params`` fixes where each cylinder is set down on the shelf: the
    i-th entry is for ``cylinder{i}`` and is either ``(x, y)``, the offset
    from the shelf centre (the base staging distance is still sampled), or
    ``(x, y, base_distance)``, which leaves the Place skill nothing to sample
    (see ``GroundPlaceController``); ``None`` keeps sampling everything for
    that cylinder. Fixed placements make planning with several cylinders
    faster and their layout on the shelf predictable.
    """
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)
    unknown_magic = set(magic_skills) - set(_SKILL_CONTROLLER_KEYS)
    if unknown_magic:
        raise ValueError(
            f"Unknown magic skill(s) {sorted(unknown_magic)}; expected a subset "
            f"of {sorted(_SKILL_CONTROLLER_KEYS)}"
        )

    if config is None:
        config = CylinderShelf3DEnvConfig()
    sim = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=num_objects, config=config, allow_state_access=True
    )

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32] | SkillCall[ObjectCentricState],
    ) -> ObjectCentricState:
        """Simulate the action, or jump to a SkillCall's predicted state."""
        if isinstance(u, SkillCall):
            return u.predicted_state.copy()
        state = x.copy()
        assert isinstance(state, CylinderShelf3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {Kinematic3DCuboidType, Kinematic3DFixtureType, Kinematic3DRobotType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    OnFixture = Predicate("OnFixture", [Kinematic3DCuboidType, Kinematic3DFixtureType])
    OnGround = Predicate("OnGround", [Kinematic3DCuboidType])
    Holding = Predicate("Holding", [Kinematic3DRobotType, Kinematic3DCuboidType])
    HandEmpty = Predicate("HandEmpty", [Kinematic3DRobotType])
    AtPreGrasp = Predicate("AtPreGrasp", [Kinematic3DRobotType, Kinematic3DCuboidType])
    # True when the robot is at no cylinder's pre-grasp pose. MoveToPreGrasp
    # requires and deletes it, so the abstract planner cannot chain two
    # MoveToPreGrasps (which would leave a stale AtPreGrasp in the abstract
    # state that no sampled trajectory can reproduce).
    NotAtPreGrasp = Predicate("NotAtPreGrasp", [Kinematic3DRobotType])
    predicates = {OnFixture, OnGround, Holding, HandEmpty, AtPreGrasp, NotAtPreGrasp}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_fixtures = x.get_objects(Kinematic3DFixtureType)

        atoms: set[GroundAtom] = set()

        assert isinstance(x, CylinderShelf3DObjectCentricState)
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

        # AtPreGrasp: empty gripper at the target's pre-grasp position.
        at_any_pre_grasp = False
        for target in target_objects:
            if is_at_pre_grasp(sim, x, target.name):
                atoms.add(GroundAtom(AtPreGrasp, [robot, target]))
                at_any_pre_grasp = True
        if not at_any_pre_grasp:
            atoms.add(GroundAtom(NotAtPreGrasp, [robot]))

        # OnFixture: within the shelf footprint and resting above floor
        # level (a floor-standing cylinder has pose_z == half_extent_z; one
        # standing on any shelf board sits at least a board thickness
        # higher).
        for target in target_objects:
            for fixture in target_fixtures:
                if (
                    np.isclose(
                        x.get(target, "pose_x") - x.get(fixture, "pose_x"),
                        0.0,
                        atol=0.15,
                    )
                    and np.isclose(
                        x.get(target, "pose_y") - x.get(fixture, "pose_y"),
                        0.0,
                        atol=0.25,
                    )
                    and x.get(target, "pose_z") > x.get(target, "half_extent_z") + 0.015
                ):
                    atoms.add(GroundAtom(OnFixture, [target, fixture]))

        objects = {robot} | set(target_objects) | set(target_fixtures)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have every cylinder on the shelf with the hand empty."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_shelf = x.get_objects(Kinematic3DFixtureType)[0]
        atoms: set[GroundAtom] = set()
        for target in target_objects:
            atoms.add(GroundAtom(OnFixture, [target, target_shelf]))
        atoms.add(GroundAtom(HandEmpty, [robot]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    MoveToPreGraspOperator = LiftedOperator(
        "MoveToPreGrasp",
        [robot, target],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(NotAtPreGrasp, [robot]),
            LiftedAtom(OnGround, [target]),
        },
        add_effects={LiftedAtom(AtPreGrasp, [robot, target])},
        delete_effects={LiftedAtom(NotAtPreGrasp, [robot])},
    )

    GraspOperator = LiftedOperator(
        "Grasp",
        [robot, target],
        preconditions={
            LiftedAtom(AtPreGrasp, [robot, target]),
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [target]),
        },
        add_effects={
            LiftedAtom(Holding, [robot, target]),
            LiftedAtom(NotAtPreGrasp, [robot]),
        },
        delete_effects={
            LiftedAtom(AtPreGrasp, [robot, target]),
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [target]),
        },
    )

    # Get lifted controllers from kinder_models, swapping in magic versions
    # where requested.
    if place_params is not None and len(place_params) != num_objects:
        raise ValueError(
            f"place_params has {len(place_params)} entries for {num_objects} "
            "cylinders"
        )
    fixed_place_params = {
        f"cylinder{i}": params
        for i, params in enumerate(place_params or [])
        if params is not None
    }
    lifted_controllers = create_lifted_controllers(
        action_space, sim, fixed_place_params
    )
    for skill_name in magic_skills:
        key = _SKILL_CONTROLLER_KEYS[skill_name]
        lifted_controllers[key] = make_magic_lifted_controller(
            lifted_controllers[key], skill_name
        )
    MoveToPreGraspController = lifted_controllers["move_to_pre_grasp"]
    GraspController = lifted_controllers["grasp"]

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    target_shelf = Variable("?target_shelf", Kinematic3DFixtureType)

    PlaceOperator = LiftedOperator(
        "Place",
        [robot, target, target_shelf],
        preconditions={LiftedAtom(Holding, [robot, target])},
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnFixture, [target, target_shelf]),
        },
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )

    PlaceController = lifted_controllers["place"]

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToPreGraspOperator, MoveToPreGraspController),
        LiftedSkill(GraspOperator, GraspController),
        LiftedSkill(PlaceOperator, PlaceController),
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
