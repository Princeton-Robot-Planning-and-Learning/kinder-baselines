"""Bilevel planning models for the dynamic push-pull-hook 2D environment."""

from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Box
from gymnasium.spaces import Space
import numpy as np
from kinder.envs.dynamic2d.dyn_pushpullhook2d import (
    DynPushPullHook2DEnvConfig,
    HookType,
    ObjectCentricDynPushPullHook2DEnv,
    TargetBlockType,
)
from kinder.envs.dynamic2d.object_types import DynRectangleType, KinRobotType, LObjectType
from kinder.envs.dynamic2d.utils import KinRobotActionSpace
from kinder.envs.utils import object_to_multibody2d
from kinder_models.dynamic2d.dynpushpullhook2d.parameterized_skills import (
    GroundHookDownController,
    create_lifted_controllers,
)
from numpy.typing import NDArray
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
from typing import Sequence


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for dynamic push-pull-hook 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, KinRobotActionSpace)

    env_config = DynPushPullHook2DEnvConfig()
    sim = ObjectCentricDynPushPullHook2DEnv(num_obstructions=num_obstructions)

    # Convert observations into states.
    def observation_to_state(o: NDArray) -> ObjectCentricState:
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray,
    ) -> ObjectCentricState:
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {KinRobotType, LObjectType, HookType, TargetBlockType, DynRectangleType}

    # State space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    HandEmpty = Predicate("HandEmpty", [KinRobotType])
    HoldingHook = Predicate("HoldingHook", [KinRobotType, HookType])
    TargetAtGoal = Predicate("TargetAtGoal", [TargetBlockType])
    predicates = {HandEmpty, HoldingHook, TargetAtGoal}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        robot = x.get_objects(KinRobotType)[0]
        hooks = x.get_objects(HookType)
        target_blocks = x.get_objects(TargetBlockType)

        atoms: set[GroundAtom] = set()

        # Holding / HandEmpty.
        holding_hook = False
        for hook in hooks:
            if x.get(hook, "held"):
                atoms.add(GroundAtom(HoldingHook, [robot, hook]))
                holding_hook = True
        if not holding_hook:
            atoms.add(GroundAtom(HandEmpty, [robot]))

        # TargetAtGoal: target block intersects the middle wall.
        middle_wall = [
            o for o in sim.initial_constant_state if o.name == "middle_wall"
        ][0]
        full_state = x.copy()
        full_state.data.update(sim.initial_constant_state.data)
        static_cache: dict[Object, object] = {}
        for tgt in target_blocks:
            tgt_body = object_to_multibody2d(tgt, full_state, static_cache)
            wall_body = object_to_multibody2d(middle_wall, full_state, static_cache)
            if tgt_body.bodies[0].geom.intersects(wall_body.bodies[0].geom):
                atoms.add(GroundAtom(TargetAtGoal, [tgt]))

        objects = {robot} | set(hooks) | set(target_blocks)
        # Include obstructions.
        for obj in x.get_objects(DynRectangleType):
            objects.add(obj)
        return RelationalAbstractState(atoms, objects)

    # Goal deriver.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        target_block = x.get_objects(TargetBlockType)[0]
        atoms = {GroundAtom(TargetAtGoal, [target_block])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Variables (names must match lifted controller variables).
    robot = Variable("?robot", KinRobotType)
    hook = Variable("?hook", HookType)
    target_block = Variable("?target_block", TargetBlockType)

    # Operators.
    GraspHookOperator = LiftedOperator(
        "GraspHook",
        [robot, hook],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingHook, [robot, hook])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PreHookOperator = LiftedOperator(
        "PreHook",
        [robot, hook, target_block],
        preconditions={LiftedAtom(HoldingHook, [robot, hook])},
        add_effects=set(),
        delete_effects=set(),
    )

    HookDownOperator = LiftedOperator(
        "HookDown",
        [robot, hook, target_block],
        preconditions={LiftedAtom(HoldingHook, [robot, hook])},
        add_effects={LiftedAtom(TargetAtGoal, [target_block])},
        delete_effects=set(),
    )

    MoveOperator = LiftedOperator(
        "Move",
        [robot],
        preconditions=set(),
        add_effects=set(),
        delete_effects=set(),
    )

    # Get lifted controllers from kinder_models.
    lifted_controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state
    )
    GraspHookController = lifted_controllers["grasp_hook"]
    PreHookController = lifted_controllers["prehook"]
    MoveController = lifted_controllers["move"]

    # HookDown controller needs [robot, hook, target_block] to match the
    # operator, but the ground controller only uses the robot.  Create a
    # lifted wrapper with the wider variable list.
    class _HookDownControllerWrapper(GroundHookDownController):
        def __init__(self, objects: Sequence[Object]) -> None:
            # Pass all objects so self.objects matches the operator parameters.
            # Only objects[0] (the robot) is used by the controller.
            super().__init__(objects, action_space, sim.initial_constant_state)

    HookDownController = LiftedParameterizedController(
        [robot, hook, target_block],
        _HookDownControllerWrapper,
        Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32),
    )

    # Skills.
    skills = {
        LiftedSkill(GraspHookOperator, GraspHookController),
        LiftedSkill(PreHookOperator, PreHookController),
        LiftedSkill(HookDownOperator, HookDownController),
        LiftedSkill(MoveOperator, MoveController),
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
