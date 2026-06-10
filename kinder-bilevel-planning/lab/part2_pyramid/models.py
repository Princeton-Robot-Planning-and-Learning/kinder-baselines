"""Part 2 domain -- build a pyramid: the target block on top of TWO obstructions.

**No AI assistants for this part.**

Unlike Part 1, you design the whole thing: the predicate(s), operator(s), and
skill(s) the task needs, plus the goal. Use your Part 1 code as a pattern -- the
machinery (env, abstraction structure, the motion planner, pick) is the same.

The catch: in ``run.py`` the two obstructions start **apart**. So think about the
*order of operations* -- what has to be true about the obstructions before the
target can sit on both, and what operator could make that true?

Holes are marked TODO(A)..TODO(D). The provided ``PickFromTable`` operator shows
the operator+skill pattern you'll copy.
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
from kinder.envs.kinematic2d.utils import (  # is_on / rectangle geom helpers live here
    CRVRobotActionSpace,
    get_suctioned_objects,
    is_on,
)
from numpy.typing import NDArray
from part2_pyramid.skills import create_pyramid_controllers
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


def create_pyramid_models(
    observation_space: Space,
    action_space: Space,
    num_obstructions: int,
    init_constant_state: ObjectCentricState | None = None,
) -> SesameModels:
    """Create the planning models for the pyramid task."""
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

    # Base predicates (provided).
    Holding = Predicate("Holding", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTable = Predicate("OnTable", [RectangleType])
    OnTarget = Predicate("OnTarget", [RectangleType])
    # TODO(A): define the predicate(s) the pyramid needs and add them to this set.
    predicates = {Holding, HandEmpty, OnTable, OnTarget}

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
        for block in blocks:
            if block in suctioned_objs:
                continue
            if is_on(x, block, target_surface, {}):
                atoms.add(GroundAtom(OnTarget, [block]))
                continue
            # TODO(B): emit your predicate atoms for `block` here (its relationship
            # to the OTHER blocks). `is_on(x, a, b, {})` is True when a rests on b;
            # `rectangle_object_to_geom(x, o, {})` (from kinder...utils) gives a
            # geom with `.vertices` and `.contains_point(px, py)` for finer checks.
            atoms.add(GroundAtom(OnTable, [block]))
        objects = {robot, target, target_surface} | obstructions
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        # TODO(D): return the goal that represents the finished pyramid (the target
        # block supported by both obstructions). Use a predicate you defined.
        del x
        raise NotImplementedError("TODO(D): the pyramid goal")

    # Operators. PickFromTable is provided as the pattern to copy.
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
    )
    # TODO(C): define the operator(s) the pyramid needs (each is a LiftedOperator
    # with preconditions / add_effects / delete_effects), and pair each with a
    # skill in the `skills` set below.

    controllers = create_pyramid_controllers(action_space, init_constant_state)
    skills = {
        LiftedSkill(PickFromTableOperator, controllers["pick"]),
        # TODO(C): add LiftedSkill(YourOperator, controllers["your_skill"]) entries.
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
