"""Bilevel planning models for the TidyBot3D Tossing3D environment.

Three things here are unlike the other dynamic3d env models, and each is deliberate.

1. The controllers are *padded* with ignored variables. LiftedSkill requires an
   operator's parameters to equal its controller's variables, and these operators are
   wider than the controllers they drive: Pick needs ?barrier to state Reachable and
   ?bin to delete NearBin, while pick_shelf only drives ?robot and ?target. Padding
   keeps the symbolic model expressive; weakening the operators instead would gut the
   Toss add effect, which is the thing the planner plans toward. Upstream's own
   move_to_target_from_other_target already carries a ?prev_target in that spirit.

2. Ground operators are supplied by name. All three scene objects -- bin_0, cube_0 and
   cuboid_barrier -- are typed MujocoMovableObjectType, and there are no fixtures in
   the state, so lifted-operator typing cannot tell a bin from a barrier. sweep3D has
   the same problem with its wiper and solves it the same way.

3. Two controller factories are called, shelf's for pick_shelf and tossing's for the
   rest, exactly as the monorepo's own test_pick_ground_toss does. Both are handed the
   same PyBulletSim: RelationalControllerGenerator grounds a fresh controller per
   sampling attempt, so an unshared sim would connect a PyBullet client and reload the
   robot on every attempt. Since #87 that client is disconnected on collection rather
   than leaked -- so this is a per-attempt cost, not a leak, but the cost is real.
"""

from pathlib import Path
from typing import Sequence

import kinder
import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder_models.dynamic3d.shelf import parameterized_skills as shelf_skills
from kinder_models.dynamic3d.tossing import parameterized_skills as tossing_skills
from kinder_models.dynamic3d.tossing.state_abstractions import (
    BARRIER_NAME,
    HandEmpty,
    Holding,
    InGoalRegion,
    NearBin,
    OnGround,
    Reachable,
    Tossing3DStateAbstractor,
)
from kinder_models.dynamic3d.utils import PyBulletSim
from numpy.typing import NDArray
from relational_structs import (
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

# The names the scene uses. All three are MujocoMovableObjectType, so these names are
# the only thing that distinguishes them -- see the module docstring. The barrier's is
# imported rather than repeated: it is the same literal the state abstractor picks the
# barrier out by, and Reachable stops being emitted if the two drift apart.
ROBOT_NAME = "robot"
CUBE_NAME = "cube_0"
BIN_NAME = "bin_0"


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
) -> SesameModels:
    """Create the env models for TidyBot Tossing3D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    task_config_path = str(
        Path(kinder.__file__).parent
        / "envs/dynamic3d/tasks/Tossing3D"
        / f"Tossing3D-o{num_objects}.json"
    )
    sim = ObjectCentricTidyBot3DEnv(
        task_config_path=task_config_path,
        num_objects=num_objects,
        allow_state_access=True,
    )

    # State and goal abstractors.
    abstractor = Tossing3DStateAbstractor(sim)
    state_abstractor = abstractor.state_abstractor
    goal_deriver = abstractor.goal_deriver

    # Need to call reset to initialize the qpos, qvel.
    initial_state, _ = sim.reset()

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
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {
        MujocoTidyBotRobotObjectType,
        MujocoObjectType,
        MujocoFixtureObjectType,
        MujocoMovableObjectType,
    }

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    predicates = {
        Holding,
        HandEmpty,
        OnGround,
        InGoalRegion,
        Reachable,
        NearBin,
    }

    # Pick operator. ?barrier is what Reachable is stated against and ?bin is what
    # NearBin is deleted against; neither reaches pick_shelf, which drives only the
    # first two parameters.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    cube = Variable("?cube", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoMovableObjectType)
    bin_obj = Variable("?bin", MujocoMovableObjectType)

    PickOperator = LiftedOperator(
        "pick",
        [robot, cube, barrier, bin_obj],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [cube]),
            LiftedAtom(Reachable, [cube, barrier]),
        },
        add_effects={
            LiftedAtom(Holding, [robot, cube]),
        },
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [cube]),
            # Picking drives the base to a grasping standoff from the cube, which is
            # not a throw pose, so any previously established NearBin is destroyed.
            LiftedAtom(NearBin, [robot, bin_obj]),
        },
    )

    # Move-to-throw-pose operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    bin_obj = Variable("?bin", MujocoMovableObjectType)
    cube = Variable("?cube", MujocoMovableObjectType)

    MoveToThrowPoseOperator = LiftedOperator(
        "move_to_throw_pose",
        [robot, bin_obj, cube],
        preconditions={
            LiftedAtom(Holding, [robot, cube]),
        },
        add_effects={
            LiftedAtom(NearBin, [robot, bin_obj]),
        },
        delete_effects=set(),
    )

    # Toss operator. InGoalRegion is the add effect the whole plan exists to achieve.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    cube = Variable("?cube", MujocoMovableObjectType)
    bin_obj = Variable("?bin", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoMovableObjectType)

    TossOperator = LiftedOperator(
        "toss",
        [robot, cube, bin_obj, barrier],
        preconditions={
            LiftedAtom(Holding, [robot, cube]),
            LiftedAtom(NearBin, [robot, bin_obj]),
        },
        add_effects={
            LiftedAtom(InGoalRegion, [cube]),
            LiftedAtom(HandEmpty, [robot]),
            # OnGround is deliberately NOT added. It requires the cube to be flat
            # (|qx|, |qy| <= ON_GROUND_TOL) as well as resting, because the grasp the
            # pick controller builds is modelled on a flat cube -- and a thrown cube
            # tumbles, so it comes to rest at an arbitrary orientation. Measured: after
            # a refined toss the abstract state is {HandEmpty, InGoalRegion, NearBin}
            # with no OnGround. Claiming it here made every refinement of this skill
            # fail, since the trajectory sampler requires the achieved abstract state
            # to equal the predicted one exactly, not merely to contain the effects.
            #
            # That cuts both ways and is this model's one soft spot: a cube that
            # happens to settle flat -- which includes flat on the bin's interior
            # floor, where z - bb_z/2 = 0.02 is inside ON_GROUND_TOL -- puts OnGround
            # back and fails refinement, and the toss samples no parameters, so a retry
            # draws the identical trajectory. Making that impossible means changing
            # OnGround in state_abstractions, not this operator.
        },
        delete_effects={
            LiftedAtom(Holding, [robot, cube]),
            # A tossed cube lands past the barrier, which is what Reachable measures.
            LiftedAtom(Reachable, [cube, barrier]),
        },
    )

    # Create the PyBullet simulator, shared by both controller factories -- see the
    # module docstring for why sharing matters here specifically.
    assert initial_state is not None
    pybullet_sim = PyBulletSim(initial_state, rendering=False)
    shelf_controllers = shelf_skills.create_lifted_controllers(
        action_space, sim.initial_constant_state, pybullet_sim=pybullet_sim
    )
    tossing_controllers = tossing_skills.create_lifted_controllers(
        action_space, sim.initial_constant_state, pybullet_sim=pybullet_sim
    )

    # Controllers, padded out to their operators' parameters.
    LiftedPickController = _pad_controller(
        shelf_controllers["pick_shelf"], PickOperator.parameters
    )
    LiftedMoveToThrowPoseController = _pad_controller(
        tossing_controllers["move_to_throw_pose"], MoveToThrowPoseOperator.parameters
    )
    LiftedTossController = _pad_controller(
        tossing_controllers["toss_from_windup"], TossOperator.parameters
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickOperator, LiftedPickController),
        LiftedSkill(MoveToThrowPoseOperator, LiftedMoveToThrowPoseController),
        LiftedSkill(TossOperator, LiftedTossController),
    }

    # Pre-compute ground operators with the known object bindings, because typing
    # cannot distinguish the bin from the barrier.
    ground_operators = _create_ground_operators(
        initial_state, [PickOperator, MoveToThrowPoseOperator, TossOperator]
    )

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
        ground_operators=ground_operators,
    )


def _pad_controller(
    controller: LiftedParameterizedController,
    parameters: Sequence[Variable],
) -> LiftedParameterizedController:
    """Widen a lifted controller to an operator's parameters.

    The controller's leading variables are the ones that reach it; any trailing operator
    parameters are bound by the operator and dropped before the ground controller is
    constructed.

    The returned controller always declares the operator's *own* parameters, even when
    the arities already match, because LiftedSkill.__post_init__ asserts
    `tuple(operator.parameters) == tuple(controller.variables)` and Variables compare by
    name as well as type. move_to_throw_pose is exactly that case: 3 variables against 3
    parameters, but named (?robot, ?target, ?held) against (?robot, ?bin, ?cube).

    The types are deliberately not asserted equal either. move_to_throw_pose's ?target
    is MujocoObjectType while the operator's ?bin is MujocoMovableObjectType, which is
    the narrower of the two; LiftedParameterizedController.ground makes the real
    is_instance check at grounding time.
    """
    num_used = len(controller.variables)
    assert num_used <= len(parameters)
    # Padding is positional, so the leading parameters must be the controller's own
    # variables. ground() type-checks against the operator's types, not these, so a
    # reordered parameter list would otherwise bind the wrong objects in silence.
    assert all(
        v.type in (p.type, p.type.parent)
        for p, v in zip(parameters[:num_used], controller.variables)
    ), "Padded controller's leading parameters do not match its variables"

    controller_cls = controller.controller_cls

    class _PaddedController(controller_cls):  # type: ignore[valid-type, misc]
        """Drop the operator-only trailing objects before grounding."""

        def __init__(self, objects: Sequence[Object]) -> None:
            super().__init__(list(objects)[:num_used])

    # Otherwise all three padded controllers report as "_PaddedController", since
    # LiftedParameterizedController.name is the class's own __name__.
    _PaddedController.__name__ = f"Padded{controller_cls.__name__}"

    return LiftedParameterizedController(
        list(parameters),
        _PaddedController,
        params_space=controller.params_space,
    )


def _create_ground_operators(
    initial_state: ObjectCentricState,
    operators: list[LiftedOperator],
) -> set[GroundOperator]:
    """Ground operators using known object bindings for this environment."""
    name_to_obj: dict[str, Object] = {obj.name: obj for obj in initial_state}
    param_to_object_name = {
        "?robot": ROBOT_NAME,
        "?cube": CUBE_NAME,
        "?bin": BIN_NAME,
        "?barrier": BARRIER_NAME,
    }
    ground_ops: set[GroundOperator] = set()
    for operator in operators:
        objects = tuple(
            name_to_obj[param_to_object_name[param.name]]
            for param in operator.parameters
        )
        ground_ops.add(operator.ground(objects))
    return ground_ops
