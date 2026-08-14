"""Bilevel planning models for the TidyBot3D Tossing3D environment.

TODO: support Tossing3D-o2. It needs a goal naming each cube and operators that say
which cube a throw is aimed at; the state abstractor asserts one cube today.
"""

from collections.abc import Sequence
from pathlib import Path

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
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder_models.dynamic3d.shelf import parameterized_skills as shelf_skills
from kinder_models.dynamic3d.tossing import parameterized_skills as tossing_skills
from kinder_models.dynamic3d.tossing.state_abstractions import (
    BARRIER_NAME,
    BIN_NAME_PREFIX,
    CUBE_NAME_PREFIX,
    HandEmpty,
    Holding,
    MovableInGoalRegion,
    MovableIsDownX,
    OnGround,
    RobotAtThrowPose,
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


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
    task_config_path: str | None = None,
) -> SesameModels:
    """Create the env models for TidyBot Tossing3D.

    `task_config_path` defaults to the installed scene. Pass one to plan against a
    variant, which must be the same scene the caller's env was built from -- the model
    and the env disagreeing about geometry is silent, not an error.
    """
    if num_objects != 1:
        raise NotImplementedError(
            f"Tossing3D bilevel planning supports one cube, got {num_objects}. The "
            "operators name a single held cube, and no operator says which cube a "
            "throw is aimed at."
        )
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    if task_config_path is None:
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
        MujocoMovableObjectType,
    }

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    predicates = {
        MovableInGoalRegion,
        OnGround,
        Holding,
        HandEmpty,
        MovableIsDownX,
        RobotAtThrowPose,
    }

    # Pick cube operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    held = Variable("?held", MujocoMovableObjectType)

    PickCubeOperator = LiftedOperator(
        "pick_cube",
        [robot, held],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [held]),
        },
        add_effects={LiftedAtom(Holding, [robot, held])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [held]),
        },
    )

    # Move to throw pose operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    held = Variable("?held", MujocoMovableObjectType)

    MoveToThrowPoseOperator = LiftedOperator(
        "move_to_throw_pose",
        [robot, target, held],
        preconditions={LiftedAtom(Holding, [robot, held])},
        add_effects={LiftedAtom(RobotAtThrowPose, [robot, target])},
        delete_effects=set(),
    )

    # Toss operator. A toss is the environment's only irreversible action, so the
    # barrier is a parameter purely to express the side change it cannot undo.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    held = Variable("?held", MujocoMovableObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoMovableObjectType)

    TossFromWindupOperator = LiftedOperator(
        "toss_from_windup",
        [robot, held, target, barrier],
        preconditions={
            LiftedAtom(Holding, [robot, held]),
            LiftedAtom(RobotAtThrowPose, [robot, target]),
            LiftedAtom(MovableIsDownX, [held, barrier]),
        },
        add_effects={
            LiftedAtom(MovableInGoalRegion, [held]),
            LiftedAtom(HandEmpty, [robot]),
        },
        delete_effects={
            LiftedAtom(Holding, [robot, held]),
            LiftedAtom(MovableIsDownX, [held, barrier]),
        },
    )

    # One PyBullet simulator for every controller: a fresh controller is ground per
    # sampling attempt, and one client connect plus robot reload each time dominates.
    assert initial_state is not None
    pybullet_sim = PyBulletSim(initial_state, rendering=False)
    shelf_controllers = shelf_skills.create_lifted_controllers(
        action_space, sim.initial_constant_state, pybullet_sim=pybullet_sim
    )
    tossing_controllers = tossing_skills.create_lifted_controllers(
        action_space, sim.initial_constant_state, pybullet_sim=pybullet_sim
    )

    # Controllers. The pick is shelf's, which already grasps a cube off the ground.
    LiftedPickCubeController = _restate_controller_variables(
        shelf_controllers["pick_shelf"], PickCubeOperator.parameters
    )
    LiftedMoveToThrowPoseController = _restate_controller_variables(
        tossing_controllers["move_to_throw_pose"], MoveToThrowPoseOperator.parameters
    )
    LiftedTossFromWindupController = _restate_controller_variables(
        tossing_controllers["toss_from_windup"], TossFromWindupOperator.parameters
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickCubeOperator, LiftedPickCubeController),
        LiftedSkill(MoveToThrowPoseOperator, LiftedMoveToThrowPoseController),
        LiftedSkill(TossFromWindupOperator, LiftedTossFromWindupController),
    }

    # Every scene object is a movable, so lifted typing cannot tell the bin from the
    # barrier and grounding exhaustively offers the planner throws at the barrier.
    ground_operators = _create_ground_operators(
        initial_state,
        [PickCubeOperator, MoveToThrowPoseOperator, TossFromWindupOperator],
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


def _restate_controller_variables(
    controller: LiftedParameterizedController,
    variables: Sequence[Variable],
) -> LiftedParameterizedController:
    """Re-declare a controller's object signature to match an operator's parameters.

    LiftedSkill requires the two to be equal and Variable compares by name and type, but
    an operator's effects can only mention its own parameters: the toss changes atoms
    about the bin and the barrier, and move_to_throw_pose's target is typed too loosely
    for RobotAtThrowPose to accept. Every controller here reads its objects by leading
    index, so the extra trailing ones are passed through untouched rather than sliced
    off -- GroundSkill separately requires the controller to keep all of them.
    """
    return LiftedParameterizedController(
        variables, controller.controller_cls, params_space=controller.params_space
    )


def _create_ground_operators(
    initial_state: ObjectCentricState,
    operators: list[LiftedOperator],
) -> set[GroundOperator]:
    """Ground operators using known object bindings for this environment."""
    name_to_obj: dict[str, Object] = {obj.name: obj for obj in initial_state}
    robots = initial_state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected one robot, got {robots}"
    param_to_object = {
        "?robot": robots[0],
        "?held": name_to_obj[_get_unique_name(name_to_obj, CUBE_NAME_PREFIX)],
        "?target": name_to_obj[_get_unique_name(name_to_obj, BIN_NAME_PREFIX)],
        "?barrier": name_to_obj[BARRIER_NAME],
    }
    ground_ops: set[GroundOperator] = set()
    for operator in operators:
        objects = tuple(param_to_object[param.name] for param in operator.parameters)
        ground_ops.add(operator.ground(objects))
    return ground_ops


def _get_unique_name(name_to_obj: dict[str, Object], prefix: str) -> str:
    """Get the one object name starting with the given prefix."""
    matches = sorted(n for n in name_to_obj if n.startswith(prefix))
    assert len(matches) == 1, f"Expected one {prefix!r} object, got {matches}"
    return matches[0]
