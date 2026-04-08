"""Bilevel planning models for the TidyBot3D sweep3D environment."""

from pathlib import Path

import kinder
import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.dynamic3d.object_types import (
    MujocoDrawerObjectType,
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from kinder_models.dynamic3d.sweep3D.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.sweep3D.state_abstractions import (
    DrawerClosed,
    DrawerOpen,
    HandEmpty,
    Holding,
    InDrawer,
    OnTable,
    Sweep3DStateAbstractor,
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
) -> SesameModels:
    """Create the env models for TidyBot base motion."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    task_config_path = str(
        Path(kinder.__file__).parent
        / "envs/dynamic3d/tasks/SweepIntoDrawer3D"
        / f"SweepIntoDrawer3D-o{num_objects}.json"
    )
    sim = ObjectCentricTidyBot3DEnv(
        task_config_path=task_config_path,
        num_objects=num_objects,
        allow_state_access=True,
    )

    # State and goal abstractors.
    abstractor = Sweep3DStateAbstractor(sim)
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
        MujocoDrawerObjectType,
        MujocoFixtureObjectType,
        MujocoMovableObjectType,
    }

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    predicates = {
        Holding,
        HandEmpty,
        OnTable,
        InDrawer,
        DrawerOpen,
        DrawerClosed,
    }

    # Open drawer operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    drawer = Variable("?drawer", MujocoDrawerObjectType)
    cube0 = Variable("?cube0", MujocoMovableObjectType)
    cube1 = Variable("?cube1", MujocoMovableObjectType)
    cube2 = Variable("?cube2", MujocoMovableObjectType)
    cube3 = Variable("?cube3", MujocoMovableObjectType)
    cube4 = Variable("?cube4", MujocoMovableObjectType)

    OpenDrawerOperator = LiftedOperator(
        "open_drawer",
        [robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [wiper]),
            LiftedAtom(OnTable, [cube0]),
            LiftedAtom(OnTable, [cube1]),
            LiftedAtom(OnTable, [cube2]),
            LiftedAtom(OnTable, [cube3]),
            LiftedAtom(OnTable, [cube4]),
            LiftedAtom(DrawerClosed, [drawer]),
        },
        add_effects={
            LiftedAtom(DrawerOpen, [drawer]),
        },
        delete_effects={
            LiftedAtom(DrawerClosed, [drawer]),
        },
    )

    # Place cupboard operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    drawer = Variable("?drawer", MujocoDrawerObjectType)
    cube0 = Variable("?cube0", MujocoMovableObjectType)
    cube1 = Variable("?cube1", MujocoMovableObjectType)
    cube2 = Variable("?cube2", MujocoMovableObjectType)
    cube3 = Variable("?cube3", MujocoMovableObjectType)
    cube4 = Variable("?cube4", MujocoMovableObjectType)

    PickWiperOperator = LiftedOperator(
        "pick_wiper",
        [robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [wiper]),
            LiftedAtom(OnTable, [cube0]),
            LiftedAtom(OnTable, [cube1]),
            LiftedAtom(OnTable, [cube2]),
            LiftedAtom(OnTable, [cube3]),
            LiftedAtom(OnTable, [cube4]),
            LiftedAtom(DrawerOpen, [drawer]),
        },
        add_effects={
            LiftedAtom(Holding, [robot, wiper]),
        },
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [wiper]),
        },
    )

    # Sweep operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    drawer = Variable("?drawer", MujocoDrawerObjectType)
    cube0 = Variable("?cube0", MujocoMovableObjectType)
    cube1 = Variable("?cube1", MujocoMovableObjectType)
    cube2 = Variable("?cube2", MujocoMovableObjectType)
    cube3 = Variable("?cube3", MujocoMovableObjectType)
    cube4 = Variable("?cube4", MujocoMovableObjectType)

    SweepOperator = LiftedOperator(
        "sweep",
        [robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4],
        preconditions={
            LiftedAtom(Holding, [robot, wiper]),
            LiftedAtom(OnTable, [cube0]),
            LiftedAtom(OnTable, [cube1]),
            LiftedAtom(OnTable, [cube2]),
            LiftedAtom(OnTable, [cube3]),
            LiftedAtom(OnTable, [cube4]),
            LiftedAtom(DrawerOpen, [drawer]),
        },
        add_effects={
            LiftedAtom(InDrawer, [cube0, drawer]),
            LiftedAtom(InDrawer, [cube1, drawer]),
            LiftedAtom(InDrawer, [cube2, drawer]),
            LiftedAtom(InDrawer, [cube3, drawer]),
            LiftedAtom(InDrawer, [cube4, drawer]),
        },
        delete_effects={
            LiftedAtom(OnTable, [cube0]),
            LiftedAtom(OnTable, [cube1]),
            LiftedAtom(OnTable, [cube2]),
            LiftedAtom(OnTable, [cube3]),
            LiftedAtom(OnTable, [cube4]),
        },
    )

    # Create the PyBullet simulator.
    assert initial_state is not None
    pybullet_sim = PyBulletSim(initial_state, rendering=False)
    controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state, pybullet_sim=pybullet_sim
    )

    # Controllers.
    LiftedOpenDrawerController = controllers["open_drawer"]
    LiftedPickWiperController = controllers["pick_wiper"]
    LiftedSweepController = controllers["sweep"]

    # Finalize the skills.
    skills = {
        LiftedSkill(OpenDrawerOperator, LiftedOpenDrawerController),
        LiftedSkill(PickWiperOperator, LiftedPickWiperController),
        LiftedSkill(SweepOperator, LiftedSweepController),
    }

    # Pre-compute ground operators with the known object bindings to avoid
    # combinatorial explosion from exhaustive grounding.
    ground_operators = _create_ground_operators(
        initial_state, [OpenDrawerOperator, PickWiperOperator, SweepOperator]
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


def _create_ground_operators(
    initial_state: ObjectCentricState,
    operators: list[LiftedOperator],
) -> set[GroundOperator]:
    """Ground operators using known object bindings for this environment."""
    name_to_obj: dict[str, Object] = {obj.name: obj for obj in initial_state}
    param_to_object_name = {
        "?robot": "robot",
        "?drawer": "kitchen_island_drawer_s1c1",
        "?wiper": "wiper_0",
        "?cube0": "cube_0",
        "?cube1": "cube_1",
        "?cube2": "cube_2",
        "?cube3": "cube_3",
        "?cube4": "cube_4",
    }
    ground_ops: set[GroundOperator] = set()
    for operator in operators:
        objects = tuple(
            name_to_obj[param_to_object_name[param.name]]
            for param in operator.parameters
        )
        ground_ops.add(operator.ground(objects))
    return ground_ops
