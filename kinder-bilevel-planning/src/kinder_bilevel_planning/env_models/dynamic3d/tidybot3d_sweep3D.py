"""Bilevel planning models for the TidyBot3D sweep3D environment."""

import kinder
import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoDrawerObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder_models.dynamic3d.sweep3D.state_abstractions import (
    Sweep3DStateAbstractor,
    HandEmpty,
    Holding,
    OnTable,
    InDrawer,
    DrawerOpen,
    DrawerClosed,
)
from kinder_models.dynamic3d.sweep3D.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)
from numpy.typing import NDArray
from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    ObjectCentricState,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
    initial_state: ObjectCentricState | None = None,
) -> SesameModels:
    """Create the env models for TidyBot base motion."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_objects}-v0", render_mode="rgb_array"
    )
    sim = env.unwrapped._object_centric_env  # type: ignore[attr-defined]  # pylint: disable=protected-access

    # State and goal abstractors.
    abstractor = Sweep3DStateAbstractor(sim)
    state_abstractor = abstractor.state_abstractor
    goal_deriver = abstractor.goal_deriver

    # Need to call reset to initialize the qpos, qvel.
    sim.reset()

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
    }  # pylint: disable=line-too-long

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
    drawer = Variable("?drawer", MujocoDrawerObjectType)

    OpenDrawerOperator = LiftedOperator(
        "open_drawer",
        [robot, drawer],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DrawerClosed, [drawer]),
        },
        add_effects={LiftedAtom(DrawerOpen, [drawer])},
        delete_effects={
            LiftedAtom(DrawerClosed, [drawer]),
        },
    )

    # Pick wiper controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    PickWiperOperator = LiftedOperator(
        "pick_wiper",
        [robot, target],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [target]),
        },
        add_effects={LiftedAtom(Holding, [robot, target])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [target]),
        },
    )

    # Sweep controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    wiper = Variable("?wiper", MujocoMovableObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    drawer = Variable("?drawer", MujocoDrawerObjectType)

    SweepOperator = LiftedOperator(
        "sweep",
        [robot, wiper, target, drawer],
        preconditions={
            LiftedAtom(Holding, [robot, wiper]),
            LiftedAtom(DrawerOpen, [drawer]),
            LiftedAtom(OnTable, [target]),
        },
        add_effects={
            LiftedAtom(InDrawer, [target, drawer]),
        },
        delete_effects={
            LiftedAtom(OnTable, [target]),
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
