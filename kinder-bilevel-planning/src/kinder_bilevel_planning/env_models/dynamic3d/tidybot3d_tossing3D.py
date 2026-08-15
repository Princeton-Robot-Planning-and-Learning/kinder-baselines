"""Bilevel planning models for the TidyBot3D Tossing3D environment.

Two operators over five predicates. The base move and the throw are one skill, so no
predicate has to name the pose between them; the pick takes no continuous parameters,
so refinement backtracks over the throw alone.

TODO: only Tossing3D-o1 is supported; no operator says which cube a throw is aimed at.
"""

from pathlib import Path

import kinder
import numpy as np
from bilevel_planning.structs import (
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
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    HandEmpty,
    Holding,
    IsThrowTarget,
    MovableInGoalRegion,
    MovableIsDownX,
    OnGround,
    Tossing3DStateAbstractor,
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
) -> SesameModels:
    """Create the env models for TidyBot Tossing3D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)
    if num_objects != 1:
        raise NotImplementedError(
            f"Tossing3D bilevel planning supports one cube, got {num_objects}. The "
            "operators do not say which cube a throw is aimed at."
        )

    task_config_path = str(
        Path(kinder.__file__).parent
        / "envs"
        / "dynamic3d"
        / "tasks"
        / "Tossing3D"
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

    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {
        MujocoTidyBotRobotObjectType,
        MujocoObjectType,
        MujocoFixtureObjectType,
        MujocoMovableObjectType,
    }

    state_space = ObjectCentricStateSpace(types)

    predicates = {
        HandEmpty,
        Holding,
        IsThrowTarget,
        MovableInGoalRegion,
        MovableIsDownX,
        OnGround,
    }

    # Pick the cube up off the ground.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    cube = Variable("?cube", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoMovableObjectType)

    PickCubeOperator = LiftedOperator(
        "pick_cube",
        [robot, cube, barrier],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [cube]),
            # Only a cube still on this side of the barrier can be reached.
            LiftedAtom(MovableIsDownX, [cube, barrier]),
        },
        add_effects={LiftedAtom(Holding, [robot, cube])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [cube]),
        },
    )

    # Drive to a pose to throw from, and throw.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)
    held = Variable("?held", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoMovableObjectType)

    MoveToTossLocationAndTossOperator = LiftedOperator(
        "move_to_toss_location_and_toss",
        [robot, target, held, barrier],
        preconditions={
            LiftedAtom(Holding, [robot, held]),
            # Upstream gives the bin the same type as the cube and the barrier, so
            # without this the grounder is free to bind ?target to either of those --
            # a discrete mistake no amount of continuous sampling can recover from.
            LiftedAtom(IsThrowTarget, [target]),
            # ?barrier is unconstrained the same way: without this, the grounder can
            # bind it to an object the held cube was never down-x of (e.g. the bin
            # itself). The delete effect below would then target an atom that was
            # never true, so the real, physical "cube crossed the barrier" fact
            # would survive refinement's exact-state-equality check unexpectedly,
            # and every sample would fail regardless of how the throw itself lands.
            # Still true here: pick_cube's own precondition requires it and no
            # effect between pick and this operator touches it.
            LiftedAtom(MovableIsDownX, [held, barrier]),
        },
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(MovableInGoalRegion, [held]),
            # Measured on 20 throws: 15/15 that scored left the cube resting on a face.
            LiftedAtom(OnGround, [held]),
        },
        delete_effects={
            LiftedAtom(Holding, [robot, held]),
            # The one-way door: past the barrier the cube cannot be picked again.
            LiftedAtom(MovableIsDownX, [held, barrier]),
        },
    )

    # Controllers.
    assert initial_state is not None
    pybullet_sim = PyBulletSim(initial_state, rendering=False)
    controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state, pybullet_sim=pybullet_sim
    )

    skills = {
        LiftedSkill(PickCubeOperator, controllers["pick_cube"]),
        LiftedSkill(
            MoveToTossLocationAndTossOperator,
            controllers["move_to_toss_location_and_toss"],
        ),
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
