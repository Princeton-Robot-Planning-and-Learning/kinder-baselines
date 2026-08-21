"""Bilevel planning models for the VegaMotion3D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d_v2.base_env import ArmJointDeltaActionSpace
from kinder.envs.kinematic3d_v2.object_types import (
    Kinematic3Dv2ArmRobotType,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_motion3d import (
    ObjectCentricVegaMotion3DEnv,
    VegaMotion3DEnvConfig,
    VegaMotion3DObjectCentricState,
)
from kinder_models.kinematic3d_v2.vega_motion3d.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.kinematic3d_v2.vega_motion3d.state_abstractions import (
    AtTarget,
    create_goal_deriver,
    create_state_abstractor,
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
    config: VegaMotion3DEnvConfig | None = None,
    prefer_ompl: bool = True,
) -> SesameModels:
    """Create the env models for VegaMotion3D.

    ``config`` overrides the env config of the internal sim. Defaults to
    ``VegaMotion3DEnvConfig()`` if not provided. ``prefer_ompl`` selects the OMPL
    motion planner when ompl is installed; set it to False to force the BiRRT
    fallback.
    """
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, ArmJointDeltaActionSpace)

    if config is None:
        config = VegaMotion3DEnvConfig()
    sim = ObjectCentricVegaMotion3DEnv(config=config, allow_state_access=True)

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
        assert isinstance(state, VegaMotion3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {Kinematic3Dv2ArmRobotType, Kinematic3Dv2PointType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    predicates = {AtTarget}

    # Abstractions.
    state_abstractor = create_state_abstractor(sim)
    goal_deriver = create_goal_deriver(state_abstractor)

    # Operators.
    robot = Variable("?robot", Kinematic3Dv2ArmRobotType)
    target = Variable("?target", Kinematic3Dv2PointType)

    MoveToTargetOperator = LiftedOperator(
        "MoveToTarget",
        [robot, target],
        preconditions=set(),
        add_effects={LiftedAtom(AtTarget, [robot, target])},
        delete_effects=set(),
    )

    # Controllers.
    controllers = create_lifted_controllers(action_space, sim, prefer_ompl=prefer_ompl)
    LiftedMoveToTargetController = controllers["move_to_target"]

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToTargetOperator, LiftedMoveToTargetController),
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
