"""Bilevel planning models for the VegaPickPlace3D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d_v2.object_types import (
    Kinematic3Dv2GraspArmRobotType,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_pickplace3d import (
    BimanualArmJointDeltaGraspActionSpace,
    ObjectCentricVegaPickPlace3DEnv,
    VegaPickPlace3DEnvConfig,
    VegaPickPlace3DObjectCentricState,
)
from kinder_models.kinematic3d_v2.vega_pickplace3d.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.kinematic3d_v2.vega_pickplace3d.state_abstractions import (
    PREDICATES,
    HandEmpty,
    Holding,
    NotHeld,
    On,
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
    config: VegaPickPlace3DEnvConfig | None = None,
    prefer_ompl: bool = True,
    grasp_approach_max_tilt: float | None = None,
) -> SesameModels:
    """Create the env models for VegaPickPlace3D.

    ``config`` overrides the env config of the internal sim. Defaults to
    ``VegaPickPlace3DEnvConfig()`` if not provided. ``prefer_ompl`` selects the OMPL
    motion planner when ompl is installed; set it to False to force the BiRRT
    fallback. ``grasp_approach_max_tilt`` bounds how far (in radians) the end
    effector may point off straight down in sampled pick and place configurations,
    with the yaw free; None (the default) leaves grasp orientations unconstrained.
    """
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, BimanualArmJointDeltaGraspActionSpace)

    if config is None:
        config = VegaPickPlace3DEnvConfig()
    sim = ObjectCentricVegaPickPlace3DEnv(config=config, allow_state_access=True)

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
        assert isinstance(state, VegaPickPlace3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {Kinematic3Dv2GraspArmRobotType, Kinematic3Dv2PointType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Abstractions.
    state_abstractor = create_state_abstractor(sim)
    goal_deriver = create_goal_deriver(state_abstractor)

    # Operators. Handing the cube to an arm that already holds it is impossible, and
    # no reachable abstract state satisfies both Holding(?giver, ?cube) and
    # HandEmpty(?giver), so the HandEmpty(?receiver) precondition also rules out
    # groundings where the giver and the receiver are the same arm.
    arm = Variable("?arm", Kinematic3Dv2GraspArmRobotType)
    giver = Variable("?giver", Kinematic3Dv2GraspArmRobotType)
    receiver = Variable("?receiver", Kinematic3Dv2GraspArmRobotType)
    cube = Variable("?cube", Kinematic3Dv2PointType)
    target = Variable("?target", Kinematic3Dv2PointType)

    PickOperator = LiftedOperator(
        "Pick",
        [arm, cube],
        preconditions={
            LiftedAtom(HandEmpty, [arm]),
            LiftedAtom(NotHeld, [cube]),
        },
        add_effects={LiftedAtom(Holding, [arm, cube])},
        delete_effects={
            LiftedAtom(HandEmpty, [arm]),
            LiftedAtom(NotHeld, [cube]),
        },
    )

    PlaceOperator = LiftedOperator(
        "Place",
        [arm, cube, target],
        preconditions={LiftedAtom(Holding, [arm, cube])},
        add_effects={
            LiftedAtom(On, [cube, target]),
            LiftedAtom(NotHeld, [cube]),
            LiftedAtom(HandEmpty, [arm]),
        },
        delete_effects={LiftedAtom(Holding, [arm, cube])},
    )

    HandoverOperator = LiftedOperator(
        "Handover",
        [giver, receiver, cube],
        preconditions={
            LiftedAtom(Holding, [giver, cube]),
            LiftedAtom(HandEmpty, [receiver]),
        },
        add_effects={
            LiftedAtom(Holding, [receiver, cube]),
            LiftedAtom(HandEmpty, [giver]),
        },
        delete_effects={
            LiftedAtom(Holding, [giver, cube]),
            LiftedAtom(HandEmpty, [receiver]),
        },
    )

    # Controllers.
    controllers = create_lifted_controllers(
        action_space,
        sim,
        prefer_ompl=prefer_ompl,
        grasp_approach_max_tilt=grasp_approach_max_tilt,
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickOperator, controllers["pick"]),
        LiftedSkill(PlaceOperator, controllers["place"]),
        LiftedSkill(HandoverOperator, controllers["handover"]),
    }

    # Finalize the models.
    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        PREDICATES,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
