"""Bilevel planning models for the TidyBot3D Tossing3D environment.

Two operators over five predicates. The base move and the throw are one skill, so no
predicate has to name the pose between them; the pick takes no continuous parameters,
so refinement backtracks over the throw alone.

Each pick and throw binds its own cube, so completed cubes remain goals while the
robot returns for the remaining cubes.
"""

# MuJoCo exposes its API through a C extension.
# pylint: disable=no-member

from pathlib import Path

import kinder
import mujoco
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
from kinder.envs.dynamic3d.objects.fixtures import FixedCuboid
from kinder.envs.dynamic3d.objects.primitive_objects import Bin, Cuboid
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    BIN_NAME,
    HandEmpty,
    Holding,
    MovableInGoalRegion,
    MovableIsDownX,
    OnGround,
    Tossing3DStateAbstractor,
)
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
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

    robot_env = sim._robot_env  # pylint: disable=protected-access
    assert robot_env is not None
    model = robot_env.sim.model.mj_model
    data = robot_env.sim.data.mj_data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(model, state_spec)

    def snapshot() -> NDArray[np.float64]:
        result = np.empty(state_size)
        mujoco.mj_getState(model, data, result, state_spec)
        return result

    reset_snapshot = snapshot()
    snapshots: dict[tuple[tuple[str, str, bytes], ...], NDArray[np.float64]] = {}

    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate with complete physics state, including on search backtracking.

        Object observations omit finger positions and contact solver state. Restoring
        only those observations at every step changes the grasp and the resulting
        throw. Preserve integration snapshots for states reached by this model.
        """
        key = tuple((obj.name, obj.type.name, x[obj].tobytes()) for obj in x)
        if key not in snapshots:
            # A new external observation starts a new planning problem. Initialize
            # its unobserved state from reset, not a previous failed refinement.
            snapshots.clear()
            mujoco.mj_setState(model, data, reset_snapshot, state_spec)
            sim.set_state(x.copy())
            snapshots[key] = snapshot()
        else:
            mujoco.mj_setState(model, data, snapshots[key], state_spec)
            mujoco.mj_forward(model, data)
        obs, _, _, _, _ = sim.step(u)
        snapshots[
            tuple((obj.name, obj.type.name, obs[obj].tobytes()) for obj in obs)
        ] = snapshot()
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
        MovableInGoalRegion,
        MovableIsDownX,
        OnGround,
    }

    # Pick the cube up off the ground.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    cube = Variable("?cube", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoObjectType)

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
    held = Variable("?held", MujocoMovableObjectType)
    barrier = Variable("?barrier", MujocoObjectType)

    MoveToTossLocationAndTossOperator = LiftedOperator(
        "move_to_toss_location_and_toss",
        [robot, held, barrier],
        preconditions={
            LiftedAtom(Holding, [robot, held]),
            # Only toss if the held cube is still on this side of the barrier.
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
    bin_state_object = initial_state.get_object_from_name(BIN_NAME)
    bin_geometry = sim.get_object(BIN_NAME)
    assert isinstance(bin_geometry, Bin)
    pybullet_sim.add_bin(
        name=BIN_NAME,
        pose=Pose(
            tuple(initial_state.get(bin_state_object, key) for key in ("x", "y", "z")),
            tuple(
                initial_state.get(bin_state_object, key)
                for key in ("qx", "qy", "qz", "qw")
            ),
        ),
        length=bin_geometry.length,
        width=bin_geometry.width,
        height=bin_geometry.height,
        wall_thickness=bin_geometry.wall_thickness,
    )
    for name, fixture in sim._fixtures_dict.items():  # pylint: disable=protected-access
        if not isinstance(fixture, FixedCuboid) or not isinstance(
            fixture.primitive, Cuboid
        ):
            continue
        obj = initial_state.get_object_from_name(name)
        pybullet_sim.add_box(
            name=name,
            pose=Pose(
                tuple(initial_state.get(obj, key) for key in ("x", "y", "z")),
                tuple(initial_state.get(obj, key) for key in ("qx", "qy", "qz", "qw")),
            ),
            dimensions=fixture.primitive.get_bounding_box_dimensions(),
        )
    # Task room geometry is static and deliberately independent of scene backgrounds.
    robot_env = sim._robot_env  # pylint: disable=protected-access
    assert robot_env is not None
    model = robot_env.sim.model.mj_model
    data = robot_env.sim.data.mj_data
    for geom_id in range(model.ngeom):
        if model.body(int(model.geom_bodyid[geom_id])).name != "tossing_room":
            continue
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, data.geom_xmat[geom_id])
        pybullet_sim.add_box(
            name=model.geom(geom_id).name,
            pose=Pose(tuple(data.geom_xpos[geom_id]), tuple(quaternion[[1, 2, 3, 0]])),
            dimensions=tuple(2 * model.geom_size[geom_id]),
            state_object=False,
        )
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
