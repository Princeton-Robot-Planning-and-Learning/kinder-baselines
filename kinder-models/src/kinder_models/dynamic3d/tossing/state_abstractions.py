"""State abstractions for the Tossing3D environment.

MovableIsDownX records that one movable is at lower x than another -- in practice which
side of cuboid_barrier (x ~ 1.3) a cube is on, so an operator model can express that a
toss past it is irreversible.

TODO: only Tossing3D-o1 is supported; no operator says which cube a throw is aimed at.
"""

import numpy as np
from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from pybullet_helpers.geometry import Quaternion
from relational_structs import (
    GroundAtom,
    Object,
    ObjectCentricState,
    Predicate,
)

from kinder_models.dynamic3d.utils import (
    END_EFFECTOR_TO_OBJECT_XY_HOLDING_TOLERANCE,
    END_EFFECTOR_TO_OBJECT_Z_HOLDING_TOLERANCE,
    GRIPPER_GRASPING_THRESHOLD,
    GRIPPER_OPEN_COMMAND_TOLERANCE,
    MINIMUM_HOLDING_HEIGHT,
    ON_GROUND_TOLERANCE,
    PyBulletSim,
    cube_tilt_from_upright,
)

# Upstream types cube, bin and barrier alike, so names state the type, not the subset.
MovableInGoalRegion = Predicate("MovableInGoalRegion", [MujocoMovableObjectType])
OnGround = Predicate("OnGround", [MujocoObjectType])
Holding = Predicate("Holding", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])
HandEmpty = Predicate("HandEmpty", [MujocoTidyBotRobotObjectType])
MovableIsDownX = Predicate(
    "MovableIsDownX", [MujocoMovableObjectType, MujocoMovableObjectType]
)
# The environment's inflated region, not the task JSON's "ranges".
GOAL_REGION_NAME = "blocks_goal_region"
# The bin the goal region is attached to (Tossing3D-o1.json's blocks_goal_region.target).
BIN_NAME = "bin_0"

CUBE_NAME_PREFIX = "cube"
BARRIER_NAME = "cuboid_barrier"


class Tossing3DStateAbstractor:
    """State abstractor for the Tossing3D environment."""

    def __init__(self, sim: ObjectCentricTidyBot3DEnv) -> None:
        """Initialize the state abstractor."""
        initial_state, _ = sim.reset()
        cubes = self._get_cubes(initial_state)
        assert len(cubes) == 1, (
            f"only Tossing3D-o1 is supported, got {len(cubes)} cubes; see this "
            "module's TODO for what o2 would need"
        )
        self._pybullet_sim = PyBulletSim(initial_state, rendering=False)
        self._robot_name = sim.robot_name
        self._sim = sim

    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        # Not pure: poses come from `state`, the goal region from the live simulator --
        # including the bin's own live pose, since the region is attached to it.
        atoms: set[GroundAtom] = set()

        self._pybullet_sim.set_state(state)

        robot = state.get_object_from_name(self._robot_name)
        fixtures = state.get_objects(MujocoFixtureObjectType)
        movables = state.get_objects(MujocoMovableObjectType)
        all_mujoco_objects = set(fixtures) | set(movables)
        cubes = self._get_cubes(state)
        barriers = [o for o in movables if o.name == BARRIER_NAME]

        holding = {cube: self._check_holding(state, robot, cube) for cube in cubes}
        if self._check_gripper_open(state, robot):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        for cube in cubes:
            if self._check_on_ground(state, cube):
                atoms.add(GroundAtom(OnGround, [cube]))
            if holding[cube]:
                atoms.add(GroundAtom(Holding, [robot, cube]))
            if self._check_in_goal_region(state, cube):
                atoms.add(GroundAtom(MovableInGoalRegion, [cube]))
            for barrier in barriers:
                if self._check_is_down_x(state, cube, barrier):
                    atoms.add(GroundAtom(MovableIsDownX, [cube, barrier]))

        objects = {robot} | all_mujoco_objects
        return RelationalAbstractState(atoms, objects)

    @staticmethod
    def _check_gripper_open(state: ObjectCentricState, robot: Object) -> bool:
        """Whether the gripper is commanded open, which implies an empty hand.

        Reads the command, not finger pose, so this and Holding are not complementary.
        """
        return bool(
            np.isclose(
                state.get(robot, "pos_gripper"),
                0.0,
                atol=GRIPPER_OPEN_COMMAND_TOLERANCE,
            )
        )

    @staticmethod
    def _check_on_ground(state: ObjectCentricState, movable: Object) -> bool:
        """Whether a movable rests flat on the ground, on any of its faces.

        Flat because the bounding box is pose-independent, so the bottom-face arithmetic
        only holds while a face is down. Which face is down is not asked: for a cube
        those are the same rest, and the grasp is derived from the upright rotation, so
        a predicate that distinguished them would refuse picks that work.
        """
        z = state.get(movable, "z")
        bounding_box_height = state.get(movable, "bb_z")
        if not np.isclose(z - bounding_box_height / 2, 0.0, atol=ON_GROUND_TOLERANCE):
            return False
        rotation: Quaternion = (
            state.get(movable, "qx"),
            state.get(movable, "qy"),
            state.get(movable, "qz"),
            state.get(movable, "qw"),
        )
        # Only a cube's faces are interchangeable; anything else keeps the strict test.
        extents = [state.get(movable, f) for f in ("bb_x", "bb_y", "bb_z")]
        if not np.allclose(extents, extents[0]):
            qx, qy, _, _ = rotation
            return bool(
                np.isclose(qx, 0.0, atol=ON_GROUND_TOLERANCE)
                and np.isclose(qy, 0.0, atol=ON_GROUND_TOLERANCE)
            )
        return bool(cube_tilt_from_upright(rotation) < ON_GROUND_TOLERANCE)

    def _check_holding(
        self, state: ObjectCentricState, robot: Object, movable: Object
    ) -> bool:
        """Whether the gripper is closed on this movable and lifting it.

        Per-axis tolerance rather than one shared scalar: planar alignment is
        empirically <1cm in this domain, but the grasp geometry leaves a real,
        roughly constant vertical TCP-to-cube-center offset, so a shared
        tolerance made z a frequent near-miss even with dx/dy well within
        bounds. A PyBullet finger-contact query would be the physically-correct
        check, but PyBulletSim.set_state() hardcodes the finger joints to 0.0
        (that sim is built for arm motion planning only), so it can't see a
        real grasp -- fixing that needs kindergarden PR #162 (unmerged).
        """
        z = state.get(movable, "z")
        pos_gripper = state.get(robot, "pos_gripper")
        if pos_gripper <= GRIPPER_GRASPING_THRESHOLD or z <= MINIMUM_HOLDING_HEIGHT:
            return False
        ee_pose = self._pybullet_sim.get_ee_pose()
        dx = abs(ee_pose.position[0] - state.get(movable, "x"))
        dy = abs(ee_pose.position[1] - state.get(movable, "y"))
        dz = abs(ee_pose.position[2] - z)
        return bool(
            dx < END_EFFECTOR_TO_OBJECT_XY_HOLDING_TOLERANCE
            and dy < END_EFFECTOR_TO_OBJECT_XY_HOLDING_TOLERANCE
            and dz < END_EFFECTOR_TO_OBJECT_Z_HOLDING_TOLERANCE
        )

    @staticmethod
    def _check_is_down_x(
        state: ObjectCentricState, movable: Object, other: Object
    ) -> bool:
        """Whether a movable is at lower x than another, read live rather than fixed."""
        return state.get(movable, "x") < state.get(other, "x")

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to toss every cube into the goal region."""
        atoms = {GroundAtom(MovableInGoalRegion, [o]) for o in self._get_cubes(state)}
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def _get_cubes(self, state: ObjectCentricState) -> list[Object]:
        """Get the cubes, told apart from the bin and the barrier by name."""
        movables = state.get_objects(MujocoMovableObjectType)
        return [o for o in movables if o.name.startswith(CUBE_NAME_PREFIX)]

    def _check_in_goal_region(self, state: ObjectCentricState, cube: Object) -> bool:
        """Check whether a cube's centre lies in the goal region, read off the bin."""
        bin_object = self._sim.get_object(BIN_NAME)
        position = np.array(
            [
                state.get(cube, "x"),
                state.get(cube, "y"),
                state.get(cube, "z"),
            ],
            dtype=np.float32,
        )
        return bin_object.check_in_region(
            position,
            GOAL_REGION_NAME,
            self._sim._robot_env,  # pylint: disable=protected-access
        )
