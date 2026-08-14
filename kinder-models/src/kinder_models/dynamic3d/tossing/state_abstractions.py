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
from prpl_utils.utils import get_signed_angle_distance
from relational_structs import (
    GroundAtom,
    Object,
    ObjectCentricState,
    Predicate,
)

from kinder_models.dynamic3d.utils import (
    END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE,
    GRIPPER_GRASPING_THRESHOLD,
    GRIPPER_OPEN_COMMAND_TOLERANCE,
    MINIMUM_HOLDING_HEIGHT,
    ON_GROUND_TOLERANCE,
    WAYPOINT_TOLERANCE,
    PyBulletSim,
)

# Upstream types cube, bin and barrier alike, so names state the type, not the subset.
MovableInGoalRegion = Predicate("MovableInGoalRegion", [MujocoMovableObjectType])
OnGround = Predicate("OnGround", [MujocoObjectType])
Holding = Predicate("Holding", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])
HandEmpty = Predicate("HandEmpty", [MujocoTidyBotRobotObjectType])
MovableIsDownX = Predicate(
    "MovableIsDownX", [MujocoMovableObjectType, MujocoMovableObjectType]
)
RobotAtThrowPose = Predicate(
    "RobotAtThrowPose", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType]
)

# The environment's inflated region, not the task JSON's "ranges".
GOAL_REGION_NAME = "blocks_goal_region"

CUBE_NAME_PREFIX = "cube"
BIN_NAME_PREFIX = "bin"
BARRIER_NAME = "cuboid_barrier"

# The achieved standoff band a toss scores from, in metres. Measured by bisecting
# _check_goals() over real rollouts on three scene seeds; see the commit message. Both
# edges move ~9 mm across seeds, so neither is quoted finer than 5 mm. A literal, not a
# live read of the region, so a scene whose bin or goal region moves needs remeasuring.
THROW_STANDOFF_BOUNDS = (1.09, 1.375)

# Wider than WAYPOINT_TOLERANCE: the sampler already spends half of it off-axis.
THROW_POSE_TOLERANCE = 2 * WAYPOINT_TOLERANCE


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
        # Not pure: poses come from `state`, the goal region from the live simulator.
        atoms: set[GroundAtom] = set()

        self._pybullet_sim.set_state(state)

        robot = state.get_object_from_name(self._robot_name)
        fixtures = state.get_objects(MujocoFixtureObjectType)
        movables = state.get_objects(MujocoMovableObjectType)
        all_mujoco_objects = set(fixtures) | set(movables)
        cubes = self._get_cubes(state)
        bins = [o for o in movables if o.name.startswith(BIN_NAME_PREFIX)]
        barriers = [o for o in movables if o.name == BARRIER_NAME]

        if self._check_gripper_open(state, robot):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        for cube in cubes:
            if self._check_on_ground(state, cube):
                atoms.add(GroundAtom(OnGround, [cube]))
            if self._check_holding(state, robot, cube):
                atoms.add(GroundAtom(Holding, [robot, cube]))
            if self._check_in_goal_region(state, cube):
                atoms.add(GroundAtom(MovableInGoalRegion, [cube]))
            for barrier in barriers:
                if self._check_is_down_x(state, cube, barrier):
                    atoms.add(GroundAtom(MovableIsDownX, [cube, barrier]))

        for target_bin in bins:
            if self._check_at_throw_pose(state, robot, target_bin):
                atoms.add(GroundAtom(RobotAtThrowPose, [robot, target_bin]))

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
        """Whether a movable rests flat on the ground.

        Flat because the bounding box is pose-independent, so the bottom-face
        arithmetic only holds while axis-aligned. A toss cannot predict this.
        """
        z = state.get(movable, "z")
        bounding_box_height = state.get(movable, "bb_z")
        return bool(
            np.isclose(z - bounding_box_height / 2, 0.0, atol=ON_GROUND_TOLERANCE)
            and np.isclose(state.get(movable, "qx"), 0.0, atol=ON_GROUND_TOLERANCE)
            and np.isclose(state.get(movable, "qy"), 0.0, atol=ON_GROUND_TOLERANCE)
        )

    def _check_holding(
        self, state: ObjectCentricState, robot: Object, movable: Object
    ) -> bool:
        """Whether the gripper is closed on this movable and lifting it."""
        z = state.get(movable, "z")
        if (
            state.get(robot, "pos_gripper") <= GRIPPER_GRASPING_THRESHOLD
            or z <= MINIMUM_HOLDING_HEIGHT
        ):
            return False
        ee_pose = self._pybullet_sim.get_ee_pose()
        return bool(
            abs(ee_pose.position[0] - state.get(movable, "x"))
            < END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE
            and abs(ee_pose.position[1] - state.get(movable, "y"))
            < END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE
            and abs(ee_pose.position[2] - z) < END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE
        )

    @staticmethod
    def _check_is_down_x(
        state: ObjectCentricState, movable: Object, other: Object
    ) -> bool:
        """Whether a movable is at lower x than another, read live rather than fixed."""
        return state.get(movable, "x") < state.get(other, "x")

    @staticmethod
    def _check_at_throw_pose(
        state: ObjectCentricState, robot: Object, target: Object
    ) -> bool:
        """Whether a throw from here would land the object in the goal region.

        A success test, not a reachability test: the band is where a throw was measured
        to score. Testing the sampler's own interval is the defect this replaced -- every
        draw satisfied it, so move_to_throw_pose's only add effect could never fail.
        """
        dx = state.get(target, "x") - state.get(robot, "pos_base_x")
        dy = state.get(target, "y") - state.get(robot, "pos_base_y")
        heading_error = abs(
            get_signed_angle_distance(
                np.arctan2(dy, dx), state.get(robot, "pos_base_rot")
            )
        )
        if abs(dy) > THROW_POSE_TOLERANCE or heading_error > THROW_POSE_TOLERANCE:
            return False
        low, high = THROW_STANDOFF_BOUNDS
        return bool(low <= dx <= high)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to toss every cube into the goal region."""
        atoms = {GroundAtom(MovableInGoalRegion, [o]) for o in self._get_cubes(state)}
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def _get_cubes(self, state: ObjectCentricState) -> list[Object]:
        """Get the cubes, told apart from the bin and the barrier by name."""
        movables = state.get_objects(MujocoMovableObjectType)
        return [o for o in movables if o.name.startswith(CUBE_NAME_PREFIX)]

    def _check_in_goal_region(self, state: ObjectCentricState, cube: Object) -> bool:
        """Check whether a cube's centre lies in the goal region."""
        ground_fixture = self._sim._ground_fixture  # pylint: disable=protected-access
        assert ground_fixture is not None, "Ground fixture not initialized"
        position = np.array(
            [
                state.get(cube, "x"),
                state.get(cube, "y"),
                state.get(cube, "z"),
            ],
            dtype=np.float32,
        )
        return ground_fixture.check_in_region(
            position,
            GOAL_REGION_NAME,
            self._sim._robot_env,  # pylint: disable=protected-access
        )
