"""State abstractions for the Tossing3D environment.

Tossing3D is the one dynamic3d domain whose goal cannot be reached by manipulation
alone: a barrier spanning the width of the scene separates the robot from the bin, and
the base cannot pass it, so a cube on the far side can never be retrieved. Reachable
below exists to make that irreversibility expressible -- an operator model that omits it
can emit a retrieve-and-retry plan the dynamics will never execute.
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

from kinder_models.dynamic3d.utils import WAYPOINT_TOL, PyBulletSim

# Predicates.
InGoalRegion = Predicate("InGoalRegion", [MujocoMovableObjectType])
OnGround = Predicate("OnGround", [MujocoObjectType])
Holding = Predicate("Holding", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])
HandEmpty = Predicate("HandEmpty", [MujocoTidyBotRobotObjectType])
Reachable = Predicate("Reachable", [MujocoMovableObjectType, MujocoMovableObjectType])
NearBin = Predicate("NearBin", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])

# The name of the region the task config asks the cubes to end up in. InGoalRegion is
# checked against this region rather than against a hardcoded box, so that it agrees
# with ObjectCentricTidyBot3DEnv._check_goals() by construction. The task JSON's own
# "ranges" entry is not the region: the environment inflates it by the ground
# placement threshold before building the region, so a predicate written against the
# literal in the file scores real successes as failures.
GOAL_REGION_NAME = "blocks_goal_region"

# Names by which the scene's movables are told apart -- see state_abstractor().
CUBE_NAME_PREFIX = "cube"
BIN_NAME_PREFIX = "bin"
BARRIER_NAME = "cuboid_barrier"

# Thresholds shared with shelf/state_abstractions.py, whose HandEmpty, OnGround and
# Holding classify the same TidyBot state that this domain reads.
HANDEMPTY_TOL = 1e-3
GRASP_THRESHOLD = 0.1
HOLDING_HEIGHT = 0.1
ON_GROUND_TOL = 0.05
EE_TO_OBJECT_TOL = 0.05

# The band of base standoffs, in metres, that NearBin treats as throwable-from. Unlike
# the grasping standoff, which MoveToTargetGroundController pins at 0.5 m, a throw
# standoff has no single right value, so this is a band rather than a number. It is
# wider than the 1.35 m the test uses and is the interval a toss sampler would draw
# from; narrowing it makes NearBin stricter, never unsound.
THROW_STANDOFF_BOUNDS = (1.20, 1.65)


class Tossing3DStateAbstractor:
    """State abstractor for the Tossing3D environment."""

    def __init__(self, sim: ObjectCentricTidyBot3DEnv) -> None:
        """Initialize the state abstractor."""
        initial_state, _ = sim.reset()  # just need to access the objects
        self._pybullet_sim = PyBulletSim(initial_state, rendering=False)
        self._robot_name = sim.robot_name
        self._sim = sim

    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state.

        Two things a caller should know. First, NearBin is not achievable through
        MoveToTargetGroundController's *sampler*, which hardcodes a 0.5 m grasping
        standoff; establishing it takes an explicit standoff parameter in the throwable
        band, so an env model that grounds that skill with its own sampler will never
        satisfy NearBin. Second, this is not a pure function of its argument: object
        poses come from `state`, but the goal region is read from the live simulator, so
        calling it on a stored or hypothetical state evaluates InGoalRegion against
        today's region. Both are inert for Tossing3D as it stands and are recorded so
        they do not have to be rediscovered.
        """
        atoms: set[GroundAtom] = set()

        # Sync the pybullet simulator.
        self._pybullet_sim.set_state(state)

        # Extract the relevant objects.
        robot = state.get_object_from_name(self._robot_name)
        fixtures = state.get_objects(MujocoFixtureObjectType)
        movables = state.get_objects(MujocoMovableObjectType)
        all_mujoco_objects = set(fixtures) | set(movables)
        # The bin and the barrier are movables too, so the cubes -- the only objects
        # here that are actually manipulated -- are picked out by name, the way sweep3D
        # picks its wiper out from its cubes. OnGround and InGoalRegion are restricted
        # to them deliberately: the bin and the barrier are scene furniture, and
        # asserting where they rest would say nothing a planner can act on.
        cubes = [o for o in movables if o.name.startswith(CUBE_NAME_PREFIX)]
        bins = [o for o in movables if o.name.startswith(BIN_NAME_PREFIX)]
        barriers = [o for o in movables if o.name == BARRIER_NAME]

        # HandEmpty.
        gripper_val = state.get(robot, "pos_gripper")
        if np.isclose(gripper_val, 0.0, atol=HANDEMPTY_TOL):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        for cube in cubes:
            # OnGround. Flatness is part of the condition because the pick controller
            # builds its grasp pose from the object's orientation, so a cube that came
            # to rest on a corner is not a cube that grasp is modelled on.
            z = state.get(cube, "z")
            bb_z = state.get(cube, "bb_z")
            if (
                np.isclose(z - bb_z / 2, 0.0, atol=ON_GROUND_TOL)
                and np.isclose(state.get(cube, "qx"), 0.0, atol=ON_GROUND_TOL)
                and np.isclose(state.get(cube, "qy"), 0.0, atol=ON_GROUND_TOL)
            ):
                atoms.add(GroundAtom(OnGround, [cube]))

            # Holding. Checking the ee pose and the target pose.
            if gripper_val > GRASP_THRESHOLD and z > HOLDING_HEIGHT:
                ee_pose = self._pybullet_sim.get_ee_pose()
                if (
                    abs(ee_pose.position[0] - state.get(cube, "x")) < EE_TO_OBJECT_TOL
                    and abs(ee_pose.position[1] - state.get(cube, "y"))
                    < EE_TO_OBJECT_TOL
                    and abs(ee_pose.position[2] - z) < EE_TO_OBJECT_TOL
                ):
                    atoms.add(GroundAtom(Holding, [robot, cube]))

            # InGoalRegion.
            if self._check_in_goal_region(state, cube):
                atoms.add(GroundAtom(InGoalRegion, [cube]))

            # Reachable. Compared against the barrier's own live x rather than a
            # constant, because the barrier's pose is sampled per episode.
            for barrier in barriers:
                if state.get(cube, "x") < state.get(barrier, "x"):
                    atoms.add(GroundAtom(Reachable, [cube, barrier]))

        # NearBin: the base is standing where a throw can actually be thrown from. All
        # three conjuncts are load-bearing, and a throw needs all three to be true.
        #
        #   dx -- the standoff, signed rather than absolute, because the robot must be
        #     on the near side of the bin. An absolute value would also accept a base
        #     past the bin (and so past the barrier), a pose that is unreachable today
        #     only because WORLD_X_BOUNDS caps base planning short of it.
        #   dy -- the base is on the bin's axis, not merely at the right radius. A
        #     radius test alone is satisfied by a whole ring of positions.
        #   heading -- the base is pointing at the bin. Position does not imply this:
        #     nothing stops the base sitting at the right (x, y) while turned away, and
        #     a throw is released along the base's own heading, so a predicate that
        #     omitted this would call the robot ready to throw when it is not.
        #
        # All three tolerances are MoveToTargetGroundController's own WAYPOINT_TOL --
        # including the heading one, since _robot_is_close_to_pose() applies that same
        # constant to x, y and theta alike. NearBin therefore admits precisely the poses
        # that controller is willing to stop at, rather than a tolerance invented here.
        low, high = THROW_STANDOFF_BOUNDS
        base_x = state.get(robot, "pos_base_x")
        base_y = state.get(robot, "pos_base_y")
        for target_bin in bins:
            dx = state.get(target_bin, "x") - base_x
            dy = abs(state.get(target_bin, "y") - base_y)
            heading_error = abs(
                get_signed_angle_distance(
                    np.arctan2(state.get(target_bin, "y") - base_y, dx),
                    state.get(robot, "pos_base_rot"),
                )
            )
            if (
                dy <= WAYPOINT_TOL
                and low - WAYPOINT_TOL <= dx <= high + WAYPOINT_TOL
                and heading_error <= WAYPOINT_TOL
            ):
                atoms.add(GroundAtom(NearBin, [robot, target_bin]))

        objects = {robot} | all_mujoco_objects
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to toss every cube into the goal region."""
        movables = state.get_objects(MujocoMovableObjectType)
        atoms = {
            GroundAtom(InGoalRegion, [o])
            for o in movables
            if o.name.startswith(CUBE_NAME_PREFIX)
        }
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def _check_in_goal_region(self, state: ObjectCentricState, cube: Object) -> bool:
        """Check whether a cube's centre lies in the goal region.

        The region is queried per state rather than cached. For a region on the ground
        the bounding box is state-independent today, so caching would be correct, but
        querying cannot go stale if that ever changes, and it is the same call
        _check_goals() makes -- which is what makes this predicate and the environment's
        own success criterion the same test rather than two tests that happen to agree.
        The cost is negligible next to a MuJoCo step.
        """
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
