"""State abstractions for the Tossing3D environment.

Tossing3D's goal region sits on the far side of a barrier that spans the scene in y:
cuboid_barrier is 5 m wide and the task config places it at x ~ 1.3, while the goal
region is x in [1.85, 2.15]. Nothing in this package models that barrier -- base motion
planning runs with an empty obstacle set -- so it is MuJoCo contact alone that keeps the
base on the near side. Reachable below records which side of it a cube is on, so an
operator model can express that a toss is irreversible rather than emitting a plan that
retrieves the cube and retries, which the dynamics will never execute.
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
# Reachable reads as "the first object is on the robot's side of the second". Both slots
# are MujocoMovableObjectType because upstream has no barrier type, so a planner will
# also ground it as (cube_0, bin_0) or (bin_0, cuboid_barrier) -- pairs state_abstractor
# never emits, since it only ever pairs a cube with BARRIER_NAME.
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

# Duplicated from shelf/state_abstractions.py, whose HandEmpty, OnGround and Holding
# classify the same TidyBot state this domain reads. They are literals there, declared
# inside state_abstractor() rather than at module scope, so nothing keeps the two sets
# in sync; hoisting them into dynamic3d/utils.py would, but that touches shelf.
HANDEMPTY_TOL = 1e-3
GRASP_THRESHOLD = 0.1
HOLDING_HEIGHT = 0.1
ON_GROUND_TOL = 0.05
EE_TO_OBJECT_TOL = 0.05

# The band of base standoffs, in metres, that NearBin treats as throwable-from. Unlike
# the grasping standoff, which MoveToTargetGroundController pins at 0.5 m, a throw
# standoff has no single right value, so this is a band rather than a number. It
# brackets the 1.35 m the test uses with room either side, and is the interval a toss
# sampler would draw from; narrowing it makes NearBin stricter, never unsound.
THROW_STANDOFF_BOUNDS = (1.20, 1.65)

# NearBin's own slack, deliberately wider than MoveToTargetGroundController's
# WAYPOINT_TOL. That controller stops when np.isclose(..., atol=WAYPOINT_TOL) holds,
# which is atol + rtol * |b| and so strictly wider than a <= test; and its own sampler
# spends half of WAYPOINT_TOL on the planned off-axis offset before the controller adds
# its termination slack on top. A pose the controller accepts can therefore sit just
# outside a strict WAYPOINT_TOL test and make NearBin false immediately after the skill
# that established it.
NEAR_BIN_TOL = 2 * WAYPOINT_TOL


class Tossing3DStateAbstractor:
    """State abstractor for the Tossing3D environment."""

    def __init__(self, sim: ObjectCentricTidyBot3DEnv) -> None:
        """Initialize the state abstractor."""
        initial_state, _ = sim.reset()  # just need to access the objects
        self._pybullet_sim = PyBulletSim(initial_state, rendering=False)
        self._robot_name = sim.robot_name
        self._sim = sim

    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        # Not a pure function of its argument: object poses come from `state`, but the
        # goal region is read from the live simulator, so calling this on a stored or
        # hypothetical state evaluates InGoalRegion against today's region.
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
        # picks its wiper out from its cubes. Every per-object predicate below
        # (OnGround, Holding, InGoalRegion, Reachable) is restricted to them
        # deliberately: the bin and the barrier are scene furniture, and asserting where
        # they rest would say nothing a planner can act on.
        cubes = self._get_cubes(state)
        bins = [o for o in movables if o.name.startswith(BIN_NAME_PREFIX)]
        barriers = [o for o in movables if o.name == BARRIER_NAME]

        # HandEmpty.
        gripper_val = state.get(robot, "pos_gripper")
        if np.isclose(gripper_val, 0.0, atol=HANDEMPTY_TOL):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        for cube in cubes:
            # OnGround. Flatness is part of the condition because the pick controller
            # builds its grasp pose from the object's orientation, so a cube that came
            # to rest on a corner is not a cube that grasp is modelled on. The cost is
            # that OnGround is not predictable after a toss: a cube that lands flat in
            # the goal region gets OnGround and InGoalRegion, one that lands tilted gets
            # only InGoalRegion, and no fixed effect set covers both under the
            # final_abstract_state == ns refinement gate.
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

            # InGoalRegion. Note for the env model that will consume this: the cube is
            # only in the region once it has come to rest, and TossController
            # terminates when its trapezoidal profile is exhausted, which is while the
            # cube may still be in flight. Under the refinement gate's
            # final_abstract_state == ns, a toss operator that adds InGoalRegion needs
            # the cube settled before the state is abstracted -- and the same goes for
            # HandEmpty, which wants pos_gripper back within 1e-3 of 0 after release.
            if self._check_in_goal_region(state, cube):
                atoms.add(GroundAtom(InGoalRegion, [cube]))

            # Reachable. Compared against the barrier's own live x rather than a
            # constant, because the barrier's placement comes from a task-config region
            # rather than from code. Today that region is 1 mm wide (barrier_init_region,
            # yaw pinned to 0), so the value barely moves; reading it keeps this correct
            # if the config ever widens it.
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
        # All three slacks are NEAR_BIN_TOL, derived from
        # MoveToTargetGroundController's own WAYPOINT_TOL -- including the heading one,
        # since _robot_is_close_to_pose() applies that same constant to x, y and theta
        # alike -- rather than a tolerance invented here. That makes NearBin's slack the
        # controller's slack; it does not make the two sets equal. The controller will
        # also stop at poses NearBin rejects: any standoff outside the band, and any
        # target_rot that swings the base off the bin's x axis (target_rot = pi/2 puts
        # it beside the bin, dx = 0).
        low, high = THROW_STANDOFF_BOUNDS
        base_x = state.get(robot, "pos_base_x")
        base_y = state.get(robot, "pos_base_y")
        for target_bin in bins:
            dx = state.get(target_bin, "x") - base_x
            dy = state.get(target_bin, "y") - base_y
            heading_error = abs(
                get_signed_angle_distance(
                    np.arctan2(dy, dx),
                    state.get(robot, "pos_base_rot"),
                )
            )
            if (
                abs(dy) <= NEAR_BIN_TOL
                and low - NEAR_BIN_TOL <= dx <= high + NEAR_BIN_TOL
                and heading_error <= NEAR_BIN_TOL
            ):
                atoms.add(GroundAtom(NearBin, [robot, target_bin]))

        objects = {robot} | all_mujoco_objects
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to toss every cube into the goal region."""
        atoms = {GroundAtom(InGoalRegion, [o]) for o in self._get_cubes(state)}
        return RelationalAbstractGoal(atoms, self.state_abstractor)

    def _get_cubes(self, state: ObjectCentricState) -> list[Object]:
        """Get the cubes, told apart from the bin and the barrier by name."""
        movables = state.get_objects(MujocoMovableObjectType)
        return [o for o in movables if o.name.startswith(CUBE_NAME_PREFIX)]

    def _check_in_goal_region(self, state: ObjectCentricState, cube: Object) -> bool:
        """Check whether a cube's centre lies in the goal region."""
        # The region is queried per state rather than cached, because this is the same
        # call _check_goals() makes -- which is what makes this predicate and the
        # environment's own success criterion the same test rather than two tests that
        # happen to agree.
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
