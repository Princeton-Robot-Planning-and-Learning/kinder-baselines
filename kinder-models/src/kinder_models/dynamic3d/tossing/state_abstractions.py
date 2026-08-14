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

from kinder_models.dynamic3d.predicate_checks import (
    check_end_effector_at_object,
    check_grasped_and_lifted,
    check_gripper_open,
    check_is_down_x,
    check_on_ground,
    check_reach_interval_hits_box,
)
from kinder_models.dynamic3d.utils import (
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

# Throwable-from standoffs, not the 0.5 m grasping one: the range a base sampler may
# draw from, which is about where a throw is *possible at all*.
#
# **This is the sampler's range only, and RobotAtThrowPose deliberately does not accept
# it.** The two were the same interval until this predicate was rewritten, and that
# identity is a defect rather than a convenience: MoveToThrowPose's only add effect is
# RobotAtThrowPose, so a predicate accepting every standoff the sampler can draw is an
# add effect satisfied by construction on every attempt. Downstream, hitl-pmp measured
# what that costs on its port of this domain -- 16/16 attempts labelled success against
# 0/16 informed draws, versus 7/20 informed for a pick skill in the same run -- because
# a per-skill success classifier trained on a constant-true label has one class to learn
# from. Widening this constant must not widen the predicate.
THROW_STANDOFF_BOUNDS = (1.20, 1.65)

# The distance the shipped toss carries the cube before its first ground contact, in
# metres. A property of the controller -- the two fixed arm configurations and the
# cube's mass -- not of the scene, so the band below can be recomputed from live
# geometry on every call and stays correct when the bin or the goal region moves.
#
# Calibrated downstream in hitl-pmp, against where success breaks rather than in free
# flight, and the difference is a trap: throwing onto open floor and recording where the
# cube comes to rest gives 1.3499 m (sd 0.0024, n = 12), which is ~0.075 m longer
# because it includes post-impact roll. The goal-region test needs the *impact* range,
# since a cube landing inside the region is caught by the bin rather than rolling on.
# Two independent sweeps bracket the impact range to (1.2608, 1.3090) and, from the two
# edges of partial solving independently, to 1.2749 from each end.
#
# That calibration was taken with the gripper opening on the first control step past
# release fraction 0.46, which is exactly what TossController does, so it describes this
# controller rather than approximating it.
THROW_REACH = 1.275

# **The band is the region's own edges, untrimmed, and the trimming a downstream port
# applies is deliberately not copied here.** hitl-pmp narrows this box by two margins it
# measured (0.025 m and 0.05 m), because a 5-seed sweep found the standoffs solving on
# every seed to be narrower than the geometric prediction at both ends. Those margins
# were measured against its own hand-composed throw and indexed by *commanded* standoff,
# and commanded standoff is not what this predicate reads: driving move_to_target to a
# commanded 1.35 m leaves the base 1.3778 m from the bin, 27.8 mm short, well inside
# WAYPOINT_TOLERANCE but a third of the width either margin would trim. Transplanting
# them would import a 28 mm indexing error along with the measurement.
#
# The untrimmed band is already what fixes the defect: it is derived from the scored
# region rather than from the sampler's range, so the add effect can fail. A consumer
# with its own reliability measurement can pass an already-trimmed box.

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
        """Whether the gripper is commanded open, which implies an empty hand."""
        return check_gripper_open(state.get(robot, "pos_gripper"))

    @staticmethod
    def _check_on_ground(state: ObjectCentricState, movable: Object) -> bool:
        """Whether a movable rests flat on the ground."""
        return check_on_ground(
            state.get(movable, "z"),
            state.get(movable, "bb_z"),
            state.get(movable, "qx"),
            state.get(movable, "qy"),
        )

    def _check_holding(
        self, state: ObjectCentricState, robot: Object, movable: Object
    ) -> bool:
        """Whether the gripper is closed on this movable and lifting it.

        The forward kinematics is why this one is not entirely in predicate_checks:
        the end-effector pose comes from the PyBullet mirror of the state, so only the
        comparison against it is simulator-free.
        """
        if not check_grasped_and_lifted(
            state.get(robot, "pos_gripper"), state.get(movable, "z")
        ):
            return False
        ee_pose = self._pybullet_sim.get_ee_pose()
        return check_end_effector_at_object(
            ee_pose.position,
            (
                state.get(movable, "x"),
                state.get(movable, "y"),
                state.get(movable, "z"),
            ),
        )

    @staticmethod
    def _check_is_down_x(
        state: ObjectCentricState, movable: Object, other: Object
    ) -> bool:
        """Whether a movable is at lower x than another, read live rather than fixed."""
        return check_is_down_x(state.get(movable, "x"), state.get(other, "x"))

    def _check_at_throw_pose(
        self, state: ObjectCentricState, robot: Object, target: Object
    ) -> bool:
        """Whether a throw from the base's pose would land the object in the region.

        A success test, not a reachability test. The standoff conjunct asks whether the
        throw's own displacement carries the object into the scored box, read live off
        the goal region, rather than whether the standoff lies in the interval a sampler
        draws from -- see THROW_STANDOFF_BOUNDS for why the latter cannot be learned
        from. Move the bin, resize the region or change the ground placement threshold
        and the band follows on its own.
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
        x_min, _, _, x_max, _, _ = self._goal_region_bbox()
        return check_reach_interval_hits_box(
            state.get(robot, "pos_base_x"), THROW_REACH, THROW_REACH, x_min, x_max
        )

    def _goal_region_bbox(self) -> tuple[float, float, float, float, float, float]:
        """The live world-frame box _check_goals() scores containment in.

        Region.bbox only reads the site's simulated position when the region carries an
        env; ground regions are constructed with env=None, so it otherwise falls back to
        an XML/parent-frame value. check_in_region handles that by swapping env in and
        back out, and so does this, rather than leaving a sim reference behind on a
        region that was deliberately left bare.
        """
        ground_fixture = self._sim._ground_fixture  # pylint: disable=protected-access
        assert ground_fixture is not None, "Ground fixture not initialized"
        found = ground_fixture.region_objects.get(GOAL_REGION_NAME, [])
        assert len(found) == 1, f"expected one {GOAL_REGION_NAME}, found {len(found)}"
        region = found[0]
        original = region.env
        region.env = self._sim._robot_env  # pylint: disable=protected-access
        try:
            bbox = tuple(float(value) for value in region.bbox)
        finally:
            region.env = original
        assert len(bbox) == 6, f"{GOAL_REGION_NAME} is not a single box: {bbox}"
        return bbox  # type: ignore[return-value]

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
