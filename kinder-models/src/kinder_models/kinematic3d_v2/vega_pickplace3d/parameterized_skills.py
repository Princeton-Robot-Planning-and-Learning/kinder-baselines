"""Parameterized skills for the VegaPickPlace3D environment.

There are three skills: picking up the cube with one arm, placing a held cube onto the
target patch, and handing the cube from one arm to the other. Each one samples a goal
configuration by rejection, drawing arm configurations uniformly within the joint limits
and accepting ones that satisfy the skill's end-effector condition (grasp-range reaches
additionally refine near misses with differential IK). It then plans a collision-free
joint path per arm and emits the path as bounded joint deltas with the grasp commands
held so that the cube stays where it is; the final action of each skill toggles one
grasp command to grasp, release, or take the cube.

The handover skill moves both arms in sequence: the holding arm carries the cube into a
region in front of the robot that both arms can reach, and the receiving arm then
reaches to the cube and takes it.

An optional grasp-approach tilt constraint (``grasp_approach_max_tilt``) restricts pick
and place configurations to top-down approaches: the end effector points at most that
angle off straight down, with the yaw free. When it is set, those two skills sample
constraint-aware, drawing a yaw and solving full-pose IK for a down-facing target,
because rejection over uniform draws essentially never produces a near-vertical end
effector that also reaches the cube.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pybullet as p
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic3d_v2.object_types import (
    ARM_NUM_JOINTS,
    Kinematic3Dv2GraspArmRobotType,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_pickplace3d import (
    ARM_SIDES,
    CUBE_NODE,
    BimanualArmJointDeltaGraspActionSpace,
    ObjectCentricVegaPickPlace3DEnv,
    VegaPickPlace3DObjectCentricState,
)
from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.ik import NumericalIK
from prpl_kinematics.planning.configuration_space import ConfigurationSpace
from prpl_kinematics.planning.joint_space import JointSpace
from prpl_kinematics.planning.motion_planner import MotionPlanner
from prpl_kinematics.tree.kinematic_tree import Configuration, KinematicTree
from relational_structs import Object, ObjectCentricState, Variable
from spatialmath import SE3

from kinder_models.kinematic3d_v2.vega_motion3d.parameterized_skills import (
    create_motion_planner,
)

# Goal configurations are drawn uniformly within the joint limits and accepted when the
# skill's end-effector condition holds, at ~0.3 ms per forward-kinematics call. A
# successful sample costs a fraction of a second; an exhausted budget (which raises
# TrajectorySamplingFailure, e.g. when the cube is out of the arm's reach) costs a few
# seconds. Bilevel planning relies on that failure to discard abstract plans that use
# the wrong arm.
DEFAULT_NUM_GOAL_CANDIDATES = 10_000

# Reaching a point with the end effector uses a two-stage sampler: a draw whose end
# effector lands within the loose radius of the point is refined with differential IK
# (keeping its orientation) and re-checked against the strict condition. Hitting the
# grasp ball directly is a ~0.05% event for a reachable cube and gets much rarer near
# the table edges, so a pure rejection sampler exhausts its budget on cubes that the
# arm can in fact reach; the loose pass is ~20x more likely per draw and the
# refinement almost always closes the remaining distance.
LOOSE_REACH_RADIUS = 0.25

# The refinement solver aims at a point this far above the reached point and stops
# within the position tolerance of it. Aiming above matters when the point is a cube
# resting on the table: an end effector driven to the cube center itself sits at table
# height and the gripper geometry collides with the table, whereas an end effector
# hovering above the cube grasps just as well (grasping is by distance alone) and
# clears the table. The strict reach distance must exceed the offset plus the
# tolerance, so a converged refinement always satisfies the strict condition.
REFINE_TARGET_Z_OFFSET = 0.04
REFINE_POSITION_TOLERANCE = 0.04
REFINE_MAX_ITERS = 50

# Sampled grasp (and take-over) configurations put the end effector within this
# fraction of the environment's grasp radius, so the grasp cannot sit on the boundary.
GRASP_RADIUS_MARGIN = 0.9

# Constraint-aware sampling under a grasp-approach tilt constraint: each candidate is a
# full-pose IK solve from a uniformly drawn seed (~0.2 s), not a forward-kinematics
# call (~0.3 ms), so the budget is far smaller than DEFAULT_NUM_GOAL_CANDIDATES.
# Measured per-candidate acceptance for reachable cubes is 10-46% depending on the
# cube's position (the soft limit margin costs several points, most at the hard
# end), so 100 candidates make a spurious exhaustion vanishingly rare (under 3e-5
# at the low end) while a genuine exhaustion (an unreachable cube) resolves in under
# half a minute. A spurious exhaustion is the costlier mistake, since it discards a
# feasible abstract plan, so the budget favors reliability over a faster failure
# path.
DOWN_FACING_NUM_GOAL_CANDIDATES = 100

# Full-pose IK from a random seed needs more iterations than the position-only
# refinement: capping at 50 iterations drops the hard-case acceptance rate from 20%
# to 16%, and beyond 100 the failure path gets slower without a matching gain.
DOWN_FACING_IK_MAX_ITERS = 100

# The physical Vega's firmware zero-torques any axis whose position passes its soft
# limit (dynamically, not only at boot), so a goal configuration on the boundary
# risks a limp joint on tracking overshoot. Constrained samples keep every arm joint
# at least this far inside its limits; the IK solver clamps to the limits each
# iteration, so without a margin accepted solutions can sit exactly on them.
DOWN_FACING_JOINT_LIMIT_MARGIN = 0.1

# Constrained placing prefers the posture the arm is already in. It tries this many
# yaws fanning out from the current yaw (plus the current yaw itself), each seeded
# from the current configuration, and keeps the solution closest to that
# configuration in max joint change; full posture resampling is the last resort. A
# nearby yaw does not imply a nearby posture (unwinding a wrist pinned against the
# margin boundary can spin the roll joint by radians while the yaw barely moves),
# hence closest-solution selection rather than first-success.
PLACE_POSTURE_SEED_ATTEMPTS = 12

# A posture-preserving solution whose max joint change is below this is accepted
# immediately, skipping the rest of the fan; tier-one successes (a short transfer at
# the current yaw) measure 0.7-0.9 rad, so the common case stays a single solve.
PLACE_POSTURE_ACCEPT_DELTA = 1.2

# Sampled place configurations put the cube center within this fraction of the target
# patch half extents, so the drop cannot land on the patch boundary.
PLACE_EXTENT_MARGIN = 0.7

# A released cube drops kinematically to its resting height from wherever it is, so
# nothing in the environment forces a gentle set-down. The sampler bounds the release
# height itself: the cube must hang no more than this far above its resting height
# when the arm lets go, so placements happen near the surface instead of as long
# drops.
PLACE_MAX_RELEASE_HEIGHT = 0.10

# The handover region: the holding arm carries the cube to a point sampled here, in
# front of the robot where both arms can reach. Bounds are on the cube center, with y
# centered on the robot's sagittal plane and z relative to the table top.
HANDOVER_Y_BOUNDS = (-0.15, 0.15)
HANDOVER_Z_ABOVE_TABLE_BOUNDS = (0.15, 0.40)

# How many carry configurations to try per handover sample: for each one, the receiving
# arm gets a fraction of the candidate budget to find a matching take-over
# configuration.
HANDOVER_NUM_CARRY_ATTEMPTS = 5


def create_collision_fn(
    sim: ObjectCentricVegaPickPlace3DEnv,
) -> Callable[[Configuration], bool]:
    """A collision check for the robot and the table in ``sim``.

    This builds a checker over the environment's kinematic tree rather than reusing the
    environment's own, which is not exposed. The tree is shared, so the two stay in
    agreement; the cost is one extra PyBullet client held for the process lifetime.
    """
    physics_client_id = p.connect(p.DIRECT)
    collision_checker = PyBulletCollisionChecker(physics_client_id)
    collision_checker.load(sim.tree)
    collision_checker.ignore(sim.robot.allowed_collision_pairs)
    return collision_checker.in_collision


def _side_of(arm: Object) -> str:
    """The side ("left" or "right") of an arm object named ``<side>_arm``."""
    side = arm.name.split("_", maxsplit=1)[0]
    assert side in ARM_SIDES, f"Not an arm object: {arm.name}"
    return side


def _down_facing_rotation(yaw: float) -> np.ndarray:
    """The end-effector rotation whose approach axis points straight down.

    The Vega tool frames carry the fingertip offset along their z axis, so a
    down-facing grasp is one whose frame z axis points along world -z; ``yaw`` spins
    the frame about the vertical.
    """
    return np.asarray((SE3.Rz(yaw) * SE3.Rx(np.pi)).R)


class _MarginJointSpace(JointSpace):
    """A JointSpace with every joint's bounds pulled in by a fixed margin.

    The constrained IK solvers use this so their solutions stay clear of the soft
    limits during the solve (the solver clamps to the space's bounds each
    iteration), rather than rejecting boundary solutions after the fact; a seeded
    solve then converges to a nearby margin-respecting solution when one exists.
    """

    def __init__(
        self, tree: KinematicTree, joint_names: Sequence[str], margin: float
    ) -> None:
        super().__init__(tree, joint_names)
        assert np.all(self._upper - self._lower > 2 * margin)
        self._lower = self._lower + margin
        self._upper = self._upper - margin


class _VegaPickPlaceControllerBase(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Shared machinery for the VegaPickPlace3D skills.

    Subclasses sample goal configurations and define a sequence of arm motions followed
    by one grasp-command toggle. Motions are planned lazily during execution, one arm at
    a time, from the state observed when the arm starts moving.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricVegaPickPlace3DEnv,
        planners: dict[str, MotionPlanner],
        collision_fn: Callable[[Configuration], bool],
        num_goal_candidates: int = DEFAULT_NUM_GOAL_CANDIDATES,
        grasp_approach_max_tilt: float | None = None,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._planners = planners
        self._collision_fn = collision_fn
        self._num_goal_candidates = num_goal_candidates
        # The tilt bound doubles as the IK orientation tolerance, so zero is
        # unsatisfiable rather than "exactly vertical".
        assert grasp_approach_max_tilt is None or grasp_approach_max_tilt > 0
        self._grasp_approach_max_tilt = grasp_approach_max_tilt
        self._ik_refiners = {
            side: NumericalIK(
                sim.tree,
                self._joint_space(side),
                sim.robot.manipulators[side].ee_frame,
                position_tolerance=REFINE_POSITION_TOLERANCE,
                orientation_tolerance=float("inf"),
                max_iters=REFINE_MAX_ITERS,
            )
            for side in ARM_SIDES
        }
        # Full-pose solvers for constraint-aware sampling. The orientation tolerance
        # is the tilt bound: a converged solve is within that angle of the down-facing
        # target rotation, and the achieved approach axis deviates from vertical by at
        # most the full rotation error, so the tilt constraint holds by construction.
        # The solvers run in a margin-shrunken joint space, so their solutions also
        # stay clear of the soft limits by construction.
        self._down_facing_iks = (
            {
                side: NumericalIK(
                    sim.tree,
                    _MarginJointSpace(
                        sim.tree,
                        self._joint_space(side).joint_names,
                        DOWN_FACING_JOINT_LIMIT_MARGIN,
                    ),
                    sim.robot.manipulators[side].ee_frame,
                    position_tolerance=REFINE_POSITION_TOLERANCE,
                    orientation_tolerance=grasp_approach_max_tilt,
                    max_iters=DOWN_FACING_IK_MAX_ITERS,
                )
                for side in ARM_SIDES
            }
            if grasp_approach_max_tilt is not None
            else {}
        )
        self._current_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        # (side, goal joints) for each arm motion, in execution order.
        self._motion_segments: list[tuple[str, np.ndarray]] = []
        # The plan for the segment currently executing, or None before planning.
        self._current_plan: list[np.ndarray] | None = None
        # (side, command) for the single grasp toggle that ends the skill.
        self._final_command: tuple[str, float] | None = None
        self._final_command_issued = False

    def _arm_space(self, side: str) -> ConfigurationSpace:
        manipulator = self._sim.robot.manipulators[side]
        return self._sim.robot.groups[manipulator.group]

    def _joint_space(self, side: str) -> JointSpace:
        space = self._arm_space(side)
        assert isinstance(space, JointSpace)
        return space

    def _sample_arm_configuration(
        self,
        side: str,
        accept: Callable[[Configuration], bool],
        rng: np.random.Generator,
        num_candidates: int,
    ) -> np.ndarray | None:
        """A collision-free configuration for ``side`` accepted by ``accept``.

        The sim must already be at the state to sample around; all other joints keep
        their current values. Returns None when the budget is exhausted.
        """
        space = self._arm_space(side)
        lower, upper = space.bounds()
        base = dict(self._sim.configuration)
        for _ in range(num_candidates):
            joints = rng.uniform(lower, upper)
            config = base | space.to_configuration(joints)
            if not accept(config):
                continue
            if self._collision_fn(config):
                continue
            return joints
        return None

    def _sample_ee_reach_configuration(
        self,
        side: str,
        position: np.ndarray,
        distance: float,
        rng: np.random.Generator,
        num_candidates: int,
    ) -> np.ndarray | None:
        """A collision-free configuration with ``side``'s end effector in range.

        Draws configurations uniformly within the joint limits. A draw whose end
        effector already satisfies the strict condition is accepted as is; a draw within
        the loose radius of ``position`` is refined with differential IK toward a point
        just above ``position``, keeping the draw's orientation, and re-checked. The sim
        must already be at the state to sample around. Returns None when the budget is
        exhausted.
        """
        assert distance > REFINE_POSITION_TOLERANCE + REFINE_TARGET_Z_OFFSET
        space = self._arm_space(side)
        lower, upper = space.bounds()
        base = dict(self._sim.configuration)
        ee_frame = self._sim.robot.manipulators[side].ee_frame
        refiner = self._ik_refiners[side]
        refine_target = position + np.array([0.0, 0.0, REFINE_TARGET_Z_OFFSET])
        for _ in range(num_candidates):
            joints = rng.uniform(lower, upper)
            config = base | space.to_configuration(joints)
            ee_pose = self._sim.tree.forward_kinematics(ee_frame, config)
            reach = float(np.linalg.norm(ee_pose.t - position))
            if reach >= LOOSE_REACH_RADIUS:
                continue
            if reach < distance and not self._collision_fn(config):
                return joints
            refined = refiner.solve(SE3.Rt(ee_pose.R, refine_target), config)
            if refined is None:
                continue
            ee_position = self._sim.tree.forward_kinematics(ee_frame, refined).t
            if np.linalg.norm(ee_position - position) >= distance:
                continue
            if self._collision_fn(refined):
                continue
            return space.to_vector(refined)
        return None

    def _solve_down_facing_candidate(
        self,
        side: str,
        target: SE3,
        seed: Configuration,
        accept: Callable[[Configuration], bool],
    ) -> np.ndarray | None:
        """One constrained candidate: solve, then check ``accept`` and collision.

        A converged solve satisfies the tilt constraint and the soft limit margin
        by construction (see the solver setup in ``__init__``). Returns the arm
        joint vector, or None if the solve failed or a check rejected it.
        """
        solved = self._down_facing_iks[side].solve(target, seed)
        if solved is None:
            return None
        if not accept(solved):
            return None
        if self._collision_fn(solved):
            return None
        return self._arm_space(side).to_vector(solved)

    def _sample_down_facing_configuration(
        self,
        side: str,
        make_target: Callable[[np.ndarray], SE3],
        accept: Callable[[Configuration], bool],
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """A collision-free configuration whose end effector points down.

        Each candidate draws a yaw and a seed configuration uniformly, solves
        full-pose IK for the target that ``make_target`` builds from the down-facing
        rotation at that yaw, and keeps solutions that pass the candidate checks.
        The sim must already be at the state to sample around. Returns None when the
        budget is exhausted.
        """
        space = self._arm_space(side)
        lower, upper = space.bounds()
        base = dict(self._sim.configuration)
        for _ in range(DOWN_FACING_NUM_GOAL_CANDIDATES):
            target = make_target(_down_facing_rotation(rng.uniform(-np.pi, np.pi)))
            seed = base | space.to_configuration(rng.uniform(lower, upper))
            joints = self._solve_down_facing_candidate(side, target, seed, accept)
            if joints is not None:
                return joints
        return None

    def reset(self, x: ObjectCentricState, params: np.ndarray) -> None:
        self._current_state = x
        self._current_params = params
        self._motion_segments = self._create_motion_segments(params)
        self._current_plan = None
        self._final_command = self._create_final_command()
        self._final_command_issued = False

    def _create_motion_segments(
        self, params: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        """The arm motions for this skill, as (side, goal joints) in order."""
        raise NotImplementedError

    def _create_final_command(self) -> tuple[str, float]:
        """The (side, grasp command) toggle that ends this skill."""
        raise NotImplementedError

    def terminated(self) -> bool:
        return self._final_command_issued

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _assemble_action(
        self,
        moving_side: str | None,
        delta: np.ndarray | None,
        grasp_overrides: dict[str, float] | None = None,
    ) -> np.ndarray:
        """An action moving one arm (or none), holding every grasp as it is.

        By default each arm's grasp command re-asserts its current grasping state, so a
        held cube stays held and a free cube stays free; ``grasp_overrides`` changes
        individual commands.
        """
        state = self._current_state
        assert isinstance(state, VegaPickPlace3DObjectCentricState)
        action = np.zeros(2 * ARM_NUM_JOINTS + 2, dtype=np.float32)
        for i, side in enumerate(ARM_SIDES):
            if side == moving_side:
                assert delta is not None
                action[i * ARM_NUM_JOINTS : (i + 1) * ARM_NUM_JOINTS] = delta
            command = 1.0 if state.grasping(side) else -1.0
            if grasp_overrides and side in grasp_overrides:
                command = grasp_overrides[side]
            action[2 * ARM_NUM_JOINTS + i] = command
        return action

    def _plan_current_segment(self) -> list[np.ndarray]:
        """Plan the first pending motion segment from the current state."""
        state = self._current_state
        assert isinstance(state, VegaPickPlace3DObjectCentricState)
        self._sim.set_state(state)
        side, goal_joints = self._motion_segments[0]
        space = self._arm_space(side)
        start = self._sim.configuration
        goal = dict(start)
        goal.update(space.to_configuration(goal_joints))
        path = self._planners[side].plan(start, goal)
        if path is None:
            raise TrajectorySamplingFailure(f"Motion planning failed for {side} arm")

        # Densify so that consecutive waypoints are within one action of each other.
        # Planners return waypoints at whatever spacing search produced, which can
        # exceed what a single action can cover.
        max_step = self._sim.config.max_action_mag / 2
        vectors = [space.to_vector(config) for config in path]
        plan: list[np.ndarray] = []
        for previous, following in zip(vectors[:-1], vectors[1:], strict=True):
            plan.extend(space.interpolate(previous, following, max_step))
        return plan

    def step(self) -> np.ndarray:
        state = self._current_state
        assert isinstance(state, VegaPickPlace3DObjectCentricState)

        # Advance through the motion segments, planning each lazily when it starts.
        while self._motion_segments:
            if self._current_plan is None:
                self._current_plan = self._plan_current_segment()
            if not self._current_plan:
                self._motion_segments.pop(0)
                self._current_plan = None
                continue
            side, _ = self._motion_segments[0]
            target_joints = self._current_plan.pop(0)
            # Every Vega arm joint is bounded, so a plain difference is the correct
            # delta; no wrapping is possible on these joints.
            current_joints = np.asarray(state.arm_joint_positions(side))
            max_magnitude = self._sim.config.max_action_mag
            delta = np.clip(
                target_joints - current_joints, -max_magnitude, max_magnitude
            )
            return self._assemble_action(side, delta.astype(np.float32))

        # All motions done: issue the single grasp toggle.
        assert self._final_command is not None
        assert not self._final_command_issued
        side, command = self._final_command
        self._final_command_issued = True
        return self._assemble_action(None, None, grasp_overrides={side: command})


class GroundPickController(_VegaPickPlaceControllerBase):
    """Move one arm's end effector to the (free) cube and grasp it."""

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> np.ndarray:
        assert isinstance(x, VegaPickPlace3DObjectCentricState)
        self._sim.set_state(x)
        side = _side_of(self.objects[0])
        cube = np.asarray(x.cube_position)
        distance = GRASP_RADIUS_MARGIN * self._sim.config.grasp_radius
        if self._grasp_approach_max_tilt is None:
            joints = self._sample_ee_reach_configuration(
                side, cube, distance, rng, self._num_goal_candidates
            )
        else:
            # Aim above the cube for the same reason the refinement does: a
            # down-facing gripper at the cube center would collide with the table.
            target_position = cube + np.array([0.0, 0.0, REFINE_TARGET_Z_OFFSET])
            assert distance > REFINE_POSITION_TOLERANCE + REFINE_TARGET_Z_OFFSET
            ee_frame = self._sim.robot.manipulators[side].ee_frame

            def in_grasp_range(config: Configuration) -> bool:
                ee = self._sim.tree.forward_kinematics(ee_frame, config).t
                return bool(np.linalg.norm(ee - cube) < distance)

            joints = self._sample_down_facing_configuration(
                side,
                lambda rotation: SE3.Rt(rotation, target_position),
                in_grasp_range,
                rng,
            )
        if joints is None:
            num_samples = (
                self._num_goal_candidates
                if self._grasp_approach_max_tilt is None
                else DOWN_FACING_NUM_GOAL_CANDIDATES
            )
            raise TrajectorySamplingFailure(
                f"No grasp configuration found for the {side} arm at cube "
                f"{tuple(cube)} after {num_samples} samples"
            )
        return joints

    def _create_motion_segments(
        self, params: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        return [(_side_of(self.objects[0]), params)]

    def _create_final_command(self) -> tuple[str, float]:
        return (_side_of(self.objects[0]), 1.0)


class GroundPlaceController(_VegaPickPlaceControllerBase):
    """Carry the held cube over the target patch and release it."""

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> np.ndarray:
        assert isinstance(x, VegaPickPlace3DObjectCentricState)
        self._sim.set_state(x)
        side = _side_of(self.objects[0])
        assert x.holder == side, "Place requires this arm to hold the cube"
        target = np.asarray(x.target_position)
        half_x, half_y = self._sim.config.target_half_extents
        resting_z = self._sim.cube_resting_z

        def cube_accepted(cube: np.ndarray) -> bool:
            return bool(
                abs(cube[0] - target[0]) < PLACE_EXTENT_MARGIN * half_x
                and abs(cube[1] - target[1]) < PLACE_EXTENT_MARGIN * half_y
                and resting_z <= cube[2] <= resting_z + PLACE_MAX_RELEASE_HEIGHT
            )

        # The bounded release height makes the acceptance region a thin slab just
        # above the patch, so raw draws are refined like grasp reaches: aim the cube
        # (via the end effector, which carries it at a fixed offset) at the slab
        # center and re-check. The cube is attached to this arm's end effector, so its
        # position under a candidate configuration comes from forward kinematics of
        # its tree node.
        place_point = np.array(
            [target[0], target[1], resting_z + PLACE_MAX_RELEASE_HEIGHT / 2]
        )

        if self._grasp_approach_max_tilt is not None:
            ee_frame = self._sim.robot.manipulators[side].ee_frame
            base = dict(self._sim.configuration)
            ee_pose = self._sim.tree.forward_kinematics(ee_frame, base)
            held_cube = self._sim.tree.forward_kinematics(CUBE_NODE, base).t
            # The cube rides rigidly on the end effector, so its offset in the
            # end-effector frame is fixed; under a candidate rotation the end
            # effector must sit at the place point minus the rotated offset for the
            # cube to land there.
            offset = np.asarray(ee_pose.R).T @ (held_cube - ee_pose.t)

            def cube_in_slab(config: Configuration) -> bool:
                cube = self._sim.tree.forward_kinematics(CUBE_NODE, config).t
                return cube_accepted(cube)

            # Prefer the posture the arm already has (see the fan constants above),
            # so the transfer stays near the current configuration rather than
            # flipping to an independently sampled posture. Yaw is free for a cube,
            # so reusing or fanning out from the current yaw loses nothing. Targets
            # use the exactly-vertical rotation at each candidate yaw, not the
            # current rotation, so the converged tilt stays within the bound even
            # though the held grasp may itself be tilted. When the held grasp is
            # not down-facing at all (an unconstrained pick or a handover), the
            # seeded fan usually fails and the resampling fallback takes over.
            rotation_now = np.asarray(ee_pose.R)
            yaw_now = float(np.arctan2(rotation_now[1, 0], rotation_now[0, 0]))
            yaw_offsets = [0.0]
            for k in range(1, PLACE_POSTURE_SEED_ATTEMPTS // 2 + 1):
                step = 2 * np.pi * k / (PLACE_POSTURE_SEED_ATTEMPTS + 1)
                yaw_offsets.extend([step, -step])
            arm_joints_now = self._arm_space(side).to_vector(base)
            best: np.ndarray | None = None
            best_delta = np.inf
            for yaw_offset in yaw_offsets:
                rotation = _down_facing_rotation(yaw_now + yaw_offset)
                joints = self._solve_down_facing_candidate(
                    side,
                    SE3.Rt(rotation, place_point - rotation @ offset),
                    base,
                    cube_in_slab,
                )
                if joints is None:
                    continue
                delta = float(np.max(np.abs(joints - arm_joints_now)))
                if delta < PLACE_POSTURE_ACCEPT_DELTA:
                    return joints
                if delta < best_delta:
                    best, best_delta = joints, delta
            if best is not None:
                return best

            joints = self._sample_down_facing_configuration(
                side,
                lambda rotation: SE3.Rt(rotation, place_point - rotation @ offset),
                cube_in_slab,
                rng,
            )
            if joints is not None:
                return joints
            raise TrajectorySamplingFailure(
                f"No place configuration found for the {side} arm over target "
                f"{tuple(target)} after {DOWN_FACING_NUM_GOAL_CANDIDATES} "
                "down-facing samples"
            )

        space = self._arm_space(side)
        lower, upper = space.bounds()
        base = dict(self._sim.configuration)
        ee_frame = self._sim.robot.manipulators[side].ee_frame
        refiner = self._ik_refiners[side]
        for _ in range(self._num_goal_candidates):
            joints = rng.uniform(lower, upper)
            config = base | space.to_configuration(joints)
            cube = self._sim.tree.forward_kinematics(CUBE_NODE, config).t
            if np.linalg.norm(cube - place_point) >= LOOSE_REACH_RADIUS:
                continue
            if cube_accepted(cube) and not self._collision_fn(config):
                return joints
            # The end-effector target that carries the cube to the place point,
            # assuming the refinement preserves the draw's orientation.
            ee_pose = self._sim.tree.forward_kinematics(ee_frame, config)
            refined = refiner.solve(
                SE3.Rt(ee_pose.R, place_point - (cube - ee_pose.t)), config
            )
            if refined is None:
                continue
            refined_cube = self._sim.tree.forward_kinematics(CUBE_NODE, refined).t
            if not cube_accepted(refined_cube):
                continue
            if self._collision_fn(refined):
                continue
            return space.to_vector(refined)
        raise TrajectorySamplingFailure(
            f"No place configuration found for the {side} arm over target "
            f"{tuple(target)} after {self._num_goal_candidates} samples"
        )

    def _create_motion_segments(
        self, params: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        return [(_side_of(self.objects[0]), params)]

    def _create_final_command(self) -> tuple[str, float]:
        return (_side_of(self.objects[0]), -1.0)


class GroundHandoverController(_VegaPickPlaceControllerBase):
    """Carry the cube to a shared region, then take it with the other arm.

    Parameters are the goal configurations of both arms, holding arm first. The holding
    arm moves first, carrying the cube; the receiving arm then reaches to the carried
    cube and the final action grasps it, which re-parents the cube onto the receiving
    arm.
    """

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> np.ndarray:
        assert isinstance(x, VegaPickPlace3DObjectCentricState)
        giver, receiver = _side_of(self.objects[0]), _side_of(self.objects[1])
        assert x.holder == giver, "Handover requires the first arm to hold the cube"
        self._sim.set_state(x)
        config = self._sim.config
        low = np.array(
            [
                config.sample_x_bounds[0],
                HANDOVER_Y_BOUNDS[0],
                config.table_height + HANDOVER_Z_ABOVE_TABLE_BOUNDS[0],
            ]
        )
        high = np.array(
            [
                config.sample_x_bounds[1],
                HANDOVER_Y_BOUNDS[1],
                config.table_height + HANDOVER_Z_ABOVE_TABLE_BOUNDS[1],
            ]
        )
        cube_frame = CUBE_NODE
        grasp_distance = GRASP_RADIUS_MARGIN * config.grasp_radius
        carry_budget = self._num_goal_candidates // HANDOVER_NUM_CARRY_ATTEMPTS
        base = dict(self._sim.configuration)
        giver_space = self._arm_space(giver)

        def carry_accept(candidate: Configuration) -> bool:
            cube = self._sim.tree.forward_kinematics(cube_frame, candidate).t
            return bool(np.all(cube >= low) and np.all(cube <= high))

        for _ in range(HANDOVER_NUM_CARRY_ATTEMPTS):
            giver_joints = self._sample_arm_configuration(
                giver, carry_accept, rng, carry_budget
            )
            if giver_joints is None:
                continue
            # Where the cube ends up under the carry configuration; the receiving
            # arm must reach within grasp range of this point while the giver is
            # at that configuration (so the arms cannot collide at the handover).
            carry_config = base | giver_space.to_configuration(giver_joints)
            cube = self._sim.tree.forward_kinematics(cube_frame, carry_config).t
            self._sim.set_arm_joint_positions(giver, giver_joints)
            receiver_joints = self._sample_ee_reach_configuration(
                receiver, cube, grasp_distance, rng, self._num_goal_candidates
            )
            self._sim.set_state(x)  # restore after moving the giver arm
            if receiver_joints is not None:
                return np.concatenate([giver_joints, receiver_joints])
        raise TrajectorySamplingFailure(
            f"No handover configurations found from the {giver} arm to the "
            f"{receiver} arm after {HANDOVER_NUM_CARRY_ATTEMPTS} carry attempts"
        )

    def _create_motion_segments(
        self, params: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        giver, receiver = _side_of(self.objects[0]), _side_of(self.objects[1])
        return [
            (giver, params[:ARM_NUM_JOINTS]),
            (receiver, params[ARM_NUM_JOINTS:]),
        ]

    def _create_final_command(self) -> tuple[str, float]:
        # The receiving arm requests a grasp; the environment re-parents the cube from
        # the holder to the requesting arm when it is within grasp range.
        return (_side_of(self.objects[1]), 1.0)


def create_lifted_controllers(
    action_space: BimanualArmJointDeltaGraspActionSpace,
    sim: ObjectCentricVegaPickPlace3DEnv,
    rng: np.random.Generator | None = None,
    prefer_ompl: bool = True,
    grasp_approach_max_tilt: float | None = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for VegaPickPlace3D.

    ``grasp_approach_max_tilt`` bounds how far (in radians) the end effector may point
    off straight down in sampled pick and place configurations, with the yaw free;
    None (the default) leaves grasp orientations unconstrained. Handover sampling is
    never constrained: the take-over happens in mid-air, where approach direction
    does not matter.
    """
    del action_space  # the action space is implied by the environment

    if rng is None:
        rng = np.random.default_rng(0)
    collision_fn = create_collision_fn(sim)
    planners = {
        side: create_motion_planner(
            sim.robot.groups[sim.robot.manipulators[side].group],
            collision_fn,
            rng,
            prefer_ompl=prefer_ompl,
        )
        for side in ARM_SIDES
    }

    class PickController(GroundPickController):
        """Pick up the cube with one arm."""

        def __init__(self, objects):
            super().__init__(
                objects,
                sim,
                planners,
                collision_fn,
                grasp_approach_max_tilt=grasp_approach_max_tilt,
            )

    class PlaceController(GroundPlaceController):
        """Place the held cube onto the target patch."""

        def __init__(self, objects):
            super().__init__(
                objects,
                sim,
                planners,
                collision_fn,
                grasp_approach_max_tilt=grasp_approach_max_tilt,
            )

    class HandoverController(GroundHandoverController):
        """Pass the cube from one arm to the other."""

        def __init__(self, objects):
            super().__init__(objects, sim, planners, collision_fn)

    # Create variables for lifted controllers.
    arm = Variable("?arm", Kinematic3Dv2GraspArmRobotType)
    giver = Variable("?giver", Kinematic3Dv2GraspArmRobotType)
    receiver = Variable("?receiver", Kinematic3Dv2GraspArmRobotType)
    cube = Variable("?cube", Kinematic3Dv2PointType)
    target = Variable("?target", Kinematic3Dv2PointType)

    pick_controller: LiftedParameterizedController = LiftedParameterizedController(
        [arm, cube],
        PickController,
        Box(-np.inf, np.inf, (ARM_NUM_JOINTS,)),
    )
    place_controller: LiftedParameterizedController = LiftedParameterizedController(
        [arm, cube, target],
        PlaceController,
        Box(-np.inf, np.inf, (ARM_NUM_JOINTS,)),
    )
    handover_controller: LiftedParameterizedController = LiftedParameterizedController(
        [giver, receiver, cube],
        HandoverController,
        Box(-np.inf, np.inf, (2 * ARM_NUM_JOINTS,)),
    )
    return {
        "pick": pick_controller,
        "place": place_controller,
        "handover": handover_controller,
    }
