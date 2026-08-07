"""PDDLStream stream implementations for LimbRepositioning3D.

See run.py for the domain design.

Streams scratch-mutate the live simulator and restore it via `_saved_sim_state`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pybullet as p
from kinder.envs.kinematic3d.limbrepositioning3d import (
    ObjectCentricLimbRepositioning3DEnv,
)
from kinder.envs.kinematic3d.utils import NUM_LIMB_JOINTS, NUM_ROBOT_JOINTS
from pybullet_helpers.geometry import Pose, SE2Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, JointVelocities
from pybullet_helpers.motion_planning import (
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
)

JointTorques = list[float]

# The grid `sample_base_pose` searches, as (dx, dy, drot) offsets from the variant's
# own placement, searched nearest-first.
BASE_SEARCH_RADIUS = 0.4
BASE_SEARCH_STEP = 0.1
BASE_SEARCH_ROTATIONS: tuple[float, ...] = (0.0, -0.2, 0.2, -0.4, 0.4)

NUM_REACH_CHECKS = 5

CLEARANCE_PROBE_DISTANCE = 0.1

COLLISION_MARGIN = 0.01


@dataclass(eq=False)
class LimbConf:
    """A configuration of the passive limb.

    `eq=False` because pddlstream identifies stream outputs by identity.
    """

    positions: tuple[float, ...]

    def __repr__(self) -> str:
        return f"qL{id(self) % 10000}"


@dataclass(eq=False)
class CoupledState:
    """A full coupled state of the robot and the limb.

    The base pose is included, since the weld is rebuilt from the live poses.
    """

    base_pose: SE2Pose
    robot_positions: JointPositions
    robot_velocities: JointVelocities
    limb_positions: JointPositions
    limb_velocities: JointVelocities
    approach: ArmTrajectory | None = None

    def __repr__(self) -> str:
        return f"s{id(self) % 10000}"


@dataclass(eq=False)
class ArmTrajectory:
    """A joint-space path that takes the arm from its retracted conf onto the grasp."""

    joint_plan: list[JointPositions]

    def __repr__(self) -> str:
        return f"at{id(self) % 10000}(waypoints={len(self.joint_plan)})"


@dataclass(eq=False)
class TorqueTrajectory:
    """An open-loop torque trajectory, one torque per environment step."""

    torques: list[JointTorques]

    def __repr__(self) -> str:
        return f"tt{id(self) % 10000}(steps={len(self.torques)})"


@dataclass
class BestAttempt:
    """The closest a rejected limb motion got, kept to visualize failures.

    `plan_limb_motion` otherwise discards every rollout that does not arrive.
    """

    state: CoupledState
    torques: list[JointTorques]
    error: float
    reason: str


@dataclass
class MPCConfig:
    """Predictive-sampling MPC hyperparameters.

    Mirrors limb-manipulation's `PredictiveSamplingPlanner`.

    Rollouts are PyBullet steps in the environment itself, not in a surrogate.
    """

    num_rollouts: int = 24
    horizon: int = 12
    action_repeat: int = 12
    num_control_points: int = 4
    noise_scale: float = 0.05
    noise_reference_error: float = 0.5
    min_noise_ratio: float = 0.1
    goal_reaching_weight: float = 100.0
    velocity_penalty_weight: float = 0.5
    velocity_penalty_threshold: float = 0.05
    max_velocity: float = 2.0
    max_velocity_penalty: float = 300.0
    joint_limit_violation_weight: float = 1e6
    collision_penalty: float = 1e4
    terminal_goal_weight: float = 100.0
    velocity_regularization_weight: float = 1.0
    replay_slack: float = 1e-2
    max_control_steps: int = 300
    divergence_factor: float = 2.0
    stall_patience: int = 60
    stall_tolerance: float = 1e-3


@dataclass
class LimbStreamContext:
    """Shared context for all streams of one `create_problem` call."""

    sim: ObjectCentricLimbRepositioning3DEnv
    start_base_pose: SE2Pose
    grasp_base_pose: SE2Pose
    # The arm's retracted configuration.
    retract_joints: JointPositions
    limb_name: str = "limb"
    motion_seed: int = 0
    mpc: MPCConfig = field(default_factory=MPCConfig)
    check_base_collisions: bool = True
    check_robot_collisions: bool = True
    num_ik_attempts: int = 12
    best_attempt: BestAttempt | None = None
    resting_penetration: dict[int, float] = field(default_factory=dict, repr=False)
    _ik_rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ik_rng = np.random.default_rng(self.motion_seed)
        self.refresh_collision_baseline()

    def refresh_collision_baseline(self) -> None:
        """Record how far the grasped limb overlaps each body it starts out touching.

        Call this only in the start state, as `reset_to_start` does.
        """
        self.resting_penetration = {
            body_id: min(
                0.0, _pair_clearance(self.sim, self.sim.limb.robot_id, body_id)
            )
            for body_id in self.human_collision_ids + self.scene_collision_ids
        }

    @property
    def scene_collision_ids(self) -> list[int]:
        """Furniture the robot must not drive into.

        This is the environment's own list, empty for the isolated and human scenes.
        """
        return list(self.sim.scene.get_scene_collision_ids())

    @property
    def human_collision_ids(self) -> list[int]:
        """The torso, and every limb except the one being moved.

        The environment models none of this, so the arm may be driven through it.

        The limb being repositioned is a target rather than an obstacle.
        """
        scene = self.sim.scene
        ids = [
            limb.robot_id
            for limb in getattr(scene, "limbs", {}).values()
            if limb.robot_id != self.sim.limb.robot_id
        ]
        torso_id = getattr(scene, "torso_id", None)
        if torso_id is not None:
            ids.append(torso_id)
        return ids

    @property
    def obstacle_ids(self) -> list[int]:
        """Everything the robot must keep out of: the person, and the furniture."""
        if not self.check_robot_collisions:
            return []
        return self.human_collision_ids + self.scene_collision_ids

    @property
    def base_obstacle_ids(self) -> list[int]:
        """What the mobile base must not be parked in.

        Kept separate from `check_robot_collisions`, which governs the arm.
        """
        return self.human_collision_ids + self.scene_collision_ids

    @property
    def goal_atol(self) -> float:
        """The joint-space distance within which the environment declares success."""
        return self.sim.config.goal_atol

    @property
    def torque_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """The robot's (lower, upper) torque limits."""
        config = self.sim.config
        return (
            np.asarray(config.torque_lower_limits, dtype=np.float64),
            np.asarray(config.torque_upper_limits, dtype=np.float64),
        )


# ------------------------------------------------------------------- simulator helpers


def extend_with_fingers(joints: JointPositions) -> JointPositions:
    """Pad arm joint positions with the six Robotiq finger joints."""
    return list(joints) + [0.0] * 6


def is_grasping(sim: ObjectCentricLimbRepositioning3DEnv) -> bool:
    """Whether the end effector is currently welded to the limb.

    The environment never tears its constraint down, so it types this as `int`.
    """
    # pylint: disable=protected-access
    constraint_id: int | None = sim._grasp_constraint_id
    return constraint_id is not None


def release_grasp(sim: ObjectCentricLimbRepositioning3DEnv) -> None:
    """Remove the weld between the end effector and the limb, if there is one."""
    if not is_grasping(sim):
        return
    p.removeConstraint(
        sim._grasp_constraint_id,  # pylint: disable=protected-access
        physicsClientId=sim.physics_client_id,
    )
    # pylint: disable=protected-access
    sim._grasp_constraint_id = None  # type: ignore[assignment]


def engage_grasp(sim: ObjectCentricLimbRepositioning3DEnv) -> None:
    """(Re)weld the end effector to the limb wherever the two currently are.

    The constraint is built from the live poses, so it starts with zero error.
    """
    release_grasp(sim)
    sim._grasp_constraint_id = (  # pylint: disable=protected-access
        sim._create_grasp_constraint()  # pylint: disable=protected-access
    )


def capture_state(sim: ObjectCentricLimbRepositioning3DEnv) -> CoupledState:
    """Snapshot the full coupled state of the simulator."""
    return CoupledState(
        base_pose=sim.robot.get_base(),
        robot_positions=list(sim.robot.arm.get_joint_positions()),
        robot_velocities=list(sim.robot.arm.get_joint_velocities()),
        limb_positions=list(sim.limb.get_joint_positions()),
        limb_velocities=list(sim.limb.get_joint_velocities()),
    )


def restore_state(
    sim: ObjectCentricLimbRepositioning3DEnv,
    state: CoupledState,
    regrasp: bool = True,
) -> None:
    """Put the simulator back into `state`.

    `regrasp` rebuilds the weld, whose stored frames are part of the restore point.
    """
    current_base = sim.robot.get_base()
    if not np.allclose(
        (current_base.x, current_base.y, current_base.rot),
        (state.base_pose.x, state.base_pose.y, state.base_pose.rot),
    ):
        release_grasp(sim)
        sim.robot.set_base(state.base_pose)
    sim.robot.arm.set_joints(
        list(state.robot_positions), joint_velocities=list(state.robot_velocities)
    )
    sim.limb.set_joints(
        list(state.limb_positions), joint_velocities=list(state.limb_velocities)
    )
    if regrasp:
        engage_grasp(sim)


@contextmanager
def _saved_sim_state(sim: ObjectCentricLimbRepositioning3DEnv) -> Iterator[None]:
    """Restore the simulator (including its grasp constraint) on the way out."""
    saved = capture_state(sim)
    was_grasping = is_grasping(sim)
    try:
        yield
    finally:
        if p.isConnected(physicsClientId=sim.physics_client_id):
            restore_state(sim, saved, regrasp=was_grasping)


def _limb_error(positions: JointPositions, goal: np.ndarray) -> float:
    """The environment's own success metric: Euclidean joint-space distance."""
    return float(np.linalg.norm(np.subtract(positions, goal)))


def advance(
    sim: ObjectCentricLimbRepositioning3DEnv, torque: JointTorques | np.ndarray
) -> None:
    """Advance the simulation by one environment step under `torque`.

    This is `sim.step()` without its observation construction or spasm sampling.
    """
    action_space = sim.torque_action_space
    clipped = np.clip(torque, action_space.low, action_space.high)
    sim._apply_torques(  # pylint: disable=protected-access
        list(clipped), sim.limb.get_muscle_tone_torque()
    )


# ------------------------------------------------------------------------------ streams


def sample_grasp(ctx: LimbStreamContext, limb: str) -> Iterator[tuple[Pose]]:
    """Yield the fixed grasp transform for this limb.

    The scene config carries one transform per limb family, so this is single-shot.

    It stays a stream so sampled grasps can be added without touching the domain.
    """
    del limb  # there is a single limb per environment
    yield (ctx.sim.scene.grasp_transform,)


def _wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi], which is what `SE2Pose` asserts its rotation lies in."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _base_candidates(ctx: LimbStreamContext) -> list[SE2Pose]:
    """Base poses to try, ordered outward from the variant's own placement."""
    nominal = ctx.grasp_base_pose
    steps = np.arange(-BASE_SEARCH_RADIUS, BASE_SEARCH_RADIUS + 1e-9, BASE_SEARCH_STEP)
    offsets = [
        (float(dx), float(dy), rot)
        for dx in steps
        for dy in steps
        for rot in BASE_SEARCH_ROTATIONS
    ]
    offsets.sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, abs(o[2])))
    return [
        SE2Pose(nominal.x + dx, nominal.y + dy, _wrap_angle(nominal.rot + rot))
        for dx, dy, rot in offsets
    ]


def _pair_clearance(sim, body_a: int, body_b: int) -> float:
    """The gap between two bodies in meters, negative when they interpenetrate.

    Bodies beyond `CLEARANCE_PROBE_DISTANCE` report that distance instead.
    """
    points = p.getClosestPoints(
        body_a,
        body_b,
        distance=CLEARANCE_PROBE_DISTANCE,
        physicsClientId=sim.physics_client_id,
    )
    if not points:
        return CLEARANCE_PROBE_DISTANCE
    return min(point[8] for point in points)


def arm_in_collision(
    ctx: LimbStreamContext, joints: JointPositions | None = None
) -> bool:
    """Whether the arm overlaps the person or the furniture.

    With `joints` the arm is moved there first, and with None it is left alone.

    The MPC needs the latter, since `set_joints` would clear its velocities.
    """
    obstacles = ctx.obstacle_ids
    if not obstacles:
        return False
    sim = ctx.sim
    if joints is not None:
        sim.robot.arm.set_joints(list(joints))
    return any(
        check_body_collisions(
            sim.robot.arm.robot_id, obstacle_id, sim.physics_client_id
        )
        for obstacle_id in obstacles
    )


def human_in_collision(ctx: LimbStreamContext) -> bool:
    """Whether the grasped limb was driven into the person or the furniture.

    Penetration is measured against `resting_penetration` rather than zero.
    """
    sim = ctx.sim
    for body_id, resting in ctx.resting_penetration.items():
        clearance = _pair_clearance(sim, sim.limb.robot_id, body_id)
        if clearance < resting - COLLISION_MARGIN:
            return True
    return False


def _random_arm_seed(ctx: LimbStreamContext) -> JointPositions:
    """A random arm configuration to restart IK from, fingers left closed.

    The coupled Robotiq finger joints would trip an assertion if randomized.
    """
    arm = ctx.sim.robot.arm
    lower = np.clip(
        np.asarray(arm.joint_lower_limits[:NUM_ROBOT_JOINTS]), -np.pi, np.pi
    )
    upper = np.clip(
        np.asarray(arm.joint_upper_limits[:NUM_ROBOT_JOINTS]), -np.pi, np.pi
    )
    return extend_with_fingers(
        ctx._ik_rng.uniform(lower, upper).tolist()  # pylint: disable=protected-access
    )


def _grasp_ik(
    ctx: LimbStreamContext,
    grasp: Pose,
    limb_positions: tuple[float, ...],
    seed_joints: JointPositions,
) -> JointPositions | None:
    """Collision-free arm joints holding `grasp` at `limb_positions`.

    Returns None if no configuration both reaches the grasp and clears the person.

    IK alone returns solutions with the forearm buried in the torso.
    """
    sim = ctx.sim
    sim.limb.set_joints(list(limb_positions), joint_velocities=[0.0] * NUM_LIMB_JOINTS)
    target_ee_pose = multiply_poses(sim.limb.get_end_effector_pose(), grasp)
    for attempt in range(max(ctx.num_ik_attempts, 1)):
        seed = list(seed_joints) if attempt == 0 else _random_arm_seed(ctx)
        sim.robot.arm.set_joints(seed)
        try:
            solution = inverse_kinematics(
                sim.robot.arm, target_ee_pose, validate=True, set_joints=False
            )
        except InverseKinematicsError:
            continue
        if not arm_in_collision(ctx, solution):
            return solution
    return None


def sample_base_pose(
    ctx: LimbStreamContext,
    limb: str,
    grasp: Pose,
    init_conf: LimbConf,
    goal_conf: LimbConf,
) -> Iterator[tuple[SE2Pose]]:
    """Yield base poses holding the grasp at both configurations.

    A rigid grasp ties the limb configuration to the end-effector pose.

    Requiring IK at the goal too is what makes `move_base` a real decision.
    """
    del limb  # there is a single limb per environment
    sim = ctx.sim

    waypoints = np.linspace(
        np.asarray(init_conf.positions),
        np.asarray(goal_conf.positions),
        NUM_REACH_CHECKS,
    )
    with _saved_sim_state(sim):
        release_grasp(sim)
        for base_conf in _base_candidates(ctx):
            sim.robot.set_base(base_conf)
            # Seed each IK from the previous solution to stay on one continuous branch,
            # since the arm cannot teleport between branches while welded to the limb.
            seed: JointPositions = ctx.retract_joints
            reachable = True
            for waypoint in waypoints:
                solution = _grasp_ik(ctx, grasp, tuple(waypoint), seed)
                if solution is None:
                    reachable = False
                    break
                seed = solution
            if not reachable:
                continue
            if ctx.check_base_collisions and _base_in_collision(ctx):
                continue
            yield (base_conf,)


def _base_in_collision(ctx: LimbStreamContext) -> bool:
    """Whether the base, where it stands, overlaps the person or the furniture.

    Only the base is checked here; the arm is handled by `arm_in_collision`.
    """
    sim = ctx.sim
    return any(
        check_body_collisions(
            sim.robot.base.robot_id, obstacle_id, sim.physics_client_id
        )
        for obstacle_id in ctx.base_obstacle_ids
    )


def plan_grasp_motion(
    ctx: LimbStreamContext,
    limb: str,
    grasp: Pose,
    base_conf: SE2Pose,
    limb_conf: LimbConf,
) -> Iterator[tuple[ArmTrajectory, CoupledState]]:
    """Find an arm path onto the grasp, and the coupled state it leaves behind.

    The path runs from the retracted conf to the limb's grasp frame.
    """
    del limb  # there is a single limb per environment
    sim = ctx.sim
    with _saved_sim_state(sim):
        release_grasp(sim)
        sim.robot.set_base(base_conf)
        grasp_joints = _grasp_ik(ctx, grasp, limb_conf.positions, ctx.retract_joints)
        if grasp_joints is None:
            return

        joint_plan = run_motion_planning(
            sim.robot.arm,
            initial_positions=list(ctx.retract_joints),
            target_positions=grasp_joints,
            collision_bodies=ctx.obstacle_ids,
            seed=ctx.motion_seed,
            physics_client_id=sim.physics_client_id,
        )
        if joint_plan is None:
            return
        trajectory = ArmTrajectory(list(joint_plan))
        state = CoupledState(
            base_pose=base_conf,
            robot_positions=list(joint_plan[-1]),
            robot_velocities=[0.0] * len(joint_plan[-1]),
            limb_positions=list(limb_conf.positions),
            limb_velocities=[0.0] * NUM_LIMB_JOINTS,
            approach=trajectory,
        )
    yield (trajectory, state)


def plan_base_motion(
    ctx: LimbStreamContext, q1: SE2Pose, q2: SE2Pose
) -> Iterator[tuple[list[SE2Pose]]]:
    """Plan a collision-free path for the mobile base from `q1` to `q2`."""
    sim = ctx.sim
    with _saved_sim_state(sim):
        # The base is only ever driven with the hand empty (see the domain)
        release_grasp(sim)
        sim.robot.arm.set_joints(list(ctx.retract_joints))
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot,
            q1,
            q2,
            collision_bodies=ctx.obstacle_ids or ctx.scene_collision_ids,
            seed=ctx.motion_seed,
        )
    if base_plan is None:
        return
    yield (list(base_plan),)


def plan_limb_motion(
    ctx: LimbStreamContext, s1: CoupledState, q2: LimbConf
) -> Iterator[tuple[TorqueTrajectory, CoupledState]]:
    """Generate a torque trajectory from state `s1` to limb configuration `q2`.

    Runs predictive-sampling MPC closed-loop, recording the torques it applies.

    It stops with enough headroom that an open-loop replay also lands in tolerance.
    """
    sim = ctx.sim
    goal = np.asarray(q2.positions, dtype=np.float64)
    mpc = PredictiveSamplingMPC(ctx, goal)

    cfg = ctx.mpc
    threshold = max(ctx.goal_atol - cfg.replay_slack, ctx.goal_atol / 2)
    with _saved_sim_state(sim):
        restore_state(sim, s1)
        torques: list[JointTorques] = []
        reached = False
        giveup = ""
        error = _limb_error(sim.limb.get_joint_positions(), goal)
        initial_error = error
        best_error = error
        steps_since_improvement = 0
        for _ in range(cfg.max_control_steps):
            if error < threshold:
                reached = True
                break
            if error > cfg.divergence_factor * initial_error:
                giveup = f"diverged (error {error:.3f}, started at {initial_error:.3f})"
                break
            if steps_since_improvement >= cfg.stall_patience:
                giveup = f"stalled at an error of {best_error:.3f}"
                break
            torque = mpc.step()
            for _ in range(cfg.action_repeat):
                advance(sim, torque)
                torques.append(list(torque))
                error = _limb_error(sim.limb.get_joint_positions(), goal)
                if error < threshold:
                    reached = True
                    break
            if ctx.resting_penetration and human_in_collision(ctx):
                reached = False
                giveup = "drove the limb into the person or the furniture"
                break
            if best_error - error > cfg.stall_tolerance:
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            best_error = min(best_error, error)
            if reached:
                break
        else:
            giveup = f"ran out of control steps at an error of {error:.3f}"

        # The MPC ran closed-loop, but execution replays the torques open-loop, so
        # certify the state that replay produces rather than the closed-loop one.
        s2: CoupledState | None = None
        if reached:
            restore_state(sim, s1)
            for torque in torques:
                advance(sim, torque)
            s2 = capture_state(sim)
            replay_error = _limb_error(s2.limb_positions, goal)
            if replay_error >= ctx.goal_atol:
                reached = False
                giveup = (
                    f"replayed open-loop to {replay_error:.3f}, outside the tolerance"
                )
            elif ctx.resting_penetration and human_in_collision(ctx):
                # The replay drifts from the closed-loop run, so check it separately.
                reached = False
                giveup = "replayed open-loop into the person or the furniture"

    if not reached:
        _record_attempt(ctx, s1, torques, best_error, giveup)
        print(f"plan-limb-motion: MPC rollout {giveup}.")
        return
    assert s2 is not None
    if not torques:
        print("plan-limb-motion: already at the target configuration.")
    else:
        print(f"plan-limb-motion: generated a trajectory with {len(torques)} steps.")
    yield (TorqueTrajectory(torques), s2)


def _record_attempt(
    ctx: LimbStreamContext,
    s1: CoupledState,
    torques: list[JointTorques],
    error: float,
    reason: str,
) -> None:
    """Keep this rejected rollout if it is the closest one seen so far."""
    if not torques:
        return
    if ctx.best_attempt is not None and ctx.best_attempt.error <= error:
        return
    ctx.best_attempt = BestAttempt(
        state=s1, torques=list(torques), error=error, reason=reason
    )


def check_torque_limits(
    ctx: LimbStreamContext,
    s1: CoupledState,
    q2: LimbConf,
    trajectory: TorqueTrajectory,
    s2: CoupledState,
) -> bool:
    """Constraint: every torque in the trajectory respects the robot's torque limits."""
    del s1, q2, s2  # the constraint is on the trajectory alone
    lower, upper = ctx.torque_limits
    tolerance = 1e-9
    for torque in trajectory.torques:
        torque_arr = np.asarray(torque, dtype=np.float64)
        if (torque_arr < lower - tolerance).any() or (
            torque_arr > upper + tolerance
        ).any():
            print("check-torque-limits: torque limit violated.")
            return False
    return True


# ---------------------------------------------------------------------------------- MPC


class PredictiveSamplingMPC:
    """Predictive-sampling MPC over robot joint torques.

    A port of limb-manipulation's `PredictiveSamplingPlanner` to KinDER.

    The dynamics model is the PyBullet environment, snapshotted and restored.
    """

    def __init__(self, ctx: LimbStreamContext, goal: np.ndarray) -> None:
        self._ctx = ctx
        self._sim = ctx.sim
        self._cfg = ctx.mpc
        self._obstacles = ctx.obstacle_ids
        self._goal = goal
        self._rng = np.random.default_rng(ctx.motion_seed)
        self._lower, self._upper = ctx.torque_limits
        self._nominal = np.zeros((self._cfg.num_control_points, NUM_ROBOT_JOINTS))
        self._control_indices = np.round(
            np.linspace(0, self._cfg.horizon - 1, self._cfg.num_control_points)
        ).astype(int)

    def step(self) -> JointTorques:
        """Choose the torque to apply in the simulator's current state."""
        error = _limb_error(self._sim.limb.get_joint_positions(), self._goal)
        candidates = self._sample_candidates(error)
        start = capture_state(self._sim)
        scores = [self._score(start, self._expand(cp)) for cp in candidates]
        restore_state(self._sim, start)
        best = candidates[int(np.argmin(scores))]
        self._nominal = best
        return list(self._expand(best)[0])

    def _expand(self, control_points: np.ndarray) -> np.ndarray:
        """Linearly interpolate control points into one torque per control step."""
        source = np.linspace(0.0, 1.0, len(control_points))
        target = np.linspace(0.0, 1.0, self._cfg.horizon)
        return np.stack(
            [
                np.interp(target, source, control_points[:, j])
                for j in range(NUM_ROBOT_JOINTS)
            ],
            axis=1,
        )

    def _sample_candidates(self, error: float) -> list[np.ndarray]:
        """Warm-started nominal, a zero-torque sequence, and noisy variations.

        Each candidate is a set of control points rather than a torque sequence.
        """
        cfg = self._cfg
        # Shift the previous solution forward one control step, repeating its last
        # torque, and re-fit control points to the result; then sample around it.
        expanded = self._expand(self._nominal)
        shifted = np.vstack([expanded[1:], expanded[-1:]])
        nominal = shifted[self._control_indices]
        candidates = [self._clip(nominal), np.zeros_like(nominal)]
        num_samples = max(cfg.num_rollouts - len(candidates), 0)
        scale = cfg.noise_scale * float(
            np.clip(error / cfg.noise_reference_error, cfg.min_noise_ratio, 1.0)
        )
        noise = self._rng.normal(
            scale=scale,
            size=(num_samples, cfg.num_control_points, NUM_ROBOT_JOINTS),
        )
        candidates.extend(self._clip(nominal + sample) for sample in noise)
        return candidates

    def _clip(self, torques: np.ndarray) -> np.ndarray:
        return np.clip(torques, self._lower, self._upper)

    def _score(self, start: CoupledState, torques: np.ndarray) -> float:
        """Roll `torques` out from `start` in the simulator and score the result."""
        restore_state(self._sim, start)
        cfg = self._cfg
        goal_cost = 0.0
        velocity_cost = 0.0
        regularization_cost = 0.0
        limit_cost = 0.0
        squared_distance = 0.0
        for torque in torques:
            for _ in range(cfg.action_repeat):
                advance(self._sim, torque)
            positions = self._sim.limb.get_joint_positions()
            velocities = self._sim.limb.get_joint_velocities()
            squared_distance = float(
                np.sum(np.square(np.subtract(positions, self._goal)))
            )
            goal_cost += squared_distance
            velocity_cost += self._velocity_penalty(velocities, squared_distance)
            regularization_cost += float(np.sum(np.square(velocities)))
            if not self._sim.limb.check_joint_limits(positions):
                limit_cost += cfg.joint_limit_violation_weight
            if self._obstacles and arm_in_collision(self._ctx):
                limit_cost += cfg.collision_penalty
            if self._ctx.resting_penetration and human_in_collision(self._ctx):
                limit_cost += cfg.collision_penalty
        return (
            cfg.goal_reaching_weight * goal_cost
            + cfg.terminal_goal_weight * squared_distance
            + cfg.velocity_penalty_weight * velocity_cost
            + cfg.velocity_regularization_weight * regularization_cost
            + limit_cost
        )

    def _velocity_penalty(
        self, velocities: JointVelocities, squared_distance: float
    ) -> float:
        """Damp the limb near the goal, and cap its speed at `max_velocity`.

        Without these the MPC blows through the goal band and whips the limb around.
        """
        cfg = self._cfg
        magnitude = float(np.linalg.norm(velocities))
        penalty = 0.0
        if squared_distance < cfg.velocity_penalty_threshold:
            penalty += magnitude * float(
                np.exp(-squared_distance / cfg.velocity_penalty_threshold)
            )
        if magnitude > cfg.max_velocity:
            penalty += cfg.max_velocity_penalty
        return penalty
