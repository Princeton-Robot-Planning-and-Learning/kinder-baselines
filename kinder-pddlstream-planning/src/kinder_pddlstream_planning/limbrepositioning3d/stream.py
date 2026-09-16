"""PDDLStream stream implementations for LimbRepositioning3D.

See run.py for the domain design, and utils.py for the simulator helpers these compose.

Streams scratch-mutate the live simulator and restore it via `saved_sim_state`.
"""

from __future__ import annotations

import itertools
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS, NUM_ROBOT_JOINTS
from kinder.envs.dynamic3d.limbrepositioning3d import (
    ObjectCentricLimbRepositioning3DEnv,
)
from pybullet_helpers.geometry import (
    Pose,
    SE2Pose,
)
from pybullet_helpers.joint import JointPositions, JointVelocities
from pybullet_helpers.motion_planning import (
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
)

from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT,
    BestAttempt,
    CoupledState,
    JointTorques,
    StreamLog,
    StreamProfile,
    advance,
    advance_corrected,
    arm_in_collision,
    base_candidates,
    base_in_collision,
    capture_state,
    commanded_robot_torque,
    control_saturation,
    default_human_torque_limit,
    exceeds_human_torque_limits,
    exceeds_robot_torque_limits,
    grasp_ik,
    human_in_collision,
    limb_directed_torque,
    limb_error,
    limb_out_of_limits,
    limb_slide_axis,
    pair_clearance,
    record_attempt,
    release_grasp,
    restore_state,
    saved_sim_state,
    slid_grasp,
)

NUM_REACH_CHECKS = 3
GRASP_SLIDE_FRACTIONS = (0.1, 0.7)
DEFAULT_GRASP_SLIDE = 0.10
MAX_DEFERRED_BASE_POSES = 20
GRASP_ROLLS: tuple[float, ...] = (0.0, np.pi)
# MPC runs per (state, goal) before plan-limb-motion gives up on it.
MAX_LIMB_MOTION_ATTEMPTS = 3


@dataclass(eq=False)
class LimbGrasp:
    """A grasp `slide` meters up the limb, rolled `roll`.

    `pose` maps the limb's grasp frame to the end effector.
    """

    pose: Pose
    slide: float
    roll: float

    def __repr__(self) -> str:
        return f"g{id(self) % 10000}(slide={self.slide:.3f}, roll={self.roll:.2f})"


@dataclass(eq=False)
class LimbConf:
    """A configuration of the passive limb.

    `eq=False` because pddlstream identifies stream outputs by identity.
    """

    positions: tuple[float, ...]

    def __repr__(self) -> str:
        return f"qL{id(self) % 10000}"


@dataclass(eq=False)
class ArmTrajectory:
    """A joint-space path that takes the arm from its retracted conf onto the grasp."""

    joint_plan: list[JointPositions]

    def __repr__(self) -> str:
        return f"at{id(self) % 10000}(waypoints={len(self.joint_plan)})"


@dataclass(eq=False)
class TorqueTrajectory:
    """An open-loop torque trajectory, one entry per environment step.

    - `robot_torques`: the corrections, each commanded on top of the live hold.
    - `human_torques`: the total torque the person's joints bear, t_h.
    - `robot_induced_torques`: the robot's share of that total, t_r.
    - `limb_path`: the limb's joint positions, starting configuration first, so one
      entry longer than the others.
    - `commanded_torques`: correction plus hold, what the robot is actually asked for
      and the only one the robot's torque limits bind.
    - `plan_seconds`: the MPC's wall time, charged to the step whose search spent it.
    """

    robot_torques: list[JointTorques]
    human_torques: list[JointTorques] = field(default_factory=list)
    robot_induced_torques: list[JointTorques] = field(default_factory=list)
    limb_path: list[JointPositions] = field(default_factory=list)
    commanded_torques: list[JointTorques] = field(default_factory=list)
    plan_seconds: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"tt{id(self) % 10000}(steps={len(self.robot_torques)})"


@dataclass
class MPCConfig:
    """Predictive-sampling MPC hyperparameters.

    Mirrors limb-manipulation's `PredictiveSamplingPlanner`.

    Rollouts are PyBullet steps in the environment itself, not in a surrogate.
    """

    num_rollouts: int = 64
    horizon: int = 12
    action_repeat: int = 12
    num_control_points: int = 4
    noise_scale: float = 0.12
    directed_gains: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    noise_reference_error: float = 1.2
    min_noise_ratio: float = 0.3
    goal_reaching_weight: float = 100.0
    velocity_penalty_weight: float = 0.5
    velocity_penalty_threshold: float = 0.05
    max_velocity: float = 2.0
    max_velocity_penalty: float = 300.0
    joint_limit_violation_weight: float = 1e6
    robot_joint_limit_violation_weight: float = 1e6
    human_torque_violation_weight: float = 1e6
    robot_torque_violation_weight: float = 1e6
    reject_violations: bool = True
    collision_penalty: float = 1e4
    terminal_goal_weight: float = 100.0
    velocity_regularization_weight: float = 1.0
    replay_slack: float = 0.05
    commit_steps: int = 1
    max_control_steps: int = 900
    divergence_patience: int = 200
    divergence_tolerance: float = 1e-5


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
    filter_saturated_bases: bool = True
    human_torque_limit: float | None = None
    robot_induced_torque_limit: float = DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT
    num_ik_attempts: int = 12
    best_attempt: BestAttempt | None = None
    base_rejections: Counter = field(default_factory=Counter)
    # Where planning time went, keyed by stream and by the stages nested inside them.
    profile: StreamProfile = field(default_factory=StreamProfile, repr=False)
    log: StreamLog = field(default_factory=StreamLog, repr=False)
    resting_penetration: dict[int, float] = field(default_factory=dict, repr=False)
    # The range of motion the planner is allowed to use
    believed_joint_limits: tuple[JointPositions, JointPositions] | None = None
    # Seeds the MPC's nominal control points.
    warm_start: Callable[[CoupledState, np.ndarray, int], np.ndarray] | None = None
    # A base pose already known to serve this task, tried before the grid sweep.
    base_pose_hint: SE2Pose | None = None
    true_joint_limits: tuple[JointPositions, JointPositions] = field(
        init=False, repr=False
    )
    limb_joint_infos: list = field(init=False, repr=False)
    grasp_slide_axis: np.ndarray = field(init=False, repr=False)
    grasp_slide_span: float = field(init=False, repr=False)
    _seeds: np.random.SeedSequence = field(init=False, repr=False)
    _ik_rng: np.random.Generator = field(init=False, repr=False)
    _grasp_rng: np.random.Generator = field(init=False, repr=False)
    _base_rng: np.random.Generator = field(init=False, repr=False)
    _reach_cloud: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Independent streams, so grasp and base draws are not correlated.
        self._seeds = np.random.SeedSequence(self.motion_seed)
        self._ik_rng, self._grasp_rng, self._base_rng = (
            np.random.default_rng(seed) for seed in self._seeds.spawn(3)
        )
        if self.human_torque_limit is None:
            self.human_torque_limit = default_human_torque_limit(
                self.sim.limb.get_name()
            )
        self.true_joint_limits = (
            list(self.sim.limb.joint_lower_limits),
            list(self.sim.limb.joint_upper_limits),
        )
        self.limb_joint_infos = self.sim.limb.get_arm_joint_infos()
        self.grasp_slide_axis, self.grasp_slide_span = limb_slide_axis(self.sim)
        self.refresh_collision_baseline()

    def spawn_rng(self) -> np.random.Generator:
        """A fresh generator, independent of every other one this context handed out."""
        return np.random.default_rng(self._seeds.spawn(1)[0])

    def refresh_collision_baseline(self) -> None:
        """Record how far the grasped limb overlaps each body it starts out touching.

        Call this only in the start state, as `reset_to_start` does.
        """
        self.resting_penetration = {
            body_id: min(0.0, pair_clearance(self.sim, self.sim.limb.robot_id, body_id))
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

    @staticmethod
    def _within(
        limits: tuple[JointPositions, JointPositions], joint_positions: JointPositions
    ) -> bool:
        lower, upper = limits
        return all(
            low <= value <= high
            for low, value, high in zip(lower, joint_positions, upper, strict=True)
        )

    def in_believed_limits(self, joint_positions: JointPositions) -> bool:
        """Whether the planner believes this configuration is inside the person."""
        if self.believed_joint_limits is None:
            return self.sim.limb.check_joint_limits(list(joint_positions))
        return self._within(self.believed_joint_limits, list(joint_positions))

    def truly_in_limits(self, joint_positions: JointPositions) -> bool:
        """Whether it actually is.

        For scoring only; planning must not read this.
        """
        return self._within(self.true_joint_limits, list(joint_positions))

    @property
    def goal_atol(self) -> float:
        """The joint-space distance within which the environment declares success."""
        return self.sim.config.goal_atol

    @property
    def robot_torque_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """The robot's (lower, upper) torque limits, from its action space."""
        config = self.sim.config
        return (
            np.asarray(config.torque_lower_limits, dtype=np.float64),
            np.asarray(config.torque_upper_limits, dtype=np.float64),
        )

    @property
    def human_torque_limits(self) -> tuple[float, float]:
        """The bounds on the total torque and on the robot's share of it."""
        assert self.human_torque_limit is not None  # set in __post_init__
        return (abs(self.human_torque_limit), abs(self.robot_induced_torque_limit))


def sample_grasp(ctx: LimbStreamContext, limb: str) -> Iterator[tuple[LimbGrasp]]:
    """Yield the default slide both ways up, then random slides without end."""
    del limb  # there is a single limb per environment
    rng = ctx._grasp_rng  # pylint: disable=protected-access
    low, high = GRASP_SLIDE_FRACTIONS
    for roll in rng.permutation(GRASP_ROLLS):
        yield (_limb_grasp(ctx, DEFAULT_GRASP_SLIDE, float(roll)),)
    while True:
        slide = float(rng.uniform(low, high)) * ctx.grasp_slide_span
        yield (_limb_grasp(ctx, slide, float(rng.choice(GRASP_ROLLS))),)


def _limb_grasp(ctx: LimbStreamContext, slide: float, roll: float) -> LimbGrasp:
    """The scene's grasp slid `slide` up the limb and rolled `roll`."""
    pose = slid_grasp(ctx, ctx.sim.scene.grasp_transform, slide, roll)
    return LimbGrasp(pose, slide, roll)


def _reach_saturation(
    ctx: LimbStreamContext, grasp: LimbGrasp, waypoints: np.ndarray
) -> float | str:
    """Peak `control_saturation` along the waypoints, or why the base cannot hold them.

    Each IK is seeded from the previous solution to stay on one continuous branch,
    since the arm cannot teleport between branches while welded to the limb.
    """
    seed: JointPositions = ctx.retract_joints
    saturation = 0.0
    for waypoint in waypoints:
        with ctx.profile.timed("grasp_ik"):
            solution = grasp_ik(ctx, grasp.pose, tuple(waypoint), seed)
        if solution is None:
            return "the arm cannot reach the grasp"
        ratio = control_saturation(ctx, solution)
        if not np.isfinite(ratio):
            return "the grasp is kinematically singular"
        saturation = max(saturation, ratio)
        seed = solution
    return saturation


def sample_base_pose(
    ctx: LimbStreamContext,
    limb: str,
    grasp: LimbGrasp,
    init_conf: LimbConf,
    goal_conf: LimbConf,
) -> Iterator[tuple[SE2Pose]]:
    """Yield base poses that hold the grasp from the initial to the goal configuration.

    Requiring IK at the goal too is what makes `move_base` a real decision.

    Bases the arm cannot hold the limb at come last, least saturated first.
    """
    del limb  # there is a single limb per environment
    sim = ctx.sim
    waypoints = np.linspace(
        np.asarray(init_conf.positions),
        np.asarray(goal_conf.positions),
        NUM_REACH_CHECKS,
    )
    rejected: list[list] = []

    def reject(base_conf: SE2Pose, reason: str) -> None:
        ctx.base_rejections[reason] += 1
        rejected.append([base_conf.x, base_conf.y, base_conf.rot, reason])

    def noted(base_conf: SE2Pose, saturation: float) -> tuple[SE2Pose]:
        ctx.log.note(saturation=saturation, rejected=list(rejected))
        rejected.clear()
        return (base_conf,)

    with saved_sim_state(sim):
        release_grasp(sim)
        candidates = ctx.profile.wrap(
            "base_candidates",
            base_candidates(ctx, grasp.pose, list(init_conf.positions)),
        )
        if ctx.base_pose_hint is not None:
            candidates = itertools.chain([ctx.base_pose_hint], candidates)
        deferred: list[tuple[float, SE2Pose]] = []
        for base_conf in candidates:
            sim.robot.set_base(base_conf)
            saturation = _reach_saturation(ctx, grasp, waypoints)
            if isinstance(saturation, str):
                reject(base_conf, saturation)
                continue
            saturated = ctx.filter_saturated_bases and saturation > 1.0
            if saturated and len(deferred) >= MAX_DEFERRED_BASE_POSES:
                reject(base_conf, "the arm cannot hold the limb's weight at the grasp")
                continue
            with ctx.profile.timed("base_collision_check"):
                base_hit = ctx.check_base_collisions and base_in_collision(ctx)
            if base_hit:
                reject(base_conf, "the base collides with the scene")
                continue
            if saturated:
                deferred.append((saturation, base_conf))
                continue
            yield noted(base_conf, saturation)
            release_grasp(sim)
        for saturation, base_conf in sorted(deferred, key=lambda entry: entry[0]):
            sim.robot.set_base(base_conf)
            yield noted(base_conf, saturation)
            release_grasp(sim)
        ctx.log.note(rejected=list(rejected))


def plan_grasp_motion(
    ctx: LimbStreamContext,
    limb: str,
    grasp: LimbGrasp,
    base_conf: SE2Pose,
    init_conf: LimbConf,
) -> Iterator[tuple[ArmTrajectory, CoupledState]]:
    """Yield the arm path from its retracted conf onto the grasp, from `base_conf`.

    A rigid grasp ties the limb configuration to the end-effector pose.
    """
    del limb  # there is a single limb per environment
    sim = ctx.sim
    with saved_sim_state(sim):
        release_grasp(sim)
        sim.robot.set_base(base_conf)
        with ctx.profile.timed("grasp_ik"):
            grasp_joints = grasp_ik(
                ctx, grasp.pose, init_conf.positions, ctx.retract_joints
            )
        joint_plan = None
        if grasp_joints is not None:
            with ctx.profile.timed("arm_motion_planning"):
                joint_plan = run_motion_planning(
                    sim.robot.arm,
                    initial_positions=list(ctx.retract_joints),
                    target_positions=grasp_joints,
                    collision_bodies=ctx.obstacle_ids,
                    seed=ctx.motion_seed,
                    physics_client_id=sim.physics_client_id,
                )
    if joint_plan is None:
        reason = (
            "the arm cannot reach the grasp"
            if grasp_joints is None
            else "no arm path onto the grasp"
        )
        ctx.base_rejections[reason] += 1
        ctx.log.note(failure=reason)
        return
    trajectory = ArmTrajectory(list(joint_plan))
    state = CoupledState(
        base_pose=base_conf,
        robot_positions=list(joint_plan[-1]),
        robot_velocities=[0.0] * len(joint_plan[-1]),
        limb_positions=list(init_conf.positions),
        limb_velocities=[0.0] * NUM_LIMB_JOINTS,
        approach=trajectory,
    )
    yield (trajectory, state)


def plan_base_motion(
    ctx: LimbStreamContext, q1: SE2Pose, q2: SE2Pose
) -> Iterator[tuple[list[SE2Pose]]]:
    """Plan a collision-free path for the mobile base from `q1` to `q2`."""
    sim = ctx.sim
    with saved_sim_state(sim):
        # The base is only ever driven with the hand empty (see the domain)
        release_grasp(sim)
        sim.robot.arm.set_joints(list(ctx.retract_joints))
        with ctx.profile.timed("base_motion_planning"):
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


def _charge_rollout(ctx: LimbStreamContext, outcome: str, start: float) -> None:
    """Record how long this rollout took and why it stopped."""
    ctx.profile.add(
        f"rollout: {re.sub(r'[0-9]+[.][0-9]+', 'N', outcome)}",
        time.time() - start,
        produced=outcome == "reached the goal",
    )


def plan_limb_motion(
    ctx: LimbStreamContext, s1: CoupledState, q2: LimbConf
) -> Iterator[tuple[TorqueTrajectory, CoupledState] | None]:
    """Generate a torque trajectory from state `s1` to limb configuration `q2`.

    Each call runs one MPC with fresh noise. A failed one yields None, so PDDLStream
    may come back for another, up to `MAX_LIMB_MOTION_ATTEMPTS`.
    """
    if not ctx.in_believed_limits(list(q2.positions)):
        reason = "the target is outside the limb's joint limits"
        _charge_rollout(ctx, reason, time.time())
        ctx.log.note(failure=reason)
        print(f"plan-limb-motion: {reason}.")
        return
    for attempt in range(MAX_LIMB_MOTION_ATTEMPTS):
        ctx.log.note(attempt=attempt)
        yield _limb_motion_attempt(ctx, s1, q2)


def _trajectory_margins(
    ctx: LimbStreamContext, trajectory: TorqueTrajectory
) -> dict[str, float]:
    """Peak load as a fraction of each limit, and the closest approach to the RoM."""
    total_limit, robot_limit = ctx.human_torque_limits
    _, upper = ctx.robot_torque_limits
    lower_rom, upper_rom = ctx.believed_joint_limits or ctx.true_joint_limits
    path = np.asarray(trajectory.limb_path)
    return {
        "human_torque_ratio": float(np.abs(trajectory.human_torques).max())
        / total_limit,
        "robot_share_ratio": float(np.abs(trajectory.robot_induced_torques).max())
        / robot_limit,
        "robot_torque_ratio": float(
            (np.abs(trajectory.commanded_torques) / upper).max()
        ),
        "rom_margin": float(
            np.minimum(path - np.asarray(lower_rom), np.asarray(upper_rom) - path).min()
        ),
    }


def _limb_motion_attempt(
    ctx: LimbStreamContext, s1: CoupledState, q2: LimbConf
) -> tuple[TorqueTrajectory, CoupledState] | None:
    """Run predictive-sampling MPC closed-loop, recording the torques it applies.

    It stops with enough headroom that an open-loop replay also lands in tolerance.
    """
    sim = ctx.sim
    call_start = time.time()
    goal = np.asarray(q2.positions, dtype=np.float64)
    mpc = PredictiveSamplingMPC(ctx, goal, s1)

    cfg = ctx.mpc
    threshold = max(ctx.goal_atol - cfg.replay_slack, ctx.goal_atol / 2)
    with saved_sim_state(sim):
        restore_state(sim, s1)
        robot_torques: list[JointTorques] = []
        plan_seconds: list[float] = []
        reached = False
        giveup = ""
        error = limb_error(ctx, sim.limb.get_joint_positions(), goal)
        best_error = error
        steps_since_improvement = 0
        control_steps = 0
        while control_steps < cfg.max_control_steps:
            if error < threshold:
                reached = True
                break
            if steps_since_improvement >= cfg.divergence_patience:
                giveup = f"diverged at an error of {best_error:.3f}"
                break
            search_start = time.time()
            committed = mpc.step()
            elapsed = time.time() - search_start
            ctx.profile.add("mpc_search", elapsed)
            overloaded = saturated = False
            for index, torque in enumerate(committed):
                control_steps += 1
                for repeat in range(cfg.action_repeat):
                    # Read before the step, which clips a saturated command away.
                    commanded = commanded_robot_torque(sim, torque)
                    saturated = exceeds_robot_torque_limits(ctx, commanded)
                    human_torque = advance(sim, commanded)
                    robot_torques.append(list(torque))
                    plan_seconds.append(elapsed if index == 0 and repeat == 0 else 0.0)
                    overloaded = exceeds_human_torque_limits(ctx, human_torque)
                    if overloaded or saturated:
                        break
                    error = limb_error(ctx, sim.limb.get_joint_positions(), goal)
                    if error < threshold:
                        reached = True
                        break
                if overloaded or saturated or reached:
                    break
            if overloaded or saturated:
                reached = False
                giveup = (
                    "loaded a limb joint past the person's torque limit"
                    if overloaded
                    else "asked the robot for more torque than it has"
                )
                break
            if limb_out_of_limits(ctx):
                reached = False
                giveup = "bent a joint of the limb past its anatomical limit"
                break
            if ctx.resting_penetration and human_in_collision(ctx):
                reached = False
                giveup = "drove the limb into the person or the furniture"
                break
            if best_error - error > cfg.divergence_tolerance:
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
        human_torques: list[JointTorques] = []
        robot_induced: list[JointTorques] = []
        limb_path: list[JointPositions] = []
        commanded_torques: list[JointTorques] = []
        if reached:
            replay_start = time.time()
            restore_state(sim, s1)
            limb_path.append(list(sim.limb.get_joint_positions()))
            for torque in robot_torques:
                # Read before the step: `advance` clips this, so afterwards there is
                # nothing left to see of a saturated command.
                step_command = commanded_robot_torque(sim, torque)
                commanded_torques.append(list(step_command))
                human_torque = advance_corrected(sim, torque)
                human_torques.append(list(human_torque.total))
                robot_induced.append(list(human_torque.robot))
                limb_path.append(list(sim.limb.get_joint_positions()))
                # The replay drifts from the closed-loop run, so it is checked
                # separately, and here at every step rather than every control step.
                if limb_out_of_limits(ctx):
                    reached = False
                    giveup = "replayed open-loop past the limb's anatomical limits"
                    break
                if exceeds_human_torque_limits(ctx, human_torque):
                    reached = False
                    giveup = "replayed open-loop past the person's torque limits"
                    break
                if exceeds_robot_torque_limits(ctx, step_command):
                    reached = False
                    giveup = "replayed open-loop past the robot's torque limits"
                    break
            ctx.profile.add("open_loop_replay", time.time() - replay_start)
        if reached:
            s2 = capture_state(sim)
            replay_error = limb_error(ctx, s2.limb_positions, goal)
            if replay_error >= ctx.goal_atol:
                reached = False
                giveup = (
                    f"replayed open-loop to {replay_error:.3f}, outside the tolerance"
                )
            elif ctx.resting_penetration and human_in_collision(ctx):
                reached = False
                giveup = "replayed open-loop into the person or the furniture"

    ctx.log.note(
        best_error=best_error,
        control_steps=control_steps,
        torque_steps=len(robot_torques),
    )
    if not reached:
        record_attempt(ctx, s1, robot_torques, best_error, giveup, plan_seconds)
        _charge_rollout(ctx, giveup, call_start)
        ctx.log.note(failure=giveup)
        print(f"plan-limb-motion: MPC rollout {giveup}.")
        return None
    if not robot_torques:
        # A zero-length LimbMotion yields a no-op move_limb PDDLStream cannot rebind.
        _charge_rollout(ctx, "already at the target configuration", call_start)
        ctx.log.note(failure="already at the target configuration")
        print("plan-limb-motion: already at the target configuration; not certified.")
        return None
    assert s2 is not None
    record_attempt(
        ctx,
        s1,
        robot_torques,
        best_error,
        "reached the goal, but the search ran out of time",
        plan_seconds,
    )
    _charge_rollout(ctx, "reached the goal", call_start)
    print(f"plan-limb-motion: generated a trajectory with {len(robot_torques)} steps.")
    trajectory = TorqueTrajectory(
        robot_torques,
        human_torques,
        robot_induced,
        limb_path,
        commanded_torques,
        plan_seconds,
    )
    ctx.log.note(**_trajectory_margins(ctx, trajectory))
    return trajectory, s2


def check_human_joint_limits(
    ctx: LimbStreamContext,
    s1: CoupledState,
    q2: LimbConf,
    trajectory: TorqueTrajectory,
    s2: CoupledState,
) -> bool:
    """Constraint: the limb stays in the person's range of motion for the whole motion.

    Every configuration of the open-loop replay is checked, not only the goal.
    """
    del s1, q2, s2  # the constraint is on the trajectory alone
    for positions in trajectory.limb_path:
        if not ctx.in_believed_limits(list(positions)):
            print("check-human-joint-limits: the limb leaves its joint limits.")
            return False
    return True


def check_human_torque_limits(
    ctx: LimbStreamContext,
    s1: CoupledState,
    q2: LimbConf,
    trajectory: TorqueTrajectory,
    s2: CoupledState,
) -> bool:
    """Constraint: no step of the trajectory overloads a joint of the person."""
    del s1, q2, s2  # the constraint is on the trajectory alone
    total_limit, robot_limit = ctx.human_torque_limits
    for total, robot in zip(
        trajectory.human_torques, trajectory.robot_induced_torques, strict=True
    ):
        if np.abs(total).max() > total_limit or np.abs(robot).max() > robot_limit:
            print("check-human-torque-limits: human torque limit violated.")
            return False
    return True


def check_robot_torque_limits(
    ctx: LimbStreamContext,
    s1: CoupledState,
    q2: LimbConf,
    trajectory: TorqueTrajectory,
    s2: CoupledState,
) -> bool:
    """Constraint: the robot is never asked for a torque outside its action space."""
    del s1, q2, s2  # the constraint is on the trajectory alone
    # A trajectory assembled without a replay, as a failed attempt is, has only the
    # corrections to offer; they are a lower bound on what was asked for.
    torques = trajectory.commanded_torques or trajectory.robot_torques
    for torque in torques:
        if exceeds_robot_torque_limits(ctx, torque):
            print("check-robot-torque-limits: robot torque limit violated.")
            return False
    return True


class PredictiveSamplingMPC:
    """Predictive-sampling MPC over robot joint torques.

    A port of limb-manipulation's `PredictiveSamplingPlanner` to KinDER.

    The dynamics model is the PyBullet environment, snapshotted and restored.
    """

    def __init__(
        self,
        ctx: LimbStreamContext,
        goal: np.ndarray,
        start: CoupledState | None = None,
    ) -> None:
        self._ctx = ctx
        self._sim = ctx.sim
        self._cfg = ctx.mpc
        self._obstacles = ctx.obstacle_ids
        self._goal = goal
        self._rng = ctx.spawn_rng()
        self._lower, self._upper = ctx.robot_torque_limits
        self._nominal = np.zeros((self._cfg.num_control_points, NUM_ROBOT_JOINTS))
        if ctx.warm_start is not None and start is not None:
            seeded = np.asarray(ctx.warm_start(start, goal, 0), dtype=np.float64)
            assert seeded.shape == self._nominal.shape
            self._nominal = np.clip(seeded, self._lower, self._upper)
        self._control_indices = np.round(
            np.linspace(0, self._cfg.horizon - 1, self._cfg.num_control_points)
        ).astype(int)
        self._commit = int(np.clip(self._cfg.commit_steps, 1, self._cfg.horizon))
        self._step_index = 0

    def step(self) -> list[JointTorques]:
        """Choose the torque to apply in the simulator's current state."""
        error = limb_error(self._ctx, self._sim.limb.get_joint_positions(), self._goal)
        with self._ctx.profile.timed("mpc_state_capture"):
            start = capture_state(self._sim)
        candidates = self._sample_candidates(error, start)
        scores = [self._score(start, self._expand(cp)) for cp in candidates]
        restore_state(self._sim, start, regrasp=False)
        best = candidates[min(range(len(scores)), key=scores.__getitem__)]
        self._nominal = best
        self._step_index += self._commit
        return [list(torque) for torque in self._expand(best)[: self._commit]]

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

    def _directed_candidates(self, scale: float) -> list[np.ndarray]:
        """One goal-directed candidate per gain, each with a noisy twin."""
        candidates = []
        for gain in self._cfg.directed_gains:
            torque = limb_directed_torque(self._sim, self._goal, gain)
            points = np.tile(torque, (self._cfg.num_control_points, 1))
            noise = self._rng.normal(scale=scale, size=points.shape)
            candidates.extend([self._clip(points), self._clip(points + noise)])
        return candidates

    def _sample_candidates(self, error: float, state=None) -> list[np.ndarray]:
        """Warm-started nominal, zero torque, goal-directed candidates, and noisy draws.

        Each candidate is a set of control points rather than a torque sequence.
        """
        cfg = self._cfg
        scale = cfg.noise_scale * float(
            np.clip(error / cfg.noise_reference_error, cfg.min_noise_ratio, 1.0)
        )
        expanded = self._expand(self._nominal)
        tail = self._rng.normal(expanded[-1], scale)
        ramp = np.linspace(expanded[-1], tail, self._commit + 1)[1:]
        shifted = np.vstack([expanded[self._commit :], ramp])
        nominal = shifted[self._control_indices]
        candidates = [self._clip(nominal), np.zeros_like(nominal)]
        if self._ctx.warm_start is not None and state is not None:
            candidates.append(
                self._clip(
                    np.asarray(
                        self._ctx.warm_start(state, self._goal, self._step_index),
                        dtype=np.float64,
                    )
                )
            )
        candidates.extend(self._directed_candidates(scale))
        num_samples = max(cfg.num_rollouts - len(candidates), 0)
        noise = self._rng.normal(
            scale=scale,
            size=(num_samples, cfg.num_control_points, NUM_ROBOT_JOINTS),
        )
        candidates.extend(self._clip(nominal + sample) for sample in noise)
        return candidates

    def _clip(self, robot_torques: np.ndarray) -> np.ndarray:
        return np.clip(robot_torques, self._lower, self._upper)

    def _score(
        self, start: CoupledState, robot_torques: np.ndarray
    ) -> tuple[int, float]:
        """Roll `robot_torques` out from `start`, rejected flag first, then its cost."""
        with self._ctx.profile.timed("mpc_state_restore"):
            restore_state(self._sim, start, regrasp=False)
        cfg = self._cfg
        goal_cost = 0.0
        velocity_cost = 0.0
        regularization_cost = 0.0
        limit_cost = 0.0
        squared_distance = 0.0
        violated = False
        for torque in robot_torques:
            overloaded = False
            saturated = False
            with self._ctx.profile.timed("mpc_physics"):
                for _ in range(cfg.action_repeat):
                    # Read before the step: `advance` clips a saturated command away.
                    commanded = commanded_robot_torque(self._sim, torque)
                    saturated |= exceeds_robot_torque_limits(self._ctx, commanded)
                    human_torque = advance(self._sim, commanded)
                    overloaded |= exceeds_human_torque_limits(self._ctx, human_torque)
            positions = self._sim.limb.get_joint_positions()
            velocities = self._sim.limb.get_joint_velocities()
            squared_distance = limb_error(self._ctx, positions, self._goal) ** 2
            goal_cost += squared_distance
            velocity_cost += self._velocity_penalty(velocities, squared_distance)
            regularization_cost += float(np.sum(np.square(velocities)))
            if not self._ctx.in_believed_limits(positions):
                limit_cost += cfg.joint_limit_violation_weight
                violated = True
            if not self._sim.robot.arm.check_joint_limits(
                list(self._sim.robot.arm.get_joint_positions())
            ):
                limit_cost += cfg.robot_joint_limit_violation_weight
                violated = True
            if overloaded:
                limit_cost += cfg.human_torque_violation_weight
                violated = True
            if saturated:
                limit_cost += cfg.robot_torque_violation_weight
                violated = True
            with self._ctx.profile.timed("mpc_collision_check"):
                arm_hit = bool(self._obstacles) and arm_in_collision(self._ctx)
                human_hit = bool(self._ctx.resting_penetration) and human_in_collision(
                    self._ctx
                )
            if arm_hit:
                limit_cost += cfg.collision_penalty
            if human_hit:
                limit_cost += cfg.collision_penalty
        cost = (
            cfg.goal_reaching_weight * goal_cost
            + cfg.terminal_goal_weight * squared_distance
            + cfg.velocity_penalty_weight * velocity_cost
            + cfg.velocity_regularization_weight * regularization_cost
            + limit_cost
        )
        return (int(violated and cfg.reject_violations), cost)

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
