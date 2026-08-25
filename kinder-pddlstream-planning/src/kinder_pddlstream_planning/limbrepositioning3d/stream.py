"""PDDLStream stream implementations for LimbRepositioning3D.

See run.py for the domain design, and utils.py for the simulator helpers these compose.

Streams scratch-mutate the live simulator and restore it via `_saved_sim_state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

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
    _base_candidates,
    _base_in_collision,
    _grasp_ik,
    _limb_error,
    _limb_slide_axis,
    _pair_clearance,
    _perturbed_grasp,
    _record_attempt,
    _saved_sim_state,
    advance,
    advance_corrected,
    arm_in_collision,
    capture_state,
    commanded_robot_torque,
    default_human_torque_limit,
    exceeds_human_torque_limits,
    exceeds_robot_torque_limits,
    human_in_collision,
    limb_is_controllable,
    limb_out_of_limits,
    release_grasp,
    restore_state,
)

NUM_REACH_CHECKS = 5

# Rolls of the nominal grasp about the tool axis
GRASP_ROLLS: tuple[float, ...] = (np.pi, 0.4, -0.4, 0.8, -0.8)

# Fraction of the grasp-to-joint span of the distal segment to slide the grasp within.
GRASP_SLIDE_FRACTIONS = (0.3, 0.8)

# Bound on the roll paired with a sampled slide, keeping it a mild perturbation.
GRASP_SAMPLE_ROLL = 0.8

# Random slid-and-rolled grasps yielded after the fixed rolls.
NUM_SAMPLED_GRASPS = 8


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
    """

    robot_torques: list[JointTorques]
    human_torques: list[JointTorques] = field(default_factory=list)
    robot_induced_torques: list[JointTorques] = field(default_factory=list)
    limb_path: list[JointPositions] = field(default_factory=list)
    commanded_torques: list[JointTorques] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"tt{id(self) % 10000}(steps={len(self.robot_torques)})"


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
    noise_reference_error: float = 1.2
    min_noise_ratio: float = 0.1
    goal_reaching_weight: float = 100.0
    velocity_penalty_weight: float = 0.5
    velocity_penalty_threshold: float = 0.3
    max_velocity: float = 2.0
    max_velocity_penalty: float = 300.0
    joint_limit_violation_weight: float = 1e6
    human_torque_violation_weight: float = 1e6
    robot_torque_violation_weight: float = 1e6
    collision_penalty: float = 1e4
    terminal_goal_weight: float = 100.0
    velocity_regularization_weight: float = 1.0
    # Headroom the closed-loop run stops with so the open-loop replay lands in tolerance.
    # The replay is re-checked explicitly and reproduces planning to ~1e-12 rad.
    replay_slack: float = 1e-3
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
    human_torque_limit: float | None = None
    robot_induced_torque_limit: float = DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT
    num_ik_attempts: int = 12
    best_attempt: BestAttempt | None = None
    resting_penetration: dict[int, float] = field(default_factory=dict, repr=False)
    limb_joint_infos: list = field(init=False, repr=False)
    grasp_slide_axis: np.ndarray = field(init=False, repr=False)
    grasp_slide_span: float = field(init=False, repr=False)
    _ik_rng: np.random.Generator = field(init=False, repr=False)
    _grasp_rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ik_rng = np.random.default_rng(self.motion_seed)
        self._grasp_rng = np.random.default_rng(self.motion_seed)
        if self.human_torque_limit is None:
            self.human_torque_limit = default_human_torque_limit(
                self.sim.limb.get_name()
            )
        self.limb_joint_infos = self.sim.limb.get_arm_joint_infos()
        self.grasp_slide_axis, self.grasp_slide_span = _limb_slide_axis(self.sim)
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
        return (abs(self.human_torque_limit), abs(self.robot_induced_torque_limit))


def sample_grasp(ctx: LimbStreamContext, limb: str) -> Iterator[tuple[Pose]]:
    """Yield the scene's grasp, fixed rolls of it, then random slides up the limb."""
    del limb  # there is a single limb per environment
    nominal = ctx.sim.scene.grasp_transform
    yield (nominal,)
    for roll in GRASP_ROLLS:
        yield (_perturbed_grasp(ctx, nominal, 0.0, roll),)
    rng = ctx._grasp_rng  # pylint: disable=protected-access
    low, high = GRASP_SLIDE_FRACTIONS
    for _ in range(NUM_SAMPLED_GRASPS):
        slide = float(rng.uniform(low, high)) * ctx.grasp_slide_span
        roll = float(rng.uniform(-GRASP_SAMPLE_ROLL, GRASP_SAMPLE_ROLL))
        yield (_perturbed_grasp(ctx, nominal, slide, roll),)


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
                if solution is None or not limb_is_controllable(ctx):
                    reachable = False
                    break
                seed = solution
            if not reachable:
                continue
            if ctx.check_base_collisions and _base_in_collision(ctx):
                continue
            yield (base_conf,)


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
    if not sim.limb.check_joint_limits(list(q2.positions)):
        print("plan-limb-motion: the target is outside the limb's joint limits.")
        return
    goal = np.asarray(q2.positions, dtype=np.float64)
    mpc = PredictiveSamplingMPC(ctx, goal)

    cfg = ctx.mpc
    threshold = max(ctx.goal_atol - cfg.replay_slack, ctx.goal_atol / 2)
    with _saved_sim_state(sim):
        restore_state(sim, s1)
        robot_torques: list[JointTorques] = []
        reached = False
        giveup = ""
        error = _limb_error(ctx, sim.limb.get_joint_positions(), goal)
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
            overloaded = saturated = False
            for _ in range(cfg.action_repeat):
                # Read before the step, which clips a saturated command away.
                commanded = commanded_robot_torque(sim, torque)
                saturated = exceeds_robot_torque_limits(ctx, commanded)
                human_torque = advance(sim, commanded)
                robot_torques.append(list(torque))
                overloaded = exceeds_human_torque_limits(ctx, human_torque)
                if overloaded or saturated:
                    break
                error = _limb_error(ctx, sim.limb.get_joint_positions(), goal)
                if error < threshold:
                    reached = True
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
        human_torques: list[JointTorques] = []
        robot_induced: list[JointTorques] = []
        limb_path: list[JointPositions] = []
        commanded: list[JointTorques] = []
        if reached:
            restore_state(sim, s1)
            limb_path.append(list(sim.limb.get_joint_positions()))
            for torque in robot_torques:
                # Read before the step: `advance` clips this, so afterwards there is
                # nothing left to see of a saturated command.
                step_command = commanded_robot_torque(sim, torque)
                commanded.append(list(step_command))
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
        if reached:
            s2 = capture_state(sim)
            replay_error = _limb_error(ctx, s2.limb_positions, goal)
            if replay_error >= ctx.goal_atol:
                reached = False
                giveup = (
                    f"replayed open-loop to {replay_error:.3f}, outside the tolerance"
                )
            elif ctx.resting_penetration and human_in_collision(ctx):
                reached = False
                giveup = "replayed open-loop into the person or the furniture"

    if not reached:
        _record_attempt(ctx, s1, robot_torques, best_error, giveup)
        print(f"plan-limb-motion: MPC rollout {giveup}.")
        return
    if not robot_torques:
        # A zero-length LimbMotion yields a no-op move_limb PDDLStream cannot rebind.
        print("plan-limb-motion: already at the target configuration; not certified.")
        return
    assert s2 is not None
    print(f"plan-limb-motion: generated a trajectory with {len(robot_torques)} steps.")
    yield (
        TorqueTrajectory(
            robot_torques, human_torques, robot_induced, limb_path, commanded
        ),
        s2,
    )


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
        if not ctx.sim.limb.check_joint_limits(list(positions)):
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

    def __init__(self, ctx: LimbStreamContext, goal: np.ndarray) -> None:
        self._ctx = ctx
        self._sim = ctx.sim
        self._cfg = ctx.mpc
        self._obstacles = ctx.obstacle_ids
        self._goal = goal
        self._rng = np.random.default_rng(ctx.motion_seed)
        self._lower, self._upper = ctx.robot_torque_limits
        self._nominal = np.zeros((self._cfg.num_control_points, NUM_ROBOT_JOINTS))
        self._control_indices = np.round(
            np.linspace(0, self._cfg.horizon - 1, self._cfg.num_control_points)
        ).astype(int)

    def step(self) -> JointTorques:
        """Choose the torque to apply in the simulator's current state."""
        error = _limb_error(self._ctx, self._sim.limb.get_joint_positions(), self._goal)
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

    def _clip(self, robot_torques: np.ndarray) -> np.ndarray:
        return np.clip(robot_torques, self._lower, self._upper)

    def _score(self, start: CoupledState, robot_torques: np.ndarray) -> float:
        """Roll `robot_torques` out from `start` and score the result."""
        restore_state(self._sim, start)
        cfg = self._cfg
        goal_cost = 0.0
        velocity_cost = 0.0
        regularization_cost = 0.0
        limit_cost = 0.0
        squared_distance = 0.0
        for torque in robot_torques:
            overloaded = False
            saturated = False
            for _ in range(cfg.action_repeat):
                # Read before the step, as `advance` clips a saturated command away.
                commanded = commanded_robot_torque(self._sim, torque)
                saturated |= exceeds_robot_torque_limits(self._ctx, commanded)
                human_torque = advance(self._sim, commanded)
                overloaded |= exceeds_human_torque_limits(self._ctx, human_torque)
            positions = self._sim.limb.get_joint_positions()
            velocities = self._sim.limb.get_joint_velocities()
            squared_distance = _limb_error(self._ctx, positions, self._goal) ** 2
            goal_cost += squared_distance
            velocity_cost += self._velocity_penalty(velocities, squared_distance)
            regularization_cost += float(np.sum(np.square(velocities)))
            if not self._sim.limb.check_joint_limits(positions):
                limit_cost += cfg.joint_limit_violation_weight
            if overloaded:
                limit_cost += cfg.human_torque_violation_weight
            if saturated:
                limit_cost += cfg.robot_torque_violation_weight
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
