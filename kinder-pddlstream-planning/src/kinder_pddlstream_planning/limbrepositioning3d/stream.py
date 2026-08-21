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
from kinder.envs.dynamic3d.limb_utils import (
    NUM_LIMB_JOINTS,
    NUM_ROBOT_JOINTS,
    joint_position_distance,
)
from kinder.envs.dynamic3d.limbrepositioning3d import (
    ObjectCentricLimbRepositioning3DEnv,
)
from pybullet_helpers.geometry import (
    Pose,
    SE2Pose,
    matrix_from_quat,
    multiply_poses,
)
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


# Rolls of the nominal grasp about the tool axis, tried when it does not reach.
GRASP_ROLLS: tuple[float, ...] = (0.4, -0.4, 0.8, -0.8)

CLEARANCE_PROBE_DISTANCE = 0.1

COLLISION_MARGIN = 0.01

# Bound on the total torque a joint of the person may bear, in N*m, per limb. Half of
# peak isometric strength: ~77 N*m shoulder flexion for arms, ~200 N*m knee extension.
HUMAN_TORQUE_LIMITS = {"arm": 40.0, "leg": 100.0}

# What the robot itself may add on top of gravity and tone, in N*m.
DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT = 5.0

# Kinova Gen3 joint effort limits, from the URDF the environment loads. Its shipped
# +-1 N*m action space cannot hold the arm up under gravity.
ROBOT_TORQUE_LIMITS = (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)

# Damping for the limb Jacobian inverse, applied only once its smallest singular value
# falls below the threshold, so well-conditioned configurations stay exact.
JACOBIAN_DAMPING = 0.1
SINGULARITY_THRESHOLD = 0.02

# Torque headroom, in N*m, a base pose must leave for the motion itself.
TORQUE_FEASIBILITY_MARGIN = 1.0

# Gravity, in m/s^2. The environment ships with this off, leaving the limb weightless.
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)

# Viscous damping at each of the limb's own joints, in N*m*s/rad. The environment zeros
# every joint's damping and friction so that torques act directly, which leaves the
# distal joints - a hand or a foot, a fraction of a kilogram at the end of the chain the
# robot grasps - free to whip. A limp limb is not frictionless, and without this the
# robot cannot steer the limb at all: see the README.
DEFAULT_LIMB_JOINT_DAMPING = 0.5


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


@dataclass(frozen=True)
class HumanTorques:
    """The torque on the human's joints, and the two references it is read against."""

    total: np.ndarray
    tone: np.ndarray
    gravity: np.ndarray

    @property
    def robot(self) -> np.ndarray:
        """What the motion adds on top of the person's own static load.

        Zero while the robot merely holds the limb still, which is what makes it a
        bound on the motion rather than on the posture. Muscle tone is not subtracted
        again: inverse dynamics already counts it inside `total`, so subtracting it
        leaves `-tone` on a limb the robot is only holding.
        """
        return self.total - self.gravity


@dataclass(eq=False)
class TorqueTrajectory:
    """An open-loop torque trajectory, one torque per environment step.

    `robot_torques` are the corrections execution replays, each commanded on top of
    whatever it takes to hold the system still at that moment. The remaining fields
    record what the open-loop replay did, so the constraint streams can check it without
    re-simulating. `human_torques` is the total t_h; `robot_induced_torques` is the t_r
    share; `commanded_torques` is the correction plus that hold, which is what the robot
    is actually asked for and the only one of the two the torque limits bind.
    """

    robot_torques: list[JointTorques]
    human_torques: list[JointTorques] = field(default_factory=list)
    robot_induced_torques: list[JointTorques] = field(default_factory=list)
    limb_path: list[JointPositions] = field(default_factory=list)
    commanded_torques: list[JointTorques] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"tt{id(self) % 10000}(steps={len(self.robot_torques)})"


@dataclass
class RolloutLog:
    """Per-step measurements of an executed `move_limb`, for plots and analysis.

    `advance_logged` fills one of these; nothing in planning reads it back.
    """

    correction_torques: list[JointTorques] = field(default_factory=list)
    commanded_torques: list[JointTorques] = field(default_factory=list)
    limb_positions: list[JointPositions] = field(default_factory=list)
    limb_velocities: list[JointVelocities] = field(default_factory=list)
    human_total: list[JointTorques] = field(default_factory=list)
    human_gravity: list[JointTorques] = field(default_factory=list)
    human_tone: list[JointTorques] = field(default_factory=list)
    human_robot_induced: list[JointTorques] = field(default_factory=list)
    grasp_wrenches: list[list[float]] = field(default_factory=list)


@dataclass
class BestAttempt:
    """The closest a rejected limb motion got, kept to visualize failures.

    `plan_limb_motion` otherwise discards every rollout that does not arrive.
    """

    state: CoupledState
    robot_torques: list[JointTorques]
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
    noise_reference_error: float = 1.2
    min_noise_ratio: float = 0.1
    goal_reaching_weight: float = 100.0
    velocity_penalty_weight: float = 0.5
    velocity_penalty_threshold: float = 0.3
    max_velocity: float = 2.0
    max_velocity_penalty: float = 300.0
    joint_limit_violation_weight: float = 1e6
    human_torque_violation_weight: float = 1e6
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


def default_human_torque_limit(limb_name: str) -> float:
    """The total-torque bound for this limb: legs bear far more than arms."""
    return HUMAN_TORQUE_LIMITS["leg" if "leg" in limb_name else "arm"]


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
    _ik_rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ik_rng = np.random.default_rng(self.motion_seed)
        if self.human_torque_limit is None:
            self.human_torque_limit = default_human_torque_limit(
                self.sim.limb.get_name()
            )
        self.limb_joint_infos = self.sim.limb.get_arm_joint_infos()
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


def _limb_error(
    ctx: LimbStreamContext, positions: JointPositions, goal: np.ndarray
) -> float:
    """The environment's own success metric, which `goal_reached` thresholds.

    It is a wrapped per-joint sum, not a Euclidean norm: for six joints the two differ
    by up to a factor of sqrt(6), so certifying against the norm accepts trajectories
    the environment then calls failures.
    """
    return joint_position_distance(ctx.limb_joint_infos, list(positions), list(goal))


def advance(
    sim: ObjectCentricLimbRepositioning3DEnv, torque: JointTorques | np.ndarray
) -> HumanTorques:
    """Advance the simulation by one environment step under `torque`, as given.

    This is `sim.step()` without its observation construction or spasm sampling. It
    applies exactly what it is handed; `hold_torque` is what adds the feedforward.
    Returns the torque the step put on each of the limb's own joints.
    """
    action_space = sim.torque_action_space
    clipped = np.clip(torque, action_space.low, action_space.high)
    positions, velocities = _limb_state(sim)
    tone = np.asarray(sim.limb.get_muscle_tone_torque())
    gravity = _gravity_torque(sim, sim.limb)
    # `_apply_torques` adds the limb's muscle tone itself; its second argument is for
    # extra torque like a spasm, which planning deliberately does not model.
    sim._apply_torques(list(clipped))  # pylint: disable=protected-access
    return measure_human_torques(sim, positions, velocities, tone, gravity)


def hold_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """The torque that holds the coupled system where it stands.

    The arm carrying its own weight, plus what the grasp needs to carry
    the limb.

    Only the arm's own term is skipped when gravity is off. The limb's
    is not: muscle tone drives the limb whether or not it has weight,
    and leaving it uncompensated is what makes a weightless limb wander
    away from a zero command.
    """
    arm = np.zeros(NUM_ROBOT_JOINTS)
    if any(sim.config.gravity):
        arm = _gravity_torque(sim, sim.robot.arm)[:NUM_ROBOT_JOINTS]
    return arm + limb_hold_torque(sim)


def advance_corrected(
    sim: ObjectCentricLimbRepositioning3DEnv, correction: JointTorques | np.ndarray
) -> HumanTorques:
    """Step under `correction`, commanded on top of holding the system still.

    This is what the planner and execution use; a zero correction means hold.
    """
    return advance(sim, commanded_robot_torque(sim, correction))


def commanded_robot_torque(
    sim: ObjectCentricLimbRepositioning3DEnv, correction: JointTorques | np.ndarray
) -> np.ndarray:
    """What `advance_corrected` asks of the robot, before the action space clips it.

    The correction shares the robot's torque budget with the hold it acts on top of, so
    this, not the correction alone, is what the robot's limits bind. `advance` clips it,
    which silently truncates the hold and breaks the premise that a zero correction
    holds the system still - hence `exceeds_robot_torque_limits` rejecting it instead.
    """
    return np.asarray(correction, dtype=np.float64) + hold_torque(sim)


def exceeds_robot_torque_limits(
    ctx: LimbStreamContext, commanded: JointTorques | np.ndarray
) -> bool:
    """Whether the robot was asked for a torque outside its action space."""
    lower, upper = ctx.robot_torque_limits
    tolerance = 1e-9
    commanded_arr = np.asarray(commanded, dtype=np.float64)
    return bool(
        (commanded_arr < lower - tolerance).any()
        or (commanded_arr > upper + tolerance).any()
    )


def grasp_wrench(sim: ObjectCentricLimbRepositioning3DEnv) -> list[float]:
    """The force and moment the grasp weld transmits, as (fx, fy, fz, mx, my, mz).

    This is the interaction wrench a force-torque sensor at the gripper would read.
    """
    if not is_grasping(sim):
        return [0.0] * 6
    return list(
        p.getConstraintState(
            sim._grasp_constraint_id,  # pylint: disable=protected-access
            physicsClientId=sim.physics_client_id,
        )
    )


def advance_logged(
    sim: ObjectCentricLimbRepositioning3DEnv,
    correction: JointTorques | np.ndarray,
    log: RolloutLog,
) -> HumanTorques:
    """`advance_corrected`, recording what the step did into `log`."""
    correction = np.asarray(correction, dtype=np.float64)
    hold = hold_torque(sim)
    human = advance(sim, correction + hold)
    positions, velocities = _limb_state(sim)
    log.correction_torques.append(list(correction))
    log.commanded_torques.append(list(correction + hold))
    log.limb_positions.append(list(positions))
    log.limb_velocities.append(list(velocities))
    log.human_total.append(list(human.total))
    log.human_gravity.append(list(human.gravity))
    log.human_tone.append(list(human.tone))
    log.human_robot_induced.append(list(human.robot))
    log.grasp_wrenches.append(grasp_wrench(sim))
    return human


def apply_limb_joint_damping(
    sim: ObjectCentricLimbRepositioning3DEnv,
    damping: float = DEFAULT_LIMB_JOINT_DAMPING,
) -> None:
    """Give the limb's joints viscous damping, which the environment zeroes out.

    `_prepare_torque_control` sets every joint's damping and friction to zero so that
    the commanded torque is the only thing acting. The muscle tone model is then the
    limb's only passive resistance, and it supplies none at its rest point, so the
    joints nearest the grasp are undamped and the robot cannot hold them still.
    """
    for joint in sim.limb.arm_joints:
        p.changeDynamics(
            sim.limb.robot_id,
            joint,
            jointDamping=damping,
            physicsClientId=sim.physics_client_id,
        )


def _gravity_torque(sim: ObjectCentricLimbRepositioning3DEnv, body) -> np.ndarray:
    """The torque gravity puts on `body`'s joints where it stands.

    Inverse dynamics with zero velocity and acceleration leaves only the gravity term.
    """
    joints = sorted(body.arm_joints)
    states = p.getJointStates(
        body.robot_id, joints, physicsClientId=sim.physics_client_id
    )
    rest = [0.0] * len(joints)
    return np.asarray(
        p.calculateInverseDynamics(
            body.robot_id,
            [state[0] for state in states],
            rest,
            rest,
            physicsClientId=sim.physics_client_id,
        )
    )


def _tool_jacobian(sim: ObjectCentricLimbRepositioning3DEnv, body) -> np.ndarray:
    """The 6xN Jacobian at `body`'s tool link, as limb-manipulation computes it."""
    joints = sorted(body.arm_joints)
    states = p.getJointStates(
        body.robot_id, joints, physicsClientId=sim.physics_client_id
    )
    positions = [state[0] for state in states]
    rest = [0.0] * len(positions)
    translational, rotational = p.calculateJacobian(
        body.robot_id,
        body.tool_link_id,
        [0.0, 0.0, 0.0],
        positions,
        rest,
        rest,
        physicsClientId=sim.physics_client_id,
    )
    return np.vstack([np.asarray(translational), np.asarray(rotational)])


def _base_twist_transform(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """R, mapping robot base-frame twists into the limb's base frame."""
    rotation = matrix_from_quat(
        sim.limb.get_base_pose().orientation
    ).T @ matrix_from_quat(sim.robot.arm.get_base_pose().orientation)
    transform = np.eye(6)
    transform[:3, :3] = rotation
    transform[3:, 3:] = rotation
    return transform


def limb_hold_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """Robot joint torque that carries the limb's weight through the grasp.

    From limb-manipulation's coupled model with the robot acceleration set to zero at
    rest. The limb's own joints already carry its muscle tone, so the grasp carries only
    what gravity leaves: the coupling term is tau = Jr^T R^T pinv(Jh^T) (g_h - t_t).

    pinv(Jh^T) g_h is a wrench in the limb's base frame, so it is R^T, not R, that
    carries it into the robot's base frame. PyBullet's Jacobians are base-frame.

    Gravity and tone are both loads on the limb, so this is skipped only when there is
    neither: with gravity off the limb still has tone, and it still has to be carried.
    """
    load = _gravity_torque(sim, sim.limb) - np.asarray(
        sim.limb.get_muscle_tone_torque()
    )
    if not load.any():
        return np.zeros(NUM_ROBOT_JOINTS)
    arm_jacobian = _tool_jacobian(sim, sim.robot.arm)
    limb_jacobian = _tool_jacobian(sim, sim.limb)
    # Damped least squares, damping only near singularities: the limb passes through
    # them (a straight leg is one), where a plain pinv returns an unbounded wrench.
    smallest = float(np.linalg.svd(limb_jacobian, compute_uv=False)[-1])
    damping = 0.0
    if smallest < SINGULARITY_THRESHOLD:
        damping = JACOBIAN_DAMPING**2 * (1.0 - (smallest / SINGULARITY_THRESHOLD) ** 2)
    gram = limb_jacobian @ limb_jacobian.T
    damped = np.linalg.solve(
        gram + damping * np.eye(gram.shape[0]), limb_jacobian @ load
    )
    coupling = arm_jacobian.T @ _base_twist_transform(sim).T @ damped
    action_space = sim.torque_action_space
    return np.clip(coupling[:NUM_ROBOT_JOINTS], action_space.low, action_space.high)


def _limb_state(
    sim: ObjectCentricLimbRepositioning3DEnv,
) -> tuple[JointPositions, JointVelocities]:
    """The limb's joint positions and velocities, in one round trip."""
    states = p.getJointStates(
        sim.limb.robot_id, sim.limb.arm_joints, physicsClientId=sim.physics_client_id
    )
    return [state[0] for state in states], [state[1] for state in states]


def measure_human_torques(
    sim: ObjectCentricLimbRepositioning3DEnv,
    positions: JointPositions,
    velocities: JointVelocities,
    tone: np.ndarray,
    gravity: np.ndarray,
) -> HumanTorques:
    """The torque on each limb joint over the step that just ran, decomposed.

    Inverse dynamics on the observed acceleration gives the total generalized torque at
    each joint. PyBullet reports no reaction moment about a revolute joint's own axis,
    so this is the only way to see what the person's joints are made to bear.
    """
    _, new_velocities = _limb_state(sim)
    accelerations = np.subtract(new_velocities, velocities) / sim.config.dt
    total = np.asarray(
        p.calculateInverseDynamics(
            sim.limb.robot_id,
            list(positions),
            list(velocities),
            list(accelerations),
            physicsClientId=sim.physics_client_id,
        )
    )
    return HumanTorques(total=total, tone=tone, gravity=gravity)


def exceeds_human_torque_limits(ctx: LimbStreamContext, human: HumanTorques) -> bool:
    """Whether a step loads the person past either bound.

    The total is what their joints must bear; the robot's share is what the trajectory
    is responsible for, and carries the tighter comfort bound.
    """
    total_limit, robot_limit = ctx.human_torque_limits
    return bool(
        np.abs(human.total).max() > total_limit
        or np.abs(human.robot).max() > robot_limit
    )


# ------------------------------------------------------------------------------ streams


def sample_grasp(ctx: LimbStreamContext, limb: str) -> Iterator[tuple[Pose]]:
    """Yield the scene's grasp transform, then rolls of it about the tool axis.

    The roll leaves the contact point alone but changes the wrist orientation, which
    gives `sample_base_pose` other IK branches to try when the nominal one fails.
    """
    del limb  # there is a single limb per environment
    nominal = ctx.sim.scene.grasp_transform
    yield (nominal,)
    for roll in GRASP_ROLLS:
        yield (
            multiply_poses(nominal, Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, roll))),
        )


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


def limb_is_controllable(ctx: LimbStreamContext) -> bool:
    """Whether the robot can hold the limb where it stands.

    Two ways to fail: a limb singularity, such as a fully straight leg, where some of
    its weight can only be borne by the person's own joints; or a hold torque the arm
    cannot supply, which for a heavy limb lands on the small wrist actuators.
    """
    sim = ctx.sim
    smallest = np.linalg.svd(_tool_jacobian(sim, sim.limb), compute_uv=False)[-1]
    if smallest < SINGULARITY_THRESHOLD:
        return False
    action_space = sim.torque_action_space
    hold = limb_hold_torque(sim)
    if not hold.any():
        return True
    return bool(
        np.all(hold > action_space.low + TORQUE_FEASIBILITY_MARGIN)
        and np.all(hold < action_space.high - TORQUE_FEASIBILITY_MARGIN)
    )


def limb_out_of_limits(ctx: LimbStreamContext) -> bool:
    """Whether the limb, as it stands, is outside the person's range of motion."""
    limb = ctx.sim.limb
    return not limb.check_joint_limits(limb.get_joint_positions())


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
                if solution is None or not limb_is_controllable(ctx):
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
            overloaded = False
            for _ in range(cfg.action_repeat):
                human_torque = advance_corrected(sim, torque)
                robot_torques.append(list(torque))
                if exceeds_human_torque_limits(ctx, human_torque):
                    overloaded = True
                    break
                error = _limb_error(ctx, sim.limb.get_joint_positions(), goal)
                if error < threshold:
                    reached = True
                    break
            if overloaded:
                reached = False
                giveup = "loaded a limb joint past the person's torque limit"
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
                    giveup = "asked the robot for more torque than it has"
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
    assert s2 is not None
    if not robot_torques:
        print("plan-limb-motion: already at the target configuration.")
    else:
        print(
            f"plan-limb-motion: generated a trajectory with "
            f"{len(robot_torques)} steps."
        )
    yield (
        TorqueTrajectory(
            robot_torques, human_torques, robot_induced, limb_path, commanded
        ),
        s2,
    )


def _record_attempt(
    ctx: LimbStreamContext,
    s1: CoupledState,
    robot_torques: list[JointTorques],
    error: float,
    reason: str,
) -> None:
    """Keep this rejected rollout if it is the closest one seen so far."""
    if not robot_torques:
        return
    if ctx.best_attempt is not None and ctx.best_attempt.error <= error:
        return
    ctx.best_attempt = BestAttempt(
        state=s1, robot_torques=list(robot_torques), error=error, reason=reason
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
    """Constraint: the robot is never asked for a torque outside its action space.

    Checked against `commanded_torques`, not `robot_torques`: the correction shares the
    budget with the hold it rides on, so a correction that is individually in bounds can
    still saturate the robot once the two are summed.
    """
    del s1, q2, s2  # the constraint is on the trajectory alone
    # A trajectory assembled without a replay, as a failed attempt is, has only the
    # corrections to offer; they are a lower bound on what was asked for.
    torques = trajectory.commanded_torques or trajectory.robot_torques
    for torque in torques:
        if exceeds_robot_torque_limits(ctx, torque):
            print("check-robot-torque-limits: robot torque limit violated.")
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
            for _ in range(cfg.action_repeat):
                human_torque = advance_corrected(self._sim, torque)
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
