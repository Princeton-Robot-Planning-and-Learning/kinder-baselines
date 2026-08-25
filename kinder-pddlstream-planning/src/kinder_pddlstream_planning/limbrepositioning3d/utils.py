"""Simulator, physics, and geometry helpers behind the LimbRepositioning3D streams."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

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

if TYPE_CHECKING:
    from kinder_pddlstream_planning.limbrepositioning3d.stream import (
        ArmTrajectory,
        LimbStreamContext,
    )

JointTorques = list[float]

# The (dx, dy, drot) grid `sample_base_pose` searches, nearest-first.
BASE_SEARCH_RADIUS = 0.4

BASE_SEARCH_STEP = 0.1

BASE_SEARCH_ROTATIONS: tuple[float, ...] = (0.0, -0.2, 0.2, -0.4, 0.4)

CLEARANCE_PROBE_DISTANCE = 0.1

COLLISION_MARGIN = 0.01

# Bound on the total torque a joint of the person may bear, in N*m, per limb.
HUMAN_TORQUE_LIMITS = {"arm": 50.0, "leg": 100.0}  # TODO: Tune these values based on realistic human joint limits.

# What the robot itself may add on top of gravity and tone, in N*m.
DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT = 10.0

# Kinova Gen3 joint effort limits, from the URDF the environment loads.
ROBOT_TORQUE_LIMITS = (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0)

# Damping for the limb Jacobian inverse, applied only near singularities.
JACOBIAN_DAMPING = 0.1

SINGULARITY_THRESHOLD = 0.02

# Torque headroom, in N*m, a base pose must leave for the motion itself.
TORQUE_FEASIBILITY_MARGIN = 1.0

DEFAULT_GRAVITY = (0.0, 0.0, -9.81)

# Viscous damping at each of the limb's own joints, in N*m*s/rad.
DEFAULT_LIMB_JOINT_DAMPING = 0.5


@dataclass(eq=False)
class CoupledState:
    """A full coupled state of the robot and limb, base pose included for the weld."""

    base_pose: SE2Pose
    robot_positions: JointPositions
    robot_velocities: JointVelocities
    limb_positions: JointPositions
    limb_velocities: JointVelocities
    approach: ArmTrajectory | None = None

    def __repr__(self) -> str:
        return f"s{id(self) % 10000}"


@dataclass(frozen=True)
class HumanTorques:
    """The torque on the human's joints, and the two references it is read against."""

    total: np.ndarray
    tone: np.ndarray
    gravity: np.ndarray

    @property
    def robot(self) -> np.ndarray:
        """What the motion adds over the static load; zero while merely holding.

        Tone is already inside `total`, so subtracting it again would leave `-tone`.
        """
        return self.total - self.gravity


@dataclass
class RolloutLog:
    """Per-step measurements of an executed `move_limb`, filled by `advance_logged`."""

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
    """The closest a rejected limb motion got, kept to visualize failures."""

    state: CoupledState
    robot_torques: list[JointTorques]
    error: float
    reason: str


def default_human_torque_limit(limb_name: str) -> float:
    """The total-torque bound for this limb: legs bear far more than arms."""
    return HUMAN_TORQUE_LIMITS["leg" if "leg" in limb_name else "arm"]


def extend_with_fingers(joints: JointPositions) -> JointPositions:
    """Pad arm joint positions with the six Robotiq finger joints."""
    return list(joints) + [0.0] * 6


def is_grasping(sim: ObjectCentricLimbRepositioning3DEnv) -> bool:
    """Whether the end effector is currently welded to the limb."""
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
    """(Re)weld the end effector to the limb where they stand, at zero error."""
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
    """Put the simulator back into `state`; `regrasp` rebuilds the weld it stored."""
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
    """The environment's own success metric: a wrapped per-joint sum, not a norm."""
    return joint_position_distance(ctx.limb_joint_infos, list(positions), list(goal))


def advance(
    sim: ObjectCentricLimbRepositioning3DEnv, torque: JointTorques | np.ndarray
) -> HumanTorques:
    """Step once under `torque` as given, returning the limb's own joint torque."""
    action_space = sim.torque_action_space
    clipped = np.clip(torque, action_space.low, action_space.high)
    positions, velocities = _limb_state(sim)
    tone = np.asarray(sim.limb.get_muscle_tone_torque())
    gravity = _gravity_torque(sim, sim.limb)
    # `_apply_torques` adds muscle tone itself; its second argument is for spasms.
    sim._apply_torques(list(clipped))  # pylint: disable=protected-access
    return measure_human_torques(sim, positions, velocities, tone, gravity)


def hold_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """The torque holding the coupled system still: the arm's weight plus the limb's."""
    arm = np.zeros(NUM_ROBOT_JOINTS)
    if any(sim.config.gravity):
        arm = _gravity_torque(sim, sim.robot.arm)[:NUM_ROBOT_JOINTS]
    return arm + limb_hold_torque(sim)


def advance_corrected(
    sim: ObjectCentricLimbRepositioning3DEnv, correction: JointTorques | np.ndarray
) -> HumanTorques:
    """Step under `correction` on top of the hold, so a zero correction holds still."""
    return advance(sim, commanded_robot_torque(sim, correction))


def commanded_robot_torque(
    sim: ObjectCentricLimbRepositioning3DEnv, correction: JointTorques | np.ndarray
) -> np.ndarray:
    """Correction plus hold - what the robot is asked for, and what its limits bind."""
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
    """The force and moment the grasp weld transmits, as (fx, fy, fz, mx, my, mz)."""
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
    """Give the limb's joints the damping `_prepare_torque_control` zeroes out."""
    for joint in sim.limb.arm_joints:
        p.changeDynamics(
            sim.limb.robot_id,
            joint,
            jointDamping=damping,
            physicsClientId=sim.physics_client_id,
        )


def _gravity_torque(sim: ObjectCentricLimbRepositioning3DEnv, body) -> np.ndarray:
    """The torque gravity puts on `body`'s joints, from inverse dynamics at rest."""
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
    """Robot joint torque carrying the limb's load through the grasp.

    tau = Jr^T R^T pinv(Jh^T) (g_h - t_t); R^T, not R, as the wrench is limb-frame.
    """
    load = _gravity_torque(sim, sim.limb) - np.asarray(
        sim.limb.get_muscle_tone_torque()
    )
    if not load.any():
        return np.zeros(NUM_ROBOT_JOINTS)
    arm_jacobian = _tool_jacobian(sim, sim.robot.arm)
    limb_jacobian = _tool_jacobian(sim, sim.limb)
    # Damped least squares, damping only near singularities (a straight leg is one).
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
    """The torque on each limb joint over the step just run, from inverse dynamics."""
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
    """Whether a step loads the person past either bound, total or the robot's share."""
    total_limit, robot_limit = ctx.human_torque_limits
    return bool(
        np.abs(human.total).max() > total_limit
        or np.abs(human.robot).max() > robot_limit
    )


def _limb_slide_axis(
    sim: ObjectCentricLimbRepositioning3DEnv,
) -> tuple[np.ndarray, float]:
    """Unit vector up the limb in the grasp frame, and the span to its joint."""
    limb = sim.limb
    client = sim.physics_client_id
    target = "lower_arm" if "arm" in limb.get_name() else "lower_leg"
    index = next(
        joint
        for joint in range(p.getNumJoints(limb.robot_id, physicsClientId=client))
        if p.getJointInfo(limb.robot_id, joint, physicsClientId=client)[12].decode()
        == target
    )
    joint_position = p.getLinkState(limb.robot_id, index, physicsClientId=client)[4]
    ee_pose = limb.get_end_effector_pose()
    offset = np.subtract(joint_position, ee_pose.position)
    span = float(np.linalg.norm(offset))
    return matrix_from_quat(ee_pose.orientation).T @ offset / span, span


def _perturbed_grasp(
    ctx: LimbStreamContext, nominal: Pose, slide: float, roll: float
) -> Pose:
    """`nominal` slid `slide` metres up the limb and rolled `roll` about the tool."""
    # The slide is left-multiplied so it lands in the limb frame, the roll right.
    offset = Pose(tuple((ctx.grasp_slide_axis * slide).tolist()))
    spin = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, roll))
    return multiply_poses(offset, nominal, spin)


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
    """Gap between two bodies in meters, negative if overlapping, capped at probe."""
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
    """Whether the arm overlaps the person or furniture; None leaves it where it is."""
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
    """Whether the robot can hold the limb here: not singular, and within arm torque."""
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
    """Whether the limb was driven into the person, past its resting overlap."""
    sim = ctx.sim
    for body_id, resting in ctx.resting_penetration.items():
        clearance = _pair_clearance(sim, sim.limb.robot_id, body_id)
        if clearance < resting - COLLISION_MARGIN:
            return True
    return False


def _random_arm_seed(ctx: LimbStreamContext) -> JointPositions:
    """A random arm configuration to restart IK from, fingers left closed."""
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
    """Collision-free arm joints holding `grasp` at `limb_positions`, or None."""
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


def _base_in_collision(ctx: LimbStreamContext) -> bool:
    """Whether the base alone overlaps the person or furniture, not the arm."""
    sim = ctx.sim
    return any(
        check_body_collisions(
            sim.robot.base.robot_id, obstacle_id, sim.physics_client_id
        )
        for obstacle_id in ctx.base_obstacle_ids
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
