"""Simulator, physics, and geometry helpers behind the LimbRepositioning3D streams."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

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


@dataclass
class StreamProfile:
    """Wall-clock seconds and call counts, keyed by stream or by inner stage.

    Stage keys nest inside stream keys, so their totals overlap by design.

    Each key also splits by whether the call produced anything, since a stream that
    burns its budget and yields nothing is what makes planning slow.
    """

    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    # The subset of the above spent on calls that produced no usable result.
    wasted_seconds: dict[str, float] = field(default_factory=dict)
    wasted_calls: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def timed(self, key: str) -> Iterator[None]:
        """Charge the wrapped block to `key`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - start)

    def wrap(self, key: str, iterator: Iterator) -> Iterator:
        """Charge each `next()` of `iterator` to `key`, leaving it lazy.

        A call cut short still charges what it spent, or a run interrupted inside a
        stream reads as if the time went to the search.
        """
        while True:
            start = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                # The call that exhausts the stream produced nothing for its time.
                self.add(key, time.perf_counter() - start, produced=False)
                return
            except BaseException:  # pylint: disable=broad-except
                self.add(key, time.perf_counter() - start, produced=False)
                raise
            self.add(key, time.perf_counter() - start, produced=True)
            yield item

    def add(self, key: str, seconds: float, produced: bool | None = None) -> None:
        """Charge `seconds` to `key`, recording it as wasted if it produced nothing."""
        self.seconds[key] = self.seconds.get(key, 0.0) + seconds
        self.calls[key] = self.calls.get(key, 0) + 1
        if produced is False:
            self.wasted_seconds[key] = self.wasted_seconds.get(key, 0.0) + seconds
            self.wasted_calls[key] = self.wasted_calls.get(key, 0) + 1

    def as_dict(self) -> dict[str, dict[str, float]]:
        """A JSON-writable view of the totals."""
        return {
            key: {
                "seconds": self.seconds[key],
                "calls": self.calls[key],
                "wasted_seconds": self.wasted_seconds.get(key, 0.0),
                "wasted_calls": self.wasted_calls.get(key, 0),
            }
            for key in sorted(self.seconds)
        }


@dataclass
class StreamLog:
    """One record per stream call, with inputs and outputs as serial object ids.

    Streams attach what they learned about a call through `note`, e.g. why it failed.
    """

    records: list[dict[str, Any]] = field(default_factory=list)
    objects: list[Any] = field(default_factory=list, repr=False)
    _serials: dict[int, int] = field(default_factory=dict, repr=False)
    _notes: dict[str, Any] = field(default_factory=dict, repr=False)

    def ref(self, obj: Any) -> int:
        """A serial id for `obj`, kept alive so `id()` is never reused."""
        if id(obj) not in self._serials:
            self._serials[id(obj)] = len(self.objects)
            self.objects.append(obj)
        return self._serials[id(obj)]

    def note(self, **info: Any) -> None:
        """Attach `info` to the call in progress."""
        self._notes.update(info)

    def record(
        self,
        stream: str,
        inputs: tuple,
        outputs: tuple | list | None,
        seconds: float,
        **info: Any,
    ) -> None:
        """Close the call in progress; `outputs` None means the stream is exhausted."""
        self.records.append(
            {
                "call": len(self.records),
                "stream": stream,
                "inputs": [self.ref(x) for x in inputs],
                "outputs": None if outputs is None else [self.ref(x) for x in outputs],
                "seconds": seconds,
                **info,
                **self._notes,
            }
        )
        self._notes = {}


NUM_BASE_SEARCH_ROTATIONS = 24
BASE_SEARCH_ROTATIONS: tuple[float, ...] = tuple(
    2.0 * np.pi * i / NUM_BASE_SEARCH_ROTATIONS
    for i in range(NUM_BASE_SEARCH_ROTATIONS)
)
NUM_SAMPLED_BASE_POSES = 400
NUM_REACH_CLOUD_SAMPLES = 2000
REACH_HEIGHT_TOLERANCE = 0.05
REACH_RADIUS_PERCENTILES = (55.0, 90.0)

IK_PROBE_ORIGIN = (50.0, 50.0)
CLEARANCE_PROBE_DISTANCE = 0.1
COLLISION_MARGIN = 0.01
HUMAN_TORQUE_LIMITS = {
    "arm": 50.0,
    "leg": 100.0,
}  # Placeholders, to be tuned to realistic human joint limits.
DEFAULT_ROBOT_INDUCED_TORQUE_LIMIT = 25.0
ROBOT_TORQUE_LIMITS = (200.0, 200.0, 200.0, 200.0, 100.0, 100.0, 100.0)
JACOBIAN_DAMPING = 0.1
SINGULARITY_THRESHOLD = 0.02
TORQUE_FEASIBILITY_MARGIN = 1.0
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
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
class BestAttempt:
    """The closest a rejected limb motion got, kept to visualize failures."""

    state: CoupledState
    robot_torques: list[JointTorques]
    error: float
    reason: str
    plan_seconds: list[float] = field(default_factory=list)


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
def saved_sim_state(sim: ObjectCentricLimbRepositioning3DEnv) -> Iterator[None]:
    """Restore the simulator (including its grasp constraint) on the way out."""
    saved = capture_state(sim)
    was_grasping = is_grasping(sim)
    try:
        yield
    finally:
        if p.isConnected(physicsClientId=sim.physics_client_id):
            restore_state(sim, saved, regrasp=was_grasping)


def limb_error(
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


def arm_gravity_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """The torque answering the arm's own weight, zero when gravity is off."""
    if not any(sim.config.gravity):
        return np.zeros(NUM_ROBOT_JOINTS)
    return _gravity_torque(sim, sim.robot.arm)[:NUM_ROBOT_JOINTS]


def hold_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """The torque holding the coupled system still: the arm's weight plus the limb's."""
    return arm_gravity_torque(sim) + limb_hold_torque(sim)


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


def _robot_torque_for_limb_load(
    sim: ObjectCentricLimbRepositioning3DEnv, load: np.ndarray
) -> np.ndarray:
    """tau = Jr^T R^T pinv(Jh^T) load, unclipped; R^T as the wrench is limb-frame."""
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
    wrench = _base_twist_transform(sim).T @ damped
    # The wrench acts at the limb's grasp frame; the arm's Jacobian is at its own tool
    # frame, and a slid grasp separates the two, so shift the moment across the offset.
    offset_world = np.subtract(
        sim.limb.get_end_effector_pose().position,
        sim.robot.arm.get_end_effector_pose().position,
    )
    offset = (
        matrix_from_quat(sim.robot.arm.get_base_pose().orientation).T @ offset_world
    )
    wrench[3:] = wrench[3:] + np.cross(offset, wrench[:3])
    return (arm_jacobian.T @ wrench)[:NUM_ROBOT_JOINTS]


def _limb_coupling_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """The robot torque answering the limb's own gravity load, net of muscle tone."""
    load = _gravity_torque(sim, sim.limb) - np.asarray(
        sim.limb.get_muscle_tone_torque()
    )
    return _robot_torque_for_limb_load(sim, load)


def limb_directed_torque(
    sim: ObjectCentricLimbRepositioning3DEnv, goal: np.ndarray, gain: float
) -> np.ndarray:
    """The robot torque driving the limb toward `goal`, through the same coupling
    chain."""
    error = np.asarray(goal) - np.asarray(sim.limb.get_joint_positions())
    return _robot_torque_for_limb_load(sim, gain * error)


def limb_hold_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """The coupling torque clipped to what the robot can actually be commanded."""
    action_space = sim.torque_action_space
    return np.clip(_limb_coupling_torque(sim), action_space.low, action_space.high)


def unclipped_hold_torque(sim: ObjectCentricLimbRepositioning3DEnv) -> np.ndarray:
    """What holding still would need, before the action space truncates it."""
    return arm_gravity_torque(sim) + _limb_coupling_torque(sim)


def hold_saturation(sim: ObjectCentricLimbRepositioning3DEnv) -> float:
    """Peak hold as a fraction of the torque limit; above 1.0 the clip truncates it."""
    return float(
        np.max(np.abs(unclipped_hold_torque(sim)) / sim.torque_action_space.high)
    )


def grasp_hold_ratio(
    ctx: LimbStreamContext, grasp: Pose, limb_positions: JointPositions
) -> float:
    """`hold_saturation` for holding `grasp` at `limb_positions`, or inf if
    unreachable."""
    solution = grasp_ik(ctx, grasp, tuple(limb_positions), ctx.retract_joints)
    if solution is None:
        return float("inf")
    ctx.sim.robot.arm.set_joints(list(solution))
    return hold_saturation(ctx.sim)


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
    return HumanTorques(
        total=total,
        tone=tone,
        gravity=gravity,
    )


def exceeds_human_torque_limits(ctx: LimbStreamContext, human: HumanTorques) -> bool:
    """Whether a step loads the person past either bound, total or the robot's share."""
    total_limit, robot_limit = ctx.human_torque_limits
    return bool(
        np.abs(human.total).max() > total_limit
        or np.abs(human.robot).max() > robot_limit
    )


def limb_slide_axis(
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


def slid_grasp(
    ctx: LimbStreamContext, nominal: Pose, slide: float, roll: float = 0.0
) -> Pose:
    """`nominal` slid `slide` up the limb and rolled `roll`, the slide in the limb
    frame."""
    offset = Pose(tuple((ctx.grasp_slide_axis * slide).tolist()))
    spin = Pose.from_rpy((0.0, 0.0, 0.0), (0.0, 0.0, roll))
    return multiply_poses(offset, nominal, spin)


def _wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi], which is what `SE2Pose` asserts its rotation lies in."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _yaw_matrix(angle: float) -> np.ndarray:
    """Rotation about z, the only rotation a level mobile base can contribute."""
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _arm_reach_cloud(ctx: LimbStreamContext) -> np.ndarray:
    """(radius, height) the tool reaches in the arm's own frame, sampled once."""
    if ctx._reach_cloud is None:  # pylint: disable=protected-access
        arm = ctx.sim.robot.arm
        points = []
        for _ in range(NUM_REACH_CLOUD_SAMPLES):
            arm.set_joints(_random_arm_seed(ctx))
            base = arm.get_base_pose()
            offset = np.subtract(arm.get_end_effector_pose().position, base.position)
            local = matrix_from_quat(base.orientation).T @ offset
            points.append((float(np.hypot(local[0], local[1])), float(local[2])))
        ctx._reach_cloud = np.asarray(points)  # pylint: disable=protected-access
    return ctx._reach_cloud  # pylint: disable=protected-access


def _arm_reach_radii(ctx: LimbStreamContext, height: float) -> tuple[float, float]:
    """The horizontal radii the tool reaches at `height`, or an empty range."""
    cloud = _arm_reach_cloud(ctx)
    near = cloud[np.abs(cloud[:, 1] - height) < REACH_HEIGHT_TOLERANCE]
    if len(near) < 2:
        return 0.0, -1.0
    low, high = np.percentile(near[:, 0], REACH_RADIUS_PERCENTILES)
    return float(low), float(high)


def base_candidates(
    ctx: LimbStreamContext, grasp: Pose, limb_positions: JointPositions
) -> Iterator[SE2Pose]:
    """Base poses that put `grasp` where the arm can reach it."""
    sim = ctx.sim
    sim.limb.set_joints(list(limb_positions))
    target = multiply_poses(sim.limb.get_end_effector_pose(), grasp)
    mount = np.asarray(sim.robot.base_to_arm_transform.position)
    height = target.position[2] - sim.robot.base.z - mount[2]
    low, high = _arm_reach_radii(ctx, height)
    if high <= low:
        return
    rng = ctx._base_rng  # pylint: disable=protected-access
    target_rotation = matrix_from_quat(target.orientation)
    for index in range(NUM_SAMPLED_BASE_POSES):
        yaw = _wrap_angle(BASE_SEARCH_ROTATIONS[index % len(BASE_SEARCH_ROTATIONS)])
        # Area-uniform, so the base is not crowded onto the target and its furniture.
        radius = float(np.sqrt(rng.uniform(low**2, high**2)))
        bearing = float(rng.uniform(-np.pi, np.pi))
        local = np.array([radius * np.cos(bearing), radius * np.sin(bearing), height])
        pose = Pose.from_matrix(
            np.block(
                [
                    [_yaw_matrix(-yaw) @ target_rotation, local.reshape(3, 1)],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            )
        )
        sim.robot.set_base(SE2Pose(*IK_PROBE_ORIGIN, yaw))
        try:
            inverse_kinematics(
                sim.robot.arm,
                multiply_poses(sim.robot.arm.get_base_pose(), pose),
                validate=True,
                set_joints=False,
            )
        except InverseKinematicsError:
            continue
        offset = _yaw_matrix(yaw) @ (mount + local)
        base_pose = SE2Pose(
            target.position[0] - offset[0], target.position[1] - offset[1], yaw
        )
        sim.robot.set_base(base_pose)
        if ctx.check_base_collisions and base_in_collision(ctx):
            continue
        yield base_pose


def pair_clearance(
    sim, body_a: int, body_b: int, probe: float = CLEARANCE_PROBE_DISTANCE
) -> float:
    """Gap between two bodies in meters, negative if overlapping, capped at probe."""
    points = p.getClosestPoints(
        body_a,
        body_b,
        distance=probe,
        physicsClientId=sim.physics_client_id,
    )
    if not points:
        return probe
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


def control_saturation(
    ctx: LimbStreamContext, arm_positions: JointPositions | None = None
) -> float:
    """How hard the arm must work to hold the limb still, as a fraction of its torque.

    1.0 is the feasibility limit, and infinity means the grasp is singular.

    Measured at `arm_positions` if given.
    """
    sim = ctx.sim
    if arm_positions is not None:
        sim.robot.arm.set_joints(list(arm_positions))
    smallest = np.linalg.svd(_tool_jacobian(sim, sim.limb), compute_uv=False)[-1]
    if smallest < SINGULARITY_THRESHOLD:
        return float("inf")
    action_space = sim.torque_action_space
    hold = unclipped_hold_torque(sim)
    if not hold.any():
        return 0.0
    upper = np.asarray(action_space.high) - TORQUE_FEASIBILITY_MARGIN
    lower = np.asarray(action_space.low) + TORQUE_FEASIBILITY_MARGIN
    return float(np.max(np.maximum(hold / upper, hold / lower)))


def limb_is_controllable(
    ctx: LimbStreamContext, arm_positions: JointPositions | None = None
) -> bool:
    """Not singular and within arm torque, measured at `arm_positions` if given."""
    return control_saturation(ctx, arm_positions) <= 1.0


def limb_out_of_limits(ctx: LimbStreamContext) -> bool:
    """Whether the limb, as it stands, is outside the person's range of motion."""
    return not ctx.in_believed_limits(ctx.sim.limb.get_joint_positions())


def human_in_collision(ctx: LimbStreamContext) -> bool:
    """Whether the limb was driven into the person, past its resting overlap."""
    sim = ctx.sim
    for body_id, resting in ctx.resting_penetration.items():
        clearance = pair_clearance(sim, sim.limb.robot_id, body_id)
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


def grasp_ik(
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


def base_in_collision(ctx: LimbStreamContext) -> bool:
    """Whether the base alone overlaps the person or furniture, not the arm."""
    sim = ctx.sim
    return any(
        check_body_collisions(
            sim.robot.base.robot_id, obstacle_id, sim.physics_client_id
        )
        for obstacle_id in ctx.base_obstacle_ids
    )


def record_attempt(
    ctx: LimbStreamContext,
    s1: CoupledState,
    robot_torques: list[JointTorques],
    error: float,
    reason: str,
    plan_seconds: list[float] | None = None,
) -> None:
    """Keep this rejected rollout if it is the closest one seen so far."""
    if not robot_torques:
        return
    if ctx.best_attempt is not None and ctx.best_attempt.error <= error:
        return
    ctx.best_attempt = BestAttempt(
        state=s1,
        robot_torques=list(robot_torques),
        error=error,
        reason=reason,
        plan_seconds=list(plan_seconds or []),
    )
