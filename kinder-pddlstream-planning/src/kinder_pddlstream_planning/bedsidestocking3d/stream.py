"""PDDLStream streams for BedsideStocking3D."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from kinder.envs.kinematic3d.bedside_stocking3d import (
    ObjectCentricBedsideStocking3DEnv,
    reach_margin,
    segment_blocked,
)
from kinder_models.kinematic3d.constants import HOME_JOINT_POSITIONS
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)

HOME_JOINTS = HOME_JOINT_POSITIONS.tolist()
GRASP_ORIENTATIONS = [(0.707, 0.707, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
GRASP_BELOW_TOP = 0.005
HOVER_HEIGHT = 0.08
BASE_DISTANCES = (0.45, 0.55, 0.65, 0.75)
BASE_ROTATIONS = (0.0, 0.4, -0.4)
CORRIDOR_X = 1.5
POSE_JITTER = 0.015
MAX_PLACE_POSE_ATTEMPTS = 100_000


@dataclass(eq=False)
class Placement:
    """An item pose; identity-hashed so streams never emit equal values."""

    obj: str
    pose: Pose
    surface: str


@dataclass(eq=False)
class Grasp:
    """An item-specific grasp transform; identity-hashed."""

    obj: str
    transform: Pose


@dataclass(eq=False)
class BaseConf:
    """A base pose; identity-hashed."""

    pose: SE2Pose


@dataclass(eq=False)
class ArmTrajectory:
    """Home -> hover -> grasp joint waypoints for one target item."""

    joint_plan: list
    target_object: str
    base: SE2Pose


@dataclass
class BedsideStreamContext:
    """Shared context for all streams of one problem."""

    sim: ObjectCentricBedsideStocking3DEnv
    theta_particles: NDArray
    delta: float
    rng: np.random.Generator
    requested: list[str]
    bedside_surfaces: list[str]
    all_surfaces: list[str]
    slots: dict[str, NDArray] = field(default_factory=dict)
    requested_slots: list[tuple[str, NDArray]] = field(default_factory=list)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@contextmanager
def _scratch_pose(sim: ObjectCentricBedsideStocking3DEnv, name: str, pose: Pose):
    body = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
    original = get_pose(body, sim.physics_client_id)
    set_pose(body, pose, sim.physics_client_id)
    try:
        yield
    finally:
        set_pose(body, original, sim.physics_client_id)


def usable_probability(ctx: BedsideStreamContext, o: str, pose: Pose) -> float:
    """P_b(m(grasp point, theta) >= 0) under the particle belief."""
    point = ctx.sim.grasp_point(o, pose)
    m = reach_margin(point[None], ctx.theta_particles, ctx.sim.shoulder)
    return float(np.mean(m >= 0))


NOMINAL_HALF_HEIGHT = 0.04
_CANDIDATE_CACHE: dict[tuple, list[tuple[str, NDArray]]] = {}


def _slot_ik_reachable(sim: ObjectCentricBedsideStocking3DEnv, slot: NDArray) -> bool:
    arm = sim.robot.arm
    static = sim.static_collision_ids
    item = Pose((slot[0], slot[1], slot[2] + NOMINAL_HALF_HEIGHT))
    for quat in GRASP_ORIENTATIONS:
        target = multiply_poses(item, Pose((0.0, 0.0, NOMINAL_HALF_HEIGHT - GRASP_BELOW_TOP), quat))
        hover = Pose((target.position[0], target.position[1], target.position[2] + HOVER_HEIGHT), quat)
        for base in _base_candidates(slot[0], slot[1]):
            sim.robot.set_base(base)
            if any(check_body_collisions(sim.robot.base.robot_id, b, sim.physics_client_id) for b in static):
                continue
            arm.set_joints(HOME_JOINTS)
            ok = True
            for ee in (hover, target):
                try:
                    inverse_kinematics(arm, ee, validate=True, set_joints=True)
                except InverseKinematicsError:
                    ok = False
                    break
                if any(check_body_collisions(arm.robot_id, b, sim.physics_client_id) for b in static):
                    ok = False
                    break
            if ok:
                return True
    return False


def requested_candidate_slots(sim: ObjectCentricBedsideStocking3DEnv) -> list[tuple[str, NDArray]]:
    """Bedside slots a requested item could ever satisfy: usable for the largest reach in
    the prior support and reachable by the robot's IK. Independent of the belief."""
    cfg = sim.config
    key = (tuple(sorted(cfg.bedside_regions.items())), cfg.reach_theta_ub, tuple(sim.shoulder))
    if key not in _CANDIDATE_CACHE:
        base, joints = sim.robot.get_base(), sim.robot.arm.get_joint_positions()
        out = []
        for region in cfg.bedside_regions:
            for slot in cfg.region_slots(region):
                top = slot + [0.0, 0.0, 2 * NOMINAL_HALF_HEIGHT]
                if reach_margin(top, cfg.reach_theta_ub, sim.shoulder) < 0:
                    continue
                if _slot_ik_reachable(sim, slot):
                    out.append((region, slot))
        sim.robot.set_base(base)
        sim.robot.arm.set_joints(joints)
        _CANDIDATE_CACHE[key] = out
    return _CANDIDATE_CACHE[key]


def sample_place_pose(ctx: BedsideStreamContext, o: str) -> Iterator[tuple[Placement]]:
    """Requested items cycle through the candidate slots; clutter samples every surface."""
    if o in ctx.requested:
        cands = [(s, slot[:2]) for s, slot in ctx.requested_slots]
        while True:
            for k in ctx.rng.permutation(len(cands)):
                s, xy = cands[k]
                xy = xy + ctx.rng.uniform(-POSE_JITTER, POSE_JITTER, size=2)
                yield (Placement(o, ctx.sim.resting_pose(o, tuple(xy), s), s),)
    surfaces = ctx.all_surfaces
    cands = [(s, xy) for s in surfaces for xy in ctx.slots[s][:, :2]]
    for k in ctx.rng.permutation(len(cands)):
        s, xy = cands[k]
        xy = xy + ctx.rng.uniform(-POSE_JITTER, POSE_JITTER, size=2)
        yield (Placement(o, ctx.sim.resting_pose(o, tuple(xy), s), s),)
    regions = ctx.sim.config.surface_regions
    for _ in range(MAX_PLACE_POSE_ATTEMPTS):
        s = surfaces[ctx.rng.integers(len(surfaces))]
        (x0, x1), (y0, y1), _ = regions[s]
        xy = (ctx.rng.uniform(x0 + 0.05, x1 - 0.05), ctx.rng.uniform(y0 + 0.05, y1 - 0.05))
        yield (Placement(o, ctx.sim.resting_pose(o, xy, s), s),)


def sample_grasp(ctx: BedsideStreamContext, o: str) -> Iterator[tuple[Grasp]]:
    z = ctx.sim._get_half_extents(o)[2] - GRASP_BELOW_TOP  # pylint: disable=protected-access
    for quat in GRASP_ORIENTATIONS:
        yield (Grasp(o, Pose((0.0, 0.0, z), quat)),)


def _base_candidates(x: float, y: float) -> list[SE2Pose]:
    facing = np.pi if x < CORRIDOR_X else 0.0
    out = []
    for dist in BASE_DISTANCES:
        for rot in BASE_ROTATIONS:
            ang = _wrap(facing + rot)
            out.append(SE2Pose(x - dist * np.cos(ang), y - dist * np.sin(ang), ang))
    return out


def inverse_kinematics_stream(
    ctx: BedsideStreamContext, o: str, p: Placement, g: Grasp
) -> Iterator[tuple[BaseConf, ArmTrajectory]]:
    sim = ctx.sim
    static = sim.static_collision_ids
    target_ee = multiply_poses(p.pose, g.transform)
    hover_ee = Pose(
        (target_ee.position[0], target_ee.position[1], target_ee.position[2] + HOVER_HEIGHT),
        target_ee.orientation,
    )
    target_id = sim._object_name_to_pybullet_id(o)  # pylint: disable=protected-access
    static = static | {target_id}
    with _scratch_pose(sim, o, p.pose):
        found = _ik_at(ctx, o, p, hover_ee, target_ee, static)
    if found is not None:
        yield found


def _ik_at(ctx, o, p, hover_ee, target_ee, static):
    sim = ctx.sim
    arm = sim.robot.arm
    x, y = p.pose.position[:2]
    for base in _base_candidates(x, y):
        sim.robot.set_base(base)
        if any(check_body_collisions(sim.robot.base.robot_id, b, sim.physics_client_id) for b in static):
            continue
        confs = []
        arm.set_joints(HOME_JOINTS)
        arm.open_fingers()
        for ee in (hover_ee, target_ee):
            try:
                q = inverse_kinematics(arm, ee, validate=True, set_joints=True)
            except InverseKinematicsError:
                break
            if any(check_body_collisions(arm.robot_id, b, sim.physics_client_id) for b in static):
                break
            confs.append(list(q))
        if len(confs) < 2:
            continue
        return (BaseConf(base), ArmTrajectory([HOME_JOINTS] + confs, o, base))
    return None


def test_usable(ctx: BedsideStreamContext, o: str, p: Placement) -> bool:
    return p.surface in ctx.bedside_surfaces and usable_probability(ctx, o, p.pose) >= 1 - ctx.delta


def test_clear(ctx: BedsideStreamContext, o: str, p: Placement, o2: str, p2: Placement) -> bool:
    if o == o2:
        return True
    sim = ctx.sim
    other = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    with _scratch_pose(sim, o2, p2.pose):
        return not segment_blocked(
            sim.shoulder,
            sim.grasp_point(o, p.pose),
            {other},
            sim.config.arm_radius,
            sim.physics_client_id,
        )


def test_cfree_pose_pose(
    ctx: BedsideStreamContext, o1: str, p1: Placement, o2: str, p2: Placement
) -> bool:
    if o1 == o2:
        return True
    if p1.surface != p2.surface:
        return True
    sim = ctx.sim
    id1 = sim._object_name_to_pybullet_id(o1)  # pylint: disable=protected-access
    id2 = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    with ExitStack() as stack:
        stack.enter_context(_scratch_pose(sim, o1, p1.pose))
        stack.enter_context(_scratch_pose(sim, o2, p2.pose))
        return not check_body_collisions(id1, id2, sim.physics_client_id, distance_threshold=0.01)


def test_cfree_traj_pose(
    ctx: BedsideStreamContext, t: ArmTrajectory, o2: str, p2: Placement
) -> bool:
    if o2 == t.target_object:
        return True
    sim = ctx.sim
    other = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    sim.robot.set_base(t.base)
    with _scratch_pose(sim, o2, p2.pose):
        for q in t.joint_plan[1:]:
            sim.robot.arm.set_joints(q)
            if check_body_collisions(sim.robot.arm.robot_id, other, sim.physics_client_id):
                return False
    return True
