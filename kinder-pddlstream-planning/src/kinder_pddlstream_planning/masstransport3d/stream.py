"""PDDLStream stream implementations for MassTransport3D.

See run.py for the domain design.

Each function takes the `MassStreamContext` built by `create_problem`. Streams move the
sim's bodies and robot as scratch space; callers reset the sim before execution.

Two stream instances must never output equal values: pddlstream would make them one
object and then trip its `update_bindings` assertion. Hence item-specific grasps and
identity-hashed base confs and trajectories.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from kinder.envs.dynamic3d.kinematic_masstransport3d import (
    MassTransport3DEnvConfig,
    ObjectCentricKinematicMassTransport3DEnv,
    holding_torques_at,
)
from kinder_models.kinematic3d.constants import HOME_JOINT_POSITIONS
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    check_body_collisions,
    inverse_kinematics,
)
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    smoothly_follow_end_effector_path,
)
from scipy.spatial.transform import Rotation

_HOME_JOINTS = HOME_JOINT_POSITIONS.tolist()

# Height of the end effector above the grasp point before the vertical descent.
HOVER_HEIGHT = 0.15

# Carry-conf parameters: distance from the base center to the tote center, and lift
# height above the table. Tried in this order before random samples.
CARRY_DISTANCES = (0.75, 0.65, 0.85)
CARRY_LIFTS = (0.15, 0.25)
CARRY_DISTANCE_BOUNDS = (0.6, 0.9)
CARRY_LIFT_BOUNDS = (0.1, 0.3)
MAX_CARRY_SAMPLES = 20

# A placement this far above its support keeps the sim's collision check from
# reverting the final step; the sim accepts a release within `min_placement_dist`.
PLACEMENT_CLEARANCE = 2.5e-3

# Margin around the tote when checking an item pose against it.
TOTE_CLEARANCE = 0.03

# Top-down grasps: rotate the gripper so its fingers close along the object's y or x.
_DOWN_CLOSE_Y = Rotation.from_quat((0.707, -0.707, 0.0, 0.0))
_DOWN_CLOSE_X = _DOWN_CLOSE_Y * Rotation.from_euler("z", np.pi / 2)


def _quat(rotation: Rotation) -> tuple[float, float, float, float]:
    x, y, z, w = rotation.as_quat()
    return (float(x), float(y), float(z), float(w))


@dataclass(frozen=True)
class ItemGrasp:
    """A top-down grasp on `item`.

    `transform` is the end effector in the item frame.
    """

    item: str
    name: str
    transform: Pose
    closing_axis: str  # "x" or "y", in the item frame

    def __repr__(self) -> str:
        return f"ItemGrasp({self.item}, {self.name})"


@dataclass(eq=False)
class BaseConf:
    """A base pose.

    Hashed by identity, so each stream output is its own object.
    """

    pose: SE2Pose

    def __repr__(self) -> str:
        return f"BaseConf({self.pose.x:.2f}, {self.pose.y:.2f}, {self.pose.rot:.2f})"


@dataclass(frozen=True)
class ToteGrasp:
    """A grasp on the rim of one tote wall.

    `transform` is the end effector in the tote frame. `yaw_in_base` is the tote's yaw
    relative to the robot base that puts the grasped wall nearest the robot, so the tote
    extends away from it.
    """

    name: str
    transform: Pose
    yaw_in_base: float

    def __repr__(self) -> str:
        return f"ToteGrasp({self.name})"


@dataclass(eq=False)
class CarryConf:
    """How the robot holds the tote, relative to its base.

    The tote rests at `tote_in_base` when the arm is at `low_joints`, and is lifted
    straight up to `high_joints` along `lift_path`. The base is only ever yawed, so the
    same carry conf serves any tote pose on either table.
    """

    grasp: ToteGrasp
    tote_in_base: Pose
    low_joints: list
    high_joints: list
    lift_path: list  # from low_joints to high_joints

    def __repr__(self) -> str:
        x, _, _ = self.tote_in_base.position
        lift = len(self.lift_path)
        return f"CarryConf({self.grasp.name}, d={x:.2f}, lift_waypoints={lift})"


@dataclass(eq=False)
class ArmTrajectory:
    """A one-way joint-space path from the home joints, run with the base at `base`.

    Its contact with `target_object` (and the tote at `tote_pose`, if any) is intended.
    """

    joint_plan: list
    base: SE2Pose
    target_object: str
    tote_pose: Pose | None = None

    def __repr__(self) -> str:
        return f"ArmTrajectory({self.target_object}, {len(self.joint_plan)} waypoints)"


@dataclass
class MassStreamContext:
    """Shared, precomputed context for all streams of one `create_problem` call."""

    sim: ObjectCentricKinematicMassTransport3DEnv
    motion_seed: int
    q_home: SE2Pose
    item_names: list[str]
    # The masses the planner believes. They need not be the sim's true masses.
    item_masses: dict[str, float]
    slots: dict[str, Pose]
    tote_grasps: list[ToteGrasp]
    item_grasps: dict[str, list[ItemGrasp]]
    dest_tote_poses: list[Pose]
    # One base conf in front of each table serves every item and slot there.
    table_bases: dict[str, BaseConf]
    dest_spots: list[Pose]
    torque_limits: np.ndarray
    # Fraction of the torque limits the planner keeps in reserve.
    torque_margin: float = 0.0
    min_item_gap: float = 0.1
    table_ids: set[int] = field(default_factory=set)

    @property
    def item_half(self) -> float:
        """Half the item cube size."""
        return self.sim.config.item_size / 2


def make_item_grasps(item: str, item_size: float) -> list[ItemGrasp]:
    """Two top-down grasps, with the fingers closing along the item's y or x."""
    position = (0.0, 0.0, 0.4 * item_size)
    return [
        ItemGrasp(item, "close-y", Pose(position, _quat(_DOWN_CLOSE_Y)), "y"),
        ItemGrasp(item, "close-x", Pose(position, _quat(_DOWN_CLOSE_X)), "x"),
    ]


def make_tote_grasps(
    half_extents: tuple[float, float, float], wall_thickness: float
) -> list[ToteGrasp]:
    """One rim grasp per wall, with the fingers straddling the wall."""
    hx, hy, hz = half_extents
    wall_x = hx - wall_thickness / 2
    wall_y = hy - wall_thickness / 2
    z = hz - 0.02
    return [
        ToteGrasp("wall-x-", Pose((-wall_x, 0.0, z), _quat(_DOWN_CLOSE_X)), 0.0),
        ToteGrasp("wall-x+", Pose((wall_x, 0.0, z), _quat(_DOWN_CLOSE_X)), np.pi),
        ToteGrasp("wall-y-", Pose((0.0, -wall_y, z), _quat(_DOWN_CLOSE_Y)), -np.pi / 2),
        ToteGrasp("wall-y+", Pose((0.0, wall_y, z), _quat(_DOWN_CLOSE_Y)), np.pi / 2),
    ]


def make_slots(
    half_extents: tuple[float, float, float],
    wall_thickness: float,
    item_size: float,
    num_x: int = 2,
    num_y: int = 3,
) -> dict[str, Pose]:
    """Item poses inside the tote, in the tote frame, on an evenly spaced grid.

    Items resting in the slots are inside the tote's bounding box, so the sim carries
    them along with it.
    """
    hx, hy, hz = half_extents
    z = -hz + wall_thickness + item_size / 2 + PLACEMENT_CLEARANCE
    inner_x = hx - wall_thickness
    inner_y = hy - wall_thickness
    xs = np.linspace(-inner_x, inner_x, num_x + 2)[1:-1] if num_x > 1 else [0.0]
    ys = np.linspace(-inner_y, inner_y, num_y + 2)[1:-1] if num_y > 1 else [0.0]
    slots: dict[str, Pose] = {}
    for x in xs:
        for y in ys:
            slots[f"slot{len(slots)}"] = Pose((float(x), float(y), z))
    return slots


def _yaw_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    return Pose((x, y, z), _quat(Rotation.from_euler("z", yaw)))


def _normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@contextmanager
def _scratch_pose(
    sim: ObjectCentricKinematicMassTransport3DEnv, name: str, pose: Pose
) -> Iterator[None]:
    """Temporarily move a body to `pose`, restoring it afterward."""
    body_id = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
    original = get_pose(body_id, sim.physics_client_id)
    set_pose(body_id, pose, sim.physics_client_id)
    try:
        yield
    finally:
        set_pose(body_id, original, sim.physics_client_id)


@contextmanager
def _parked_bodies(sim: ObjectCentricKinematicMassTransport3DEnv) -> Iterator[None]:
    """Move every movable body far away, so streams see only what they place."""
    with ExitStack() as stack:
        for index, name in enumerate(sorted(sim.item_names) + ["tote"]):
            stack.enter_context(
                _scratch_pose(sim, name, Pose((100.0 + index, 100.0, 0.0)))
            )
        yield


def _base_collides(ctx: MassStreamContext, base: SE2Pose) -> bool:
    """Whether the mobile base at `base` hits a table."""
    sim = ctx.sim
    sim.robot.set_base(base)
    return any(
        check_body_collisions(sim.robot.base.robot_id, table, sim.physics_client_id)
        for table in ctx.table_ids
    )


def _payload_offset(transform_to_ee: Pose) -> Pose:
    """The payload's center of mass in the end-effector frame, from its grasp."""
    return transform_to_ee.invert()


def _within_torque_limits(
    ctx: MassStreamContext,
    joint_plan: list,
    payload_mass: float,
    payload_offset: Pose | None,
) -> bool:
    """Whether every conf in `joint_plan` can hold the payload."""
    sim = ctx.sim
    limits = (1.0 - ctx.torque_margin) * ctx.torque_limits
    original = sim.robot.arm.get_joint_positions()
    try:
        for joints in joint_plan:
            torques = holding_torques_at(
                sim.robot.arm,
                sim.physics_client_id,
                joints,
                payload_mass,
                payload_offset,
                gravity=sim.config.gravity,
            )
            if np.any(np.abs(torques) > limits):
                return False
        return True
    finally:
        sim.robot.arm.set_joints(original)


def _plan_vertical_reach(
    ctx: MassStreamContext,
    base: SE2Pose,
    target_ee_pose: Pose,
    collision_bodies: set[int],
) -> list | None:
    """A path from the home joints to `target_ee_pose`, ending in a straight descent."""
    sim = ctx.sim
    hover_ee_pose = Pose(
        (
            target_ee_pose.position[0],
            target_ee_pose.position[1],
            target_ee_pose.position[2] + HOVER_HEIGHT,
        ),
        target_ee_pose.orientation,
    )
    sim.robot.set_base(base)
    sim.robot.arm.set_joints(_HOME_JOINTS)
    try:
        hover_joints = inverse_kinematics(
            sim.robot.arm, hover_ee_pose, validate=True, set_joints=False
        )
    except InverseKinematicsError:
        return None
    general_plan = _plan_arm_motion(ctx, _HOME_JOINTS, hover_joints, collision_bodies)
    if general_plan is None:
        return None
    sim.robot.arm.set_joints(general_plan[-1])
    try:
        vertical_plan = smoothly_follow_end_effector_path(
            sim.robot.arm,
            [hover_ee_pose, target_ee_pose],
            initial_joints=general_plan[-1],
            collision_ids=collision_bodies,
            joint_distance_fn=create_joint_distance_fn(sim.robot.arm),
            seed=ctx.motion_seed,
            include_start=False,
        )
    except InverseKinematicsError:
        return None
    if not vertical_plan:
        return None
    return remap_joint_position_plan_to_constant_distance(
        general_plan + vertical_plan,
        sim.robot.arm,
        max_distance=sim.config.max_action_mag / 2,
    )


def _plan_arm_motion(
    ctx: MassStreamContext, start: list, goal: list, collision_bodies: set[int]
) -> list | None:
    """A collision-free arm path, trying the straight joint-space path first."""
    sim = ctx.sim
    for direct_path_only in (True, False):
        plan = run_motion_planning(
            sim.robot.arm,
            initial_positions=start,
            target_positions=goal,
            collision_bodies=collision_bodies,
            seed=ctx.motion_seed,
            physics_client_id=sim.physics_client_id,
            direct_path_only=direct_path_only,
        )
        if plan is not None:
            return plan
    return None


def make_dest_spots(
    config: MassTransport3DEnvConfig, dest_tote_poses: list[Pose]
) -> list[Pose]:
    """Item spots on the destination table: a grid like the source's, off the tote."""
    half = config.item_size / 2
    x, y, _ = config.dest_table_pose.position
    hx, hy, _ = config.table_half_extents
    step = config.item_grid_step
    edge = step / 2
    reach = half + config.item_tote_clearance
    thx, thy, _ = config.tote_half_extents
    z = config.table_top_z + half + PLACEMENT_CLEARANCE
    spots = []
    for cx in np.arange(x - hx + edge, x + hx - edge + 1e-9, step):
        for cy in np.arange(y - hy + edge, y + hy - edge + 1e-9, step):
            if all(
                abs(cx - tote.position[0]) >= thx + reach
                or abs(cy - tote.position[1]) >= thy + reach
                for tote in dest_tote_poses
            ):
                spots.append(Pose((float(cx), float(cy), z)))
    return spots


def _item_reach(
    ctx: MassStreamContext,
    o: str,
    item_pose: Pose,
    g: ItemGrasp,
    base: SE2Pose,
    collision_bodies: set[int],
) -> list | None:
    """A path from base conf `base` that reaches grasp `g` on item `o`, or None."""
    if _base_collides(ctx, base):
        return None
    target_ee_pose = multiply_poses(item_pose, g.transform)
    joint_plan = _plan_vertical_reach(ctx, base, target_ee_pose, collision_bodies)
    if joint_plan is None:
        return None
    # The arm must hold the item along the whole path, out and back.
    if not _within_torque_limits(
        ctx, joint_plan, ctx.item_masses[o], _payload_offset(g.transform)
    ):
        return None
    return joint_plan


def sample_grasp(ctx: MassStreamContext, o: str) -> Iterator[tuple[ItemGrasp]]:
    """Yield the top-down grasps on item `o`."""
    for grasp in ctx.item_grasps[o]:
        yield (grasp,)


def sample_carry_conf(
    ctx: MassStreamContext, g: ToteGrasp
) -> Iterator[tuple[CarryConf]]:
    """Yield ways to hold the tote with grasp `g`, relative to the base."""
    sim = ctx.sim
    # The tote rests at the same height on either table.
    tote_z = sim.config.tote_initial_pose.position[2]
    candidates = [(d, h) for d in CARRY_DISTANCES for h in CARRY_LIFTS]
    rng = np.random.default_rng(ctx.motion_seed)
    while len(candidates) < MAX_CARRY_SAMPLES:
        candidates.append(
            (rng.uniform(*CARRY_DISTANCE_BOUNDS), rng.uniform(*CARRY_LIFT_BOUNDS))
        )
    origin = SE2Pose(0.0, 0.0, 0.0)
    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    for distance, lift in candidates:
        tote_in_base = _yaw_pose(float(distance), 0.0, tote_z, g.yaw_in_base)
        low_ee = multiply_poses(tote_in_base, g.transform)
        high_ee = Pose(
            (low_ee.position[0], low_ee.position[1], low_ee.position[2] + lift),
            low_ee.orientation,
        )
        with _parked_bodies(sim):
            sim.robot.set_base(origin)
            sim.robot.arm.set_joints(_HOME_JOINTS)
            try:
                high_joints = inverse_kinematics(
                    sim.robot.arm, high_ee, validate=True, set_joints=False
                )
                sim.robot.arm.set_joints(high_joints)
                down_path = smoothly_follow_end_effector_path(
                    sim.robot.arm,
                    [high_ee, low_ee],
                    initial_joints=high_joints,
                    collision_ids=set(),
                    joint_distance_fn=joint_distance_fn,
                    seed=ctx.motion_seed,
                    include_start=True,
                )
            except InverseKinematicsError:
                continue
        if not down_path:
            continue
        lift_path = list(reversed(down_path))
        yield (
            CarryConf(
                grasp=g,
                tote_in_base=tote_in_base,
                low_joints=lift_path[0],
                high_joints=lift_path[-1],
                lift_path=lift_path,
            ),
        )


def inverse_kinematics_stream(
    ctx: MassStreamContext, o: str, p: Pose, g: ItemGrasp, r: str, q: BaseConf
) -> Iterator[tuple[ArmTrajectory]]:
    """Yield an arm path from base conf `q` that reaches grasp `g` on item `o` at `p`."""
    del r  # only ties the base to the table the pose is on
    with _parked_bodies(ctx.sim):
        joint_plan = _item_reach(ctx, o, p, g, q.pose, set(ctx.table_ids))
    if joint_plan is not None:
        yield (ArmTrajectory(joint_plan, q.pose, target_object=o),)


def slot_inverse_kinematics_stream(
    ctx: MassStreamContext,
    o: str,
    s: str,
    tp: Pose,
    g: ItemGrasp,
    r: str,
    q: BaseConf,
) -> Iterator[tuple[ArmTrajectory]]:
    """Yield an arm path from base conf `q` that reaches item `o` in tote slot `s`."""
    del r  # only ties the base to the table the tote is on
    # An item in a slot takes the tote's orientation, and the slots leave room for
    # the fingers only along the tote's x.
    if g.closing_axis != "x":
        return
    item_pose = multiply_poses(tp, ctx.slots[s])
    sim = ctx.sim
    collision_bodies = set(ctx.table_ids) | {sim.tote_id}
    with _parked_bodies(sim), _scratch_pose(sim, "tote", tp):
        joint_plan = _item_reach(ctx, o, item_pose, g, q.pose, collision_bodies)
    if joint_plan is not None:
        yield (ArmTrajectory(joint_plan, q.pose, target_object=o, tote_pose=tp),)


def tote_inverse_kinematics_stream(
    ctx: MassStreamContext, tp: Pose, r: str, g: ToteGrasp, a: CarryConf
) -> Iterator[tuple[BaseConf, ArmTrajectory]]:
    """Yield the base conf that puts the tote at `tp` for carry conf `a`.

    The arm path runs from the home joints to the lifted conf above the tote.
    """
    del r  # certified on the base conf, so it is on one side only
    assert a.grasp == g
    sim = ctx.sim
    tote_yaw = Rotation.from_quat(tp.orientation).as_euler("xyz")[2]
    base_yaw = _normalize_angle(tote_yaw - g.yaw_in_base)
    offset = a.tote_in_base.position
    base = SE2Pose(
        tp.position[0] - np.cos(base_yaw) * offset[0] + np.sin(base_yaw) * offset[1],
        tp.position[1] - np.sin(base_yaw) * offset[0] - np.cos(base_yaw) * offset[1],
        base_yaw,
    )
    with _parked_bodies(sim), _scratch_pose(sim, "tote", tp):
        if _base_collides(ctx, base):
            return
        sim.robot.set_base(base)
        collision_bodies = set(ctx.table_ids)
        # The lifted conf must clear the tables and the tote.
        sim.robot.arm.set_joints(a.high_joints)
        if any(
            check_body_collisions(sim.robot.arm.robot_id, body, sim.physics_client_id)
            for body in collision_bodies | {sim.tote_id}
        ):
            return
        joint_plan = _plan_arm_motion(
            ctx, _HOME_JOINTS, a.high_joints, collision_bodies | {sim.tote_id}
        )
    if joint_plan is None:
        return
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=sim.config.max_action_mag / 2
    )
    yield (
        BaseConf(base),
        ArmTrajectory(joint_plan, base, target_object="tote", tote_pose=tp),
    )


def plan_base_motion(
    ctx: MassStreamContext, q1: BaseConf, q2: BaseConf
) -> Iterator[tuple[list]]:
    """Yield a base path from `q1` to `q2` that keeps the base clear of the tables."""
    sim = ctx.sim
    with _parked_bodies(sim):
        sim.robot.arm.set_joints(_HOME_JOINTS)
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot,
            q1.pose,
            q2.pose,
            collision_bodies=ctx.table_ids,
            seed=ctx.motion_seed,
        )
    if base_plan is None:
        return
    yield (base_plan,)


def test_cfree_pose_pose(
    ctx: MassStreamContext, o1: str, p1: Pose, o2: str, p2: Pose
) -> bool:
    """Whether items at `p1` and `p2` leave room for the fingers between them."""
    if o1 == o2:
        return True
    if abs(p1.position[2] - p2.position[2]) > ctx.sim.config.item_size:
        return True
    gap = max(
        abs(p1.position[0] - p2.position[0]), abs(p1.position[1] - p2.position[1])
    )
    # Grid cells sit exactly `min_item_gap` apart, so allow for rounding.
    return gap >= ctx.min_item_gap - 1e-6


def test_cfree_pose_tote(ctx: MassStreamContext, o: str, p: Pose, tp: Pose) -> bool:
    """Whether item `o` at `p` stays clear of the tote at `tp`."""
    sim = ctx.sim
    with _scratch_pose(sim, o, p), _scratch_pose(sim, "tote", tp):
        return not check_body_collisions(
            sim._object_name_to_pybullet_id(o),  # pylint: disable=protected-access
            sim.tote_id,
            sim.physics_client_id,
            distance_threshold=TOTE_CLEARANCE,
        )


def _traj_hits(ctx: MassStreamContext, t: ArmTrajectory, body_id: int) -> bool:
    sim = ctx.sim
    sim.robot.set_base(t.base)
    for joints in t.joint_plan:
        sim.robot.arm.set_joints(joints)
        if check_body_collisions(
            sim.robot.arm.robot_id, body_id, sim.physics_client_id
        ):
            return True
    return False


def test_cfree_traj_pose(
    ctx: MassStreamContext, t: ArmTrajectory, o2: str, p2: Pose
) -> bool:
    """Whether trajectory `t` avoids item `o2` resting at pose `p2`."""
    if o2 == t.target_object:
        return True
    sim = ctx.sim
    original = sim.robot.arm.get_joint_positions()
    try:
        with _scratch_pose(sim, o2, p2):
            return not _traj_hits(
                ctx,
                t,
                sim._object_name_to_pybullet_id(o2),  # pylint: disable=protected-access
            )
    finally:
        sim.robot.arm.set_joints(original)


def test_cfree_traj_tote(ctx: MassStreamContext, t: ArmTrajectory, tp: Pose) -> bool:
    """Whether trajectory `t` avoids the tote at `tp`."""
    # Trajectories planned around the tote at this very pose already avoid it.
    if t.tote_pose is not None and t.tote_pose == tp:
        return True
    sim = ctx.sim
    original = sim.robot.arm.get_joint_positions()
    try:
        with _scratch_pose(sim, "tote", tp):
            return not _traj_hits(ctx, t, sim.tote_id)
    finally:
        sim.robot.arm.set_joints(original)


def tote_payload(
    ctx: MassStreamContext, g: ToteGrasp, slot_items: dict[str, str]
) -> tuple[float, Pose]:
    """Mass and end-effector-frame center of mass of the tote and its contents.

    `slot_items` maps slot names to the items in them.
    """
    tote_in_ee = g.transform.invert()
    masses = [ctx.sim.config.tote_mass]
    centers = [np.asarray(tote_in_ee.position)]
    for slot, item in slot_items.items():
        masses.append(ctx.item_masses[item])
        centers.append(np.asarray(multiply_poses(tote_in_ee, ctx.slots[slot]).position))
    com = np.average(np.asarray(centers), axis=0, weights=masses)
    return float(sum(masses)), Pose((float(com[0]), float(com[1]), float(com[2])))


def test_carry_torque(
    ctx: MassStreamContext, g: ToteGrasp, a: CarryConf, fluents=()
) -> bool:
    """Whether the arm can lift and hold the tote with its current contents."""
    slot_items = {fact[2]: fact[1] for fact in fluents if fact[0].lower() == "inslot"}
    mass, offset = tote_payload(ctx, g, slot_items)
    return _within_torque_limits(ctx, a.lift_path, mass, offset)
