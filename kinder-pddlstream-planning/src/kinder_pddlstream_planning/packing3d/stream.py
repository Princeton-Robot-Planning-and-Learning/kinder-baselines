"""PDDLStream stream implementations for Packing3D.

See run.py for the domain design.

Each function takes the `PackingStreamContext` built by `create_problem`.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from kinder.envs.kinematic3d.packing3d import ObjectCentricPacking3DEnv
from kinder_models.kinematic3d.constants import HOME_JOINT_POSITIONS
from kinder_models.kinematic3d.utils import get_target_robot_pose_from_parameters
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

_HOME_JOINTS = HOME_JOINT_POSITIONS.tolist()

# Grasp transform (object anchor frame -> end-effector frame) onto the molded peg.
# A lower z closes the fingers on the part's wide base instead of the peg.
PEG_HEIGHT = 0.05
GRASP_TRANSFORM_TO_OBJECT = Pose((0.0, 0.0, 0.075), (0.707, 0.707, 0, 0))

# Height above the grasp/place point for the non-vertical reach, clear of the walls.
HOVER_HEIGHT = 0.15

# Fallback (distance, rotation) base standoffs if the home pose can't reach via IK.
BASE_STANDOFFS = [(0.5, 0.0), (0.55, np.pi / 6), (0.55, -np.pi / 6)]

MAX_PLACE_POSE_ATTEMPTS = 100_000

# Inward margin (meters) between a part's footprint and the inner face of the rack wall.
# A flush footprint stalls the descent, since the sim reverts any contacting step.
WALL_CLEARANCE = 1e-2


@dataclass(eq=False)
class ArmTrajectory:
    """A one-way joint-space path from `HOME_JOINT_POSITIONS` to a grasp conf.

    It is tagged with its target object, whose contact is intended.
    """

    joint_plan: list
    target_object: str


@dataclass
class PackingStreamContext:
    """Shared, precomputed context for all streams of one `create_problem` call."""

    sim: ObjectCentricPacking3DEnv
    motion_seed: int
    q_home: SE2Pose
    rack_pose: Pose
    # The cavity's half extents, i.e. the outer ones shrunk by the wall thickness.
    rack_interior_half_extents: tuple[float, float, float]
    part_xy_bounds: dict[str, tuple[float, float, float, float]]
    place_z: dict[str, float]
    grasp_anchor_offset: dict[str, tuple[float, float]]
    part_names: list[str] = field(default_factory=list)


@contextmanager
def _scratch_part_pose(
    sim: ObjectCentricPacking3DEnv, name: str, pose: Pose
) -> Iterator[None]:
    """Temporarily move a part's PyBullet body to `pose`, restoring it afterward."""
    body_id = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
    original = get_pose(body_id, sim.physics_client_id)
    set_pose(body_id, pose, sim.physics_client_id)
    try:
        yield
    finally:
        set_pose(body_id, original, sim.physics_client_id)


def _poses_from_fluents(fluents, exclude: str | None) -> dict[str, Pose]:
    """Extract {part_name: current_pose} from an `AtPose` fluent list."""
    poses: dict[str, Pose] = {}
    for fact in fluents:
        name, obj_name, pose = fact[0], fact[1], fact[2]
        if name.lower() == "atpose" and obj_name != exclude:
            poses[obj_name] = pose
    return poses


def _placement_xy_window(
    rack_pose: Pose,
    rack_interior_half_extents: tuple[float, float, float],
    xy_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Placement origins keeping a part's footprint inside the rack cavity.

    These half extents must describe the cavity, not the outer bounding box.
    """
    min_x, max_x, min_y, max_y = xy_bounds
    lower_x, upper_x = _apply_wall_clearance(
        rack_pose.position[0] - rack_interior_half_extents[0] - min_x,
        rack_pose.position[0] + rack_interior_half_extents[0] - max_x,
    )
    lower_y, upper_y = _apply_wall_clearance(
        rack_pose.position[1] - rack_interior_half_extents[1] - min_y,
        rack_pose.position[1] + rack_interior_half_extents[1] - max_y,
    )
    return lower_x, upper_x, lower_y, upper_y


def _apply_wall_clearance(lower: float, upper: float) -> tuple[float, float]:
    """Shrink `[lower, upper]` by `WALL_CLEARANCE` at both ends.

    A part wider than the cavity collapses to one centered candidate.
    """
    lower, upper = lower + WALL_CLEARANCE, upper - WALL_CLEARANCE
    if lower > upper:
        lower = upper = (lower + upper) / 2
    return lower, upper


def _rack_grid_positions(
    window: tuple[float, float, float, float],
    xy_bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Evenly-spaced candidate (x, y) placement origins covering `window`.

    The step is half a footprint, since two triangles tile only when offset.
    """
    lower_x, upper_x, lower_y, upper_y = window
    min_x, max_x, min_y, max_y = xy_bounds
    xs = _grid_axis(lower_x, upper_x, max_x - min_x)
    ys = _grid_axis(lower_y, upper_y, max_y - min_y)
    return [(float(x), float(y)) for x in xs for y in ys]


def _grid_axis(lower: float, upper: float, footprint: float) -> np.ndarray:
    """Samples spanning `[lower, upper]`, roughly half a `footprint` apart.

    The count is rounded down, since it dominates the 3-part solve time.
    """
    step = max(footprint / 2, 1e-6)
    # The epsilon keeps e.g. 0.16 / 0.05 == 3.1999999999999997 from losing a cell.
    num_cells = int(np.floor((upper - lower) / step + 1e-9)) + 1
    return np.linspace(lower, upper, max(num_cells, 1))


def sample_place_pose(ctx: PackingStreamContext, o: str) -> Iterator[tuple[Pose]]:
    """Yield candidate placement poses for part `o` inside the rack."""
    # No fluents: pddlstream forbids a fluent stream from certifying `Pose`, which is
    # also inverse-kinematics' domain input. Conflicts are caught by `place` instead.
    xy_bounds = ctx.part_xy_bounds[o]
    z = ctx.place_z[o]
    window = _placement_xy_window(
        ctx.rack_pose, ctx.rack_interior_half_extents, xy_bounds
    )
    grid = _rack_grid_positions(window, xy_bounds)
    # A distinct starting cell per part keeps two parts from proposing the same first
    # candidate, whose retry trips a pddlstream bug in `skeleton.update_bindings`.
    part_index = ctx.part_names.index(o)
    order = [(part_index + i) % len(grid) for i in range(len(grid))]
    for idx in order:
        x, y = grid[idx]
        yield (Pose((x, y, z)),)
    # Fall back to continuous sampling if the grid is exhausted.
    lower_x, upper_x, lower_y, upper_y = window
    for _ in range(MAX_PLACE_POSE_ATTEMPTS):
        x = ctx.sim.np_random.uniform(lower_x, upper_x)
        y = ctx.sim.np_random.uniform(lower_y, upper_y)
        yield (Pose((x, y, z)),)


def sample_grasp(ctx: PackingStreamContext, o: str) -> Iterator[tuple[Pose]]:
    """Yield the grasp transform for part `o`, which is the same for every part."""
    del ctx, o  # the grasp transform is the same for every part
    yield (GRASP_TRANSFORM_TO_OBJECT,)


def inverse_kinematics_stream(
    ctx: PackingStreamContext, o: str, p: Pose, g: Pose
) -> Iterator[tuple[SE2Pose, ArmTrajectory]]:
    """Yield a base pose and an arm path that reaches grasp `g` on `o` at `p`."""
    sim = ctx.sim
    # Only the static rack/table here; other parts are checked by test_cfree_traj_pose.
    static_collision_bodies = {
        sim.table_id,
        sim._rack_id,  # pylint: disable=protected-access
    }
    offset_x, offset_y = ctx.grasp_anchor_offset[o]
    anchor_pose = Pose(
        (p.position[0] + offset_x, p.position[1] + offset_y, p.position[2]),
        p.orientation,
    )
    target_ee_pose = multiply_poses(anchor_pose, g)
    hover_ee_pose = Pose(
        (
            target_ee_pose.position[0],
            target_ee_pose.position[1],
            target_ee_pose.position[2] + HOVER_HEIGHT,
        ),
        target_ee_pose.orientation,
    )
    target_se2 = p.to_se2()
    candidates = [ctx.q_home] + [
        get_target_robot_pose_from_parameters(target_se2, dist, rot)
        for dist, rot in BASE_STANDOFFS
    ]
    joint_distance_fn = create_joint_distance_fn(sim.robot.arm)
    for base_conf in candidates:
        sim.robot.set_base(base_conf)
        sim.robot.arm.set_joints(_HOME_JOINTS)
        try:
            hover_joints = inverse_kinematics(
                sim.robot.arm, hover_ee_pose, validate=True, set_joints=False
            )
        except InverseKinematicsError:
            continue
        general_plan = run_motion_planning(
            sim.robot.arm,
            initial_positions=_HOME_JOINTS,
            target_positions=hover_joints,
            collision_bodies=static_collision_bodies,
            seed=ctx.motion_seed,
            physics_client_id=sim.physics_client_id,
        )
        if general_plan is None:
            continue
        sim.robot.arm.set_joints(general_plan[-1])
        try:
            vertical_plan = smoothly_follow_end_effector_path(
                sim.robot.arm,
                [hover_ee_pose, target_ee_pose],
                initial_joints=general_plan[-1],
                collision_ids=static_collision_bodies,
                joint_distance_fn=joint_distance_fn,
                seed=ctx.motion_seed,
                include_start=False,
            )
        except InverseKinematicsError:
            continue
        if not vertical_plan:
            continue
        joint_plan = remap_joint_position_plan_to_constant_distance(
            general_plan + vertical_plan,
            sim.robot.arm,
            max_distance=sim.config.max_action_mag / 2,
        )
        yield (base_conf, ArmTrajectory(joint_plan, target_object=o))
        return


def test_cfree_pose_pose(
    ctx: PackingStreamContext, o1: str, p1: Pose, o2: str, p2: Pose
) -> bool:
    """Whether parts `o1` and `o2` are collision-free at poses `p1` and `p2`."""
    if o1 == o2:
        return True
    sim = ctx.sim
    id1 = sim._object_name_to_pybullet_id(o1)  # pylint: disable=protected-access
    id2 = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    with ExitStack() as stack:
        stack.enter_context(_scratch_part_pose(sim, o1, p1))
        stack.enter_context(_scratch_part_pose(sim, o2, p2))
        return not check_body_collisions(id1, id2, sim.physics_client_id)


def test_cfree_traj_pose(
    ctx: PackingStreamContext, t: ArmTrajectory, o2: str, p2: Pose
) -> bool:
    """Whether trajectory `t` avoids part `o2` while it rests at pose `p2`."""
    # The trajectory intentionally contacts its own target object at the grasp waypoint.
    if o2 == t.target_object:
        return True
    sim = ctx.sim
    other_id = sim._object_name_to_pybullet_id(o2)  # pylint: disable=protected-access
    robot_id = sim.robot.arm.robot_id
    original_joints = sim.robot.arm.get_joint_positions()
    try:
        with _scratch_part_pose(sim, o2, p2):
            for joints in t.joint_plan:
                sim.robot.arm.set_joints(joints)
                if check_body_collisions(robot_id, other_id, sim.physics_client_id):
                    return False
        return True
    finally:
        sim.robot.arm.set_joints(original_joints)


def plan_base_motion(
    ctx: PackingStreamContext, q1: SE2Pose, q2: SE2Pose, fluents=()
) -> Iterator[tuple[list]]:
    """Yield a collision-free base path from `q1` to `q2`."""
    sim = ctx.sim
    occupied = _poses_from_fluents(fluents, exclude=None)
    with ExitStack() as stack:
        for other_name, other_pose in occupied.items():
            stack.enter_context(_scratch_part_pose(sim, other_name, other_pose))
        sim.robot.arm.set_joints(_HOME_JOINTS)
        collision_bodies = (
            sim._get_collision_object_ids()  # pylint: disable=protected-access
        )
        base_plan = run_single_arm_mobile_base_motion_planning(
            sim.robot,
            q1,
            q2,
            collision_bodies=collision_bodies,
            seed=ctx.motion_seed,
        )
    if base_plan is None:
        return
    yield (base_plan,)
