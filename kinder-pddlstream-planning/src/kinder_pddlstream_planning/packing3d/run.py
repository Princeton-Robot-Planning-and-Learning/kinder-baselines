"""Solve KinDER's Packing3D environment with PDDLStream.

Mirrors pddlstream's pick-and-place examples, with the streams in stream.py.

`Kin`'s trajectory runs forward then reversed, so arm conf is never a fluent.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from kinder.envs.kinematic3d.packing3d import (
    ObjectCentricPacking3DEnv,
    Packing3DEnvConfig,
    Packing3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
    remove_fingers_from_extended_joints,
)
from numpy.typing import NDArray
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import AND, PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read
from pybullet_helpers.geometry import SE2Pose, get_pose, multiply_poses
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.joint import get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import get_triangle_vertices

from kinder_pddlstream_planning.packing3d.stream import (
    PackingStreamContext,
    inverse_kinematics_stream,
    plan_base_motion,
    sample_grasp,
    sample_place_pose,
    test_cfree_pose_pose,
    test_cfree_traj_pose,
)
from kinder_pddlstream_planning.rendering import (
    DEFAULT_GIF_DIR,
    gif_output_path,
    render_frame,
    save_gif,
)

_HERE = Path(__file__).parent
DOMAIN_PDDL = read(str(_HERE / "domain.pddl"))
STREAM_PDDL = read(str(_HERE / "stream.pddl"))

HOME_BASE_POSE = SE2Pose(-1.0, 0.0, 0.0)


def _joint_infos(sim: ObjectCentricPacking3DEnv):
    return sim.robot.arm.get_arm_joint_infos()[:7]


def _arm(sim: ObjectCentricPacking3DEnv) -> FingeredSingleArmPyBulletRobot:
    """The arm, narrowed to the fingered robot type Packing3D actually builds."""
    arm = sim.robot.arm
    assert isinstance(arm, FingeredSingleArmPyBulletRobot)
    return arm


def _clip_action(sim: ObjectCentricPacking3DEnv, action) -> NDArray[np.float32]:
    """Clip `action` into the environment's action space."""
    action_space = sim.action_space
    assert isinstance(action_space, Kinematic3DRobotActionSpace)
    return np.clip(
        np.asarray(action, dtype=np.float32), action_space.low, action_space.high
    )


def _part_xy_bounds(
    state: Packing3DObjectCentricState, name: str
) -> tuple[float, float, float, float]:
    """The part's footprint bounds in its raw PyBullet body frame.

    A triangle's raw origin is a vertex, so its bounds are asymmetric.
    """
    obj = state.get_object_from_name(name)
    if obj.type == Kinematic3DCuboidType:
        half_extent_x, half_extent_y, _, _ = state.get_object_half_extents_packing3d(
            name
        )
        return (-half_extent_x, half_extent_x, -half_extent_y, half_extent_y)
    side_a, side_b, _, triangle_type = state.get_object_triangle_features(name)
    vertices = get_triangle_vertices(
        {0: "equilateral", 1: "right"}[int(triangle_type)], (side_a, side_b)
    )
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    return (min(xs), max(xs), min(ys), max(ys))


def build_stream_context(
    sim: ObjectCentricPacking3DEnv,
    state: Packing3DObjectCentricState,
    motion_seed: int = 0,
) -> PackingStreamContext:
    """Precompute the rack and part geometry every stream shares.

    Split out of `create_problem` so tests can assert on the planner's numbers.
    """
    part_names = sorted(state.part_poses)
    q_home = sim.robot.get_base()
    rack_pose = state.rack_pose
    rack_half_extents = state.rack_half_extents
    part_half_extents = {
        name: state.get_object_half_extents_packing3d(name)[:3] for name in part_names
    }
    part_xy_bounds = {name: _part_xy_bounds(state, name) for name in part_names}

    # pylint: disable=protected-access
    grasp_anchor_offset = {
        name: (
            state.get_object_pose(name).position[0]
            - get_pose(
                sim._object_name_to_pybullet_id(name),
                sim.physics_client_id,
            ).position[0],
            state.get_object_pose(name).position[1]
            - get_pose(
                sim._object_name_to_pybullet_id(name),
                sim.physics_client_id,
            ).position[1],
        )
        for name in part_names
    }

    # `rack_half_extents` is the outer bounding box, but parts fit in the cavity it
    # encloses, which is a wall thickness smaller on every side.
    rack_interior_half_extents = (
        rack_half_extents[0] - sim.config.rack_wall_thickness,
        rack_half_extents[1] - sim.config.rack_wall_thickness,
        rack_half_extents[2],
    )
    rack_floor_z = (
        rack_pose.position[2] - rack_half_extents[2] + sim.config.rack_wall_thickness
    )
    # A part flush on the rack floor trips the sim's collision-revert check, so it
    # never lands. Aim for mid-window: placed, but not close enough to touch.
    placement_clearance = sim.config.min_placement_dist / 2
    place_z = {
        name: rack_floor_z + part_half_extents[name][2] + placement_clearance
        for name in part_names
    }

    return PackingStreamContext(
        sim=sim,
        motion_seed=motion_seed,
        q_home=q_home,
        rack_pose=rack_pose,
        rack_interior_half_extents=rack_interior_half_extents,
        part_xy_bounds=part_xy_bounds,
        place_z=place_z,
        grasp_anchor_offset=grasp_anchor_offset,
        part_names=part_names,
    )


def create_problem(
    sim: ObjectCentricPacking3DEnv,
    state: Packing3DObjectCentricState,
    motion_seed: int = 0,
) -> PDDLProblem:
    """Build a PDDLProblem for packing every part in `state` into the rack."""
    ctx = build_stream_context(sim, state, motion_seed=motion_seed)

    init: list[tuple] = [
        ("BConf", ctx.q_home),
        ("AtBConf", ctx.q_home),
        ("HandEmpty",),
        ("CanMove",),
    ]
    goal_literals: list[tuple] = [("HandEmpty",)]
    for name in ctx.part_names:
        pose = get_pose(
            sim._object_name_to_pybullet_id(name),  # pylint: disable=protected-access
            sim.physics_client_id,
        )
        init += [
            ("Packable", name),
            ("Graspable", name),
            ("Pose", name, pose),
            ("AtPose", name, pose),
        ]
        goal_literals.append(("On", name))
    goal = (AND, *goal_literals)

    stream_map = {
        "sample-place-pose": from_gen_fn(partial(sample_place_pose, ctx)),
        "sample-grasp": from_gen_fn(partial(sample_grasp, ctx)),
        "inverse-kinematics": from_gen_fn(partial(inverse_kinematics_stream, ctx)),
        "plan-base-motion": from_gen_fn(partial(plan_base_motion, ctx)),
        "test-cfree-pose-pose": from_test(partial(test_cfree_pose_pose, ctx)),
        "test-cfree-traj-pose": from_test(partial(test_cfree_traj_pose, ctx)),
    }

    return PDDLProblem(DOMAIN_PDDL, {}, STREAM_PDDL, stream_map, init, goal)


def plan_packing3d(
    sim: ObjectCentricPacking3DEnv,
    state: Packing3DObjectCentricState,
    max_time: float = 60.0,
    verbose: bool = False,
    **problem_kwargs,
) -> list[tuple[str, tuple]] | None:
    """Solve for a sequence of move_base/pick/place actions to pack all parts."""
    problem = create_problem(sim, state, **problem_kwargs)
    solution = solve(
        problem,
        algorithm="adaptive",
        unit_costs=True,
        max_time=max_time,
        verbose=verbose,
    )
    if verbose:
        print_solution(solution)
    plan, _, _ = solution
    return plan


def _run_gripper(
    sim: ObjectCentricPacking3DEnv,
    close: bool,
    frames: list | None = None,
    max_steps: int = 5,
    num_interp_frames: int = 8,
) -> None:
    arm = _arm(sim)
    finger_before = arm.get_finger_state()
    action = _clip_action(sim, [0.0] * 10 + [-1.0 if close else 1.0])
    succeeded = False
    for _ in range(max_steps):
        sim.step(action)
        grasped = sim._grasped_object  # pylint: disable=protected-access
        if close and grasped is not None:
            succeeded = True
            break
        if not close and grasped is None:
            succeeded = True
            break
    if not succeeded or frames is None:
        return
    # Interpolate finger_state purely for the GIF, ending on the real final value.
    finger_after = arm.get_finger_state()
    for i in range(1, num_interp_frames + 1):
        arm.set_finger_state(
            finger_before + (finger_after - finger_before) * i / num_interp_frames
        )
        frames.append(render_frame(sim))


def execute_plan(
    sim: ObjectCentricPacking3DEnv, plan, frames: list | None = None
) -> bool:
    """Execute a move_base/pick/place plan in the real environment."""

    def step(action) -> None:
        sim.step(_clip_action(sim, action))
        if frames is not None:
            frames.append(render_frame(sim))

    def step_to_joint_target(
        target_joints, tol: float = 1e-3, max_attempts: int = 5
    ) -> None:
        # Re-derive the delta from the current joints on each attempt, since float32
        # rounding and per-step clipping leave a residual that compounds otherwise.
        for _ in range(max_attempts):
            current = remove_fingers_from_extended_joints(
                sim.robot.arm.get_joint_positions()
            )
            delta = get_jointwise_difference(
                _joint_infos(sim), list(target_joints[:7]), current
            )
            step([0.0, 0.0, 0.0] + delta + [0.0])
            if max(abs(d) for d in delta) < tol:
                break

    for name, args in plan:
        if name == "move_base":
            _, _, base_plan = args
            for target_base in base_plan[1:]:
                current_base = sim.robot.get_base()
                delta = target_base - current_base
                step([delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0])
        elif name == "pick":
            _, _, _, _, traj = args
            joint_plan = traj.joint_plan
            for target_joints in joint_plan[1:]:
                step_to_joint_target(target_joints)
            _run_gripper(sim, close=True, frames=frames)
            for target_joints in reversed(joint_plan[:-1]):
                step_to_joint_target(target_joints)
        elif name == "place":
            _, target_pose, _, _, traj = args
            joint_plan = list(traj.joint_plan)
            for target_joints in joint_plan[1:-1]:
                step_to_joint_target(target_joints)
            final_joints = joint_plan[-1]
            # Correct the final conf with the grasp transform recorded at grasp time, as
            # Shelf3D's place skill does, or the release misses the placement tolerance.

            # Re-planning rather than a bare IK call keeps the correction on the same
            # elbow branch, since a branch switch needs a jump that hits the rack.

            # pylint: disable=protected-access
            real_transform = sim._grasped_object_transform
            grasped_object_id = sim._grasped_object_id
            if real_transform is not None:
                corrected_ee = multiply_poses(target_pose, real_transform.invert())
                # Motion planning wants the full extended (arm + finger) joint vector,
                # matching how joint_plan waypoints are produced upstream.
                current_joints = sim.robot.arm.get_joint_positions()
                try:
                    correction_plan = smoothly_follow_end_effector_path(
                        sim.robot.arm,
                        [corrected_ee],
                        initial_joints=current_joints,
                        collision_ids={sim.table_id, sim._rack_id},
                        joint_distance_fn=create_joint_distance_fn(sim.robot.arm),
                        held_object=grasped_object_id,
                        base_link_to_held_obj=real_transform,
                        include_start=False,
                    )
                except InverseKinematicsError:
                    correction_plan = []
                # The search leaves the arm wherever its last IK attempt landed, and
                # step_to_joint_target derives its delta from those live joints.
                sim.robot.arm.set_joints(current_joints)

                # A goal-only BiRRT query can settle for the closest tree node rather
                # than raising, so verify via forward kinematics before trusting it.
                if correction_plan and not sim.robot.arm.forward_kinematics(
                    correction_plan[-1]
                ).allclose(corrected_ee, atol=1e-3):
                    correction_plan = []
                if correction_plan:
                    final_joints = correction_plan[-1]
            step_to_joint_target(final_joints)
            _run_gripper(sim, close=False, frames=frames)
            for target_joints in reversed(joint_plan[:-1]):
                step_to_joint_target(target_joints)
        else:
            raise ValueError(f"Unknown action: {name}")
    return sim.goal_reached()


def solve_and_execute(
    num_parts: int = 2,
    seed: int = 0,
    max_time: float = 60.0,
    use_gui: bool = False,
    verbose: bool = False,
    gif_path: str | Path | None = None,
    **problem_kwargs,
) -> bool:
    """Reset Packing3D, plan with PDDLStream, and execute the plan.

    Returns whether every part ended up supported by the rack.
    """
    sim = ObjectCentricPacking3DEnv(
        num_parts=num_parts,
        config=Packing3DEnvConfig(robot_base_home_pose=HOME_BASE_POSE),
        use_gui=use_gui,
        allow_state_access=True,
    )
    state, _ = sim.reset(seed=seed)
    plan = plan_packing3d(
        sim, state, max_time=max_time, verbose=verbose, **problem_kwargs
    )
    if plan is None:
        return False
    # Planning scratch-mutates the live state, so rewind by re-seeding. Restoring via
    # reset(options={"init_state": state}) instead would shift triangle parts.
    sim.reset(seed=seed)
    frames = [render_frame(sim)] if gif_path is not None else None
    try:
        return execute_plan(sim, plan, frames=frames)
    finally:
        if frames is not None:
            assert gif_path is not None
            save_gif(gif_path, frames)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-parts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="Show the PyBullet GUI (default: off).",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save a GIF of the rollout (default: off).",
    )
    parser.add_argument(
        "--gif-dir",
        type=Path,
        default=DEFAULT_GIF_DIR,
        help="Directory to write the GIF into (default: %(default)s).",
    )
    args = parser.parse_args()
    gif_path = (
        gif_output_path("packing3d", f"p{args.num_parts}", args.gif_dir)
        if args.save_gif
        else None
    )
    success = solve_and_execute(
        num_parts=args.num_parts,
        seed=args.seed,
        max_time=args.max_time,
        use_gui=args.use_gui,
        gif_path=gif_path,
        verbose=True,
    )
    print(f"Reached goal: {success}")


if __name__ == "__main__":
    main()
