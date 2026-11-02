"""Solve KinDER's BedsideStocking3D environment with PDDLStream.

The planner takes a particle belief over the user's reach ellipsoid; a requested item's
pose is usable if P_b(reachable) >= 1 - delta. Motion is IK-only: arm trajectories are
home -> hover -> grasp waypoints and base moves are unchecked straight lines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from kinder.envs.kinematic3d.bedside_stocking3d import (
    BedsideStocking3DEnvConfig,
    BedsideStocking3DObjectCentricState,
    ObjectCentricBedsideStocking3DEnv,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
    remove_fingers_from_extended_joints,
)
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import AND, PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.utils import read
from pybullet_helpers.geometry import get_pose
from pybullet_helpers.joint import get_jointwise_difference
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    smoothly_follow_end_effector_path,
)

from kinder_pddlstream_planning.bedsidestocking3d.stream import (
    BaseConf,
    BedsideStreamContext,
    requested_candidate_slots,
    Placement,
    inverse_kinematics_stream,
    sample_grasp,
    sample_place_pose,
    test_cfree_pose_pose,
    test_cfree_traj_pose,
    test_clear,
    test_usable,
)

_HERE = Path(__file__).parent
DOMAIN_PDDL = read(str(_HERE / "domain.pddl"))
STREAM_PDDL = read(str(_HERE / "stream.pddl"))


def _surface_of(sim: ObjectCentricBedsideStocking3DEnv, name: str) -> str:
    body = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
    for s in sim.config.surface_regions:
        if sim._get_surfaces_supporting_object(body) & {sim.surface_id(s)}:  # pylint: disable=protected-access
            return s
    return "unknown"


def build_stream_context(
    sim: ObjectCentricBedsideStocking3DEnv,
    state: BedsideStocking3DObjectCentricState,
    theta_particles: np.ndarray | None = None,
    delta: float = 0.1,
    seed: int = 0,
) -> BedsideStreamContext:
    if theta_particles is None:
        theta_particles = sim.reach_theta[None]
    cfg = sim.config
    surfaces = list(cfg.surface_regions)
    return BedsideStreamContext(
        sim=sim,
        theta_particles=np.asarray(theta_particles, dtype=float),
        delta=delta,
        rng=np.random.default_rng(seed),
        requested=state.requested_objects,
        bedside_surfaces=list(cfg.bedside_regions),
        all_surfaces=surfaces,
        slots={s: cfg.region_slots(s) for s in surfaces},
        requested_slots=requested_candidate_slots(sim),
    )


def create_problem(
    sim: ObjectCentricBedsideStocking3DEnv,
    state: BedsideStocking3DObjectCentricState,
    **ctx_kwargs,
) -> PDDLProblem:
    ctx = build_stream_context(sim, state, **ctx_kwargs)
    q0 = BaseConf(sim.robot.get_base())
    init: list[tuple] = [("BConf", q0), ("AtBConf", q0), ("HandEmpty",), ("CanMove",)]
    goal: list[tuple] = [("HandEmpty",)]
    for name in state.requested_objects + state.clutter_objects:
        body = sim._object_name_to_pybullet_id(name)  # pylint: disable=protected-access
        p0 = Placement(name, get_pose(body, sim.physics_client_id), _surface_of(sim, name))
        init += [("Item", name), ("Pose", name, p0), ("AtPose", name, p0)]
        if name in ctx.requested:
            init.append(("Requested", name))
            goal.append(("Delivered", name))
    stream_map = {
        "sample-place-pose": from_gen_fn(partial(sample_place_pose, ctx)),
        "sample-grasp": from_gen_fn(partial(sample_grasp, ctx)),
        "inverse-kinematics": from_gen_fn(partial(inverse_kinematics_stream, ctx)),
        "test-usable": from_test(partial(test_usable, ctx)),
        "test-clear": from_test(partial(test_clear, ctx)),
        "test-cfree-pose-pose": from_test(partial(test_cfree_pose_pose, ctx)),
        "test-cfree-traj-pose": from_test(partial(test_cfree_traj_pose, ctx)),
    }
    return PDDLProblem(DOMAIN_PDDL, {}, STREAM_PDDL, stream_map, init, (AND, *goal))


def plan_bedside(
    sim: ObjectCentricBedsideStocking3DEnv,
    state: BedsideStocking3DObjectCentricState,
    max_time: float = 600.0,
    algorithm: str = "adaptive",
    search_sample_ratio: float = 1.0,
    max_complexity: float = float("inf"),
    verbose: bool = False,
    **ctx_kwargs,
):
    problem = create_problem(sim, state, **ctx_kwargs)
    # pddlstream walks its binding tree recursively; long plans exceed the default limit.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    solution = solve(
        problem,
        algorithm=algorithm,
        unit_costs=True,
        max_time=max_time,
        search_sample_ratio=search_sample_ratio,
        max_complexity=max_complexity,
        verbose=verbose,
    )
    if verbose:
        print_solution(solution)
    plan, _, _ = solution
    return plan


def _clip(sim, action):
    space = sim.action_space
    assert isinstance(space, Kinematic3DRobotActionSpace)
    return np.clip(np.asarray(action, dtype=np.float32), space.low, space.high)


def _arm_path(sim, target, exclude: set[int], seed: int) -> list:
    """BiRRT from the current arm conf to `target`; straight line if that fails."""
    # pylint: disable=protected-access
    arm = sim.robot.arm
    base, cur = sim.robot.get_base(), list(arm.get_joint_positions())
    finger = arm.get_finger_state()
    held = sim._grasped_object_id
    bodies = sim._get_collision_object_ids() - exclude - ({held} if held is not None else set())
    path = run_motion_planning(
        arm,
        cur,
        list(target),
        bodies,
        seed=seed,
        physics_client_id=sim.physics_client_id,
        held_object=held,
        base_link_to_held_obj=sim._grasped_object_transform,
    )
    sim._set_robot_and_held_object(base, remove_fingers_from_extended_joints(cur), finger)
    if path is None:
        path = [cur, list(target)]
    return remap_joint_position_plan_to_constant_distance(
        path, arm, max_distance=sim.config.max_action_mag / 2
    )


def _descent(sim, hover, grasp, target: int, seed: int) -> list:
    """Straight-line end-effector path from the hover conf down to the grasp conf."""
    # pylint: disable=protected-access
    arm = sim.robot.arm
    base, cur = sim.robot.get_base(), list(arm.get_joint_positions())
    finger = arm.get_finger_state()
    held = sim._grasped_object_id
    bodies = sim._get_collision_object_ids() - {target} - ({held} if held is not None else set())
    waypoints = [arm.forward_kinematics(list(hover)), arm.forward_kinematics(list(grasp))]
    try:
        path = smoothly_follow_end_effector_path(
            arm,
            waypoints,
            initial_joints=list(hover),
            collision_ids=bodies,
            joint_distance_fn=create_joint_distance_fn(arm),
            seed=seed,
            include_start=False,
            held_object=held,
            base_link_to_held_obj=sim._grasped_object_transform,
        )
    except InverseKinematicsError:
        path = [list(grasp)]
    sim._set_robot_and_held_object(base, remove_fingers_from_extended_joints(cur), finger)
    return remap_joint_position_plan_to_constant_distance(
        [list(hover)] + list(path), arm, max_distance=0.02
    )[1:]


def execute_plan(sim: ObjectCentricBedsideStocking3DEnv, plan, seed: int = 0) -> bool:
    """Execute a move_base/pick/place plan; arm paths are refined with BiRRT."""

    def to_joints(target, tol=1e-3, attempts=40):
        for _ in range(attempts):
            cur = remove_fingers_from_extended_joints(sim.robot.arm.get_joint_positions())
            d = get_jointwise_difference(
                sim.robot.arm.get_arm_joint_infos()[:7], list(target[:7]), cur
            )
            sim.step(_clip(sim, [0.0, 0.0, 0.0] + d + [0.0]))
            if max(abs(x) for x in d) < tol:
                return

    def to_base(target, attempts=200):
        for _ in range(attempts):
            d = target - sim.robot.get_base()
            if max(abs(d.x), abs(d.y), abs(d.rot)) < 1e-3:
                return
            sim.step(_clip(sim, [d.x, d.y, d.rot] + [0.0] * 8))

    def gripper(close: bool):
        for _ in range(5):
            sim.step(_clip(sim, [0.0] * 10 + [-1.0 if close else 1.0]))
            held = sim._grasped_object  # pylint: disable=protected-access
            if (held is not None) == close:
                return

    for name, args in plan:
        if name == "move_base":
            to_base(args[1].pose)
        elif name in ("pick", "place"):
            traj = args[4]
            target = sim._object_name_to_pybullet_id(args[0])  # pylint: disable=protected-access
            _, hover, grasp = traj.joint_plan
            approach = _arm_path(sim, hover, {target}, seed)
            for q in approach[1:] + _descent(sim, hover, grasp, target, seed):
                to_joints(q)
            gripper(close=name == "pick")
            to_joints(hover)
            retreat = _arm_path(sim, traj.joint_plan[0], set(), seed)
            for q in retreat[1:]:
                to_joints(q)
        else:
            raise ValueError(f"Unknown action: {name}")
    return sim.goal_reached()


def solve_and_execute(
    num_requested: int = 4,
    num_clutter: int = 2,
    seed: int = 0,
    max_time: float = 600.0,
    delta: float = 0.1,
    search_sample_ratio: float = 1.0,
    max_complexity: float = float("inf"),
    execute: bool = False,
    use_gui: bool = False,
    verbose: bool = False,
) -> dict:
    sim = ObjectCentricBedsideStocking3DEnv(
        num_requested=num_requested,
        num_clutter=num_clutter,
        config=BedsideStocking3DEnvConfig(),
        use_gui=use_gui,
        allow_state_access=True,
    )
    state, _ = sim.reset(seed=seed)
    t0 = time.perf_counter()
    plan = plan_bedside(
        sim,
        state,
        max_time=max_time,
        search_sample_ratio=search_sample_ratio,
        max_complexity=max_complexity,
        verbose=verbose,
        delta=delta,
        seed=seed,
    )
    plan_time = time.perf_counter() - t0
    result = {
        "num_requested": num_requested,
        "num_clutter": num_clutter,
        "seed": seed,
        "search_sample_ratio": search_sample_ratio,
        "max_complexity": max_complexity,
        "plan_found": plan is not None,
        "plan_time": plan_time,
        "plan_length": None if plan is None else len(plan),
        "num_moves": None if plan is None else sum(a[0] == "place" for a in plan),
        "executed": None,
    }
    if plan is not None and execute:
        sim.reset(seed=seed)
        result["executed"] = bool(execute_plan(sim, plan))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-requested", type=int, default=4)
    parser.add_argument("--num-clutter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--search-sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-complexity", type=float, default=float("inf"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    result = solve_and_execute(
        num_requested=args.num_requested,
        num_clutter=args.num_clutter,
        seed=args.seed,
        max_time=args.max_time,
        delta=args.delta,
        search_sample_ratio=args.search_sample_ratio,
        max_complexity=args.max_complexity,
        execute=args.execute,
        use_gui=args.use_gui,
        verbose=args.verbose,
    )
    print(json.dumps(result))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
