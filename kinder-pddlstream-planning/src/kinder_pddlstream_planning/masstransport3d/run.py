"""Solve KinDER's MassTransport3D environment with PDDLStream.

Items go from the source table to the destination table, either one at a time in the
hand or several at a time in the tote. The tote has a fixed set of slots.

Item masses enter only through torque checks: on item paths in `inverse-kinematics` and
`slot-inverse-kinematics`, and on the tote lift in the fluent test `test-carry-torque`,
which sees which items are in the tote. A skeleton that loads too much therefore looks
fine symbolically and fails only there. The planner uses the masses it is given, which
need not be the sim's true ones.

Crossing between the tables costs `travel_cost` each way, while every other action
costs 1, so sharing trips pays off and the task becomes bin packing under the torque
limits. `max_trips` can also bound the trips outright.

Item spots and the base conf in front of each table are fixed, so the geometry is
cheap and shared across skeletons, and the search spends its effort on which items
travel together.

As in Packing3D, item and tote arm paths start at the home joints and run forward then
reversed, so the arm conf is never a fluent. The one exception is the tote: the arm
stays at the carry conf's lifted joints while it holds the tote.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
from kinder.envs.dynamic3d.kinematic_masstransport3d import (
    MassTransport3DEnvConfig,
    MassTransport3DObjectCentricState,
    ObjectCentricKinematicMassTransport3DEnv,
)
from kinder.envs.dynamic3d.masstransport3d import ObjectCentricMassTransport3DEnv
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
    remove_fingers_from_extended_joints,
)
from numpy.typing import NDArray
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import AND, PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test
from pddlstream.language.stream import PartialInputs, StreamInfo
from pddlstream.utils import read
from pybullet_helpers.geometry import Pose, SE2Pose, get_pose, multiply_poses
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.joint import get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot

from kinder_pddlstream_planning.masstransport3d.stream import (
    HOVER_HEIGHT,
    BaseConf,
    ItemGrasp,
    MassStreamContext,
    inverse_kinematics_stream,
    make_dest_spots,
    make_item_grasps,
    make_slots,
    make_tote_grasps,
    plan_base_motion,
    sample_carry_conf,
    sample_grasp,
    slot_inverse_kinematics_stream,
    test_carry_torque,
    test_cfree_pose_pose,
    test_cfree_pose_tote,
    test_cfree_traj_pose,
    test_cfree_traj_tote,
    tote_inverse_kinematics_stream,
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

# Where the tote goes on the destination table: mirroring its source pose.
DEST_TOTE_OFFSET = (0.005, 0.25)

# x of the base conf in front of each table, facing it. From there the arm reaches
# every item spot on the table.
TABLE_BASE_X = 0.05

# What one trip across costs, relative to 1 for every other action.
DEFAULT_TRAVEL_COST = 10.0

# By default pddlstream shares one optimistic output object across every instance
# of a stream. A skeleton that uses two instances then binds that object twice and
# trips the `update_bindings` assertion, so these streams get their own objects.
UNIQUE_OUTPUT_STREAMS = (
    "sample-carry-conf",
    "inverse-kinematics",
    "slot-inverse-kinematics",
    "tote-inverse-kinematics",
    "plan-base-motion",
)


def _joint_infos(sim: ObjectCentricKinematicMassTransport3DEnv):
    return sim.robot.arm.get_arm_joint_infos()[:7]


def _arm(
    sim: ObjectCentricKinematicMassTransport3DEnv,
) -> FingeredSingleArmPyBulletRobot:
    """The arm, narrowed to the fingered robot type the env builds."""
    arm = sim.robot.arm
    assert isinstance(arm, FingeredSingleArmPyBulletRobot)
    return arm


def _wrap(angle: float) -> float:
    """`angle` wrapped into [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _fk(arm, joints) -> Pose:
    """End-effector pose at `joints`, leaving the arm where it was.

    The arm's own `forward_kinematics` leaves it at `joints`, which under physics
    teleports it into whatever it was about to grasp.
    """
    current = arm.get_joint_positions()
    try:
        return arm.forward_kinematics(joints)
    finally:
        arm.set_joints(current)


def _clip_action(
    sim: ObjectCentricKinematicMassTransport3DEnv, action
) -> NDArray[np.float32]:
    """Clip `action` into the environment's action space."""
    action_space = sim.action_space
    assert isinstance(action_space, Kinematic3DRobotActionSpace)
    return np.clip(
        np.asarray(action, dtype=np.float32), action_space.low, action_space.high
    )


def build_stream_context(
    sim: ObjectCentricKinematicMassTransport3DEnv,
    state: MassTransport3DObjectCentricState,
    item_masses: dict[str, float] | None = None,
    motion_seed: int = 0,
    torque_margin: float = 0.0,
    slot_grid: tuple[int, int] = (2, 3),
) -> MassStreamContext:
    """Precompute what every stream shares.

    `item_masses` are the masses the planner assumes; by default the sim's true ones.
    """
    config = sim.config
    item_names = sim.item_names
    if item_masses is None:
        item_masses = sim.item_masses
    assert set(item_masses) == set(item_names), "Need one planning mass per item"
    tote_z = config.tote_initial_pose.position[2]
    dest_x, dest_y, _ = config.dest_table_pose.position
    dest_tote_poses = [
        Pose((dest_x + DEST_TOTE_OFFSET[0], dest_y + DEST_TOTE_OFFSET[1], tote_z))
    ]
    table_bases = {
        region: BaseConf(SE2Pose(TABLE_BASE_X, table.position[1], 0.0))
        for region, table in (
            ("src", config.source_table_pose),
            ("dst", config.dest_table_pose),
        )
    }
    del state  # the sim already holds the state; kept for a uniform signature
    return MassStreamContext(
        sim=sim,
        motion_seed=motion_seed,
        q_home=sim.robot.get_base(),
        item_names=list(item_names),
        item_masses=dict(item_masses),
        slots=make_slots(
            config.tote_half_extents,
            config.tote_wall_thickness,
            config.item_size,
            num_x=slot_grid[0],
            num_y=slot_grid[1],
        ),
        tote_grasps=make_tote_grasps(
            config.tote_half_extents, config.tote_wall_thickness
        ),
        item_grasps={
            name: make_item_grasps(name, config.item_size) for name in item_names
        },
        dest_tote_poses=dest_tote_poses,
        table_bases=table_bases,
        dest_spots=make_dest_spots(config, dest_tote_poses),
        torque_limits=config.scaled_torque_limits,
        torque_margin=torque_margin,
        table_ids={sim.source_table_id, sim.dest_table_id},
    )


def create_problem(
    sim: ObjectCentricKinematicMassTransport3DEnv,
    state: MassTransport3DObjectCentricState,
    max_trips: int | None = None,
    travel_cost: float = DEFAULT_TRAVEL_COST,
    **context_kwargs,
) -> PDDLProblem:
    """Build a PDDLProblem for moving every item onto the destination table.

    `max_trips` defaults to one trip per item, which never binds.
    """
    ctx = build_stream_context(sim, state, **context_kwargs)
    if max_trips is None:
        max_trips = len(ctx.item_names)
    trips = [f"trips{n}" for n in range(max_trips + 1)]

    tote_pose = state.get_object_pose("tote")
    q_home = BaseConf(ctx.q_home)
    init: list[tuple] = [
        ("BConf", q_home),
        ("AtBConf", q_home),
        ("HandEmpty",),
        ("CanMove",),
        ("BConfIn", q_home, "src"),
        ("TotePose", tote_pose),
        ("TotePoseIn", tote_pose, "src"),
        ("AtTotePose", tote_pose),
        ("TripsLeft", trips[-1]),
        ("=", ("TravelCost",), travel_cost),
    ]
    for region, base in ctx.table_bases.items():
        init += [
            ("BConf", base),
            ("BConfIn", base, region),
            ("TableBase", base, region),
        ]
    init += [("NextTrip", trips[n + 1], trips[n]) for n in range(max_trips)]
    init += [("ToteGrasp", g) for g in ctx.tote_grasps]
    # A fixed set rather than a stream: an input-free stream used twice in one
    # skeleton trips pddlstream's `update_bindings` assertion.
    for pose in ctx.dest_tote_poses:
        init += [("TotePose", pose), ("TotePoseIn", pose, "dst")]
    init += [("PoseIn", spot, "dst") for spot in ctx.dest_spots]
    for slot in ctx.slots:
        init += [("Slot", slot), ("SlotFree", slot)]
    goal_literals: list[tuple] = [("HandEmpty",)]
    for name in ctx.item_names:
        pose = state.get_object_pose(name)
        init += [
            ("Item", name),
            ("Pose", name, pose),
            ("PoseIn", pose, "src"),
            ("AtPose", name, pose),
        ]
        for spot in ctx.dest_spots:
            init += [("Pose", name, spot), ("DestPose", name, spot)]
        goal_literals.append(("OnDest", name))
    goal = (AND, *goal_literals)

    stream_map = {
        "sample-grasp": from_gen_fn(partial(sample_grasp, ctx)),
        "sample-carry-conf": from_gen_fn(partial(sample_carry_conf, ctx)),
        "inverse-kinematics": from_gen_fn(partial(inverse_kinematics_stream, ctx)),
        "slot-inverse-kinematics": from_gen_fn(
            partial(slot_inverse_kinematics_stream, ctx)
        ),
        "tote-inverse-kinematics": from_gen_fn(
            partial(tote_inverse_kinematics_stream, ctx)
        ),
        "plan-base-motion": from_gen_fn(partial(plan_base_motion, ctx)),
        "test-cfree-pose-pose": from_test(partial(test_cfree_pose_pose, ctx)),
        "test-cfree-pose-tote": from_test(partial(test_cfree_pose_tote, ctx)),
        "test-cfree-traj-pose": from_test(partial(test_cfree_traj_pose, ctx)),
        "test-cfree-traj-tote": from_test(partial(test_cfree_traj_tote, ctx)),
        "test-carry-torque": from_test(partial(test_carry_torque, ctx)),
    }

    constant_map = {"src": "src", "dst": "dst"}
    return PDDLProblem(DOMAIN_PDDL, constant_map, STREAM_PDDL, stream_map, init, goal)


def plan_mass_transport3d(
    sim: ObjectCentricKinematicMassTransport3DEnv,
    state: MassTransport3DObjectCentricState,
    max_time: float = 300.0,
    verbose: bool = False,
    **problem_kwargs,
) -> list[tuple[str, tuple]] | None:
    """Solve for a plan that moves every item onto the destination table."""
    problem = create_problem(sim, state, **problem_kwargs)
    stream_info = {
        name: StreamInfo(opt_gen_fn=PartialInputs(unique=True))
        for name in UNIQUE_OUTPUT_STREAMS
    }
    solution = solve(
        problem,
        stream_info=stream_info,
        algorithm="adaptive",
        unit_costs=False,
        max_time=max_time,
        verbose=verbose,
    )
    if verbose:
        print_solution(solution)
    plan, _, _ = solution
    return plan


def _run_gripper(
    sim: ObjectCentricKinematicMassTransport3DEnv,
    close: bool,
    frames: list | None = None,
    max_steps: int = 5,
    num_interp_frames: int = 8,
) -> bool:
    """Close or open the gripper; return whether the grasp state changed."""
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
    if succeeded and frames is not None:
        # Interpolate finger_state purely for the GIF, ending on the real value.
        finger_after = arm.get_finger_state()
        for i in range(1, num_interp_frames + 1):
            arm.set_finger_state(
                finger_before + (finger_after - finger_before) * i / num_interp_frames
            )
            frames.append(render_frame(sim))
    return succeeded


class _Executor:
    """Steps the sim through plan actions, recording frames and failures."""

    def __init__(
        self, sim: ObjectCentricKinematicMassTransport3DEnv, frames: list | None
    ) -> None:
        self.sim = sim
        self.frames = frames
        self.torque_violations = 0

    def step(self, action) -> None:
        """Take one env step."""
        _, _, _, _, info = self.sim.step(_clip_action(self.sim, action))
        self.torque_violations += int(info.get("torque_violation", False))
        if self.frames is not None:
            self.frames.append(render_frame(self.sim))

    def to_joints(
        self, target_joints, tol: float = 1e-3, max_attempts: int = 5
    ) -> None:
        """Step until the arm reaches `target_joints`, or give up."""
        # Re-derive the delta from the current joints on each attempt, since float32
        # rounding and per-step clipping leave a residual that compounds otherwise.
        for _ in range(max_attempts):
            current = remove_fingers_from_extended_joints(
                self.sim.robot.arm.get_joint_positions()
            )
            delta = get_jointwise_difference(
                _joint_infos(self.sim), list(target_joints[:7]), current
            )
            self.step([0.0, 0.0, 0.0] + delta + [0.0])
            if max(abs(d) for d in delta) < tol:
                break

    def follow(self, joint_plan) -> None:
        """Step through every waypoint of `joint_plan` after the first."""
        for target_joints in joint_plan[1:]:
            self.to_joints(target_joints)

    def move_base(self, base_plan, tol: float = 1e-3, max_steps: int = 50) -> None:
        """Step the base through every waypoint of `base_plan`.

        Waypoints can be farther apart than one (clipped) step, so each takes as many
        steps as it needs.
        """
        for target_base in base_plan[1:]:
            for _ in range(max_steps):
                current = self.sim.robot.get_base()
                delta = [
                    target_base.x - current.x,
                    target_base.y - current.y,
                    _wrap(target_base.rot - current.rot),
                ]
                if max(abs(d) for d in delta) < tol:
                    break
                self.step(delta + [0.0] * 7 + [0.0])

    def gripper(self, close: bool) -> bool:
        """Close or open the gripper."""
        return _run_gripper(self.sim, close=close, frames=self.frames)

    def corrected_final_joints(self, target_object_pose: Pose, final_joints) -> list:
        """Joints that put the held object at its target, given the real grasp.

        The grasp recorded at pick time can differ slightly from the nominal one.
        """
        sim = self.sim
        # pylint: disable=protected-access
        real_transform = sim._grasped_object_transform
        held_id = sim._grasped_object_id
        if real_transform is None:
            return final_joints
        corrected_ee = multiply_poses(target_object_pose, real_transform.invert())
        current = sim.robot.arm.get_joint_positions()
        try:
            correction = smoothly_follow_end_effector_path(
                sim.robot.arm,
                [corrected_ee],
                initial_joints=current,
                collision_ids={sim.source_table_id, sim.dest_table_id},
                joint_distance_fn=create_joint_distance_fn(sim.robot.arm),
                held_object=held_id,
                base_link_to_held_obj=real_transform,
                include_start=False,
            )
        except InverseKinematicsError:
            correction = []
        sim.robot.arm.set_joints(current)
        if correction and _fk(sim.robot.arm, correction[-1]).allclose(
            corrected_ee, atol=1e-3
        ):
            return correction[-1]
        return final_joints

    def pick_along(self, joint_plan, target_ee_pose: Pose | None = None) -> bool:
        """Reach along `joint_plan`, grasp, and retreat back along it.

        If the object has moved since planning (items slide in a carried tote), the
        final descent is re-planned onto `target_ee_pose`, the grasp at the object's
        observed pose.
        """
        arm = self.sim.robot.arm
        if target_ee_pose is None or _fk(arm, joint_plan[-1]).allclose(
            target_ee_pose, atol=3e-3
        ):
            self.follow(joint_plan)
            grasped = self.gripper(close=True)
            self.follow(list(reversed(joint_plan)))
            return grasped
        approach = self._approach_part(joint_plan)
        self.follow(approach)
        start = arm.get_joint_positions()
        hover = Pose(
            (
                target_ee_pose.position[0],
                target_ee_pose.position[1],
                target_ee_pose.position[2] + HOVER_HEIGHT,
            ),
            target_ee_pose.orientation,
        )
        sim = self.sim
        try:
            descent = smoothly_follow_end_effector_path(
                arm,
                [hover, target_ee_pose],
                initial_joints=start,
                collision_ids={sim.source_table_id, sim.dest_table_id},
                joint_distance_fn=create_joint_distance_fn(arm),
                include_start=True,
            )
        except InverseKinematicsError:
            descent = []
        arm.set_joints(start)
        descent = [start] + list(descent)
        self.follow(descent)
        grasped = self.gripper(close=True)
        self.follow(list(reversed(descent)))
        self.follow(list(reversed(approach)))
        return grasped

    def descent_to(self, target_ee_pose: Pose, planned_descent: list) -> list:
        """A vertical descent from the current conf onto `target_ee_pose`.

        Falls back to `planned_descent` if the target has not moved or no straight path
        exists.
        """
        arm = self.sim.robot.arm
        if _fk(arm, planned_descent[-1]).allclose(target_ee_pose, atol=3e-3):
            return planned_descent
        start = arm.get_joint_positions()
        start_ee = _fk(arm, start)
        hover = Pose(
            (
                target_ee_pose.position[0],
                target_ee_pose.position[1],
                start_ee.position[2],
            ),
            target_ee_pose.orientation,
        )
        sim = self.sim
        try:
            path = smoothly_follow_end_effector_path(
                arm,
                [hover, target_ee_pose],
                initial_joints=start,
                collision_ids={sim.source_table_id, sim.dest_table_id},
                joint_distance_fn=create_joint_distance_fn(arm),
                include_start=False,
            )
        except InverseKinematicsError:
            path = []
        arm.set_joints(start)
        return [start] + list(path) if path else planned_descent

    def _approach_part(self, joint_plan) -> list:
        """The waypoints of `joint_plan` up to the hover point above its target."""
        arm = self.sim.robot.arm
        target_z = _fk(arm, joint_plan[-1]).position[2]
        hover_index = len(joint_plan) - 1
        for index, joints in enumerate(joint_plan):
            if _fk(arm, joints).position[2] >= target_z + HOVER_HEIGHT - 1e-3:
                hover_index = index
        return list(joint_plan[: hover_index + 1])

    def place_along(self, joint_plan, target_object_pose: Pose) -> bool:
        """Reach along `joint_plan`, release at the target, and retreat."""
        joint_plan = list(joint_plan)
        self.follow(joint_plan[:-1])
        self.to_joints(self.corrected_final_joints(target_object_pose, joint_plan[-1]))
        released = self.gripper(close=False)
        self.follow(list(reversed(joint_plan)))
        return released


def _observed_grasp(sim, item: str, grasp: ItemGrasp) -> Pose:
    """The end-effector pose for `grasp` on `item` where it actually is now."""
    # pylint: disable=protected-access
    item_pose = get_pose(sim._object_name_to_pybullet_id(item), sim.physics_client_id)
    return multiply_poses(item_pose, grasp.transform)


def execute_plan(
    sim: ObjectCentricKinematicMassTransport3DEnv,
    plan,
    slots: dict[str, Pose],
    frames: list | None = None,
) -> bool:
    """Execute a plan in the real environment; return whether the goal is reached."""
    run = _Executor(sim, frames)
    for name, args in plan:
        if name in ("move_base", "travel", "return"):
            run.move_base(args[2] if name != "move_base" else args[3])
        elif name == "pick":
            run.pick_along(args[4].joint_plan, _observed_grasp(sim, args[0], args[2]))
        elif name == "place":
            run.place_along(args[4].joint_plan, args[1])
        elif name == "load":
            _, slot, tote_pose, _, _, traj = args
            run.place_along(traj.joint_plan, multiply_poses(tote_pose, slots[slot]))
        elif name == "unload":
            run.pick_along(args[5].joint_plan, _observed_grasp(sim, args[0], args[3]))
        elif name == "pick_tote":
            _, grasp, carry, _, traj = args
            run.follow(traj.joint_plan)
            # The tote may have been nudged since planning; descend onto where it is.
            tote_pose = get_pose(sim.tote_id, sim.physics_client_id)
            descent = run.descent_to(
                multiply_poses(tote_pose, grasp.transform),
                list(reversed(carry.lift_path)),
            )
            run.follow(descent)
            run.gripper(close=True)
            run.follow(list(reversed(descent)))
        elif name == "place_tote":
            _, _, carry, _, traj = args
            run.follow(list(reversed(carry.lift_path)))
            run.gripper(close=False)
            run.follow(carry.lift_path)
            run.follow(list(reversed(traj.joint_plan)))
        else:
            raise ValueError(f"Unknown action: {name}")
    return sim.goal_reached()


def make_sim(
    num_items: int,
    use_gui: bool = False,
    config: MassTransport3DEnvConfig | None = None,
    kinematic: bool = False,
) -> ObjectCentricKinematicMassTransport3DEnv:
    """The env to plan and execute in, with state access for the streams.

    By default this is the physics env; `kinematic` picks the fast variant.
    """
    env_cls = (
        ObjectCentricKinematicMassTransport3DEnv
        if kinematic
        else ObjectCentricMassTransport3DEnv
    )
    return env_cls(
        num_items=num_items,
        config=config if config is not None else MassTransport3DEnvConfig(),
        use_gui=use_gui,
        allow_state_access=True,
    )


def solve_and_execute(
    num_items: int = 2,
    seed: int = 0,
    max_time: float = 300.0,
    use_gui: bool = False,
    verbose: bool = False,
    gif_path: str | Path | None = None,
    config: MassTransport3DEnvConfig | None = None,
    kinematic: bool = False,
    **problem_kwargs,
) -> bool:
    """Reset MassTransport3D, plan with PDDLStream, and execute the plan.

    Returns whether every item ended up on the destination table.
    """
    sim = make_sim(num_items, use_gui=use_gui, config=config, kinematic=kinematic)
    state, _ = sim.reset(seed=seed)
    plan = plan_mass_transport3d(
        sim, state, max_time=max_time, verbose=verbose, **problem_kwargs
    )
    if plan is None:
        return False
    # Planning scratch-mutates the live state, so rewind by re-seeding.
    sim.reset(seed=seed)
    slots = build_stream_context(
        sim, state, slot_grid=problem_kwargs.get("slot_grid", (2, 3))
    ).slots
    frames = [render_frame(sim)] if gif_path is not None else None
    try:
        return execute_plan(sim, plan, slots, frames=frames)
    finally:
        if frames is not None:
            assert gif_path is not None
            save_gif(gif_path, frames)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-items", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=300.0)
    parser.add_argument(
        "--max-trips",
        type=int,
        default=None,
        help="Bound on source-to-destination trips (default: one per item).",
    )
    parser.add_argument(
        "--travel-cost",
        type=float,
        default=DEFAULT_TRAVEL_COST,
        help="Cost of one crossing between the tables; other actions cost 1.",
    )
    parser.add_argument(
        "--torque-limit-scale",
        type=float,
        default=1.0,
        help="Scale on the arm's torque limits; lower is harder.",
    )
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Run the kinematic env instead of the physics one (default: physics).",
    )
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
        gif_output_path("masstransport3d", f"o{args.num_items}", args.gif_dir)
        if args.save_gif
        else None
    )
    success = solve_and_execute(
        num_items=args.num_items,
        seed=args.seed,
        max_time=args.max_time,
        use_gui=args.use_gui,
        gif_path=gif_path,
        verbose=True,
        config=MassTransport3DEnvConfig(torque_limit_scale=args.torque_limit_scale),
        max_trips=args.max_trips,
        travel_cost=args.travel_cost,
        kinematic=args.kinematic,
    )
    print(f"Reached goal: {success}")


if __name__ == "__main__":
    main()
