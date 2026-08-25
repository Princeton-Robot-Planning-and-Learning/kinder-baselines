"""Tests for the PDDLStream LimbRepositioning3D integration."""

import itertools

import numpy as np
import pybullet as p
import pytest
from kinder.envs.dynamic3d.limb_utils import NUM_LIMB_JOINTS, NUM_ROBOT_JOINTS
from kinder.envs.dynamic3d.limbrepositioning3d import create_variant_config
from pybullet_helpers.geometry import SE2Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)

from kinder_pddlstream_planning.limbrepositioning3d.run import (
    LIMB_NAME,
    build_stream_context,
    create_env,
    reset_to_start,
    start_base_pose,
    variant_kwargs,
)
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    GRASP_ROLLS,
    GRASP_SLIDE_FRACTIONS,
    NUM_SAMPLED_GRASPS,
    LimbConf,
    MPCConfig,
    PredictiveSamplingMPC,
    TorqueTrajectory,
    check_human_joint_limits,
    check_human_torque_limits,
    check_robot_torque_limits,
    plan_base_motion,
    plan_grasp_motion,
    sample_base_pose,
    sample_grasp,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    _base_in_collision,
    advance_corrected,
    arm_in_collision,
    capture_state,
    engage_grasp,
    human_in_collision,
    limb_out_of_limits,
    release_grasp,
    restore_state,
)

# A variant with no furniture, so the tests that build an environment stay fast and are
# not at the mercy of the bed/wheelchair meshes.
VARIANT = "isolated-right-arm"

# A variant whose goal is out of reach from the base placement it ships with.
MOVE_BASE_REQUIRED_VARIANT = "wheelchair-left-arm"


@pytest.fixture(name="limb_env", scope="module")
def _limb_env():
    """A reset LimbRepositioning3D environment with its base at the start pose."""
    sim = create_env(VARIANT)
    sim.reset(seed=0)
    ctx = build_stream_context(sim, motion_seed=0)
    reset_to_start(ctx)
    yield sim, ctx
    sim.close()


def _confs(ctx):
    """The limb's initial and goal configurations, as stream objects."""
    scene = ctx.sim.config.scene
    return (
        LimbConf(tuple(scene.limb_init_joint_positions)),
        LimbConf(tuple(scene.limb_goal_joint_positions)),
    )


def test_the_sweep_applies_the_per_variant_overrides():
    """One shared setting cannot solve all sixteen, so the sweep fills in per
    variant."""
    assert variant_kwargs("wheelchair-left-arm", seed=0)["robot_base_z"] == 0.4
    assert variant_kwargs("wheelchair-right-arm", seed=0)["robot_base_z"] == 0.55
    assert variant_kwargs("bed-left-leg", seed=0)["check_base_collisions"] is False
    assert variant_kwargs("bed-right-leg", seed=0)["check_base_collisions"] is False

    # Variants the defaults already solve are passed through untouched.
    assert variant_kwargs("isolated-right-arm", seed=0) == {"seed": 0}


def test_start_pose_is_behind_the_variant_placement():
    """move_base has to have something to do: the base starts away from the limb."""
    grasp_base = create_variant_config(VARIANT).robot_base_home_pose
    start = start_base_pose(grasp_base, standoff=1.0)
    assert np.isclose(start.rot, grasp_base.rot)
    assert np.isclose(np.hypot(start.x - grasp_base.x, start.y - grasp_base.y), 1.0)
    # Backing off along the base's own heading, so the robot approaches head-on.
    assert np.isclose(start.x, grasp_base.x - np.cos(grasp_base.rot))
    assert np.isclose(start.y, grasp_base.y - np.sin(grasp_base.rot))


def test_the_shipped_base_placement_cannot_reach_the_goal():
    """Regression test for the reason `move_base` exists in this domain.

    Each variant's placement is solved for the limb's initial configuration only.

    A planner holding the base fixed cannot solve the variants whose goal it misses.
    """
    sim = create_env(MOVE_BASE_REQUIRED_VARIANT)
    try:
        sim.reset(seed=0)
        ctx = build_stream_context(sim, motion_seed=0)
        reset_to_start(ctx)
        scene = sim.config.scene
        (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
        release_grasp(sim)
        sim.robot.set_base(ctx.grasp_base_pose)
        reached = {}
        for label, conf in [
            ("init", scene.limb_init_joint_positions),
            ("goal", scene.limb_goal_joint_positions),
        ]:
            sim.limb.set_joints(list(conf))
            target = multiply_poses(sim.limb.get_end_effector_pose(), grasp)
            sim.robot.arm.set_joints(list(ctx.retract_joints))
            try:
                inverse_kinematics(
                    sim.robot.arm, target, validate=True, set_joints=False
                )
                reached[label] = True
            except InverseKinematicsError:
                reached[label] = False
        assert reached["init"], "the shipped placement should reach the initial grasp"
        assert not reached["goal"], "the shipped placement should not reach the goal"
    finally:
        sim.close()


def test_sampled_base_poses_reach_the_whole_motion(limb_env):
    """Every certified base pose must hold the grasp throughout the motion.

    Endpoints alone are not enough, so this checks at a finer resolution.
    """
    sim, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    candidates = list(
        itertools.islice(
            sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf), 2
        )
    )
    assert candidates, "no base pose reaches the whole motion"

    release_grasp(sim)
    try:
        for (base_conf,) in candidates:
            sim.robot.set_base(base_conf)
            seed = list(ctx.retract_joints)
            for waypoint in np.linspace(
                np.asarray(init_conf.positions), np.asarray(goal_conf.positions), 9
            ):
                sim.limb.set_joints(list(waypoint))
                target = multiply_poses(sim.limb.get_end_effector_pose(), grasp)
                sim.robot.arm.set_joints(seed)
                seed = inverse_kinematics(
                    sim.robot.arm, target, validate=True, set_joints=False
                )
    finally:
        reset_to_start(ctx)


def test_grasp_stream_offers_the_scene_transform_first(limb_env):
    """The scene's transform is tried first, then rolls of it, then slides up the
    limb."""
    sim, ctx = limb_env
    grasps = [g for (g,) in sample_grasp(ctx, LIMB_NAME)]
    assert grasps[0] == sim.scene.grasp_transform
    assert len(grasps) == 1 + len(GRASP_ROLLS) + NUM_SAMPLED_GRASPS
    nominal = sim.scene.grasp_transform
    for grasp in grasps[1 : 1 + len(GRASP_ROLLS)]:
        assert np.allclose(grasp.position, nominal.position)
    low, high = GRASP_SLIDE_FRACTIONS
    for grasp in grasps[1 + len(GRASP_ROLLS) :]:
        offset = np.subtract(grasp.position, nominal.position)
        along = float(offset @ ctx.grasp_slide_axis)
        # The slide runs up the limb and nowhere else.
        assert np.allclose(offset - along * ctx.grasp_slide_axis, 0.0, atol=1e-6)
        assert low * ctx.grasp_slide_span <= along <= high * ctx.grasp_slide_span


def test_grasp_motion_ends_on_the_limb(limb_env):
    """The planned path ends with the end effector on the grasp frame.

    This is what lets the executed `grasp` action weld with zero error.
    """
    sim, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    (base_conf,) = next(sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf))
    trajectory, state = next(
        plan_grasp_motion(ctx, LIMB_NAME, grasp, base_conf, init_conf)
    )

    assert state.base_pose == base_conf
    assert list(state.limb_positions) == list(init_conf.positions)
    assert not any(state.limb_velocities)
    assert not any(state.robot_velocities)
    assert list(trajectory.joint_plan[0]) == list(ctx.retract_joints)
    assert list(trajectory.joint_plan[-1]) == list(state.robot_positions)

    restore_state(sim, state, regrasp=False)
    try:
        error = np.linalg.norm(
            np.subtract(
                sim.robot.arm.get_end_effector_pose().position,
                multiply_poses(sim.limb.get_end_effector_pose(), grasp).position,
            )
        )
        assert error < 1e-3
    finally:
        reset_to_start(ctx)


def test_base_motion_connects_its_endpoints(limb_env):
    """`plan-base-motion` returns a path that starts and ends where it was asked to."""
    _, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    (base_conf,) = next(sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf))
    (base_plan,) = next(plan_base_motion(ctx, ctx.start_base_pose, base_conf))

    assert len(base_plan) >= 2
    for planned, requested in [
        (base_plan[0], ctx.start_base_pose),
        (base_plan[-1], base_conf),
    ]:
        assert np.allclose(
            (planned.x, planned.y, planned.rot),
            (requested.x, requested.y, requested.rot),
            atol=1e-5,
        )


def test_state_restore_round_trips_through_the_grasp(limb_env):
    """`capture_state` and `restore_state` form a complete restore point.

    The MPC rewinds between rollouts, so a missed weld frame corrupts them all.
    """
    sim, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    (base_conf,) = next(sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf))
    _, state = next(plan_grasp_motion(ctx, LIMB_NAME, grasp, base_conf, init_conf))

    restore_state(sim, state)
    torque = [0.3] * NUM_ROBOT_JOINTS
    for _ in range(50):
        advance_corrected(sim, torque)
    first = capture_state(sim)

    restore_state(sim, state)
    for _ in range(50):
        advance_corrected(sim, torque)
    second = capture_state(sim)

    for field in ("robot_positions", "limb_positions", "limb_velocities"):
        assert np.allclose(getattr(first, field), getattr(second, field), atol=1e-9)
    reset_to_start(ctx)


def test_mpc_makes_progress_toward_the_goal(limb_env):
    """A short predictive-sampling run closes some of the gap to the goal.

    Not a convergence test, since driving all the way in takes tens of seconds.
    """
    sim, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    (base_conf,) = next(sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf))
    _, state = next(plan_grasp_motion(ctx, LIMB_NAME, grasp, base_conf, init_conf))

    goal = np.asarray(goal_conf.positions)
    ctx.mpc = MPCConfig(num_rollouts=12)
    restore_state(sim, state)
    start_error = float(
        np.linalg.norm(np.subtract(sim.limb.get_joint_positions(), goal))
    )
    mpc = PredictiveSamplingMPC(ctx, goal)
    for _ in range(25):
        torque = mpc.step()
        assert len(torque) == NUM_ROBOT_JOINTS
        for _ in range(ctx.mpc.action_repeat):
            advance_corrected(sim, torque)
    end_error = float(np.linalg.norm(np.subtract(sim.limb.get_joint_positions(), goal)))
    assert end_error < start_error
    reset_to_start(ctx)


def test_torques_stay_inside_the_environment_limits(limb_env):
    """The MPC only ever proposes torques `check-robot-torque-limits` accepts."""
    sim, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    (base_conf,) = next(sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf))
    _, state = next(plan_grasp_motion(ctx, LIMB_NAME, grasp, base_conf, init_conf))

    lower, upper = ctx.robot_torque_limits
    ctx.mpc = MPCConfig(num_rollouts=8)
    restore_state(sim, state)
    mpc = PredictiveSamplingMPC(ctx, np.asarray(goal_conf.positions))
    for _ in range(5):
        torque = np.asarray(mpc.step())
        assert (torque >= lower - 1e-9).all() and (torque <= upper + 1e-9).all()
    reset_to_start(ctx)


def test_reset_to_start_restores_the_planning_start_state(limb_env):
    """Planning scratch-mutates the simulator; execution has to rewind it exactly."""
    sim, ctx = limb_env
    scene = sim.config.scene

    engage_grasp(sim)
    for _ in range(20):
        advance_corrected(sim, [0.5] * NUM_ROBOT_JOINTS)

    reset_to_start(ctx)
    assert sim._grasp_constraint_id is None  # pylint: disable=protected-access
    base = sim.robot.get_base()
    assert np.allclose(
        (base.x, base.y, base.rot),
        (ctx.start_base_pose.x, ctx.start_base_pose.y, ctx.start_base_pose.rot),
    )
    assert np.allclose(sim.robot.arm.get_joint_positions(), ctx.retract_joints)
    assert np.allclose(sim.limb.get_joint_positions(), scene.limb_init_joint_positions)
    assert np.allclose(sim.limb.get_joint_velocities(), [0.0] * NUM_LIMB_JOINTS)


# ---------------------------------------------------------------- human-body collisions

# The isolated scenes contain nothing but the limb being repositioned, so the collision
# tests need a scene that actually has a person in it.
HUMAN_VARIANT = "human-right-arm"


@pytest.fixture(name="human_env", scope="module")
def _human_env():
    """A scene containing a torso and three resting limbs besides the grasped one."""
    sim = create_env(HUMAN_VARIANT)
    sim.reset(seed=0)
    ctx = build_stream_context(sim, motion_seed=0)
    reset_to_start(ctx)
    yield sim, ctx
    sim.close()


def test_the_person_is_an_obstacle_but_the_grasped_limb_is_not(human_env):
    """The body must be avoided; the limb being repositioned must not be.

    The environment reports none of this and zeroes their collision filter masks.
    """
    sim, ctx = human_env
    obstacles = ctx.human_collision_ids

    assert (
        sim.limb.robot_id not in obstacles
    ), "the grasped limb is a target, not an obstacle"
    assert sim.scene.torso_id in obstacles
    for limb in sim.scene.limbs.values():
        if limb.robot_id != sim.limb.robot_id:
            assert limb.robot_id in obstacles
    assert set(ctx.obstacle_ids) >= set(obstacles)

    # The environment itself reports none of them.
    assert not set(sim.scene.get_scene_collision_ids()) & set(obstacles)


def test_planned_grasp_approach_stays_out_of_the_person(human_env):
    """Every waypoint of the approach must clear the body.

    IK alone anchors the motion to a conf with the forearm inside the torso.
    """
    sim, ctx = human_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    (base_conf,) = next(sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf))
    trajectory, _ = next(plan_grasp_motion(ctx, LIMB_NAME, grasp, base_conf, init_conf))

    release_grasp(sim)
    try:
        # `arm_in_collision` reads the arm where it stands, so the base has to be put
        # back at the sampled pose first -- the streams restore it on the way out.
        sim.robot.set_base(base_conf)
        sim.limb.set_joints(list(init_conf.positions))
        for waypoint in trajectory.joint_plan:
            assert not arm_in_collision(ctx, waypoint)
    finally:
        reset_to_start(ctx)


def test_collision_checking_can_be_turned_off(human_env):
    """Disabling it restores the environment's own (absent) collision model."""
    sim, _ = human_env
    ctx = build_stream_context(sim, check_robot_collisions=False)
    assert ctx.obstacle_ids == []
    assert ctx.human_collision_ids, "the bodies still exist; they are just not enforced"
    assert ctx.base_obstacle_ids, "where the base parks is a separate question"


# ------------------------------------------------------- base, self and body collisions

# The variant that prompted these checks, where the base was parked inside the leg.
LEG_VARIANT = "human-left-leg"


@pytest.fixture(name="leg_env", scope="module")
def _leg_env():
    """A seated person whose left leg is the limb being repositioned."""
    sim = create_env(LEG_VARIANT)
    sim.reset(seed=0)
    ctx = build_stream_context(sim, motion_seed=0)
    reset_to_start(ctx)
    yield sim, ctx
    sim.close()


def test_the_base_may_not_be_parked_in_the_person(leg_env):
    """Regression test for base poses that overlapped the patient.

    `_base_in_collision` used to consult the empty `scene_collision_ids` alone.
    """
    sim, ctx = leg_env
    try:
        assert not _base_in_collision(ctx), "the start pose is clear"

        for body_id in (sim.scene.torso_id, sim.limb.robot_id):
            position, _ = p.getBasePositionAndOrientation(
                body_id, physicsClientId=sim.physics_client_id
            )
            sim.robot.set_base(
                SE2Pose(position[0], position[1], ctx.grasp_base_pose.rot)
            )
            assert _base_in_collision(ctx), f"base parked inside body {body_id}"
    finally:
        reset_to_start(ctx)


def test_human_collisions_are_measured_against_the_resting_overlap(leg_env):
    """The person's parts overlap, so only extra penetration counts.

    A check against zero would call the shipped start state a collision.
    """
    sim, ctx = leg_env
    try:
        assert ctx.resting_penetration, "the baseline is recorded at the start state"
        assert min(ctx.resting_penetration.values()) < 0, "parts do overlap at rest"
        assert not human_in_collision(ctx), "the start state is not a collision"

        init = np.asarray(sim.config.scene.limb_init_joint_positions)
        sim.limb.set_joints(list(init + 0.3))
        assert not human_in_collision(ctx), "settling motion is not a collision"
        sim.limb.set_joints(list(init + 1.5))
        assert human_in_collision(ctx), "the leg is driven into the torso here"
    finally:
        reset_to_start(ctx)


def _trajectory(*limb_path):
    """A trajectory whose replay passed through `limb_path`, with no torque on it."""
    return TorqueTrajectory(
        robot_torques=[], human_torques=[], limb_path=[list(q) for q in limb_path]
    )


def test_the_limb_configurations_of_the_task_are_anatomically_possible(limb_env):
    """Both ends of the task pass `check-human-joint-limits`."""
    _, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    path = _trajectory(init_conf.positions, goal_conf.positions)
    assert check_human_joint_limits(ctx, None, goal_conf, path, None)


def test_configurations_outside_the_range_of_motion_are_rejected(limb_env):
    """A hyperextended elbow is rejected wherever in the trajectory it appears.

    The whole path is checked, so a legal goal does not excuse an illegal waypoint.
    """
    _, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    hyperextended = list(init_conf.positions)
    hyperextended[3] = -hyperextended[3] + 1.0
    path = _trajectory(init_conf.positions, hyperextended, goal_conf.positions)
    assert not check_human_joint_limits(ctx, None, goal_conf, path, None)


def test_human_torque_limits_are_checked_over_the_whole_trajectory(limb_env):
    """`check-human-torque-limits` rejects a single overloaded step."""
    _, ctx = limb_env
    _, goal_conf = _confs(ctx)
    within = [0.0] * NUM_LIMB_JOINTS
    over = [0.0] * NUM_LIMB_JOINTS
    over[0] = ctx.human_torque_limit + 1.0

    safe = TorqueTrajectory(
        robot_torques=[],
        human_torques=[within, within],
        robot_induced_torques=[within, within],
    )
    assert check_human_torque_limits(ctx, None, goal_conf, safe, None)
    unsafe = TorqueTrajectory(
        robot_torques=[],
        human_torques=[within, over],
        robot_induced_torques=[within, within],
    )
    assert not check_human_torque_limits(ctx, None, goal_conf, unsafe, None)


def test_robot_torque_limits_are_checked_over_the_whole_trajectory(limb_env):
    """`check-robot-torque-limits` rejects a single step outside the action space."""
    _, ctx = limb_env
    _, goal_conf = _confs(ctx)
    _, upper = ctx.robot_torque_limits
    within = list(upper)
    over = list(upper + 1.0)

    safe = TorqueTrajectory(robot_torques=[within, within])
    assert check_robot_torque_limits(ctx, None, goal_conf, safe, None)
    unsafe = TorqueTrajectory(robot_torques=[within, over])
    assert not check_robot_torque_limits(ctx, None, goal_conf, unsafe, None)


def test_robot_torque_limits_bind_the_commanded_torque(limb_env):
    """The correction shares the robot's budget with the hold it rides on.

    A correction inside the action space can still saturate the robot once the hold is
    added, and the check has to see that rather than the correction alone.
    """
    _, ctx = limb_env
    _, goal_conf = _confs(ctx)
    _, upper = ctx.robot_torque_limits
    within = list(upper)
    over = list(upper + 1.0)

    saturating = TorqueTrajectory(
        robot_torques=[within, within], commanded_torques=[within, over]
    )
    assert not check_robot_torque_limits(ctx, None, goal_conf, saturating, None)


def test_holding_the_limb_still_costs_the_person_nothing_extra(limb_env):
    """A zero correction leaves the person bearing gravity and nothing more.

    `robot` is the bound the motion is checked against, so it has to read zero while
    the robot only holds: subtracting the muscle tone from it as well would leave
    `-tone` here and spend the whole budget before the limb moves.
    """
    sim, ctx = limb_env
    init_conf, goal_conf = _confs(ctx)
    (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
    (base_conf,) = next(sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf))
    _, state = next(plan_grasp_motion(ctx, LIMB_NAME, grasp, base_conf, init_conf))
    try:
        restore_state(sim, state)
        human = advance_corrected(sim, [0.0] * NUM_ROBOT_JOINTS)
        assert np.abs(human.gravity).max() > 0.0, "the limb has weight"
        assert np.allclose(human.total, human.gravity, atol=1e-2), "the person bears it"
        assert np.abs(human.robot).max() < 1e-2, "the robot adds nothing while holding"
    finally:
        reset_to_start(ctx)


def test_limb_motions_that_leave_the_range_of_motion_are_rejected(limb_env):
    """`plan-limb-motion` will not certify a trajectory that bends a joint too far.

    The MPC penalizes the violation, but only rejecting it keeps an out- of-range
    trajectory out of the plan.
    """
    sim, ctx = limb_env
    try:
        init_conf, goal_conf = _confs(ctx)
        (grasp,) = next(sample_grasp(ctx, LIMB_NAME))
        (base_conf,) = next(
            sample_base_pose(ctx, LIMB_NAME, grasp, init_conf, goal_conf)
        )
        _, state = next(plan_grasp_motion(ctx, LIMB_NAME, grasp, base_conf, init_conf))

        restore_state(sim, state)
        assert not limb_out_of_limits(ctx)
        hyperextended = list(sim.limb.get_joint_positions())
        hyperextended[3] = -hyperextended[3] + 1.0
        sim.limb.set_joints(hyperextended)
        assert limb_out_of_limits(ctx)
    finally:
        reset_to_start(ctx)
