"""Collision checking of every motion a Tossing3D pick or toss executes.

Two hitl-pmp Tossing3D runs (t3-final-20261023, expectimax seed 2 and determinized seed
4) collapsed on one chain. An in-bin pick's lift passed the held cube through the bin
wall between two BiRRT nodes, timed out, and left the arm stretched. No skill ever
retracted it. The next pick's base motion, planned in 2-D with the arm ignored, swung
the stretched gripper into the bin. Then every later pick timed out on the rim.

`t3_final_stuck_pick_states.json` holds the recorded robot, cube and bin just before s2
seq 482 (the first snag), seq 484 (base motion onto the rim) and seq 485 (stuck). s4
before seq 808 is the stretched arm whose fingers the planner poses inside the cube.
"""

# These tests inspect planning geometry and phase state directly.
# pylint: disable=protected-access

import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# pylint: disable=wrong-import-position
import kinder
import numpy as np
import pytest
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectTypeFeatures,
    MujocoStaticColliderType,
    MujocoTidyBotRobotObjectType,
)
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object
from pybullet_helpers.joint import get_joint_infos, iter_between_joint_positions
from relational_structs import Object, ObjectCentricState
from spatialmath import SE2

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
    PickCubeController,
    create_lifted_controllers,
    first_dense_joint_path_collision,
)
from kinder_models.dynamic3d.utils import (
    MINIMUM_HOLDING_HEIGHT,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    run_base_motion_planning,
    wrap_arm_joint_difference,
)

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

_STATES = json.loads(
    Path(__file__).with_name("t3_final_stuck_pick_states.json").read_text("utf-8")
)
_HOME = np.deg2rad([0, -20, 180, -146, 0, -50, 90])
_ARM = tuple(f"pos_arm_joint{i}" for i in range(1, 8))
Phase = PickCubeController.PickCubeControllerPhase


@pytest.fixture(name="env", scope="module")
def env_fixture():
    """One Tossing3D env for the module; each test restores its own state."""
    env = kinder.make(
        "kinder/Tossing3D-o1-v0",
        render_mode="rgb_array",
        num_objects=1,
        allow_state_access=True,
    )
    yield env
    env.close()


def _restore(env, key: str) -> ObjectCentricState:
    """Seed 125's scene with the recorded robot, cube and bin, set in MuJoCo."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    for name, features in _STATES[key]["state"].items():
        obj = state.get_object_from_name(name)
        for feature, value in features.items():
            state.set(obj, feature, float(value))
    env.unwrapped.set_state(env.observation_space.vectorize(state))
    return env.observation_space.devectorize(env.unwrapped.get_state())


def _bin_aware_sim(env, state: ObjectCentricState) -> PyBulletSim:
    scene = env.unwrapped._object_centric_env
    geometry = scene.get_object("bin_0")
    bin_obj = state.get_object_from_name("bin_0")
    sim = PyBulletSim(state)
    sim.add_bin(
        name="bin_0",
        pose=Pose(
            tuple(state.get(bin_obj, k) for k in ("x", "y", "z")),
            tuple(state.get(bin_obj, k) for k in ("qx", "qy", "qz", "qw")),
        ),
        length=geometry.length,
        width=geometry.width,
        height=geometry.height,
        wall_thickness=geometry.wall_thickness,
    )
    return sim


def _objects(state: ObjectCentricState):
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    return (
        robot,
        state.get_object_from_name("cube_0"),
        state.get_object_from_name("cuboid_barrier"),
    )


def _planned_pick(env, state: ObjectCentricState):
    sim = _bin_aware_sim(env, state)
    controller = create_lifted_controllers(env.action_space, state, pybullet_sim=sim)[
        "pick_cube"
    ].ground(_objects(state))
    controller.reset(state, None)
    return controller, sim


def _arm(state: ObjectCentricState) -> np.ndarray:
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    return np.array([state.get(robot, k) for k in _ARM])


def _near(a, b, atol: float = 1e-6) -> bool:
    diff = wrap_arm_joint_difference(np.asarray(a)[:7] - np.asarray(b)[:7])
    return bool(np.max(np.abs(diff)) <= atol)


def _base_sweep_collisions(sim, state, base_plan, joints) -> int:
    """Poses along the executed base path (5 mm / 0.01 rad) at which the robot, with the
    arm and fingers at `joints`, touches anything in the planning scene."""
    robot = _objects(state)[0]
    poses = []
    for start, end in zip(base_plan[:-1], base_plan[1:]):
        dth = (end.theta() - start.theta() + np.pi) % (2 * np.pi) - np.pi
        steps = max(
            1,
            int(
                max(
                    abs(end.x - start.x) / 0.005,
                    abs(end.y - start.y) / 0.005,
                    abs(dth) / 0.01,
                )
            ),
        )
        for t in np.linspace(0.0, 1.0, steps + 1)[:-1]:
            poses.append(
                (
                    start.x + t * (end.x - start.x),
                    start.y + t * (end.y - start.y),
                    start.theta() + t * dth,
                )
            )
    poses.append((base_plan[-1].x, base_plan[-1].y, base_plan[-1].theta()))
    hits = 0
    for x, y, theta in poses:
        moved = state.copy()
        moved.set(robot, "pos_base_x", x)
        moved.set(robot, "pos_base_y", y)
        moved.set(robot, "pos_base_rot", theta)
        sim.set_state(moved)
        hits += int(
            check_collisions_with_held_object(
                sim.robot,
                sim.get_collision_bodies(),
                sim.physics_client_id,
                None,
                None,
                list(joints),
            )
        )
    return hits


def _dense_arm_collisions(sim, plan, *, held: int | None, threshold: float, bodies):
    """Points of `plan`, at 200 per radian, at which the planner's own collision
    function reports a collision."""
    robot = sim.robot
    infos = get_joint_infos(robot.robot_id, robot.arm_joints, sim.physics_client_id)
    dense = [list(plan[0])]
    for q1, q2 in zip(plan[:-1], plan[1:]):
        dense += list(
            iter_between_joint_positions(
                infos, list(q1), list(q2), num_interp_per_unit=200, include_start=False
            )
        )
    return sum(
        int(
            check_collisions_with_held_object(
                robot,
                bodies,
                sim.physics_client_id,
                held,
                sim.base_link_to_held_obj if held is not None else None,
                q,
                distance_threshold=threshold,
            )
        )
        for q in dense
    )


# --- A: a pick starts by opening the gripper and retracting the arm ---------------


@pytest.mark.parametrize("key", ["expectimax_s2_seq484", "expectimax_s2_seq485"])
def test_a_stretched_pick_plans_a_retract_from_the_real_arm_to_home(env, key):
    """RETRACT is planned from the arm the robot really has, to home_joints."""
    state = _restore(env, key)
    assert not _near(_arm(state), _HOME, atol=0.5)  # The recorded arm is stretched.
    controller, sim = _planned_pick(env, state)
    try:
        retract = controller.plans[Phase.RETRACT]
        assert _near(retract[0], _arm(state))
        assert _near(retract[-1], _HOME)
    finally:
        sim.close()


def test_the_pick_opens_the_gripper_then_retracts_before_the_base_moves(env):
    """Executed in MuJoCo from the recorded state before s2 seq 484, whose base motion
    carried the stretched arm onto the bin rim and timed out: the arm reaches home
    before the base moves, and the cube is held."""
    state = _restore(env, "expectimax_s2_seq484")
    robot, cube, _ = _objects(state)
    base0 = np.array([state.get(robot, k) for k in ("pos_base_x", "pos_base_y")])
    controller, sim = _planned_pick(env, state)
    phases, retracted_before_base_moved = [], None
    try:
        for _ in range(400):
            phases.append(controller.current_phase)
            obs, _, _, _, _ = env.step(controller.step())
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
            base = np.array([state.get(robot, k) for k in ("pos_base_x", "pos_base_y")])
            if (
                retracted_before_base_moved is None
                and np.linalg.norm(base - base0) > 0.01
            ):
                retracted_before_base_moved = _near(_arm(state), _HOME, atol=0.05)
            if controller.terminated():
                break
    finally:
        sim.close()
    order = list(dict.fromkeys(phases))
    assert order[:3] == [Phase.OPEN_GRIPPER, Phase.RETRACT, Phase.BASE_MOTION]
    assert retracted_before_base_moved
    assert controller.terminated()
    assert state.get(cube, "z") > MINIMUM_HOLDING_HEIGHT


def test_a_retracted_pick_spends_no_time_retracting(env):
    """From a fresh episode the arm is already home: the retract is a single point."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    controller, sim = _planned_pick(env, state)
    try:
        retract = controller.plans[Phase.RETRACT]
        assert all(_near(q, _HOME, atol=0.05) for q in retract)
    finally:
        sim.close()


# --- B: the base motion is checked in 3-D with the arm attached --------------------


def test_the_base_planner_rejects_a_path_that_sweeps_the_arm_through_the_bin(env):
    """S2 seq 484: the 2-D planner's path, with the recorded stretched arm carried
    rigidly, puts the gripper in the bin.

    Given the robot's 3-D body, it does not.
    """
    state = _restore(env, "expectimax_s2_seq484")
    controller, sim = _planned_pick(env, state)
    target = controller.plans[Phase.BASE_MOTION][-1]
    stretched = list(_arm(state)) + [0.0] * 6
    try:
        kwargs = {
            "state": state,
            "target_base_pose": target,
            "x_bounds": WORLD_X_BOUNDS,
            "y_bounds": WORLD_Y_BOUNDS,
            "seed": 0,
            "disable_collision_objects": ["cube_0"],
            "bounding_boxes": sim.bounding_boxes,
        }
        flat = run_base_motion_planning(**kwargs)
        assert flat is not None
        assert _base_sweep_collisions(sim, state, flat, stretched) > 0

        def body_collides(pose: SE2) -> bool:
            return sim.robot_collides_at_base(pose, stretched)

        checked = run_base_motion_planning(**kwargs, body_collision_fn=body_collides)
        if checked is not None:
            assert _base_sweep_collisions(sim, state, checked, stretched) == 0
    finally:
        sim.close()


@pytest.mark.parametrize("key", ["expectimax_s2_seq484", "expectimax_s2_seq485"])
def test_the_pick_base_motion_is_clear_with_the_arm_it_carries(env, key):
    """The arm the base motion carries is the retract's end; every pose along the
    executed base path is clear of everything with it attached."""
    state = _restore(env, key)
    controller, sim = _planned_pick(env, state)
    try:
        carried = controller.plans[Phase.RETRACT][-1]
        base = controller.plans[Phase.BASE_MOTION]
        assert _base_sweep_collisions(sim, state, base, carried) == 0
    finally:
        sim.close()


# --- C: plan from the real configuration, and check it --------------------------


@pytest.mark.parametrize("key", ["expectimax_s2_seq484", "expectimax_s2_seq485"])
def test_the_hover_starts_where_the_retract_leaves_the_arm(env, key):
    """Not the current (stretched) arm teleported to the final base pose."""
    state = _restore(env, key)
    controller, sim = _planned_pick(env, state)
    try:
        hover = controller.plans[Phase.MOVE_ARM_TO_HOVER_OVER_CUBE]
        assert _near(hover[0], controller.plans[Phase.RETRACT][-1])
    finally:
        sim.close()


def test_the_current_configuration_is_collision_checked(env):
    """The robot's real configuration is checked: free at seed 125, and in collision
    once the cube is put inside the gripper's fingers."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    controller, sim = _planned_pick(env, state)
    try:
        assert controller.current_configuration_collision(state) is None
        sim.set_state(state)
        ee = sim.get_ee_pose().position
        inside = state.copy()
        cube = _objects(state)[1]
        for axis, value in zip(("x", "y", "z"), ee):
            inside.set(cube, axis, value)
        assert controller.current_configuration_collision(inside) is not None
    finally:
        sim.close()


# --- D: arm plans are checked densely, not only at BiRRT's nodes ---------------------


def test_dense_check_finds_a_collision_between_two_clear_waypoints(env):
    """A probe midway along a joint-1 sweep is found, though both ends are clear."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    sim = PyBulletSim(state)
    try:
        # Reaching forward, so a turn of the first joint sweeps the gripper 0.56 m
        # out from its axis.
        start = list(np.deg2rad([0, 40, 180, -90, 0, -40, 90])) + [0.0] * 6
        end = list(start)
        end[0] += 0.8
        sim.robot.set_joints(start)
        a = np.array(sim.get_ee_pose().position)
        sim.robot.set_joints(end)
        b = np.array(sim.get_ee_pose().position)
        middle = list(start)
        middle[0] += 0.4
        sim.robot.set_joints(middle)
        c = sim.get_ee_pose().position
        assert np.linalg.norm(a - b) > 0.1
        box = sim.add_box(
            name="probe",
            pose=Pose(tuple(c)),
            dimensions=(0.02, 0.02, 0.02),
            state_object=False,
        )

        def collides(q) -> bool:
            return check_collisions_with_held_object(
                sim.robot, {box}, sim.physics_client_id, None, None, q
            )

        assert not collides(start) and not collides(end)
        hit = first_dense_joint_path_collision(sim.robot, [start, end], collides)
        assert hit is not None and hit[0] == 0 and 0.3 < hit[1] < 0.7
    finally:
        sim.close()


def test_the_first_snag_pick_now_lifts_the_cube(env):
    """S2 seq 482 timed out in the run, with the cube pressed against the wall."""
    state = _restore(env, "expectimax_s2_seq482")
    cube = _objects(state)[1]
    controller, sim = _planned_pick(env, state)
    try:
        for _ in range(400):
            obs, _, _, _, _ = env.step(controller.step())
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
            if controller.terminated():
                break
    finally:
        sim.close()
    assert controller.terminated()
    assert state.get(cube, "z") > MINIMUM_HOLDING_HEIGHT


def test_the_first_snag_lift_is_collision_free_when_checked_densely(env):
    """S2 seq 482: the executed lift used to pass the held cube 39 mm into the bin wall
    between two nodes (55/1281 points in collision at 40 per radian)."""
    state = _restore(env, "expectimax_s2_seq482")
    controller, sim = _planned_pick(env, state)
    try:
        cube_id = sim._cubes["cube_0"]
        robot = _objects(state)[0]
        after_base = state.copy()
        end = controller.plans[Phase.BASE_MOTION][-1]
        after_base.set(robot, "pos_base_x", end.x)
        after_base.set(robot, "pos_base_y", end.y)
        after_base.set(robot, "pos_base_rot", end.theta())
        sim.set_state(after_base)
        everything = sim.get_collision_bodies()
        checks = {
            Phase.MOVE_ARM_TO_HOVER_OVER_CUBE: (None, 1e-6, everything),
            Phase.MOVE_ARM_DOWN_AROUND_CUBE: (None, 1e-6, everything),
            Phase.LIFT_CUBE_TO_HOME: (
                cube_id,
                controller.BIN_HELD_OBJECT_COLLISION_THRESHOLD,
                sim.get_collision_bodies(held_object=cube_id),
            ),
        }
        for phase, (held, threshold, bodies) in checks.items():
            hits = _dense_arm_collisions(
                sim,
                controller.plans[phase],
                held=held,
                threshold=threshold,
                bodies=bodies,
            )
            assert hits == 0, f"{phase.name}: {hits} dense points in collision"
    finally:
        sim.close()


# --- A, for the toss: retract holding the cube, before the base moves ---------------


def _held_toss_state(*, turn: float) -> ObjectCentricState:
    """A recorded toss start (cube held, arm near home), with the arm's first joint
    turned by `turn` and the cube carried along by the same grasp."""
    rows = json.loads(
        Path(__file__).with_name("held_cube_regression_states.json").read_text("utf-8")
    )
    data = {}
    for name, values in rows[0]["before_state"].items():
        if name == "robot":
            kind = MujocoTidyBotRobotObjectType
        elif name in ("cube_0", "bin_0"):
            kind = MujocoMovableObjectType
        elif name.startswith("collider:"):
            kind = MujocoStaticColliderType
        else:
            kind = MujocoFixtureObjectType
        data[Object(name, kind)] = np.asarray(values, dtype=np.float32)
    state = ObjectCentricState(data, MujocoObjectTypeFeatures)
    robot = state.get_object_from_name("robot")
    cube = state.get_object_from_name("cube_0")
    sim = PyBulletSim(state)
    try:
        sim.set_state(state)
        before = sim.get_ee_pose()
        cube_pose = Pose(
            tuple(state.get(cube, k) for k in ("x", "y", "z")),
            tuple(state.get(cube, k) for k in ("qx", "qy", "qz", "qw")),
        )
        grasp = multiply_poses(before.invert(), cube_pose)
        state.set(robot, "pos_arm_joint1", state.get(robot, "pos_arm_joint1") + turn)
        sim.set_state(state)
        moved = multiply_poses(sim.get_ee_pose(), grasp)
    finally:
        sim.close()
    for key, value in zip(("x", "y", "z"), moved.position):
        state.set(cube, key, value)
    for key, value in zip(("qx", "qy", "qz", "qw"), moved.orientation):
        state.set(cube, key, value)
    return state


def test_the_toss_retracts_holding_the_cube_before_its_base_motion():
    """The toss retracts with the cube still held, and the base waits for it."""
    state = _held_toss_state(turn=0.4)
    robot = state.get_object_from_name("robot")
    controller = MoveToTossLocationAndTossController(
        tuple(
            state.get_object_from_name(n) for n in ("robot", "cube_0", "cuboid_barrier")
        )
    )
    phase = controller.MoveToTossLocationAndTossControllerPhase
    params = np.array([2.1344, -0.0012, np.deg2rad(214.82), 577.2])  # As recorded.
    try:
        controller.reset(state, params)
        retract = controller._retract_plan
        assert _near(retract[0], _arm(state))
        assert _near(retract[-1], _HOME)
        assert np.allclose(controller._windup_start_joint_angles, _HOME)
        assert controller._phase is phase.RETRACT
        controller.observe(state)
        action = controller.step()
        assert controller._phase is phase.RETRACT
        assert np.allclose(action[:3], 0.0)  # The base waits for the retract.
        assert action[10] == pytest.approx(state.get(robot, "pos_gripper"))
    finally:
        if controller._pybullet_sim is not None:
            controller._pybullet_sim.close()
