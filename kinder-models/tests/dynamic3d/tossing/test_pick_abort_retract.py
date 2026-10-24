"""A Tossing3D pick that touches something it should not, or runs out of steps, backs
out the way it came; one that starts in contact escapes before it plans.

`t3_relaunch_barrier_stall_states.json` holds the t3-relaunch-20261023 EES seed-2 pick
that descended onto the barrier and pushed against it for the rest of its 400 steps,
before and after. Every later pick refused ("Robot starts in collision"), and the run
ended on repeated human resets that moved the cube but never the arm.
"""

# These tests inspect phase state directly.
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
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from pybullet_helpers.geometry import Pose
from relational_structs import ObjectCentricState

from kinder_models.dynamic3d.tossing import parameterized_skills
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    PickCubeController,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.utils import (
    MINIMUM_HOLDING_HEIGHT,
    PyBulletSim,
    wrap_arm_joint_difference,
)

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

_STATES = json.loads(
    Path(__file__).with_name("t3_relaunch_barrier_stall_states.json").read_text("utf-8")
)
_HOME = np.deg2rad([0, -20, 180, -146, 0, -50, 90])
_ARM = tuple(f"pos_arm_joint{i}" for i in range(1, 8))
# The caller's budget in Tossing3D practice, and what the retrace may add to it.
_STEP_LIMIT = 400
_RETRACE_LIMIT = 400
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


def _pick(env, state: ObjectCentricState) -> tuple[PickCubeController, PyBulletSim]:
    sim = _bin_aware_sim(env, state)
    controller = create_lifted_controllers(env.action_space, state, pybullet_sim=sim)[
        "pick_cube"
    ].ground(_objects(state))
    assert isinstance(controller, PickCubeController)
    return controller, sim


def _drive(env, controller, state, limit: int) -> tuple[ObjectCentricState, int, list]:
    """Step as Tossing3D practice does: until terminated() or `limit` steps."""
    actions = []
    for step in range(limit):
        action = controller.step()
        actions.append(np.asarray(action).copy())
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
        if controller.terminated():
            return state, step + 1, actions
    return state, limit, actions


def _contact(env, state: ObjectCentricState) -> str | None:
    """What a fresh pick finds the robot touching in `state`."""
    controller, sim = _pick(env, state)
    try:
        return controller.current_configuration_collision(state)
    finally:
        sim.close()


def _arm(state: ObjectCentricState) -> np.ndarray:
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    return np.array([state.get(robot, k) for k in _ARM])


def test_the_recorded_stall_ends_against_the_barrier(env):
    """The fixture is what it says: the stalled pick left the arm on the barrier."""
    state = _restore(env, "ees_s2_seq68_stall_end")
    assert "cuboid_barrier" in (_contact(env, state) or "")
    assert _contact(env, _restore(env, "ees_s2_seq68_pick_start")) is None


def test_the_barrier_stall_aborts_at_first_contact_and_backs_out(env):
    """The pick stops at the first tick its arm touches the barrier, retraces its own
    path out, and reports a failure; the next pick is not refused."""
    state = _restore(env, "ees_s2_seq68_pick_start")
    cube = _objects(state)[1]
    cube_x = state.get(cube, "x")
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        state, steps, _ = _drive(env, controller, state, _STEP_LIMIT)
        assert controller.terminated()
        assert controller.failure is not None
        assert "cuboid_barrier" in controller.failure
        assert steps < _STEP_LIMIT
    finally:
        sim.close()
    assert _contact(env, state) is None
    assert np.max(np.abs(wrap_arm_joint_difference(_arm(state) - _HOME))) < 0.05
    # Not pushed toward, let alone over, the barrier.
    assert state.get(cube, "x") <= cube_x + 0.005
    following, sim = _pick(env, state)
    try:
        following.reset(state, None)  # would raise "Robot starts in collision"
    finally:
        sim.close()


def test_without_the_contact_abort_the_stall_retraces_out_at_the_step_limit(
    env, monkeypatch
):
    """With only retract-on-timeout, the stall runs to the limit as before, and the
    caller's on_step_limit() then backs the arm out along the path it came by."""
    monkeypatch.setattr(parameterized_skills, "PICK_CUBE_ABORT_ON_CONTACT", False)
    state = _restore(env, "ees_s2_seq68_pick_start")
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        state, steps, _ = _drive(env, controller, state, _STEP_LIMIT)
        assert steps == _STEP_LIMIT and not controller.terminated()
        assert "cuboid_barrier" in (_contact(env, state) or "")
        controller.on_step_limit()
        state, _, _ = _drive(env, controller, state, _RETRACE_LIMIT)
        assert controller.terminated()
        assert controller.failure is not None and "step limit" in controller.failure
    finally:
        sim.close()
    assert _contact(env, state) is None
    assert np.max(np.abs(wrap_arm_joint_difference(_arm(state) - _HOME))) < 0.05


def test_a_step_limit_mid_lift_keeps_a_cube_the_closed_hand_has_raised(env):
    """Out of steps mid-lift, a closed hand with the cube raised is left alone.

    Retracing would lower the cube and reopen the hand on it. Of the 26 relaunch picks
    that used all 400 steps, 2/26 were lifts holding a raised cube at the limit, and
    retracing every lift lost both. With the cube still down, the lift retraces like
    any other phase: 3/26 left a closed hand in the cube otherwise.
    """
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    cube = _objects(state)[1]
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        raised = state.copy()
        raised.set(cube, "z", MINIMUM_HOLDING_HEIGHT + 0.05)
        controller.observe(raised)
        controller.current_phase = Phase.LIFT_CUBE_TO_HOME
        controller._closed_gripper = True
        controller.on_step_limit()
        assert controller.failure is None
        assert controller.current_phase == Phase.LIFT_CUBE_TO_HOME
        controller.observe(state)
        controller.current_phase = Phase.LIFT_CUBE_TO_HOME
        controller.on_step_limit()
        assert controller.failure is not None
        assert controller.current_phase == Phase.ABORT_RETRACE
    finally:
        sim.close()


def test_with_both_off_the_step_limit_leaves_the_arm_on_the_barrier(env, monkeypatch):
    """The behaviour before: the switches are what changes it."""
    monkeypatch.setattr(parameterized_skills, "PICK_CUBE_ABORT_ON_CONTACT", False)
    monkeypatch.setattr(parameterized_skills, "PICK_CUBE_RETRACT_ON_TIMEOUT", False)
    state = _restore(env, "ees_s2_seq68_pick_start")
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        state, _, _ = _drive(env, controller, state, _STEP_LIMIT)
        controller.on_step_limit()
        assert controller.failure is None
        state, _, _ = _drive(env, controller, state, 5)
    finally:
        sim.close()
    assert "cuboid_barrier" in (_contact(env, state) or "")


def test_a_pick_that_starts_on_the_barrier_escapes_first(env):
    """From the stalled state the pick no longer refuses: it plans an escape whose
    clearance never decreases, then the normal retract; executed, the arm ends clear."""
    state = _restore(env, "ees_s2_seq68_stall_end")
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        assert controller.escape_waypoints > 0
        state, steps, _ = _drive(env, controller, state, _STEP_LIMIT)
        if not controller.terminated():
            controller.on_step_limit()
            state, _, _ = _drive(env, controller, state, _RETRACE_LIMIT)
        assert controller.terminated(), f"{steps=} {controller.current_phase}"
    finally:
        sim.close()
    assert _contact(env, state) is None


def test_with_the_escape_off_a_start_in_contact_is_still_refused(env, monkeypatch):
    """No escaping motion, or the escape switched off, keeps the loud refusal."""
    monkeypatch.setattr(parameterized_skills, "PICK_CUBE_ESCAPE_RETRACT", False)
    state = _restore(env, "ees_s2_seq68_stall_end")
    controller, sim = _pick(env, state)
    try:
        with pytest.raises(ValueError, match="Robot starts in collision"):
            controller.reset(state, None)
    finally:
        sim.close()


def test_the_escape_never_lets_clearance_decrease(env):
    """Along the planned escape, the robot's signed distance to the scene only grows
    until it is clear, and the escape ends clear."""
    state = _restore(env, "ees_s2_seq68_stall_end")
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        retract = controller.plans[Phase.RETRACT]
        assert retract is not None
        escape = retract[: controller.escape_waypoints]
        sim.set_state(state)
        distances = [controller.signed_clearance(q) for q in escape]
        assert distances[0] < 0.0
        assert distances[-1] > 0.0
        assert all(b >= a - 1e-9 for a, b in zip(distances, distances[1:]))
    finally:
        sim.close()


def test_the_target_cube_is_an_expected_contact_only_once_the_hand_descends(env):
    """Touching the cube to pick is expected from the descent on, and nowhere before;
    any other body is unexpected in every phase."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        sim.set_state(state)
        ee = sim.get_ee_pose().position
        inside = state.copy()
        cube = _objects(state)[1]
        for axis, value in zip(("x", "y", "z"), ee):
            inside.set(cube, axis, value)
        expected = {
            Phase.OPEN_GRIPPER: "cube_0",
            Phase.RETRACT: "cube_0",
            Phase.BASE_MOTION: "cube_0",
            Phase.MOVE_ARM_TO_HOVER_OVER_CUBE: "cube_0",
            Phase.MOVE_ARM_DOWN_AROUND_CUBE: None,
            Phase.CLOSE_GRIPPER_TO_GRASP_CUBE: None,
            Phase.LIFT_CUBE_TO_HOME: None,
        }
        for phase, contact in expected.items():
            controller.current_phase = phase
            assert controller.unexpected_contact(inside) == contact, phase
        controller.current_phase = Phase.MOVE_ARM_DOWN_AROUND_CUBE
        assert controller.unexpected_contact(state) is None
    finally:
        sim.close()


def test_a_normal_pick_is_unchanged_by_the_abort_switches(env, monkeypatch):
    """Seed 125's pick, with every switch on and with every switch off, commands the
    same actions bit for bit and holds the cube."""
    runs = []
    for enabled in (True, False):
        for name in (
            "PICK_CUBE_ABORT_ON_CONTACT",
            "PICK_CUBE_RETRACT_ON_TIMEOUT",
            "PICK_CUBE_ESCAPE_RETRACT",
        ):
            monkeypatch.setattr(parameterized_skills, name, enabled)
        obs, _ = env.reset(seed=125)
        state = env.observation_space.devectorize(obs)
        controller, sim = _pick(env, state)
        try:
            controller.reset(state, None)
            state, _, actions = _drive(env, controller, state, _STEP_LIMIT)
            assert controller.terminated() and controller.failure is None
        finally:
            sim.close()
        runs.append(np.stack(actions))
    assert runs[0].shape == runs[1].shape
    assert runs[0].tobytes() == runs[1].tobytes()
