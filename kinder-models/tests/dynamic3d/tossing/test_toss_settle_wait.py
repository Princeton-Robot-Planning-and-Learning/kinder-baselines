"""The toss ends only once the thrown cube has come to rest, or a few steps later.

`MoveToTossLocationAndTossController` used to terminate as soon as the arm was home,
while the cube could still be bouncing or rolling: 64/3053 recorded tosses in hitl-pmp's
t3-final-20261023 expectimax runs ended with it moving, and 25 of the 114 human resets
there followed one. Holding still a few steps lets 19/25 of those settle somewhere
pickable.

`t3_final_unsettled_toss_states.json` holds the recorded robot, cube and bin at the last
tick of one such toss: seed 1 seq 84, the cube at 3.5 m/s.
"""

# These tests drive the controller's last phase directly.
# pylint: disable=protected-access

import json
import os
from pathlib import Path
from typing import Any

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# pylint: disable=wrong-import-position
import kinder
import numpy as np
import pytest
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import ObjectCentricState

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    TOSS_SETTLE_MAX_STEPS,
    TOSS_SETTLE_SPEED,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.utils import PyBulletSim

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

_STATES = json.loads(
    Path(__file__).with_name("t3_final_unsettled_toss_states.json").read_text("utf-8")
)


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
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    for name, features in _STATES[key]["state"].items():
        obj = state.get_object_from_name(name)
        for feature, value in features.items():
            state.set(obj, feature, float(value))
    env.unwrapped.set_state(env.observation_space.vectorize(state))
    return env.observation_space.devectorize(env.unwrapped.get_state())


def _speed(state: ObjectCentricState) -> float:
    cube = state.get_object_from_name("cube_0")
    return float(np.linalg.norm([state.get(cube, k) for k in ("vx", "vy", "vz")]))


def _after_the_swing(env, state: ObjectCentricState):
    """The toss controller as it stands once the swing is over: returning home."""
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    objects = (
        robot,
        state.get_object_from_name("cube_0"),
        state.get_object_from_name("cuboid_barrier"),
    )
    sim = PyBulletSim(state)
    controller: Any = create_lifted_controllers(
        env.action_space, state, pybullet_sim=sim
    )["move_to_toss_location_and_toss"].ground(objects)
    phase = controller.MoveToTossLocationAndTossControllerPhase
    controller._current_base_motion_plan = []
    controller._phase = phase.RETURN_HOME
    controller.observe(state)
    return controller, sim


def _run_out(env, controller, state, limit: int = 100):
    steps = 0
    while not controller.terminated() and steps < limit:
        obs, _, _, _, _ = env.step(controller.step())
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
        steps += 1
    return steps, state


def test_the_toss_does_not_end_while_the_cube_is_still_moving(env):
    """S1 seq 84: the arm is home, but the cube is still moving at 3.5 m/s."""
    state = _restore(env, "expectimax_s1_seq84")
    assert _speed(state) > TOSS_SETTLE_SPEED
    controller, sim = _after_the_swing(env, state)
    try:
        assert not controller.terminated()
    finally:
        sim.close()


def test_the_toss_ends_once_the_cube_comes_to_rest(env):
    """S1 seq 84: at rest within the wait, and the toss ends as soon as it is."""
    state = _restore(env, "expectimax_s1_seq84")
    controller, sim = _after_the_swing(env, state)
    try:
        steps, state = _run_out(env, controller, state)
    finally:
        sim.close()
    assert 0 < steps <= TOSS_SETTLE_MAX_STEPS
    assert _speed(state) < TOSS_SETTLE_SPEED


def test_the_wait_is_capped(env, monkeypatch):
    """A cube that never stops (one jittering in a contact) ends the toss at the cap."""
    state = _restore(env, "expectimax_s1_seq84")
    controller, sim = _after_the_swing(env, state)
    monkeypatch.setattr(controller, "_thrown_cube_speed", lambda: 1.0)
    try:
        steps, _ = _run_out(env, controller, state)
    finally:
        sim.close()
    assert steps == TOSS_SETTLE_MAX_STEPS


def test_a_cube_already_at_rest_costs_no_wait(env):
    """The same state with the cube still: the toss ends as soon as the arm is home."""
    state = _restore(env, "expectimax_s1_seq84")
    cube = state.get_object_from_name("cube_0")
    for key in ("vx", "vy", "vz", "wx", "wy", "wz"):
        state.set(cube, key, 0.0)
    controller, sim = _after_the_swing(env, state)
    try:
        assert controller.terminated()
    finally:
        sim.close()
