"""The pick may stand on any of the cube's four sides.

The 2F-85 grips a cube the same way on faces 180 and 270 degrees as on 0 and 90; what
changes is where the base stands -- on the opposite side of the cube. With only the
first two faces, a cube whose near sides are blocked (by the bin, a room wall or the
barrier) cannot be picked at all: in hitl-pmp's t3-final-20261023 expectimax runs, 55
floor states had no plannable pick, and with all four faces 51 of them plan.

`t3_final_floor_pick_states.json` holds two of them: the recorded robot, cube and bin
just before expectimax seed 2 seq 64 and seed 5 seq 16, each a human reset taken because
no pick planned.
"""

# These tests inspect planning internals directly.
# pylint: disable=protected-access

import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# pylint: disable=wrong-import-position
import kinder
import pytest
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from pybullet_helpers.geometry import Pose
from relational_structs import ObjectCentricState

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.utils import (
    MINIMUM_HOLDING_HEIGHT,
    PyBulletSim,
    upright_grasp_rotations,
)

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

_STATES = json.loads(
    Path(__file__).with_name("t3_final_floor_pick_states.json").read_text("utf-8")
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


def _objects(state: ObjectCentricState):
    return (
        state.get_objects(MujocoTidyBotRobotObjectType)[0],
        state.get_object_from_name("cube_0"),
        state.get_object_from_name("cuboid_barrier"),
    )


def _pick(env, state: ObjectCentricState):
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
    controller = create_lifted_controllers(env.action_space, state, pybullet_sim=sim)[
        "pick_cube"
    ].ground(_objects(state))
    return controller, sim


def test_the_two_far_sides_follow_every_near_side_grasp_in_the_same_order(env):
    """Today's list (the real robot's, faces 0 and 90) comes first and unchanged, so any
    pick that planned before plans the same grasp; then the same list again on faces 180
    and 270."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    controller, sim = _pick(env, state)
    try:
        cube = _objects(state)[1]
        faces = upright_grasp_rotations(
            tuple(state.get(cube, k) for k in ("qx", "qy", "qz", "qw"))
        )
        candidates = controller.grasp_candidates(state)
        half = len(candidates) // 2
        assert len(candidates) == 2 * 1612
        near, far = candidates[:half], candidates[half:]
        assert {c.face_quat for c in near} == {faces[0], faces[1]}
        assert {c.face_quat for c in far} == {faces[2], faces[3]}
        # upright_grasp_rotations orders the faces yaw, +90, -90, +180.
        swap = {faces[0]: faces[3], faces[1]: faces[2]}
        for a, b in zip(near, far):
            assert (b.yaw_offset, b.tilt, b.opening) == (
                a.yaw_offset,
                a.tilt,
                a.opening,
            )
            assert b.face_quat == swap[a.face_quat]
    finally:
        sim.close()


@pytest.mark.parametrize("key", ["expectimax_s2_seq64", "expectimax_s5_seq16"])
def test_a_cube_reachable_only_from_its_far_sides_plans(env, key):
    """Both recorded states were human resets because no pick planned; now one plans,
    from a far side."""
    state = _restore(env, key)
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
        cube = _objects(state)[1]
        faces = upright_grasp_rotations(
            tuple(state.get(cube, k) for k in ("qx", "qy", "qz", "qw"))
        )
        assert controller.grasp_candidate.face_quat in (faces[2], faces[3])
    finally:
        sim.close()


def test_a_far_side_pick_holds_the_cube(env):
    """Executed in MuJoCo from the recorded s2 seq 64 state."""
    state = _restore(env, "expectimax_s2_seq64")
    cube = _objects(state)[1]
    controller, sim = _pick(env, state)
    try:
        controller.reset(state, None)
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
