"""The pick may stand on any of the cube's four sides.

The 2F-85 grips a cube the same way on faces 180 and 270 degrees as on 0 and 90; what
changes is where the base stands -- on the opposite side of the cube. With only the
first two faces, a cube whose near sides are blocked (by the bin, a room wall or the
barrier) cannot be picked at all: in hitl-pmp's t3-final-20261023 expectimax runs, 55
floor states had no plannable pick, and with all four faces 51 of them plan.

The two far sides are a rescue, tried only once every first grasp has failed, in the
real robot's order (`pick_rescue_order_cases.json`).

`t3_final_floor_pick_states.json` holds three of those states: the recorded robot, cube
and bin just before expectimax seed 2 seq 64 and seed 5 seq 16 (a far side plans) and
seed 1 seq 830 (no side plans), each a human reset taken because no pick planned.
"""

# These tests inspect planning internals directly.
# pylint: disable=protected-access

import json
import os
import re
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

_HERE = Path(__file__).parent
_STATES = json.loads((_HERE / "t3_final_floor_pick_states.json").read_text("utf-8"))
_ORDER = json.loads((_HERE / "pick_rescue_order_cases.json").read_text("utf-8"))


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


def _faces(state: ObjectCentricState):
    cube = _objects(state)[1]
    return upright_grasp_rotations(
        tuple(state.get(cube, k) for k in ("qx", "qy", "qz", "qw"))
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


def test_the_first_grasps_stand_only_on_faces_0_and_90(env):
    """The first list, today's (and the real robot's), is unchanged: faces 0 and 90."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    controller, sim = _pick(env, state)
    try:
        faces = _faces(state)
        candidates = controller.grasp_candidates(state)
        assert len(candidates) == 1612
        assert {c.face_quat for c in candidates} == {faces[0], faces[1]}
    finally:
        sim.close()


@pytest.mark.parametrize("case", _ORDER["cases"], ids=str)
def test_the_rescue_follows_the_real_robots_order(env, case):
    """h_k = cube yaw + k * 90 deg; sim's first grasps stand on h_0 and h_1, so the
    rescue tries h_2 and h_3, nearest the robot's heading first, ties to lower k."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    faces = _faces(state)
    cube_yaw = 2.0 * float(np.arctan2(faces[0][2], faces[0][3]))
    robot = _objects(state)[0]
    state.set(
        robot,
        "pos_base_rot",
        cube_yaw + np.deg2rad(case["base_yaw_minus_cube_yaw_deg"]),
    )
    controller, sim = _pick(env, state)
    try:
        sides = controller.rescue_sides(state)
        assert [side.k for side in sides] == case["order"]
        # upright_grasp_rotations orders the faces yaw, +90, -90, +180.
        normal = {2: faces[3], 3: faces[2]}
        assert [side.face_quat for side in sides] == [normal[k] for k in case["order"]]
    finally:
        sim.close()


def test_each_rescue_side_tries_the_first_grasps_of_its_opposite_face(env):
    """At a far stand the grasps are its opposite face's, in the first list's order."""
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    controller, sim = _pick(env, state)
    try:
        faces = _faces(state)
        first = controller.grasp_candidates(state)
        opposite = {2: faces[0], 3: faces[1]}
        for side in controller.rescue_sides(state):
            near = [c for c in first if c.face_quat == opposite[side.k]]
            assert len(side.candidates) == len(near) == 806
            assert all(c.face_quat == side.face_quat for c in side.candidates)
            assert [(c.yaw_offset, c.tilt, c.opening) for c in side.candidates] == [
                (c.yaw_offset, c.tilt, c.opening) for c in near
            ]
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
        faces = _faces(state)
        assert controller.grasp_candidate.face_quat in (faces[2], faces[3])
    finally:
        sim.close()


def test_a_refusal_lists_every_side_tried(env):
    """As the real robot's does: today's reason, then each rescue side and why."""
    state = _restore(env, "expectimax_s1_seq830")
    controller, sim = _pick(env, state)
    try:
        with pytest.raises(ValueError) as refusal:
            controller.reset(state, None)
    finally:
        sim.close()
    message = str(refusal.value)
    head, _, tail = message.partition("; other sides tried: ")
    assert head == "No collision-free cube grasp"
    sides = tail.split("; ")
    assert len(sides) == 2
    for side in sides:
        assert re.fullmatch(
            r"-?\d+ deg: (base motion refused|no collision-free grasp)",
            side,
        ), side


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
