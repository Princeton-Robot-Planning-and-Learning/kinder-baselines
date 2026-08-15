"""Tests for tossing state_abstractions.py."""

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import GroundAtom, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.shelf import parameterized_skills as shelf_skills
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    HandEmpty,
    Holding,
    MovableInGoalRegion,
    MovableIsDownX,
    OnGround,
    Tossing3DStateAbstractor,
)


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def _make_env_and_abstractor(record_name: str | None = None):
    """A reset Tossing3D-o1 env, its abstractor, and the initial state."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array")
    if MAKE_VIDEOS and record_name is not None:
        env.unwrapped._object_centric_env.set_render_camera("task_view")  # type: ignore # pylint: disable=protected-access
        env = RecordVideo(env, "unit_test_videos", name_prefix=record_name)
    sim = env.unwrapped._object_centric_env  # type: ignore # pylint: disable=protected-access
    abstractor = Tossing3DStateAbstractor(sim)
    obs, _ = env.reset(seed=125)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    return env, abstractor, state


def test_hand_empty_holds_when_the_gripper_is_open():
    """The gripper starts commanded open, so the hand reads empty."""
    env, abstractor, state = _make_env_and_abstractor()
    robot = _get_robot_from_state(state)
    atoms = abstractor.state_abstractor(state).atoms
    assert GroundAtom(HandEmpty, [robot]) in atoms
    env.close()


def test_on_ground_holds_for_a_cube_resting_flat():
    """A settled cube is both at floor height and unrotated about x and y."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    atoms = abstractor.state_abstractor(state).atoms
    assert GroundAtom(OnGround, [cube]) in atoms
    env.close()


def test_on_ground_fails_for_a_tilted_cube_at_the_same_height():
    """Tilt alone breaks it, which is also what keeps the bb arithmetic valid."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    tilted = state.copy()
    tilted.set(cube, "qx", 0.5)
    atoms = abstractor.state_abstractor(tilted).atoms
    assert GroundAtom(OnGround, [cube]) not in atoms
    env.close()


def test_movable_is_down_x_holds_short_of_the_barrier():
    """The cube starts at lower x than cuboid_barrier."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    atoms = abstractor.state_abstractor(state).atoms
    assert GroundAtom(MovableIsDownX, [cube, barrier]) in atoms
    env.close()


def test_movable_is_down_x_fails_past_the_barrier():
    """Losing this atom is what makes a toss irreversible rather than retryable."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    thrown = state.copy()
    thrown.set(cube, "x", state.get(barrier, "x") + 0.5)
    atoms = abstractor.state_abstractor(thrown).atoms
    assert GroundAtom(MovableIsDownX, [cube, barrier]) not in atoms
    env.close()


def test_holding_holds_for_a_lifted_cube_at_the_end_effector():
    """Needs all three: gripper commanded closed, cube lifted, cube at the ee."""
    env, abstractor, state = _make_env_and_abstractor()
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube_0")
    # Sync the sim first, so the ee pose is the one the abstractor will read.
    abstractor.state_abstractor(state)
    ee = abstractor._pybullet_sim.get_ee_pose()  # pylint: disable=protected-access
    grasped = state.copy()
    grasped.set(robot, "pos_gripper", 1.0)
    grasped.set(cube, "x", ee.position[0])
    grasped.set(cube, "y", ee.position[1])
    grasped.set(cube, "z", ee.position[2])
    atoms = abstractor.state_abstractor(grasped).atoms
    assert GroundAtom(Holding, [robot, cube]) in atoms
    env.close()


def test_movable_in_goal_region_is_absent_before_any_throw():
    """The abstraction must agree with the environment's own success check."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    atoms = abstractor.state_abstractor(state).atoms
    assert GroundAtom(MovableInGoalRegion, [cube]) not in atoms
    assert not abstractor._sim._check_goals()  # pylint: disable=protected-access
    env.close()


def test_movable_in_goal_region_uses_the_inflated_region():
    """x = 2.14 is outside the task JSON's literal 2.10 but inside the inflated 2.15.

    That gap is the point: the environment inflates "ranges" by the ground placement
    threshold, and the predicate has to inflate with it.
    """
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    landed = state.copy()
    landed.set(cube, "x", 2.14)
    landed.set(cube, "y", 0.0)
    landed.set(cube, "z", 0.025)
    atoms = abstractor.state_abstractor(landed).atoms
    assert GroundAtom(MovableInGoalRegion, [cube]) in atoms
    env.close()


def test_the_goal_is_every_cube_in_the_goal_region():
    """And the initial state does not satisfy it."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    goal = abstractor.goal_deriver(state)
    assert goal.atoms == {GroundAtom(MovableInGoalRegion, [cube])}
    assert not goal.check_state(state)
    env.close()


def test_the_abstractor_rejects_the_two_cube_variant():
    """O2 is out of scope: no operator says which cube a throw is aimed at."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o2-v0", render_mode="rgb_array")
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    with pytest.raises(AssertionError, match="only Tossing3D-o1 is supported"):
        Tossing3DStateAbstractor(sim)
    env.close()


def _rest(axis, deg):
    """A quaternion (qx, qy, qz, qw) rotating `deg` about `axis`."""
    half = np.deg2rad(deg) / 2
    vec = np.array(axis, dtype=float)
    return tuple(float(v) for v in np.sin(half) * vec) + (float(np.cos(half)),)


def test_on_ground_holds_for_a_cube_resting_on_any_of_its_faces():
    """A cube on its side is the same cube on the same floor. The grasp no longer cares
    which face is up, so neither should the predicate."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    for axis, deg in [
        ([0, 0, 1], 0),
        ([0, 0, 1], 90),
        ([1, 0, 0], 90),
        ([1, 0, 0], 180),
        ([1, 0, 0], -90),
        ([0, 1, 0], 90),
        ([0, 1, 0], -90),
        ([0, 1, 0], 180),
    ]:
        rested = state.copy()
        for feature, value in zip(("qx", "qy", "qz", "qw"), _rest(axis, deg)):
            rested.set(cube, feature, value)
        atoms = abstractor.state_abstractor(rested).atoms
        assert GroundAtom(OnGround, [cube]) in atoms, (axis, deg)
    env.close()


def test_on_ground_still_fails_for_a_cube_balanced_between_faces():
    """Canonicalising is not the same as ignoring orientation: a cube on an edge rests
    on no face at all, and the bounding-box arithmetic stops meaning anything."""
    env, abstractor, state = _make_env_and_abstractor()
    cube = state.get_object_from_name("cube_0")
    balanced = state.copy()
    for feature, value in zip(("qx", "qy", "qz", "qw"), _rest([1, 0, 0], 45)):
        balanced.set(cube, feature, value)
    atoms = abstractor.state_abstractor(balanced).atoms
    assert GroundAtom(OnGround, [cube]) not in atoms
    env.close()
