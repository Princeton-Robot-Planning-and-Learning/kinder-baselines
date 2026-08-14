"""Tests for tossing state_abstractions.py."""

from unittest import mock

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
    THROW_REACH,
    THROW_STANDOFF_BOUNDS,
    HandEmpty,
    Holding,
    MovableInGoalRegion,
    MovableIsDownX,
    OnGround,
    RobotAtThrowPose,
    Tossing3DStateAbstractor,
)

# The standoff the pick-and-toss tests drive to, inside THROW_STANDOFF_BOUNDS.
THROW_STANDOFF = 1.35

# Also inside THROW_STANDOFF_BOUNDS, and far enough back that the throw falls short.
UNREACHING_STANDOFF = 1.60


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


def test_robot_at_throw_pose_holds_after_driving_to_a_throw_standoff():
    """Checked against the controller that establishes it, not a hand-set state."""
    env, abstractor, state = _make_env_and_abstractor("Tossing3D-state-abstraction")
    robot = _get_robot_from_state(state)
    target_bin = state.get_object_from_name("bin_0")
    controller = create_lifted_controllers(env.action_space)["move_to_target"].ground(
        (robot, target_bin)
    )
    # cube_0 rests at x = 0.71 and the base is headed for x = 0.65, so it is excluded
    # the way MoveToThrowPoseController excludes the object it holds.
    controller.reset(
        state,
        np.array([THROW_STANDOFF, 0.0]),
        disable_collision_objects=["cube_0"],
    )
    for _ in range(300):
        obs, _, _, _, _ = env.step(controller.step())
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    atoms = abstractor.state_abstractor(state).atoms
    assert GroundAtom(RobotAtThrowPose, [robot, target_bin]) in atoms
    env.close()


def test_robot_at_throw_pose_fails_at_a_standoff_the_sampler_can_draw():
    """The add effect has to be able to fail, or its sampler has nothing to learn.

    1.60 m sits inside THROW_STANDOFF_BOUNDS, so a base sampler drawing from that
    range can produce it, and the predicate accepted it until the band was derived
    from the scored region instead. From there the throw lands ~0.175 m short of the
    region, which is why accepting it made MoveToThrowPose's only add effect
    constant-true.
    """
    env, abstractor, state = _make_env_and_abstractor()
    robot = _get_robot_from_state(state)
    target_bin = state.get_object_from_name("bin_0")
    assert THROW_STANDOFF_BOUNDS[0] <= UNREACHING_STANDOFF <= THROW_STANDOFF_BOUNDS[1]
    controller = create_lifted_controllers(env.action_space)["move_to_target"].ground(
        (robot, target_bin)
    )
    controller.reset(
        state,
        np.array([UNREACHING_STANDOFF, 0.0]),
        disable_collision_objects=["cube_0"],
    )
    for _ in range(300):
        obs, _, _, _, _ = env.step(controller.step())
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    atoms = abstractor.state_abstractor(state).atoms
    assert GroundAtom(RobotAtThrowPose, [robot, target_bin]) not in atoms
    env.close()


def test_the_accepted_band_follows_the_goal_region_rather_than_a_literal():
    """Move the scored box and the band moves with it, since it is derived from it.

    The band is not written down anywhere: it falls out of the region's own edges and
    THROW_REACH. Shifting the region 1 m up-x must therefore shift the accepted
    standoffs by the same metre, which a hard-coded interval would not do.
    """
    env, abstractor, state = _make_env_and_abstractor()
    robot = _get_robot_from_state(state)
    target_bin = state.get_object_from_name("bin_0")
    bbox = abstractor._goal_region_bbox()  # pylint: disable=protected-access
    x_min, y_min, z_min, x_max, y_max, z_max = bbox

    def accepts(base_x: float) -> bool:
        moved = state.copy()
        moved.set(robot, "pos_base_x", base_x)
        moved.set(robot, "pos_base_y", state.get(target_bin, "y"))
        moved.set(robot, "pos_base_rot", 0.0)
        return GroundAtom(RobotAtThrowPose, [robot, target_bin]) in (
            abstractor.state_abstractor(moved).atoms
        )

    on_axis_hit = (x_min + x_max) / 2 - THROW_REACH
    assert accepts(on_axis_hit)
    assert not accepts(on_axis_hit + 1.0)

    shifted = (x_min + 1.0, y_min, z_min, x_max + 1.0, y_max, z_max)
    with mock.patch.object(abstractor, "_goal_region_bbox", return_value=shifted):
        assert accepts(on_axis_hit + 1.0)
        assert not accepts(on_axis_hit)
    env.close()


@pytest.mark.parametrize(
    ("standoff", "expected"), [(THROW_STANDOFF, True), (UNREACHING_STANDOFF, False)]
)
def test_robot_at_throw_pose_agrees_with_whether_the_throw_scores(standoff, expected):
    """The calibration guard for THROW_REACH, against real episode outcomes.

    Everything else in the band is read from live geometry, so this one constant is
    the only thing that can go stale -- and it goes stale silently, since a wrong value
    still yields a plausible-looking band. Changing the toss controller, the windup
    configuration, the cube's mass or the physics step is what fails here.

    Asserting the predicate against _check_goals() rather than against a recorded
    number is what makes it a check that the symbolic layer still describes the
    dynamics: both standoffs are ones a base sampler may draw, and the predicate has to
    say the same thing the throw does at each.
    """
    env, abstractor, state = _make_env_and_abstractor()
    robot = _get_robot_from_state(state)
    cube = state.get_object_from_name("cube_0")
    target_bin = state.get_object_from_name("bin_0")
    controllers = create_lifted_controllers(env.action_space)

    def run(controller, steps: int) -> None:
        nonlocal state
        for _ in range(steps):
            obs, _, _, _, _ = env.step(controller.step())
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
            if controller.terminated():
                return
        assert False, "Controller did not terminate"

    picker = shelf_skills.create_lifted_controllers(env.action_space)[
        "pick_shelf"
    ].ground((robot, cube))
    picker.reset(state, picker.sample_parameters(state, np.random.default_rng(123)))
    run(picker, 400)

    mover = controllers["move_to_target"].ground((robot, target_bin))
    mover.reset(state, np.array([standoff, 0.0]), disable_collision_objects=["cube_0"])
    run(mover, 300)

    at_throw_pose = GroundAtom(RobotAtThrowPose, [robot, target_bin]) in (
        abstractor.state_abstractor(state).atoms
    )

    windup = controllers["move_arm_to_conf"].ground((robot,))
    windup.reset(state, np.deg2rad([0, 50, 180, -110, 0, -100, 90]))
    run(windup, 200)
    tosser = controllers["toss"].ground((robot,))
    tosser.reset(state, np.deg2rad([0, 20, 180, -35, 0, 25, 90]))
    run(tosser, 200)

    scored = bool(abstractor._sim._check_goals())  # pylint: disable=protected-access
    assert at_throw_pose == expected
    assert scored == expected, (
        f"at standoff {standoff} the throw scored {scored} but the predicate said "
        f"{at_throw_pose}; THROW_REACH = {THROW_REACH} is no longer calibrated"
    )
    env.close()


def test_the_abstractor_rejects_the_two_cube_variant():
    """o2 is out of scope: no operator says which cube a throw is aimed at."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o2-v0", render_mode="rgb_array")
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    with pytest.raises(AssertionError, match="only Tossing3D-o1 is supported"):
        Tossing3DStateAbstractor(sim)
    env.close()
