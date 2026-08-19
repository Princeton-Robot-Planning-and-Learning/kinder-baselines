"""Tests for kinematic3d utils.py."""

import numpy as np
from pybullet_helpers.geometry import SE2Pose

from prpl_utils.utils import wrap_angle

from kinder_models.kinematic3d.utils import (
    get_target_robot_pose_from_parameters,
    step_toward_se2_waypoint,
)

_MAX_ACTION_MAG = 0.1


def test_waypoint_within_bound_is_popped():
    """A reachable waypoint yields the exact delta and is popped."""
    current = SE2Pose(0.0, 0.0, 0.0)
    plan = [SE2Pose(0.05, -0.03, 0.02)]

    delta, exhausted = step_toward_se2_waypoint(current, plan, _MAX_ACTION_MAG)

    assert np.allclose(delta, [0.05, -0.03, 0.02])
    assert not plan
    assert exhausted


def test_waypoint_beyond_bound_is_clamped_and_kept():
    """An unreachable waypoint yields a clamped delta and stays in the plan;
    executing each delta exactly converges in the expected number of steps."""
    current = SE2Pose(0.0, 0.0, 0.0)
    waypoint = SE2Pose(0.35, 0.1, 0.0)
    plan = [waypoint]

    delta, exhausted = step_toward_se2_waypoint(current, plan, _MAX_ACTION_MAG)

    assert np.allclose(delta, [0.1, 0.1, 0.0])
    assert plan == [waypoint]
    assert not exhausted

    # Closed loop: x needs ceil(0.35 / 0.1) = 4 steps; the final short step
    # returns the exact remaining delta and pops the waypoint.
    for step in range(1, 5):
        delta, exhausted = step_toward_se2_waypoint(current, plan, _MAX_ACTION_MAG)
        current = current + SE2Pose(*delta)
        assert exhausted == (step == 4)

    assert np.allclose([current.x, current.y, current.rot], [0.35, 0.1, 0.0])
    assert not plan


def test_multi_waypoint_plan_consumed_one_at_a_time():
    """Waypoints are consumed in order and the plan is exhausted at the end."""
    first = SE2Pose(0.05, 0.0, 0.0)
    second = SE2Pose(0.1, 0.05, 0.01)
    plan = [first, second]

    current = SE2Pose(0.0, 0.0, 0.0)
    delta, exhausted = step_toward_se2_waypoint(current, plan, _MAX_ACTION_MAG)
    assert np.allclose(delta, [0.05, 0.0, 0.0])
    assert plan == [second]
    assert not exhausted

    current = current + SE2Pose(*delta)
    delta, exhausted = step_toward_se2_waypoint(current, plan, _MAX_ACTION_MAG)
    assert np.allclose(delta, [0.05, 0.05, 0.01])
    assert not plan
    assert exhausted


def test_short_step_is_retried_not_skipped():
    """If the pose does not advance, the same waypoint stays at plan[0] and the
    same clamped delta is returned again."""
    current = SE2Pose(0.0, 0.0, 0.0)
    waypoint = SE2Pose(0.5, 0.0, 0.0)
    plan = [waypoint]

    first_delta, _ = step_toward_se2_waypoint(current, plan, _MAX_ACTION_MAG)
    second_delta, exhausted = step_toward_se2_waypoint(current, plan, _MAX_ACTION_MAG)

    assert np.allclose(first_delta, second_delta)
    assert plan == [waypoint]
    assert not exhausted


def test_get_target_robot_pose_from_parameters_places_and_faces_target():
    """The robot is target_distance away from the target and faces it."""
    target = SE2Pose(1.0, -2.0, 0.7)
    for distance, rot in [(0.5, 0.0), (0.55, np.pi / 4), (0.6, -np.pi / 4)]:
        robot = get_target_robot_pose_from_parameters(target, distance, rot)
        dx, dy = target.x - robot.x, target.y - robot.y
        assert np.isclose(np.hypot(dx, dy), distance)
        # Heading points at the target.
        assert np.isclose(wrap_angle(np.arctan2(dy, dx) - robot.rot), 0.0, atol=1e-9)
        # The approach direction is offset from the target's rotation by rot.
        assert np.isclose(wrap_angle(robot.rot - (target.rot + rot)), 0.0, atol=1e-9)
