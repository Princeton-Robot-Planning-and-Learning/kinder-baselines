"""Tests for toss_swing.py."""

import numpy as np

from kinder_models.dynamic3d.tossing.toss_profile import TOSS_MAX_VELOCITY
from kinder_models.dynamic3d.tossing.toss_swing import (
    TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
    TOSS_RELEASE_ARM_CONFIGURATION,
    TOSS_SLICES_PER_CONTROL_STEP,
    TOSS_WINDUP_ARM_CONFIGURATION,
    plan_toss_swing,
    toss_swing_action,
)


def _straight_swing(release_ms=TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS):
    """A planned swing between the two demonstrated confs, without a simulator."""
    start = list(TOSS_WINDUP_ARM_CONFIGURATION) + [0.0] * 6
    end = list(TOSS_RELEASE_ARM_CONFIGURATION) + [0.0] * 6
    return plan_toss_swing([end], start, TOSS_MAX_VELOCITY, release_ms)


def test_plan_toss_swing_splits_the_release_millisecond_into_step_and_slice():
    """The millisecond names a slice inside a control step, not a step boundary."""
    swing = _straight_swing(release_ms=725)
    step, slice_ = divmod(725, TOSS_SLICES_PER_CONTROL_STEP)
    assert swing.release_step == step
    assert swing.release_slice == slice_


def test_plan_toss_swing_direction_is_a_unit_vector_along_the_swing():
    """The profile carries the distance; the direction carries only the heading."""
    swing = _straight_swing()
    assert np.isclose(np.linalg.norm(swing.direction), 1.0)


def test_plan_toss_swing_direction_is_zero_for_a_swing_that_does_not_move():
    """Otherwise the unit vector is a division by zero."""
    conf = list(TOSS_WINDUP_ARM_CONFIGURATION) + [0.0] * 6
    swing = plan_toss_swing(
        [conf], conf, TOSS_MAX_VELOCITY, TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS
    )
    assert np.allclose(swing.direction, 0.0)


def test_toss_swing_action_holds_the_gripper_shut_before_the_release_step():
    swing = _straight_swing()
    action = toss_swing_action(swing, 0, [0.0] * 13, 1.0, False)
    assert action.shape == (18,)
    assert action[10] == 1.0


def test_toss_swing_action_opens_the_gripper_after_the_release_step():
    swing = _straight_swing()
    action = toss_swing_action(swing, swing.release_step + 1, [0.0] * 13, 1.0, True)
    assert action.shape == (18,)
    assert action[10] == 0.0


def test_toss_swing_action_emits_a_schedule_on_the_step_the_release_falls_inside():
    """A (slices, 18) schedule, so the millisecond means the millisecond."""
    swing = _straight_swing(release_ms=725)
    assert swing.release_slice != 0
    schedule = toss_swing_action(swing, swing.release_step, [0.0] * 13, 1.0, False)
    assert schedule.shape == (TOSS_SLICES_PER_CONTROL_STEP, 18)
    assert np.all(schedule[: swing.release_slice, 10] == 1.0)
    assert np.all(schedule[swing.release_slice :, 10] == 0.0)


def test_toss_swing_action_stays_flat_when_the_release_lands_on_a_step_boundary():
    """No schedule is needed when no millisecond inside the step is special."""
    swing = _straight_swing(release_ms=TOSS_SLICES_PER_CONTROL_STEP * 7)
    assert swing.release_slice == 0
    action = toss_swing_action(swing, swing.release_step, [0.0] * 13, 1.0, False)
    assert action.shape == (18,)
    assert action[10] == 0.0


def test_gripper_release_ms_splits_into_a_control_step_and_a_slice():
    """The parameter is absolute wall-clock milliseconds from the start of the swing.

    reset() only decomposes it; nothing rounds it to a control-step boundary.
    """
    assert TOSS_SLICES_PER_CONTROL_STEP == 100
    for ms, expected in [(0, (0, 0)), (723, (7, 23)), (100, (1, 0)), (2399, (23, 99))]:
        assert divmod(ms, TOSS_SLICES_PER_CONTROL_STEP) == expected
