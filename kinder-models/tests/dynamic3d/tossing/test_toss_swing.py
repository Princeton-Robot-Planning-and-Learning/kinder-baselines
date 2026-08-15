"""Tests for toss_swing.py."""

import numpy as np

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
)
from kinder_models.dynamic3d.tossing.toss_swing import (
    TOSS_DEFAULT_GRIPPER_RELEASE_MILLISECONDS,
    TOSS_MAX_ACCELERATION,
    TOSS_MAX_DECELERATION,
    TOSS_MAX_VELOCITY,
    TOSS_RELEASE_ARM_CONFIGURATION,
    TOSS_SLICES_PER_CONTROL_STEP,
    TOSS_WINDUP_ARM_CONFIGURATION,
    plan_toss_swing,
    toss_profile_limits,
    toss_swing_action,
)
from kinder_models.dynamic3d.utils import _CONTROL_TIMESTEP, _trapezoidal_motion_profile


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
    """Before the release step, the gripper command still matches the caller's own."""
    swing = _straight_swing()
    action = toss_swing_action(swing, 0, [0.0] * 13, 1.0, False)
    assert action.shape == (18,)
    assert action[10] == 1.0


def test_toss_swing_action_opens_the_gripper_after_the_release_step():
    """Once has_released is set, the gripper stays open regardless of step_idx."""
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


def test_toss_release_speed_default_rebuilds_the_unscaled_profile():
    """The default must rebuild the profile the literal 140/300/200 deg/s limits give.

    Asserts equality of the sampled trajectory rather than of the three limits, so a
    refactor of how the limits reach the profile still has to keep the motion.
    """
    total_dist = float(
        np.linalg.norm(TOSS_RELEASE_ARM_CONFIGURATION - TOSS_WINDUP_ARM_CONFIGURATION)
    )
    expected = _trapezoidal_motion_profile(
        total_dist,
        max_vel=np.deg2rad(140),
        max_accel=np.deg2rad(300),
        max_decel=np.deg2rad(200),
        step_size=_CONTROL_TIMESTEP,
    )
    max_vel, max_accel, max_decel = toss_profile_limits()
    actual = _trapezoidal_motion_profile(
        total_dist,
        max_vel=max_vel,
        max_accel=max_accel,
        max_decel=max_decel,
        step_size=_CONTROL_TIMESTEP,
    )
    assert np.array_equal(actual, expected)


def test_toss_release_speed_scales_every_limit_by_the_same_factor():
    """A release speed is an effort scale on the whole profile, not on max_vel alone.

    Scaling max_vel alone moves the release out of the cruise phase, where the
    acceleration limits set the speed instead, so it stops tracking max_vel.
    """
    # The invariant is the profile's shape. Asserted a few ULP wide rather than bitwise:
    # recovering a ratio divides a factor back out, and a/(b*c)*c need not round back.
    shape = np.array([TOSS_MAX_ACCELERATION, TOSS_MAX_DECELERATION]) / TOSS_MAX_VELOCITY
    for scale in (0.25, 0.5, 1.0, 1.7, 2.0, 3.0):
        max_vel, max_accel, max_decel = toss_profile_limits(scale * TOSS_MAX_VELOCITY)
        assert np.allclose([max_accel, max_decel] / max_vel, shape, rtol=1e-15), scale
        assert max_vel == min(scale, 1.0) * TOSS_MAX_VELOCITY

    # Proportional through the origin, not merely affine, or "twice the speed" would
    # mean other than twice the effort. Below the clamp only; clamping is not
    # homogeneous.
    rng = np.random.default_rng(0)
    for _ in range(200):
        factor = rng.uniform(0.05, 1.0)
        speed = rng.uniform(0.05, 1.0) * TOSS_MAX_VELOCITY
        scaled = np.array(toss_profile_limits(factor * speed))
        assert np.allclose(
            scaled, factor * np.array(toss_profile_limits(speed)), rtol=1e-12
        )
    assert np.array_equal(np.array(toss_profile_limits(0.0)), np.zeros(3))


def test_toss_release_speed_clamps_the_effort_to_zero_and_one():
    """The arm's own ceiling, and no reverse swing below it."""
    at_ceiling = toss_profile_limits(TOSS_MAX_VELOCITY)
    assert toss_profile_limits(2.0 * TOSS_MAX_VELOCITY) == at_ceiling
    assert toss_profile_limits(-TOSS_MAX_VELOCITY) == toss_profile_limits(0.0)
    assert np.array_equal(np.array(toss_profile_limits(-1.0)), np.zeros(3))


def test_the_release_speeds_the_sampler_draws_are_never_clamped():
    """SPEED_BOUNDS' top edge is the clamp point, so it must pass through."""
    for speed in np.linspace(*MoveToTossLocationAndTossController.SPEED_BOUNDS, 25):
        assert np.isclose(toss_profile_limits(speed)[0], speed)
