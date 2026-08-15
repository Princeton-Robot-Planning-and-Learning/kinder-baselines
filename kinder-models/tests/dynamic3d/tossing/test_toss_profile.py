"""Tests for toss_profile.py."""

import numpy as np
import pytest

from kinder_models.dynamic3d.tossing.parameterized_skills import TOSS_SPEED_BOUNDS
from kinder_models.dynamic3d.tossing.toss_profile import (
    TOSS_MAX_ACCELERATION,
    TOSS_MAX_DECELERATION,
    TOSS_MAX_VELOCITY,
    toss_profile_limits,
)
from kinder_models.dynamic3d.tossing.toss_swing import (
    TOSS_RELEASE_ARM_CONFIGURATION,
    TOSS_WINDUP_ARM_CONFIGURATION,
)
from kinder_models.dynamic3d.utils import (
    _CONTROL_TIMESTEP,
    _trapezoidal_motion_profile,
)


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
    """TOSS_SPEED_BOUNDS' top edge is the clamp point, so it must pass through."""
    for speed in np.linspace(*TOSS_SPEED_BOUNDS, 25):
        assert np.isclose(toss_profile_limits(speed)[0], speed)
