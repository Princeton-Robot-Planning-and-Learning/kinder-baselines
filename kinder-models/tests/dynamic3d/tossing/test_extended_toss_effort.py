"""Explicit simulation-only effort extension, preserving the default profile."""

import numpy as np
import pytest

from kinder_models.dynamic3d.tossing.toss_swing import (
    TOSS_MAX_VELOCITY,
    plan_toss_swing,
    toss_profile_limits,
)


def test_default_effort_ceiling_is_unchanged() -> None:
    assert toss_profile_limits(3 * TOSS_MAX_VELOCITY) == toss_profile_limits()


def test_extended_effort_scales_all_limits_and_still_caps() -> None:
    nominal = np.array(toss_profile_limits())
    np.testing.assert_allclose(
        toss_profile_limits(2 * TOSS_MAX_VELOCITY, max_effort=2.5),
        2 * nominal,
    )
    np.testing.assert_allclose(
        toss_profile_limits(3 * TOSS_MAX_VELOCITY, max_effort=2.5),
        2.5 * nominal,
    )


@pytest.mark.parametrize("max_effort", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_effort_ceiling_is_rejected(max_effort: float) -> None:
    with pytest.raises(ValueError, match="max_effort"):
        toss_profile_limits(max_effort=max_effort)


def test_swing_propagates_explicit_ceiling() -> None:
    start = [0.0] * 7
    end = [1.0] * 7
    nominal = plan_toss_swing([start, end], start, 2 * TOSS_MAX_VELOCITY)
    extended = plan_toss_swing(
        [start, end], start, 2 * TOSS_MAX_VELOCITY, max_effort=2.5
    )
    assert len(extended.trajectory) < len(nominal.trajectory)
