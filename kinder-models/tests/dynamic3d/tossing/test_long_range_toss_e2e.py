"""The coverage demo exercises the same live path as these paired regressions."""

import pytest

from kinder_models.dynamic3d.tossing.coverage import run_trial


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize(
    "seed, speed", [(10125, 360.0), (10126, 380.0), (10127, 358.0)]
)
def test_far_receiver_requires_extended_effort(
    extended: bool, seed: int, speed: float
) -> None:
    """Calibrated cases establish reachability, not an unbiased success rate."""
    row = run_trial(
        seed=seed,
        distance=2.5,
        speed=speed,
        release_ms=500.0,
        max_effort=3.0 if extended else 1.0,
    )
    assert row["pickup_holding"], row
    assert row["pickup_steps"] > 0 and row["toss_steps"] > 0
    assert row["status"] == ("success" if extended else "toss_miss"), row


def test_coverage_records_planning_failure_without_claiming_a_throw() -> None:
    """An obstructed target is logged as a planning failure with zero toss ticks."""
    row = run_trial(
        seed=10125, distance=1.35, speed=360.0, release_ms=500.0, max_effort=3.0
    )
    assert row["status"] == "toss_planning_failure", row
    assert row["pickup_holding"] and row["toss_steps"] == 0
    assert "Base motion planning failed" in row["error"]


def test_coverage_records_pickup_timeout() -> None:
    """A short control budget is a timeout, not a physical toss failure."""
    row = run_trial(
        seed=10125,
        distance=2.5,
        speed=360.0,
        release_ms=500.0,
        max_effort=3.0,
        step_limit=1,
    )
    assert row["status"] == "pickup_timeout", row
    assert row["pickup_steps"] == 1 and row["toss_steps"] == 0
