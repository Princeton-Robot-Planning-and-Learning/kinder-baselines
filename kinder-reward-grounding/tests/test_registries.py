"""Tests for predicate and progress metric registries."""

from kinder_reward_grounding.predicates import PREDICATES
from kinder_reward_grounding.rewards import PROGRESS_METRICS


def test_predicate_registry_contains_basic_predicates() -> None:
    assert "held" in PREDICATES
    assert "near" in PREDICATES
    assert "keypoint_near" in PREDICATES
    assert "vertical_gap_below" in PREDICATES


def test_progress_registry_contains_basic_metrics() -> None:
    assert "distance_progress" in PROGRESS_METRICS
    assert "vertical_gap_progress" in PROGRESS_METRICS
    assert "keypoint_distance_progress" in PROGRESS_METRICS
