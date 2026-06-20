"""Compatibility wrapper for progress metrics."""

from kinder_reward_grounding.rewards.progress_metrics import (
    PROGRESS_METRICS,
    ProgressMetricFn,
    distance_progress,
    get_progress_metric,
    keypoint_distance_progress,
    passage_progress,
    register_progress_metric,
    vertical_gap_progress,
)

__all__ = [
    "PROGRESS_METRICS",
    "ProgressMetricFn",
    "distance_progress",
    "get_progress_metric",
    "keypoint_distance_progress",
    "passage_progress",
    "register_progress_metric",
    "vertical_gap_progress",
]
