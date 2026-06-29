"""Reward evaluation and composition."""

from kinder_reward_grounding.rewards.composers import (
    ComposerInput,
    ComposerOutput,
    RewardComposer,
    SequentialComposer,
    SubgoalEvaluation,
)
from kinder_reward_grounding.rewards.evaluator import RewardEvaluator
from kinder_reward_grounding.rewards.progress_metrics import (
    PROGRESS_METRICS,
    ProgressMetricFn,
    distance_progress,
    get_progress_metric,
    keypoint_distance_progress,
    register_progress_metric,
    vertical_gap_progress,
)

__all__ = [
    "PROGRESS_METRICS",
    "ComposerInput",
    "ComposerOutput",
    "ProgressMetricFn",
    "RewardComposer",
    "RewardEvaluator",
    "SequentialComposer",
    "SubgoalEvaluation",
    "distance_progress",
    "get_progress_metric",
    "keypoint_distance_progress",
    "register_progress_metric",
    "vertical_gap_progress",
]
