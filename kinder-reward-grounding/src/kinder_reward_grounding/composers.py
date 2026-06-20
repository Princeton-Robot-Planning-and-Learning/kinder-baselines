"""Compatibility wrapper for reward composers."""

from kinder_reward_grounding.rewards.composers import (
    ComposerInput,
    ComposerOutput,
    HybridComposer,
    RewardComposer,
    SequentialComposer,
    SoftWeightedComposer,
    SubgoalEvaluation,
)

__all__ = [
    "ComposerInput",
    "ComposerOutput",
    "HybridComposer",
    "RewardComposer",
    "SequentialComposer",
    "SoftWeightedComposer",
    "SubgoalEvaluation",
]
