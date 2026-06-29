"""Compatibility wrapper for reward composers."""

from kinder_reward_grounding.rewards.composers import (
    ComposerInput,
    ComposerOutput,
    RewardComposer,
    SequentialComposer,
    SubgoalEvaluation,
)

__all__ = [
    "ComposerInput",
    "ComposerOutput",
    "RewardComposer",
    "SequentialComposer",
    "SubgoalEvaluation",
]
