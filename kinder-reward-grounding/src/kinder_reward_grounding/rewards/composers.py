"""Reward composition strategies."""

# Composer implementations and their protocol intentionally expose one method.
# pylint: disable=too-few-public-methods

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SubgoalEvaluation:
    """Evaluation result for one grounded subgoal."""

    name: str
    progress: float
    completed_before: bool
    completed_after: bool
    weight: float
    completion_bonus: float
    diagnostics: dict[str, float | bool | str]


@dataclass(frozen=True)
class ComposerInput:
    """Inputs required by a reward composer."""

    subgoals: tuple[SubgoalEvaluation, ...]
    terminated: bool
    time_penalty: float
    success_bonus: float


@dataclass(frozen=True)
class ComposerOutput:
    """Output produced by a reward composer."""

    reward: float
    active_subgoal_idx: int
    diagnostics: dict[str, float | bool | str]


class RewardComposer(Protocol):
    """Protocol for reward composition strategies."""

    def compose(self, composer_input: ComposerInput) -> ComposerOutput:
        """Compose evaluated subgoals into one scalar reward."""
        raise NotImplementedError


class SequentialComposer:
    """Compose rewards using only the current active subgoal."""

    def compose(self, composer_input: ComposerInput) -> ComposerOutput:
        """Return reward for the first subgoal incomplete before transition."""
        active_idx = next(
            (
                idx
                for idx, subgoal in enumerate(composer_input.subgoals)
                if not subgoal.completed_before
            ),
            len(composer_input.subgoals),
        )

        if active_idx >= len(composer_input.subgoals):
            reward = composer_input.success_bonus if composer_input.terminated else 0.0
            return ComposerOutput(
                reward=float(reward),
                active_subgoal_idx=active_idx,
                diagnostics={
                    "active_subgoal": "",
                    "active_subgoal_idx": active_idx,
                    "active_progress": 0.0,
                    "active_completed_before": True,
                    "active_completed_after": True,
                },
            )

        active_subgoal = composer_input.subgoals[active_idx]
        reward = (
            composer_input.time_penalty
            + active_subgoal.weight * active_subgoal.progress
        )

        if active_subgoal.completed_after:
            reward += active_subgoal.completion_bonus

        if composer_input.terminated:
            reward += composer_input.success_bonus

        return ComposerOutput(
            reward=float(reward),
            active_subgoal_idx=active_idx,
            diagnostics={
                "active_subgoal": active_subgoal.name,
                "active_subgoal_idx": active_idx,
                "active_progress": active_subgoal.progress,
                "active_completed_before": active_subgoal.completed_before,
                "active_completed_after": active_subgoal.completed_after,
            },
        )


class SoftWeightedComposer:
    """Skeleton for soft weighted subgoal composition."""

    def compose(self, composer_input: ComposerInput) -> ComposerOutput:
        """Compose evaluated subgoals into one scalar reward."""
        raise NotImplementedError


class HybridComposer:
    """Skeleton for hybrid sequential and weighted composition."""

    def compose(self, composer_input: ComposerInput) -> ComposerOutput:
        """Compose evaluated subgoals into one scalar reward."""
        raise NotImplementedError
