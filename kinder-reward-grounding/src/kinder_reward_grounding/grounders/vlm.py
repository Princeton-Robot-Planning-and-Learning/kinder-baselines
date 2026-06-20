"""Placeholder VLM grounding interface.

The real VLM implementation is intentionally absent. This module defines the
boundary expected by reward code and provides a deterministic mock for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GroundingQuery:
    """Minimal query sent to a visual-language grounder."""

    prompt: str
    objects: tuple[str, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)


class VLMGrounder(Protocol):
    """Interface for scoring whether a visual scene satisfies a query."""

    def score(self, scene: Any, query: GroundingQuery) -> float:
        """Return a normalized score in [0, 1]."""


class MockVLMGrounder:
    """Deterministic stand-in for VLM scoring in tests and offline demos."""

    def __init__(self, default_score: float = 0.5) -> None:
        self.default_score = self._clamp(default_score)
        self._scores: dict[str, float] = {}

    def set_score(self, prompt: str, score: float) -> None:
        """Set a deterministic score for a prompt."""
        self._scores[prompt] = self._clamp(score)

    def score(self, scene: Any, query: GroundingQuery) -> float:
        """Return a deterministic normalized score for the query prompt."""
        del scene
        return self._scores.get(query.prompt, self.default_score)

    @staticmethod
    def _clamp(score: float) -> float:
        return min(1.0, max(0.0, float(score)))
