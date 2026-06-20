"""Dataclasses for structured reward grounding specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

GroundingBackend = Literal["oracle_object", "vlm", "pixel", "distance_3d"]


@dataclass(frozen=True)
class ObjectRef:
    """Reference to a task-relevant object.

    Attributes:
        role: Semantic role used by reward specs, e.g. "tool" or "target".
        name: Optional concrete object name in the environment.
        type_name: Optional object type name.
    """

    role: str
    name: str | None = None
    type_name: str | None = None


@dataclass(frozen=True)
class PredicateSpec:
    """Structured specification for a subgoal completion predicate."""

    name: str
    objects: dict[str, str]
    backend: GroundingBackend = "oracle_object"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressMetricSpec:
    """Structured specification for a dense progress metric."""

    name: str
    objects: dict[str, str]
    backend: GroundingBackend = "oracle_object"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubgoalSpec:
    """A semantic subgoal with both completion and progress definitions."""

    name: str
    completion: PredicateSpec
    progress: ProgressMetricSpec
    weight: float = 1.0
    completion_bonus: float = 1.0
    preconditions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardSpec:
    """Full reward specification for one environment or task."""

    env_id: str
    objects: tuple[ObjectRef, ...]
    subgoals: tuple[SubgoalSpec, ...]
    composer: str = "sequential"
    time_penalty: float = -0.01
    success_bonus: float = 50.0
