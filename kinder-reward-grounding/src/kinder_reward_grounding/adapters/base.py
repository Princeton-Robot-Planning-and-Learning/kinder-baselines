"""Base adapter interfaces for reward grounding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias

import numpy as np

StateLike: TypeAlias = np.ndarray | Mapping[str, float]


class EnvAdapter(Protocol):
    """Interface for decoding states and computing object-centric quantities."""

    def decode(self, env: Any, state: StateLike) -> Any:
        """Decode a planner or object-centric state into a structured state."""

    def get(self, state: Any, obj: Any, feature: str) -> Any:
        """Get a feature value for an object."""

    def get_object_by_name(self, state: Any, name: str) -> Any:
        """Get an object by name from a structured state."""

    def get_constant_by_name(self, env: Any, name: str) -> tuple[Any, Any]:
        """Get a constant object and its constant state by name."""

    def position(self, state: Any, obj: Any) -> np.ndarray:
        """Get a 2D object position."""

    def distance(self, state: Any, obj_a: Any, obj_b: Any) -> float:
        """Compute distance between two objects."""

    def vertical_gap(self, state: Any, obj_a: Any, obj_b: Any) -> float:
        """Compute vertical geometric gap between two rectangle-like objects."""

    def min_keypoint_distance(
        self,
        state: Any,
        source: Any,
        target: Any,
        keypoint_type: str = "heuristic",
    ) -> float:
        """Compute minimum distance from source keypoints to target."""

    def intersects(self, state: Any, obj_a: Any, obj_b: Any) -> bool:
        """Return whether two objects intersect."""
