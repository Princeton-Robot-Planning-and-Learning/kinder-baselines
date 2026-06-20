"""Progress metric library for grounded reward specifications."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressMetricFn = Callable[[Any, Any, dict[str, Any], dict[str, Any], Any], float]

PROGRESS_METRICS: dict[str, ProgressMetricFn] = {}


def register_progress_metric(
    name: str,
) -> Callable[[ProgressMetricFn], ProgressMetricFn]:
    """Register a progress metric function."""

    def decorator(fn: ProgressMetricFn) -> ProgressMetricFn:
        PROGRESS_METRICS[name] = fn
        return fn

    return decorator


def get_progress_metric(name: str) -> ProgressMetricFn:
    """Look up a progress metric function by name."""
    if name not in PROGRESS_METRICS:
        raise KeyError(f"Unknown progress metric: {name}")
    return PROGRESS_METRICS[name]


@register_progress_metric("distance_progress")
def distance_progress(
    state: Any,
    next_state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> float:
    """Return positive progress when distance between two objects decreases."""
    del params
    obj_a = objects["a"]
    obj_b = objects["b"]
    before = adapter.distance(state, obj_a, obj_b)
    after = adapter.distance(next_state, obj_a, obj_b)
    return float(before - after)


@register_progress_metric("vertical_gap_progress")
def vertical_gap_progress(
    state: Any,
    next_state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> float:
    """Return positive progress when vertical gap decreases."""
    del params
    moving = objects["moving"]
    target = objects["target"]
    before = adapter.vertical_gap(state, moving, target)
    after = adapter.vertical_gap(next_state, moving, target)
    return float(before - after)


@register_progress_metric("keypoint_distance_progress")
def keypoint_distance_progress(
    state: Any,
    next_state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> float:
    """Return positive progress when keypoint-target distance decreases."""
    source = objects["source"]
    target = objects["target"]
    keypoint_type = str(params.get("keypoint_type", "heuristic"))
    before = adapter.min_keypoint_distance(
        state,
        source,
        target,
        keypoint_type=keypoint_type,
    )
    after = adapter.min_keypoint_distance(
        next_state,
        source,
        target,
        keypoint_type=keypoint_type,
    )
    return float(before - after)


@register_progress_metric("passage_progress")
def passage_progress(
    state: Any,
    next_state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> float:
    """Return positive progress when an object gets closer to a passage."""
    del params
    before = adapter.passage_distance(
        state,
        objects["robot"],
        objects["bottom"],
        objects["top"],
    )
    after = adapter.passage_distance(
        next_state,
        objects["robot"],
        objects["bottom"],
        objects["top"],
    )
    return float(before - after)
