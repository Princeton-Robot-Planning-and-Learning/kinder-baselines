"""Predicate library for grounded reward specifications."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

PredicateFn = Callable[[Any, dict[str, Any], dict[str, Any], Any], bool]

PREDICATES: dict[str, PredicateFn] = {}


def register_predicate(name: str) -> Callable[[PredicateFn], PredicateFn]:
    """Register a predicate function."""

    def decorator(fn: PredicateFn) -> PredicateFn:
        PREDICATES[name] = fn
        return fn

    return decorator


def get_predicate(name: str) -> PredicateFn:
    """Look up a predicate function by name."""
    if name not in PREDICATES:
        raise KeyError(f"Unknown predicate: {name}")
    return PREDICATES[name]


@register_predicate("held")
def held(
    state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> bool:
    """Return whether an object is held."""
    obj = objects["obj"]
    threshold = float(params.get("threshold", 0.5))
    return float(adapter.get(state, obj, "held")) > threshold


@register_predicate("near")
def near(
    state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> bool:
    """Return whether two objects are within a distance threshold."""
    obj_a = objects["a"]
    obj_b = objects["b"]
    threshold = float(params["threshold"])
    return float(adapter.distance(state, obj_a, obj_b)) <= threshold


@register_predicate("keypoint_near")
def keypoint_near(
    state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> bool:
    """Return whether a source object's keypoint is near a target object."""
    source = objects["source"]
    target = objects["target"]
    threshold = float(params["threshold"])
    keypoint_type = str(params.get("keypoint_type", "heuristic"))
    distance = adapter.min_keypoint_distance(
        state,
        source,
        target,
        keypoint_type=keypoint_type,
    )
    return float(distance) <= threshold


@register_predicate("vertical_gap_below")
def vertical_gap_below(
    state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> bool:
    """Return whether the vertical gap is below a threshold."""
    moving = objects["moving"]
    target = objects["target"]
    threshold = float(params["threshold"])
    return float(adapter.vertical_gap(state, moving, target)) <= threshold


@register_predicate("near_passage")
def near_passage(
    state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> bool:
    """Return whether an object is near a passage center."""
    threshold = float(params["threshold"])
    distance = adapter.passage_distance(
        state,
        objects["robot"],
        objects["bottom"],
        objects["top"],
    )
    return float(distance) <= threshold


@register_predicate("intersects")
def intersects(
    state: Any,
    objects: dict[str, Any],
    params: dict[str, Any],
    adapter: Any,
) -> bool:
    """Return whether two objects geometrically intersect."""
    del params
    return bool(adapter.intersects(state, objects["a"], objects["b"]))
