"""Compatibility wrapper for predicate functions."""

from kinder_reward_grounding.predicates import (
    PREDICATES,
    PredicateFn,
    get_predicate,
    held,
    intersects,
    keypoint_near,
    near,
    near_passage,
    register_predicate,
    vertical_gap_below,
)

__all__ = [
    "PREDICATES",
    "PredicateFn",
    "get_predicate",
    "held",
    "intersects",
    "keypoint_near",
    "near",
    "near_passage",
    "register_predicate",
    "vertical_gap_below",
]
