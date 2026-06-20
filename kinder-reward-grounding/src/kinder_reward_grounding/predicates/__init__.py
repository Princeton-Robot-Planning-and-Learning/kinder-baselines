"""Predicate functions and registry for grounded subgoals."""

from kinder_reward_grounding.predicates.registry import (
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
