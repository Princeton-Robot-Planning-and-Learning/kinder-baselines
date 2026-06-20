"""Tests for predicate output contracts."""

from typing import Any

from kinder_reward_grounding.predicates import held, near, vertical_gap_below


class PredicateAdapter:
    """Minimal adapter for predicate tests."""

    def get(self, state: dict[str, dict[str, float]], obj: str, feature: str) -> float:
        return state[obj][feature]

    def distance(self, state: dict[str, float], obj_a: str, obj_b: str) -> float:
        return abs(state[obj_a] - state[obj_b])

    def vertical_gap(self, state: dict[str, float], obj_a: str, obj_b: str) -> float:
        del obj_a, obj_b
        return state["gap"]


def assert_bool(value: Any) -> None:
    assert isinstance(value, bool)


def test_predicates_return_bools() -> None:
    adapter = PredicateAdapter()

    assert_bool(
        held(
            {"tool": {"held": 1.0}},
            {"obj": "tool"},
            {"threshold": 0.5},
            adapter,
        )
    )
    assert_bool(
        near(
            {"robot": 0.0, "goal": 0.25},
            {"a": "robot", "b": "goal"},
            {"threshold": 0.5},
            adapter,
        )
    )
    assert_bool(
        vertical_gap_below(
            {"gap": 0.1},
            {"moving": "block", "target": "wall"},
            {"threshold": 0.2},
            adapter,
        )
    )
