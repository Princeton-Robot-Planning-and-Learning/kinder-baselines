"""Tests for reward evaluator wiring."""

from typing import Any

import numpy as np

from kinder_reward_grounding.evaluator import RewardEvaluator
from kinder_reward_grounding.specs import (
    ObjectRef,
    PredicateSpec,
    ProgressMetricSpec,
    RewardSpec,
    SubgoalSpec,
)


class DummyAdapter:
    """Minimal object adapter for evaluator tests."""

    def decode(self, env: Any, flat_state: Any) -> dict[str, float]:
        del env
        return flat_state

    def get_object_by_name(self, state: dict[str, float], name: str) -> str:
        if name not in state:
            raise KeyError(name)
        return name

    def get_constant_by_name(self, env: Any, name: str) -> tuple[Any, str]:
        del env
        return {}, name

    def distance(self, state: dict[str, float], obj_a: str, obj_b: str) -> float:
        return abs(state[obj_a] - state[obj_b])


def test_reward_evaluator_runs_with_dummy_adapter() -> None:
    spec = RewardSpec(
        env_id="dummy",
        objects=(
            ObjectRef(role="robot", name="robot"),
            ObjectRef(role="target", name="target"),
        ),
        subgoals=(
            SubgoalSpec(
                name="approach_target",
                completion=PredicateSpec(
                    name="near",
                    objects={"a": "robot", "b": "target"},
                    params={"threshold": 1.0},
                ),
                progress=ProgressMetricSpec(
                    name="distance_progress",
                    objects={"a": "robot", "b": "target"},
                ),
                weight=2.0,
                completion_bonus=5.0,
            ),
        ),
        time_penalty=-0.1,
        success_bonus=50.0,
    )
    evaluator = RewardEvaluator(spec=spec, adapter=DummyAdapter())

    reward = evaluator(
        state={"robot": 0.0, "target": 10.0},
        action=np.array([]),
        next_state={"robot": 0.0, "target": 0.5},
        env_reward=0.0,
        terminated=False,
        env=None,
    )

    assert reward == 23.9
    assert not hasattr(evaluator, "active_subgoal_idx")
