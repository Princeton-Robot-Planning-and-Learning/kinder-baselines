"""Tests for reward specifications."""

# Test names document behavior; per-test docstrings would duplicate them.
# pylint: disable=missing-function-docstring

from kinder_reward_grounding.env_specs.dyn_pushpullhook2d import (
    make_dyn_pushpullhook2d_reward_spec,
)
from kinder_reward_grounding.env_specs.motion2d import make_motion2d_reward_spec
from kinder_reward_grounding.specs import PredicateSpec, ProgressMetricSpec


def test_dyn_pushpullhook2d_spec_has_subgoals() -> None:
    spec = make_dyn_pushpullhook2d_reward_spec()

    assert spec.env_id == "kinder/DynPushPullHook2D-o0-v0"
    assert len(spec.subgoals) == 3
    assert spec.subgoals[0].name == "grasp_hook"
    assert spec.subgoals[1].name == "bring_hook_to_target"
    assert spec.subgoals[2].name == "move_target_to_wall"
    assert spec.subgoals[2].completion.name == "vertical_gap_below"


def test_dyn_pushpullhook2d_spec_uses_oracle_object_backend() -> None:
    spec = make_dyn_pushpullhook2d_reward_spec()

    for subgoal in spec.subgoals:
        assert subgoal.completion.backend == "oracle_object"
        assert subgoal.progress.backend == "oracle_object"


def test_grounding_specs_default_to_oracle_object_backend() -> None:
    predicate = PredicateSpec(name="near", objects={"a": "robot", "b": "target"})
    progress = ProgressMetricSpec(
        name="distance_progress",
        objects={"a": "robot", "b": "target"},
    )

    assert predicate.backend == "oracle_object"
    assert progress.backend == "oracle_object"


def test_motion2d_spec_guides_robot_through_passages_before_goal() -> None:
    spec = make_motion2d_reward_spec()

    assert spec.env_id == "kinder/Motion2D-p5-v0"
    assert len(spec.subgoals) == 6

    subgoal = spec.subgoals[0]
    assert subgoal.name == "reach_passage_1"
    assert subgoal.completion.name == "near_passage"
    assert subgoal.progress.name == "passage_progress"
    assert subgoal.completion.backend == "oracle_object"
    assert subgoal.progress.backend == "oracle_object"

    final_subgoal = spec.subgoals[-1]
    assert final_subgoal.name == "reach_goal"
    assert final_subgoal.completion.name == "near"
    assert final_subgoal.progress.name == "distance_progress"
