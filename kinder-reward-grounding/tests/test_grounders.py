"""Tests for VLM grounder placeholders."""

# Test names document behavior; per-test docstrings would duplicate them.
# pylint: disable=missing-function-docstring

from kinder_reward_grounding.grounders import GroundingQuery, MockVLMGrounder


def test_mock_vlm_grounder_returns_default_score() -> None:
    grounder = MockVLMGrounder(default_score=0.25)

    score = grounder.score(scene=None, query=GroundingQuery(prompt="is near target?"))

    assert score == 0.25


def test_mock_vlm_grounder_returns_deterministic_prompt_score() -> None:
    grounder = MockVLMGrounder(default_score=0.25)
    grounder.set_score("is near target?", 0.75)

    query = GroundingQuery(prompt="is near target?")

    assert grounder.score(scene={"frame": 1}, query=query) == 0.75
    assert grounder.score(scene={"frame": 2}, query=query) == 0.75


def test_mock_vlm_grounder_clamps_scores_to_unit_interval() -> None:
    grounder = MockVLMGrounder(default_score=2.0)
    grounder.set_score("bad score", -1.0)

    assert grounder.score(scene=None, query=GroundingQuery(prompt="unknown")) == 1.0
    assert grounder.score(scene=None, query=GroundingQuery(prompt="bad score")) == 0.0
