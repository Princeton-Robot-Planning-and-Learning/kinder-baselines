"""Tests for reward composers."""

from kinder_reward_grounding.composers import (
    ComposerInput,
    SequentialComposer,
    SubgoalEvaluation,
)


def test_sequential_composer_rewards_only_active_subgoal() -> None:
    composer = SequentialComposer()
    output = composer.compose(
        ComposerInput(
            subgoals=(
                SubgoalEvaluation(
                    name="first",
                    progress=1.0,
                    completed_before=False,
                    completed_after=False,
                    weight=2.0,
                    completion_bonus=5.0,
                    diagnostics={},
                ),
                SubgoalEvaluation(
                    name="second",
                    progress=100.0,
                    completed_before=False,
                    completed_after=True,
                    weight=100.0,
                    completion_bonus=500.0,
                    diagnostics={},
                ),
            ),
            terminated=False,
            time_penalty=-0.1,
            success_bonus=50.0,
        )
    )

    assert output.reward == 1.9
    assert output.active_subgoal_idx == 0
    assert output.diagnostics["active_subgoal"] == "first"
    assert output.diagnostics["active_progress"] == 1.0
    assert output.diagnostics["active_completed_before"] is False
    assert output.diagnostics["active_completed_after"] is False


def test_sequential_composer_selects_first_subgoal_incomplete_before() -> None:
    composer = SequentialComposer()
    output = composer.compose(
        ComposerInput(
            subgoals=(
                SubgoalEvaluation(
                    name="first",
                    progress=100.0,
                    completed_before=True,
                    completed_after=True,
                    weight=100.0,
                    completion_bonus=500.0,
                    diagnostics={},
                ),
                SubgoalEvaluation(
                    name="second",
                    progress=1.0,
                    completed_before=False,
                    completed_after=False,
                    weight=2.0,
                    completion_bonus=5.0,
                    diagnostics={},
                ),
            ),
            terminated=False,
            time_penalty=-0.1,
            success_bonus=50.0,
        )
    )

    assert output.reward == 1.9
    assert output.active_subgoal_idx == 1
    assert output.diagnostics["active_subgoal"] == "second"


def test_sequential_composer_adds_completion_bonus_when_completed_after() -> None:
    composer = SequentialComposer()
    output = composer.compose(
        ComposerInput(
            subgoals=(
                SubgoalEvaluation(
                    name="first",
                    progress=1.0,
                    completed_before=False,
                    completed_after=True,
                    weight=2.0,
                    completion_bonus=5.0,
                    diagnostics={},
                ),
            ),
            terminated=False,
            time_penalty=-0.1,
            success_bonus=50.0,
        )
    )

    assert output.reward == 6.9
    assert output.active_subgoal_idx == 0
    assert output.diagnostics["active_completed_before"] is False
    assert output.diagnostics["active_completed_after"] is True


def test_sequential_composer_is_stateless_across_calls() -> None:
    composer = SequentialComposer()
    complete_first = ComposerInput(
        subgoals=(
            SubgoalEvaluation(
                name="first",
                progress=1.0,
                completed_before=True,
                completed_after=True,
                weight=2.0,
                completion_bonus=5.0,
                diagnostics={},
            ),
            SubgoalEvaluation(
                name="second",
                progress=2.0,
                completed_before=False,
                completed_after=False,
                weight=3.0,
                completion_bonus=7.0,
                diagnostics={},
            ),
        ),
        terminated=False,
        time_penalty=0.0,
        success_bonus=50.0,
    )
    incomplete_first = ComposerInput(
        subgoals=(
            SubgoalEvaluation(
                name="first",
                progress=1.0,
                completed_before=False,
                completed_after=False,
                weight=2.0,
                completion_bonus=5.0,
                diagnostics={},
            ),
            SubgoalEvaluation(
                name="second",
                progress=2.0,
                completed_before=False,
                completed_after=False,
                weight=3.0,
                completion_bonus=7.0,
                diagnostics={},
            ),
        ),
        terminated=False,
        time_penalty=0.0,
        success_bonus=50.0,
    )

    assert composer.compose(complete_first).active_subgoal_idx == 1
    assert composer.compose(incomplete_first).active_subgoal_idx == 0
