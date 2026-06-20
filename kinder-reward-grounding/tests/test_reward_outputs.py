"""Tests for reward output contracts."""

import math

from kinder_reward_grounding.rewards import (
    ComposerInput,
    SequentialComposer,
    SubgoalEvaluation,
)


def test_sequential_reward_output_is_finite_for_known_input() -> None:
    output = SequentialComposer().compose(
        ComposerInput(
            subgoals=(
                SubgoalEvaluation(
                    name="reach",
                    progress=0.25,
                    completed_before=False,
                    completed_after=True,
                    weight=4.0,
                    completion_bonus=3.0,
                    diagnostics={},
                ),
            ),
            terminated=False,
            time_penalty=-0.1,
            success_bonus=50.0,
        )
    )

    assert math.isfinite(output.reward)
    assert output.reward == 3.9
    assert output.active_subgoal_idx == 0
