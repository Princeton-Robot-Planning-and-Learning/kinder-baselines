"""Reward evaluator that compiles RewardSpec into a reward_fn."""

# The MPC-compatible callable signature and subgoal evaluation loop are
# intentionally explicit in this prototype.
# pylint: disable=too-few-public-methods,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals

from __future__ import annotations

from typing import Any

import numpy as np

from kinder_reward_grounding.adapters.base import StateLike
from kinder_reward_grounding.predicates import get_predicate
from kinder_reward_grounding.rewards.composers import (
    ComposerInput,
    RewardComposer,
    SequentialComposer,
    SubgoalEvaluation,
)
from kinder_reward_grounding.rewards.progress_metrics import get_progress_metric
from kinder_reward_grounding.specs import RewardSpec


class RewardEvaluator:
    """Evaluate a structured RewardSpec as an MPC-compatible reward function."""

    def __init__(
        self,
        spec: RewardSpec,
        adapter: Any,
        composer: RewardComposer | None = None,
    ) -> None:
        self.spec = spec
        self.adapter = adapter
        self.composer = composer or SequentialComposer()

    def __call__(
        self,
        state: StateLike,
        action: np.ndarray,
        next_state: StateLike,
        env_reward: float,
        terminated: bool,
        env: Any,
    ) -> float:
        """Return scalar reward for one transition."""
        del action, env_reward

        decoded_state = self.adapter.decode(env, state)
        decoded_next_state = self.adapter.decode(env, next_state)

        subgoal_evaluations: list[SubgoalEvaluation] = []

        for subgoal in self.spec.subgoals:
            objects = self._bind_objects(decoded_state, subgoal.progress.objects)

            progress_fn = get_progress_metric(subgoal.progress.name)
            progress = progress_fn(
                decoded_state,
                decoded_next_state,
                objects,
                subgoal.progress.params,
                self.adapter,
            )

            completion_objects = self._bind_objects(
                decoded_state,
                subgoal.completion.objects,
            )
            next_completion_objects = self._bind_objects(
                decoded_next_state,
                subgoal.completion.objects,
            )
            predicate_fn = get_predicate(subgoal.completion.name)
            completed_before = predicate_fn(
                decoded_state,
                completion_objects,
                subgoal.completion.params,
                self.adapter,
            )
            completed_after = predicate_fn(
                decoded_next_state,
                next_completion_objects,
                subgoal.completion.params,
                self.adapter,
            )

            subgoal_evaluations.append(
                SubgoalEvaluation(
                    name=subgoal.name,
                    progress=float(progress),
                    completed_before=bool(completed_before),
                    completed_after=bool(completed_after),
                    weight=subgoal.weight,
                    completion_bonus=subgoal.completion_bonus,
                    diagnostics={
                        "progress_backend": subgoal.progress.backend,
                        "completion_backend": subgoal.completion.backend,
                    },
                )
            )

        composer_input = ComposerInput(
            subgoals=tuple(subgoal_evaluations),
            terminated=terminated,
            time_penalty=self.spec.time_penalty,
            success_bonus=self.spec.success_bonus,
        )
        output = self.composer.compose(composer_input)

        return output.reward

    def _bind_objects(
        self,
        state: Any,
        object_bindings: dict[str, str],
    ) -> dict[str, Any]:
        """Bind role names in a spec to concrete KinDER objects."""
        bound: dict[str, Any] = {}

        role_to_ref = {obj.role: obj for obj in self.spec.objects}

        for local_role, global_role in object_bindings.items():
            ref = role_to_ref[global_role]
            if ref.name is None:
                raise ValueError(f"ObjectRef for role {global_role!r} has no name.")

            bound[local_role] = self.adapter.get_object_by_name(state, ref.name)

        return bound
