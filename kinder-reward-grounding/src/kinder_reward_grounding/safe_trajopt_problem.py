"""Reward-grounding TrajOptProblem wrapper with safe invalid-transition handling."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
from kinder_trajopt.trajopt_problem import KinderTrajOptProblem
from prpl_utils.trajopt.trajopt_problem import TrajOptAction, TrajOptState

RewardFn = Callable[
    [TrajOptState, TrajOptAction, TrajOptState, float, bool, Any],
    float,
]


class SafeKinderTrajOptProblem(KinderTrajOptProblem):
    """TrajOpt wrapper for reward-grounded planning.

    This keeps reward-grounding-specific reward replacement and invalid dynamic
    geometry handling out of the generic ``kinder_trajopt`` package.
    """

    def __init__(
        self,
        *args: Any,
        reward_fn: RewardFn | None = None,
        invalid_transition_reward: float = -1e6,
        max_logged_invalid_transitions: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._reward_fn = reward_fn
        self.invalid_transition_reward = invalid_transition_reward
        self.max_logged_invalid_transitions = max_logged_invalid_transitions
        self.num_invalid_transitions = 0

    def get_next_state(
        self,
        state: TrajOptState,
        action: TrajOptAction,
    ) -> TrajOptState:
        """Return the next state, applying reward override and safe penalties."""
        try:
            next_state = super().get_next_state(state, action)
        except AssertionError as err:
            if not self._is_known_geometry_assertion(err):
                raise

            self.num_invalid_transitions += 1
            if self.num_invalid_transitions <= self.max_logged_invalid_transitions:
                print(
                    "[SafeKinderTrajOptProblem] "
                    f"invalid geometry transition #{self.num_invalid_transitions}; "
                    f"reward={self.invalid_transition_reward:.1e}"
                )

            step = self._cache_step
            self._cached_rewards[step] = float(self.invalid_transition_reward)
            self._cached_terminated[step] = True
            self._cache_step += 1
            return np.array(state, copy=True)

        if self._reward_fn is not None:
            step = self._cache_step - 1
            env_reward = self._cached_rewards[step]
            terminated = self._cached_terminated[step]
            self._cached_rewards[step] = float(
                self._reward_fn(
                    state,
                    action,
                    next_state,
                    env_reward,
                    terminated,
                    self._env,
                )
            )

        return next_state

    @staticmethod
    def _is_known_geometry_assertion(err: AssertionError) -> bool:
        """Return whether an AssertionError came from known dynamic geometry code."""
        geometry_markers = (
            "tomsgeoms2d",
            "dyn_pushpullhook2d.py",
            "envs/utils.py",
        )
        return any(
            any(marker in frame.filename for marker in geometry_markers)
            for frame in traceback.extract_tb(err.__traceback__)
        )
