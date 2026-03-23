"""TrajOptProblem implementation backed by a KinDER gymnasium env."""

import logging
from typing import Any

from gymnasium.spaces import Box
from prpl_utils.structs import Image
from prpl_utils.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)


class KinderTrajOptProblem(TrajOptProblem):
    """Wraps a KinDER env as a TrajOptProblem.

    Uses `get_transition` for dynamics, reward, and termination. Caches
    reward and termination from each `get_next_state` call so that
    `get_traj_cost` does not need to re-simulate the trajectory.
    """

    def __init__(
        self,
        env: Any,
        initial_state: TrajOptState,
        horizon: int,
    ) -> None:
        self._env = env
        self._initial_state = initial_state
        self._horizon = horizon
        self._cached_rewards: dict[int, float] = {}
        self._cached_terminated: dict[int, bool] = {}
        self._cache_step = 0
        self._num_rollouts_scored = 0
        self._num_goals_found = 0
        self._best_cost_this_step = float("inf")

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def state_space(self) -> Box:
        space = self._env.observation_space
        assert isinstance(space, Box)
        return space

    @property
    def action_space(self) -> Box:
        space = self._env.action_space
        assert isinstance(space, Box)
        return space

    @property
    def initial_state(self) -> TrajOptState:
        return self._initial_state

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        next_state, reward, terminated = self._env.unwrapped.get_transition(
            state, action
        )
        step = self._cache_step
        self._cached_rewards[step] = float(reward)
        self._cached_terminated[step] = terminated
        self._cache_step += 1
        return next_state

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        total_reward = 0.0
        horizon = len(traj.actions)
        start_step = self._cache_step - horizon
        terminated_early = False
        for idx in range(horizon):
            step = start_step + idx
            total_reward += self._cached_rewards[step]
            if self._cached_terminated[step]:
                terminated_early = True
                break
        cost = -total_reward
        self._num_rollouts_scored += 1
        if terminated_early:
            self._num_goals_found += 1
        self._best_cost_this_step = min(self._best_cost_this_step, cost)
        return cost

    @property
    def num_rollouts_scored(self) -> int:
        """Number of rollouts scored since the last reset."""
        return self._num_rollouts_scored

    def log_and_reset_step_stats(self, timestep: int) -> None:
        """Log MPC step stats and reset counters."""
        logging.debug(
            "MPC step %d: best_cost=%.1f, goals_found=%d/%d rollouts",
            timestep,
            self._best_cost_this_step,
            self._num_goals_found,
            self._num_rollouts_scored,
        )
        self._num_rollouts_scored = 0
        self._num_goals_found = 0
        self._best_cost_this_step = float("inf")

    def render_state(self, state: TrajOptState) -> Image:
        raise NotImplementedError
