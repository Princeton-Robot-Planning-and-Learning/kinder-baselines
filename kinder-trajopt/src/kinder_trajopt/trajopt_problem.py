"""TrajOptProblem implementation backed by a KinDER gymnasium env."""

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

    Uses `get_transition` for dynamics, reward, and termination. The cost
    of a trajectory is the negated cumulative reward (lower cost = better).
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
        next_state, _, _ = self._env.unwrapped.get_transition(state, action)
        return next_state

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        total_reward = 0.0
        for idx in range(len(traj.actions)):
            _, reward, terminated = self._env.unwrapped.get_transition(
                traj.states[idx], traj.actions[idx]
            )
            total_reward += float(reward)
            if terminated:
                break
        return -total_reward

    def render_state(self, state: TrajOptState) -> Image:
        raise NotImplementedError
