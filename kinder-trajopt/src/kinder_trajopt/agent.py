"""Trajectory optimization agent using predictive sampling."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from prpl_utils.gym_agent import Agent
from prpl_utils.trajopt.mpc_wrapper import MPCWrapper
from prpl_utils.trajopt.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)

from kinder_trajopt.trajopt_problem import KinderTrajOptProblem


class TrajOptAgent(Agent[NDArray[np.float32], NDArray[np.float32]]):
    """An agent that uses predictive sampling for trajectory optimization.

    Uses the MPC wrapper to re-plan at every timestep in a receding-horizon
    fashion. Works with any KinDER environment that supports
    `get_transition`.
    """

    def __init__(
        self,
        env: Any,
        seed: int,
        horizon: int = 50,
        num_rollouts: int = 100,
        noise_fraction: float = 1.0,
        num_control_points: int = 10,
        warm_start: bool = True,
    ) -> None:
        super().__init__(seed)
        self._env = env
        self._horizon = horizon
        action_range = env.action_space.high - env.action_space.low
        noise_scale = action_range * noise_fraction
        config = PredictiveSamplingHyperparameters(
            num_rollouts=num_rollouts,
            noise_scale=noise_scale,
            num_control_points=num_control_points,
        )
        solver = PredictiveSamplingSolver(seed, config, warm_start)
        self._mpc = MPCWrapper(solver)

    def reset(
        self,
        obs: NDArray[np.float32],
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        self._problem = KinderTrajOptProblem(
            env=self._env,
            initial_state=obs,
            horizon=self._horizon,
        )
        self._mpc.reset(self._problem)

    def _get_action(self) -> NDArray[np.float32]:
        assert self._last_observation is not None
        action = self._mpc.step(self._last_observation)
        self._problem.log_and_reset_step_stats()
        return action
