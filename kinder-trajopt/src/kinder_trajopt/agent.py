"""Trajectory optimization agent using predictive sampling."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from prpl_utils.gym_agent import Agent
from prpl_utils.trajopt.mpc_wrapper import MPCWrapper
from prpl_utils.trajopt.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)
from prpl_utils.trajopt.trajopt_problem import TrajOptAction, TrajOptState

from kinder_trajopt.trajopt_problem import KinderTrajOptProblem


def _make_wm_transition(
    wm_checkpoint: str,
    term_checkpoint: str,
    term_threshold: float,
) -> Callable[[TrajOptState, TrajOptAction], tuple[TrajOptState, float, bool]]:
    """Build a transition function backed by a learned world model.

    Imports kinder_mbrl lazily so that the simulator path (wm_checkpoint="")
    works without kinder_mbrl installed.

    The returned callable mirrors the contract of env.unwrapped.get_transition:
        next_state, reward, terminated = transition_fn(state, action)

    Reward follows the sparse reward structure: reward = -1 + term_prob,
    which approaches -1 far from the goal and 0 at the goal. Termination fires
    when term_prob exceeds term_threshold.

    Args:
        wm_checkpoint: Path to the dynamics model checkpoint (wm.pt).
        term_checkpoint: Path to the termination classifier checkpoint (term.pt).
        term_threshold: Probability threshold above which a state is considered
            terminal.

    Returns:
        A callable (state, action) -> (next_state, reward, terminated).
    """
    from kinder_mbrl.planning import (  # pylint: disable=import-outside-toplevel
        load_termination_classifier,
        load_world_model,
        wm_get_next_state,
        wm_get_termination_prob,
    )

    wm_model, wm_norms = load_world_model(wm_checkpoint)
    term_model, term_norms = load_termination_classifier(term_checkpoint)

    def transition(
        state: TrajOptState, action: TrajOptAction
    ) -> tuple[TrajOptState, float, bool]:
        next_state = wm_get_next_state(state, action, wm_model, wm_norms)
        prob = wm_get_termination_prob(next_state, term_model, term_norms)
        return next_state, -1.0 + prob, prob > term_threshold

    return transition


class TrajOptAgent(Agent[NDArray[np.float32], NDArray[np.float32]]):
    """An agent that uses predictive sampling for trajectory optimization.

    Uses the MPC wrapper to re-plan at every timestep in a receding-horizon
    fashion. Works with any KinDER environment that supports `get_transition`.

    When wm_checkpoint and term_checkpoint are provided the agent replaces the
    simulator's get_transition with a learned world model and termination
    classifier, using a soft sparse reward (reward = -1 + term_prob). When
    they are omitted (the default) behaviour is identical to before.
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
        replan_interval: int = 1,
        wm_checkpoint: str = "",
        term_checkpoint: str = "",
        term_threshold: float = 0.5,
    ) -> None:
        super().__init__(seed)
        self._env = env
        self._horizon = horizon
        self._problem: KinderTrajOptProblem | None = None
        self._wm_checkpoint = wm_checkpoint
        self._term_checkpoint = term_checkpoint
        self._term_threshold = term_threshold
        action_range = env.action_space.high - env.action_space.low
        noise_scale = action_range * noise_fraction
        config = PredictiveSamplingHyperparameters(
            num_rollouts=num_rollouts,
            noise_scale=noise_scale,
            num_control_points=num_control_points,
        )
        solver = PredictiveSamplingSolver(seed, config, warm_start)
        self._mpc = MPCWrapper(solver, replan_interval=replan_interval)

    def reset(
        self,
        obs: NDArray[np.float32],
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        transition_fn = None
        if self._wm_checkpoint and self._term_checkpoint:
            transition_fn = _make_wm_transition(
                self._wm_checkpoint, self._term_checkpoint, self._term_threshold
            )
        self._problem = KinderTrajOptProblem(
            env=self._env,
            initial_state=obs,
            horizon=self._horizon,
            transition_fn=transition_fn,
        )
        self._mpc.reset(self._problem)

    def _get_action(self) -> NDArray[np.float32]:
        assert self._last_observation is not None
        action = self._mpc.step(self._last_observation)
        assert self._problem is not None
        if self._problem.num_rollouts_scored > 0:
            self._problem.log_and_reset_step_stats(self._timestep)
        return action
