"""Tests for the reward-grounding SafeKinderTrajOptProblem wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
trajopt_problem = pytest.importorskip("prpl_utils.trajopt.trajopt_problem")
pytest.importorskip("kinder_trajopt.trajopt_problem")
safe_trajopt_problem = pytest.importorskip(
    "kinder_reward_grounding.safe_trajopt_problem"
)

Box = gymnasium.spaces.Box
TrajOptTraj = trajopt_problem.TrajOptTraj
SafeKinderTrajOptProblem = safe_trajopt_problem.SafeKinderTrajOptProblem


class FakeEnv:
    """Minimal KinDER-like env for trajopt wrapper tests."""

    observation_space = Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    unwrapped: FakeEnv

    def __init__(self, reward: float = -2.0, terminated: bool = False) -> None:
        self.reward = reward
        self.terminated = terminated
        self.unwrapped = self

    def get_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        return state + action, self.reward, self.terminated


class RaisingEnv(FakeEnv):
    """Env that raises during transition simulation."""

    def __init__(self, err: AssertionError) -> None:
        super().__init__()
        self.err = err

    def get_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        raise self.err


def make_one_step_traj(
    state: np.ndarray,
    next_state: np.ndarray,
    action: np.ndarray,
) -> Any:
    """Build a one-step trajectory."""
    return TrajOptTraj(np.array([state, next_state]), np.array([action]))


def test_reward_fn_overrides_env_reward() -> None:
    """reward_fn should replace the env reward cached by KinderTrajOptProblem."""
    state = np.array([1.0, 2.0])
    action = np.array([0.5, -0.25])

    def reward_fn(
        prev_state: np.ndarray,
        prev_action: np.ndarray,
        next_state: np.ndarray,
        env_reward: float,
        terminated: bool,
        env: Any,
    ) -> float:
        assert np.allclose(prev_state, state)
        assert np.allclose(prev_action, action)
        assert np.allclose(next_state, state + action)
        assert env_reward == -2.0
        assert not terminated
        assert isinstance(env, FakeEnv)
        return 7.5

    problem = SafeKinderTrajOptProblem(
        env=FakeEnv(),
        initial_state=state,
        horizon=1,
        reward_fn=reward_fn,
    )

    next_state = problem.get_next_state(state, action)
    cost = problem.get_traj_cost(make_one_step_traj(state, next_state, action))

    assert cost == -7.5


def test_without_reward_fn_uses_env_reward() -> None:
    """No reward_fn should preserve the wrapped env reward."""
    state = np.array([1.0, 2.0])
    action = np.array([0.5, -0.25])
    problem = SafeKinderTrajOptProblem(
        env=FakeEnv(reward=-3.0),
        initial_state=state,
        horizon=1,
    )

    next_state = problem.get_next_state(state, action)
    cost = problem.get_traj_cost(make_one_step_traj(state, next_state, action))

    assert cost == 3.0


def test_known_geometry_assertion_gets_penalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known geometry assertions should become terminated penalty transitions."""
    state = np.array([1.0, 2.0])
    action = np.array([0.5, -0.25])
    problem = SafeKinderTrajOptProblem(
        env=RaisingEnv(AssertionError("invalid geometry")),
        initial_state=state,
        horizon=1,
        invalid_transition_reward=-123.0,
        max_logged_invalid_transitions=0,
    )
    monkeypatch.setattr(
        SafeKinderTrajOptProblem,
        "_is_known_geometry_assertion",
        staticmethod(lambda err: True),
    )

    next_state = problem.get_next_state(state, action)
    cost = problem.get_traj_cost(make_one_step_traj(state, next_state, action))

    assert np.allclose(next_state, state)
    assert next_state is not state
    assert cost == 123.0
    assert problem.num_invalid_transitions == 1


def test_unknown_assertion_is_reraised() -> None:
    """Non-geometry assertions should not be hidden by the safe wrapper."""
    state = np.array([1.0, 2.0])
    action = np.array([0.5, -0.25])
    problem = SafeKinderTrajOptProblem(
        env=RaisingEnv(AssertionError("real bug")),
        initial_state=state,
        horizon=1,
    )

    with pytest.raises(AssertionError, match="real bug"):
        problem.get_next_state(state, action)
