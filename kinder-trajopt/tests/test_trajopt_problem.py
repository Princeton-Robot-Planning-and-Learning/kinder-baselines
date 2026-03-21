"""Tests for the KinderTrajOptProblem wrapper."""

import kinder
import numpy as np
from prpl_utils.trajopt.trajopt_problem import TrajOptTraj

from kinder_trajopt.trajopt_problem import KinderTrajOptProblem

kinder.register_all_environments()


def test_problem_properties():
    """Problem exposes correct spaces and initial state."""
    env = kinder.make("kinder/Motion2D-p0-v0", allow_state_access=True)
    obs, _ = env.reset(seed=42)
    problem = KinderTrajOptProblem(env, obs, horizon=10)

    assert problem.horizon == 10
    assert problem.action_space.shape == env.action_space.shape
    assert problem.state_space.shape == env.observation_space.shape
    assert np.allclose(problem.initial_state, obs)
    env.close()


def test_get_next_state_deterministic():
    """get_next_state should be deterministic for the same inputs."""
    env = kinder.make("kinder/Motion2D-p0-v0", allow_state_access=True)
    obs, _ = env.reset(seed=42)
    problem = KinderTrajOptProblem(env, obs, horizon=10)

    action = env.action_space.sample()
    result1 = problem.get_next_state(obs, action)
    result2 = problem.get_next_state(obs, action)
    assert np.allclose(result1, result2)
    env.close()


def test_get_traj_cost_positive():
    """Cost should be positive (negated negative reward)."""
    env = kinder.make("kinder/Motion2D-p0-v0", allow_state_access=True)
    obs, _ = env.reset(seed=42)
    problem = KinderTrajOptProblem(env, obs, horizon=5)

    state = obs
    states = [state]
    actions = []
    for _ in range(5):
        action = env.action_space.sample()
        actions.append(action)
        state = problem.get_next_state(state, action)
        states.append(state)

    traj = TrajOptTraj(np.array(states), np.array(actions))
    cost = problem.get_traj_cost(traj)
    assert cost > 0
    env.close()
