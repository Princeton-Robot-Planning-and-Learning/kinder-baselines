"""Tests for the random-shooting planning utilities."""

import numpy as np
import pytest

from kinder_mbrl.models import MLPDynamics, TerminationClassifier
from kinder_mbrl.planning import state_cost, wm_get_next_state, wm_get_termination_prob


@pytest.fixture(name="model_and_norms")
def model_and_norms_fixture():
    """Return a small two-head MLPDynamics and identity-like norms for testing."""
    robot_dim, env_dim, action_dim = 9, 3, 4
    state_dim = robot_dim + env_dim
    model = MLPDynamics(state_dim, action_dim, robot_dim=robot_dim, env_dim=env_dim)
    model.eval()
    norms = {
        "s_mean": np.zeros(state_dim, dtype=np.float32),
        "s_std": np.ones(state_dim, dtype=np.float32),
        "a_mean": np.zeros(action_dim, dtype=np.float32),
        "a_std": np.ones(action_dim, dtype=np.float32),
        "dr_mean": np.zeros(robot_dim, dtype=np.float32),
        "dr_std": np.ones(robot_dim, dtype=np.float32),
        "de_mean": np.zeros(env_dim, dtype=np.float32),
        "de_std": np.ones(env_dim, dtype=np.float32),
    }
    return model, norms


@pytest.fixture(name="term_model_and_norms")
def term_model_and_norms_fixture():
    """Return a small TerminationClassifier and identity-like norms for testing."""
    state_dim = 12
    model = TerminationClassifier(state_dim)
    model.eval()
    norms = {
        "s_mean": np.zeros(state_dim, dtype=np.float32),
        "s_std": np.ones(state_dim, dtype=np.float32),
    }
    return model, norms


def test_wm_get_termination_prob_range(term_model_and_norms):
    """wm_get_termination_prob should return a float in [0, 1]."""
    model, norms = term_model_and_norms
    state_dim = norms["s_mean"].shape[0]
    next_state = np.zeros(state_dim, dtype=np.float32)
    prob = wm_get_termination_prob(next_state, model, norms)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_wm_get_termination_prob_does_not_mutate_input(term_model_and_norms):
    """wm_get_termination_prob should not modify the input next_state array."""
    model, norms = term_model_and_norms
    state_dim = norms["s_mean"].shape[0]
    next_state = np.ones(state_dim, dtype=np.float32)
    next_state_copy = next_state.copy()
    wm_get_termination_prob(next_state, model, norms)
    np.testing.assert_array_equal(next_state, next_state_copy)


def test_state_cost_at_goal():
    """Cost should be zero when robot is exactly at the target."""
    state = np.zeros(12, dtype=np.float32)
    state[:2] = [1.0, 2.0]
    state[9:11] = [1.0, 2.0]
    assert state_cost(state) == pytest.approx(0.0)


def test_state_cost_positive():
    """Cost should be positive when robot is not at the target."""
    state = np.zeros(12, dtype=np.float32)
    state[:2] = [0.0, 0.0]
    state[9:11] = [3.0, 4.0]
    assert state_cost(state) == pytest.approx(5.0)


def test_wm_get_next_state_shape(model_and_norms):
    """wm_get_next_state should return a state of the same shape."""
    model, norms = model_and_norms
    state_dim = norms["s_mean"].shape[0]
    action_dim = norms["a_mean"].shape[0]
    state = np.zeros(state_dim, dtype=np.float32)
    action = np.zeros(action_dim, dtype=np.float32)
    next_state = wm_get_next_state(state, action, model, norms)
    assert next_state.shape == state.shape


def test_wm_get_next_state_does_not_mutate_input(model_and_norms):
    """wm_get_next_state should not modify the input state array."""
    model, norms = model_and_norms
    state_dim = norms["s_mean"].shape[0]
    action_dim = norms["a_mean"].shape[0]
    state = np.ones(state_dim, dtype=np.float32)
    state_copy = state.copy()
    action = np.zeros(action_dim, dtype=np.float32)
    wm_get_next_state(state, action, model, norms)
    np.testing.assert_array_equal(state, state_copy)
