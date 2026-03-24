"""Tests for the random-shooting planning utilities."""

import numpy as np
import pytest

from kinder_mbrl.models import MLPDynamics
from kinder_mbrl.planning import state_cost, wm_get_next_state


@pytest.fixture(name="model_and_norms")
def model_and_norms_fixture():
    """Return a small MLPDynamics and identity-like norms for testing."""
    state_dim, action_dim = 12, 4
    model = MLPDynamics(state_dim, action_dim, output_dim=state_dim)
    model.eval()
    norms = {
        "s_mean": np.zeros(state_dim, dtype=np.float32),
        "s_std": np.ones(state_dim, dtype=np.float32),
        "a_mean": np.zeros(action_dim, dtype=np.float32),
        "a_std": np.ones(action_dim, dtype=np.float32),
        "d_mean": np.zeros(state_dim, dtype=np.float32),
        "d_std": np.ones(state_dim, dtype=np.float32),
    }
    return model, norms


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
