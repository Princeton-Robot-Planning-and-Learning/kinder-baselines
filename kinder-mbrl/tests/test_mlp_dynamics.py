"""Tests for MLPDynamics and Normalizer."""

import numpy as np
import pytest
import torch

from kinder_mbrl.models import MLPDynamics, Normalizer


@pytest.fixture(name="dims")
def dims_fixture():
    """Return (state_dim, action_dim) used across tests."""
    return 12, 4


def test_mlp_dynamics_output_shape(dims):
    """Forward pass produces the expected output shape."""
    state_dim, action_dim = dims
    model = MLPDynamics(state_dim, action_dim, output_dim=state_dim)
    batch_size = 8
    state = torch.zeros(batch_size, state_dim)
    action = torch.zeros(batch_size, action_dim)
    delta = model(state, action)
    assert delta.shape == (batch_size, state_dim)


def test_mlp_dynamics_deterministic(dims):
    """Same inputs always produce the same output."""
    state_dim, action_dim = dims
    model = MLPDynamics(state_dim, action_dim, output_dim=state_dim)
    model.eval()
    state = torch.randn(4, state_dim)
    action = torch.randn(4, action_dim)
    with torch.no_grad():
        out1 = model(state, action)
        out2 = model(state, action)
    assert torch.allclose(out1, out2)


def test_normalizer_round_trip():
    """Normalizing then denormalizing recovers the original values."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((100, 8)).astype(np.float32)
    norm = Normalizer(data)
    normalized = norm.normalize(data)
    recovered = norm.denormalize(normalized)
    np.testing.assert_allclose(recovered, data, atol=1e-5)


def test_normalizer_zero_mean_unit_std():
    """Normalized training data should be approximately zero-mean unit-std."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((1000, 5)).astype(np.float32) * 3 + 7
    norm = Normalizer(data)
    normalized = norm.normalize(data)
    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(5), atol=1e-5)
    np.testing.assert_allclose(normalized.std(axis=0), np.ones(5), atol=1e-2)
