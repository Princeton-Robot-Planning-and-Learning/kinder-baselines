"""MLP delta dynamics model and per-feature normalizer."""

import numpy as np
import torch
from torch import nn


class MLPDynamics(nn.Module):
    """Two-layer MLP that predicts the full state delta given (state, action).

    The network takes the concatenation of the normalized state and action vectors as
    input and outputs a predicted delta for the entire state vector, i.e. delta_s such
    that s_{t+1} = s_t + delta_s.
    """

    def __init__(
        self, state_dim: int, action_dim: int, output_dim: int, hidden_dim: int = 256
    ):
        """Initialize the MLP dynamics model.

        Args:
            state_dim: Dimensionality of the full state vector (robot + env).
            action_dim: Dimensionality of the action vector.
            output_dim: Dimensionality of the predicted delta (usually equal
                to state_dim).
            hidden_dim: Width of each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the state delta for a batch of (state, action) pairs.

        Args:
            state: Normalized state tensor of shape (B, state_dim).
            action: Normalized action tensor of shape (B, action_dim).

        Returns:
            Predicted normalized state delta of shape (B, output_dim).
        """
        return self.net(torch.cat([state, action], dim=-1))


class Normalizer:
    """Per-feature zero-mean unit-variance normalizer.

    Computes mean and std from a reference dataset and applies them to normalize or
    denormalize arrays of the same feature dimension. A small epsilon (1e-8) is added to
    std to avoid division by zero for constant features.
    """

    def __init__(self, data: np.ndarray):
        """Fit the normalizer to a dataset.

        Args:
            data: Array of shape (N, D) used to compute mean and std along
                the sample axis.
        """
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = (data.std(axis=0) + 1e-8).astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply zero-mean unit-variance normalization.

        Args:
            x: Array of shape (..., D).

        Returns:
            Normalized array of the same shape.
        """
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Invert normalization to recover original-scale values.

        Args:
            x: Normalized array of shape (..., D).

        Returns:
            Denormalized array of the same shape.
        """
        return x * self.std + self.mean
