"""MLP delta dynamics model and per-feature normalizer."""

from typing import Tuple

import numpy as np
import torch
from torch import nn


class MLPDynamics(nn.Module):
    """MLP that separately predicts delta robot state and delta env state.

    The network uses a shared trunk that ingests the concatenation of the
    normalized state and action vectors, then branches into two independent
    output heads: one for the robot-state delta and one for the env-state
    delta. This allows each head to specialize in its own dynamics regime
    (kinematic/actuator changes vs. object/scene changes).

    Next-state prediction:
        delta_robot, delta_env = model(state, action)
        s_{t+1} = s_t + concat(delta_robot, delta_env)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        robot_dim: int,
        env_dim: int,
        hidden_dim: int = 256,
    ):
        """Initialize the two-head MLP dynamics model.

        Args:
            state_dim: Dimensionality of the full state vector (robot + env).
            action_dim: Dimensionality of the action vector.
            robot_dim: Number of robot-state dimensions; the robot head
                predicts a delta of this size.
            env_dim: Number of env-state dimensions; the env head predicts
                a delta of this size.
            hidden_dim: Width of each hidden layer in the shared trunk.
        """
        super().__init__()
        self.robot_dim = robot_dim
        self.env_dim = env_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.robot_head = nn.Linear(hidden_dim, robot_dim)
        self.env_head = nn.Linear(hidden_dim, env_dim)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict separate robot and env state deltas for a batch of inputs.

        Args:
            state: Normalized state tensor of shape (B, state_dim).
            action: Normalized action tensor of shape (B, action_dim).

        Returns:
            A tuple (delta_robot, delta_env) where delta_robot has shape
            (B, robot_dim) and delta_env has shape (B, env_dim). Both are
            in normalized space.
        """
        features = self.trunk(torch.cat([state, action], dim=-1))
        return self.robot_head(features), self.env_head(features)


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
