"""Random-shooting MPC planning utilities.

Provides helpers for loading a trained world model and performing one-step
next-state prediction, used by the random-shooting MPC experiment script.
"""

from typing import Tuple

import numpy as np
import torch

from kinder_mbrl.models.mlp_dynamics import MLPDynamics


def state_cost(state: np.ndarray) -> float:
    """Compute the planning cost for a given state.

    Returns the Euclidean distance between the robot's (x, y) position
    and the center of the target region.

    Args:
        state: Full environment state vector. Indices 0:2 are robot (x, y)
            and indices 9:11 are target (x, y).

    Returns:
        Scalar cost value; lower is better.
    """
    robot_xy = state[:2]
    target_xy = state[9:11]
    return float(np.linalg.norm(robot_xy - target_xy))


def load_world_model(checkpoint: str) -> Tuple[MLPDynamics, dict]:
    """Load a trained MLPDynamics model and its normalizers from a checkpoint.

    Args:
        checkpoint: Path to the .pt file saved by the training script.

    Returns:
        A tuple (model, norms) where model is the MLPDynamics instance in eval
        mode and norms is a dict containing the mean/std arrays for states,
        actions, and deltas.
    """
    ckpt = torch.load(checkpoint, weights_only=False)
    model = MLPDynamics(ckpt["state_dim"], ckpt["action_dim"], ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norms = {
        "s_mean": ckpt["s_norm"]["mean"],
        "s_std": ckpt["s_norm"]["std"],
        "a_mean": ckpt["a_norm"]["mean"],
        "a_std": ckpt["a_norm"]["std"],
        "d_mean": ckpt["d_norm"]["mean"],
        "d_std": ckpt["d_norm"]["std"],
    }
    return model, norms


def wm_get_next_state(
    state: np.ndarray,
    action: np.ndarray,
    model: MLPDynamics,
    norms: dict,
) -> np.ndarray:
    """Predict the next full state using the learned world model.

    Normalizes (state, action), runs a forward pass to predict the state
    delta, denormalizes the delta, and adds it to the current state.

    Args:
        state: Current full state vector (robot + env concatenated).
        action: Action vector to apply.
        model: Trained MLPDynamics instance.
        norms: Normalizer statistics dict returned by load_world_model.

    Returns:
        Predicted next full state vector.
    """
    s_in = torch.tensor(
        (state - norms["s_mean"]) / norms["s_std"], dtype=torch.float32
    )
    a_in = torch.tensor(
        (action - norms["a_mean"]) / norms["a_std"], dtype=torch.float32
    )
    with torch.no_grad():
        d_pred = model(s_in.unsqueeze(0), a_in.unsqueeze(0)).squeeze(0).numpy()
    delta = d_pred * norms["d_std"] + norms["d_mean"]
    next_state = state.copy()
    next_state += delta
    return next_state
