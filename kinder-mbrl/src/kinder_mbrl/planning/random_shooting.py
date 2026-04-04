"""Random-shooting MPC planning utilities.

Provides helpers for loading a trained world model and performing one-step next-state
prediction, used by the random-shooting MPC experiment script.
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
        A tuple (model, norms) where model is the two-head MLPDynamics instance
        in eval mode and norms is a dict containing the mean/std arrays for
        states, actions, robot deltas, and env deltas.
    """
    ckpt = torch.load(checkpoint, weights_only=False)
    model = MLPDynamics(
        ckpt["state_dim"], ckpt["action_dim"], ckpt["robot_dim"], ckpt["env_dim"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norms = {
        "s_mean": ckpt["s_norm"]["mean"],
        "s_std": ckpt["s_norm"]["std"],
        "a_mean": ckpt["a_norm"]["mean"],
        "a_std": ckpt["a_norm"]["std"],
        "dr_mean": ckpt["dr_norm"]["mean"],
        "dr_std": ckpt["dr_norm"]["std"],
        "de_mean": ckpt["de_norm"]["mean"],
        "de_std": ckpt["de_norm"]["std"],
    }
    return model, norms


def wm_get_next_state(
    state: np.ndarray,
    action: np.ndarray,
    model: MLPDynamics,
    norms: dict,
) -> np.ndarray:
    """Predict the next full state using the learned world model.

    Normalizes (state, action), runs a forward pass through both heads to
    predict the robot and env state deltas separately, denormalizes each
    delta with its own statistics, concatenates them, and adds the result to
    the current state.

    Args:
        state: Current full state vector (robot + env concatenated).
        action: Action vector to apply.
        model: Trained two-head MLPDynamics instance.
        norms: Normalizer statistics dict returned by load_world_model.

    Returns:
        Predicted next full state vector.
    """
    s_in = torch.tensor((state - norms["s_mean"]) / norms["s_std"], dtype=torch.float32)
    a_in = torch.tensor(
        (action - norms["a_mean"]) / norms["a_std"], dtype=torch.float32
    )
    with torch.no_grad():
        pred_dr, pred_de = model(s_in.unsqueeze(0), a_in.unsqueeze(0))
    dr = pred_dr.squeeze(0).numpy() * norms["dr_std"] + norms["dr_mean"]
    de = pred_de.squeeze(0).numpy() * norms["de_std"] + norms["de_mean"]
    next_state = state.copy()
    next_state += np.concatenate([dr, de])
    return next_state
