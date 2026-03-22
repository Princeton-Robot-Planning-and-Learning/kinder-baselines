"""MLP delta dynamics model for Motion2D-p0.

Trains f(s_t, a_t) -> delta_full_state, where:
    s_{t+1} = s_t + delta_full_state

Usage:
    python scripts/state_wm.py --mode train
    python scripts/state_wm.py --mode eval --checkpoint output/wm.pt
"""

import argparse
import os
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_HDF5 = (
    "/home/yixuan/prbench_dir/prpl-mono/prbench-models/datasets/motion2d_p0.hdf5"
)


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


def load_transitions(hdf5_path: str):
    """Load all transitions from an HDF5 demo file.

    Reads every episode, concatenates robot and env observations into a single
    full state vector, and computes per-step state deltas.

    Args:
        hdf5_path: Path to the HDF5 file produced by demos_to_hdf5.py.
            Each episode must contain obs/robot_state, obs/env_state,
            and actions datasets.

    Returns:
        A tuple (states, actions, deltas), each a float32 array of shape
        (N, D) where N is the total number of transitions across all episodes.
        states  — full state at time t (robot + env concatenated)
        actions — action at time t
        deltas  — full state delta: state[t+1] - state[t]
    """
    states, actions, deltas = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys())
        print(f"Loading {len(keys)} episodes...")
        for key in keys:
            ep = f["data"][key]
            robot = ep["obs/robot_state"][:]  # (T, 9)
            env = ep["obs/env_state"][:]  # (T, env_dim)
            acts = ep["actions"][:]  # (T, action_dim)
            full = np.concatenate([robot, env], -1)  # (T, state_dim)
            delta = full[1:] - full[:-1]  # (T-1, state_dim)
            states.append(full[:-1])
            actions.append(acts[:-1])
            deltas.append(delta)
    return (
        np.concatenate(states).astype(np.float32),
        np.concatenate(actions).astype(np.float32),
        np.concatenate(deltas).astype(np.float32),
    )


def train(
    hdf5_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> None:
    """Train an MLPDynamics model and save the checkpoint.

    Loads transitions from the HDF5 file, fits per-feature normalizers for
    states, actions, and deltas, then trains the model with MSE loss on
    normalized delta predictions. Prints loss every 10 epochs and saves the
    model weights together with normalizer statistics.

    Args:
        hdf5_path: Path to the HDF5 demo file.
        output_dir: Directory where the checkpoint (wm.pt) will be saved.
        epochs: Number of full passes over the training data.
        batch_size: Number of transitions per mini-batch.
        lr: Adam learning rate.
    """
    states, actions, deltas = load_transitions(hdf5_path)
    print(
        f"Transitions: {len(states)}  |  "
        f"state_dim={states.shape[1]}  action_dim={actions.shape[1]}"
    )

    s_norm = Normalizer(states)
    a_norm = Normalizer(actions)
    d_norm = Normalizer(deltas)

    s = torch.tensor(s_norm.normalize(states))
    a = torch.tensor(a_norm.normalize(actions))
    d = torch.tensor(d_norm.normalize(deltas))

    loader = DataLoader(TensorDataset(s, a, d), batch_size=batch_size, shuffle=True)
    model = MLPDynamics(states.shape[1], actions.shape[1], output_dim=states.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0.0
        for sb, ab, db in loader:
            loss = nn.functional.mse_loss(model(sb, ab), db)
            opt.zero_grad()
            loss.backward()  # type: ignore
            opt.step()
            total += loss.item() * len(sb)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  loss={total/len(s):.6f}")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = Path(output_dir) / "wm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "state_dim": states.shape[1],
            "action_dim": actions.shape[1],
            "output_dim": states.shape[1],
            "s_norm": {"mean": s_norm.mean, "std": s_norm.std},
            "a_norm": {"mean": a_norm.mean, "std": a_norm.std},
            "d_norm": {"mean": d_norm.mean, "std": d_norm.std},
        },
        ckpt_path,
    )
    print(f"Checkpoint saved → {ckpt_path}")


def eval_rollout(hdf5_path: str, checkpoint: str, num_episodes: int = 5) -> None:
    """Evaluate the world model via open-loop multi-step rollouts.

    For each episode, starts from the ground-truth initial state and rolls out
    using only model predictions (no simulator corrections). Reports the L2
    error between the predicted and ground-truth full state at each step.

    Args:
        hdf5_path: Path to the HDF5 demo file used for evaluation.
        checkpoint: Path to the .pt checkpoint saved by train().
        num_episodes: Number of episodes to evaluate.
    """
    ckpt = torch.load(checkpoint, weights_only=False)
    model = MLPDynamics(ckpt["state_dim"], ckpt["action_dim"], ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    s_mean, s_std = ckpt["s_norm"]["mean"], ckpt["s_norm"]["std"]
    a_mean, a_std = ckpt["a_norm"]["mean"], ckpt["a_norm"]["std"]
    d_mean, d_std = ckpt["d_norm"]["mean"], ckpt["d_norm"]["std"]

    all_errors = []
    with h5py.File(hdf5_path, "r") as f:
        for key in sorted(f["data"].keys())[:num_episodes]:
            ep = f["data"][key]
            robot = ep["obs/robot_state"][:]
            env = ep["obs/env_state"][:]
            acts = ep["actions"][:]

            pred_state = np.concatenate([robot[0], env[0]])  # full state at t=0
            ep_errors = []
            for t in range(len(robot) - 1):
                s_in = torch.tensor((pred_state - s_mean) / s_std, dtype=torch.float32)
                a_in = torch.tensor((acts[t] - a_mean) / a_std, dtype=torch.float32)
                with torch.no_grad():
                    d_pred = (
                        model(s_in.unsqueeze(0), a_in.unsqueeze(0)).squeeze(0).numpy()
                    )
                pred_state = pred_state + d_pred * d_std + d_mean
                gt_state = np.concatenate([robot[t + 1], env[t + 1]])
                ep_errors.append(np.linalg.norm(pred_state - gt_state))

            print(
                f"  {key}: mean_err={np.mean(ep_errors):.4f}  "
                f"final_err={ep_errors[-1]:.4f}  steps={len(ep_errors)}"
            )
            all_errors.extend(ep_errors)

    print(
        f"\nOverall mean rollout error ({num_episodes} episodes): {np.mean(all_errors):.4f}"  # pylint: disable=line-too-long
    )


def main() -> None:
    """Parse arguments and dispatch to train() or eval_rollout()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", default=DEFAULT_HDF5)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--checkpoint", default="output/wm.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_eval_episodes", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "train":
        train(args.hdf5_path, args.output_dir, args.epochs, args.batch_size, args.lr)
    else:
        eval_rollout(args.hdf5_path, args.checkpoint, args.num_eval_episodes)


if __name__ == "__main__":
    main()
