"""MLP delta dynamics model for Motion2D-p0.

Trains f(s_t, a_t) -> delta_robot_state, where:
    s_{t+1}[:9] = s_t[:9] + delta_robot_state
    s_{t+1}[9:] = s_t[9:]  (env state is static)

Usage:
    python scripts/state_wm.py --mode train
    python scripts/state_wm.py --mode eval --checkpoint output/wm.pt
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROBOT_STATE_DIM = 9
DEFAULT_HDF5 = "/home/yixuan/prbench_dir/prpl-mono/prbench-models/datasets/motion2d_p0.hdf5"


class MLPDynamics(nn.Module):
    """Predicts delta_robot_state given (full_state, action)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, ROBOT_STATE_DIM),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class Normalizer:
    def __init__(self, data: np.ndarray):
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std  = (data.std(axis=0) + 1e-8).astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def load_transitions(hdf5_path: str):
    """Build (s_t, a_t, delta_robot) tuples from all episodes."""
    states, actions, deltas = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(f["data"].keys())
        print(f"Loading {len(keys)} episodes...")
        for key in keys:
            ep    = f["data"][key]
            robot = ep["obs/robot_state"][:]          # (T, 9)
            env   = ep["obs/env_state"][:]            # (T, env_dim)
            acts  = ep["actions"][:]                  # (T, action_dim)
            full  = np.concatenate([robot, env], -1)  # (T, state_dim)
            delta = robot[1:] - robot[:-1]            # (T-1, 9)
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
):
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
    model  = MLPDynamics(states.shape[1], actions.shape[1])
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0.0
        for sb, ab, db in loader:
            loss = nn.functional.mse_loss(model(sb, ab), db)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(sb)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  loss={total/len(s):.6f}")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = Path(output_dir) / "wm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "state_dim":   states.shape[1],
            "action_dim":  actions.shape[1],
            "s_norm": {"mean": s_norm.mean, "std": s_norm.std},
            "a_norm": {"mean": a_norm.mean, "std": a_norm.std},
            "d_norm": {"mean": d_norm.mean, "std": d_norm.std},
        },
        ckpt_path,
    )
    print(f"Checkpoint saved → {ckpt_path}")


def eval_rollout(hdf5_path: str, checkpoint: str, num_episodes: int = 5):
    """Multi-step open-loop rollout: compare predicted vs. ground-truth robot state."""
    ckpt  = torch.load(checkpoint, weights_only=False)
    model = MLPDynamics(ckpt["state_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    s_mean, s_std = ckpt["s_norm"]["mean"], ckpt["s_norm"]["std"]
    a_mean, a_std = ckpt["a_norm"]["mean"], ckpt["a_norm"]["std"]
    d_mean, d_std = ckpt["d_norm"]["mean"], ckpt["d_norm"]["std"]

    all_errors = []
    with h5py.File(hdf5_path, "r") as f:
        for key in sorted(f["data"].keys())[:num_episodes]:
            ep    = f["data"][key]
            robot = ep["obs/robot_state"][:]
            env   = ep["obs/env_state"][:]
            acts  = ep["actions"][:]

            pred_robot = robot[0].copy()
            ep_errors  = []
            for t in range(len(robot) - 1):
                full = np.concatenate([pred_robot, env[t]])
                s_in = torch.tensor((full    - s_mean) / s_std, dtype=torch.float32)
                a_in = torch.tensor((acts[t] - a_mean) / a_std, dtype=torch.float32)
                with torch.no_grad():
                    d_pred = model(s_in.unsqueeze(0), a_in.unsqueeze(0)).squeeze(0).numpy()
                pred_robot = pred_robot + d_pred * d_std + d_mean
                ep_errors.append(np.linalg.norm(pred_robot - robot[t + 1]))

            print(
                f"  {key}: mean_err={np.mean(ep_errors):.4f}  "
                f"final_err={ep_errors[-1]:.4f}  steps={len(ep_errors)}"
            )
            all_errors.extend(ep_errors)

    print(f"\nOverall mean rollout error ({num_episodes} episodes): {np.mean(all_errors):.4f}")


def main():
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
