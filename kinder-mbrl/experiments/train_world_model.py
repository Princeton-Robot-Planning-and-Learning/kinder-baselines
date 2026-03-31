"""Train and evaluate a two-head MLP delta dynamics world model.

Trains two independent output heads on a shared trunk:
    delta_robot, delta_env = f(s_t, a_t)
    s_{t+1} = s_t + concat(delta_robot, delta_env)

Each head is trained with its own MSE loss and its own per-feature normalizer,
allowing robot-state and env-state dynamics to be learned independently.

Usage:
    python experiments/train_world_model.py --mode train
    python experiments/train_world_model.py --mode eval --checkpoint output/wm.pt
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import h5py  # type: ignore
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kinder_mbrl.data_utils import load_transitions
from kinder_mbrl.models import MLPDynamics, Normalizer


def train(
    hdf5_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> None:
    """Train a two-head MLPDynamics model and save the checkpoint.

    Loads transitions from the HDF5 file, fits per-feature normalizers for
    states, actions, robot deltas, and env deltas independently, then trains
    the model with separate MSE losses on each normalized delta head. Prints
    the combined loss every 10 epochs and saves the model weights together
    with all normalizer statistics.

    Args:
        hdf5_path: Path to the HDF5 demo file.
        output_dir: Directory where the checkpoint (wm.pt) will be saved.
        epochs: Number of full passes over the training data.
        batch_size: Number of transitions per mini-batch.
        lr: Adam learning rate.
    """
    states, actions, deltas, robot_dim = load_transitions(hdf5_path)
    env_dim = states.shape[1] - robot_dim
    robot_deltas = deltas[:, :robot_dim]
    env_deltas = deltas[:, robot_dim:]
    print(
        f"Transitions: {len(states)}  |  "
        f"state_dim={states.shape[1]}  action_dim={actions.shape[1]}  "
        f"robot_dim={robot_dim}  env_dim={env_dim}"
    )

    s_norm = Normalizer(states)
    a_norm = Normalizer(actions)
    dr_norm = Normalizer(robot_deltas)
    de_norm = Normalizer(env_deltas)

    s_t = torch.tensor(s_norm.normalize(states))
    a_t = torch.tensor(a_norm.normalize(actions))
    dr_t = torch.tensor(dr_norm.normalize(robot_deltas))
    de_t = torch.tensor(de_norm.normalize(env_deltas))

    loader = DataLoader(
        TensorDataset(s_t, a_t, dr_t, de_t), batch_size=batch_size, shuffle=True
    )
    model = MLPDynamics(
        states.shape[1], actions.shape[1], robot_dim=robot_dim, env_dim=env_dim
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total = 0.0
        for s_b, a_b, dr_b, de_b in loader:
            pred_dr, pred_de = model(s_b, a_b)
            loss = nn.functional.mse_loss(pred_dr, dr_b) + nn.functional.mse_loss(
                pred_de, de_b
            )
            opt.zero_grad()
            loss.backward()  # type: ignore
            opt.step()
            total += loss.item() * len(s_b)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}  loss={total/len(s_t):.6f}")

    os.makedirs(output_dir, exist_ok=True)
    task_name = Path(hdf5_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = Path(output_dir) / f"wm_{task_name}_ep{epochs}_{timestamp}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "state_dim": states.shape[1],
            "action_dim": actions.shape[1],
            "robot_dim": robot_dim,
            "env_dim": env_dim,
            "s_norm": {"mean": s_norm.mean, "std": s_norm.std},
            "a_norm": {"mean": a_norm.mean, "std": a_norm.std},
            "dr_norm": {"mean": dr_norm.mean, "std": dr_norm.std},
            "de_norm": {"mean": de_norm.mean, "std": de_norm.std},
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
    model = MLPDynamics(
        ckpt["state_dim"], ckpt["action_dim"], ckpt["robot_dim"], ckpt["env_dim"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    s_mean, s_std = ckpt["s_norm"]["mean"], ckpt["s_norm"]["std"]
    a_mean, a_std = ckpt["a_norm"]["mean"], ckpt["a_norm"]["std"]
    dr_mean, dr_std = ckpt["dr_norm"]["mean"], ckpt["dr_norm"]["std"]
    de_mean, de_std = ckpt["de_norm"]["mean"], ckpt["de_norm"]["std"]

    all_errors = []
    with h5py.File(hdf5_path, "r") as file_handle:
        for key in sorted(file_handle["data"].keys())[:num_episodes]:
            ep = file_handle["data"][key]
            robot = ep["obs/robot_state"][:]
            env = ep["obs/env_state"][:]
            acts = ep["actions"][:]

            pred_state = np.concatenate([robot[0], env[0]])
            ep_errors = []
            for t_idx in range(len(robot) - 1):
                s_in = torch.tensor((pred_state - s_mean) / s_std, dtype=torch.float32)
                a_in = torch.tensor((acts[t_idx] - a_mean) / a_std, dtype=torch.float32)
                with torch.no_grad():
                    pred_dr, pred_de = model(s_in.unsqueeze(0), a_in.unsqueeze(0))
                dr = pred_dr.squeeze(0).numpy() * dr_std + dr_mean
                de = pred_de.squeeze(0).numpy() * de_std + de_mean
                pred_state = pred_state + np.concatenate([dr, de])
                gt_state = np.concatenate([robot[t_idx + 1], env[t_idx + 1]])
                ep_errors.append(np.linalg.norm(pred_state - gt_state))

            print(
                f"  {key}: mean_err={np.mean(ep_errors):.4f}  "
                f"final_err={ep_errors[-1]:.4f}  steps={len(ep_errors)}"
            )
            all_errors.extend(ep_errors)

    print(
        f"\nOverall mean rollout error ({num_episodes} episodes): "
        f"{np.mean(all_errors):.4f}"
    )


def main() -> None:
    """Parse arguments and dispatch to train() or eval_rollout()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", default="../../../prpl-mono/prbench-models/datasets/motion2d_p0.hdf5")
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
