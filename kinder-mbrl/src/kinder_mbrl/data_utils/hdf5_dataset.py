"""HDF5 dataset loading for model-based RL training."""

from typing import Tuple

import h5py  # type: ignore
import numpy as np


def load_transitions(
    hdf5_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load all transitions from an HDF5 demo file.

    Reads every episode, concatenates robot and env observations into a single
    full state vector, and computes per-step state deltas.

    Args:
        hdf5_path: Path to the HDF5 file produced by demos_to_hdf5.py.
            Each episode must contain obs/robot_state, obs/env_state,
            and actions datasets.

    Returns:
        A tuple (states, actions, deltas, robot_dim) where states, actions, and
        deltas are float32 arrays of shape (N, D) with N being the total number
        of transitions across all episodes, and robot_dim is the number of robot
        state dimensions. Slicing deltas[:, :robot_dim] gives the robot delta
        and deltas[:, robot_dim:] gives the env delta.
        states    — full state at time t (robot + env concatenated)
        actions   — action at time t
        deltas    — full state delta: state[t+1] - state[t]
        robot_dim — number of robot-state dimensions in the state vector
    """
    states, actions, deltas = [], [], []
    robot_dim: int = 0
    with h5py.File(hdf5_path, "r") as file_handle:
        keys = sorted(file_handle["data"].keys())
        print(f"Loading {len(keys)} episodes...")
        for key in keys:
            ep = file_handle["data"][key]
            robot = ep["obs/robot_state"][:]
            env = ep["obs/env_state"][:]
            acts = ep["actions"][:]
            robot_dim = robot.shape[-1]
            full = np.concatenate([robot, env], -1)
            delta = full[1:] - full[:-1]
            states.append(full[:-1])
            actions.append(acts[:-1])
            deltas.append(delta)
    return (
        np.concatenate(states).astype(np.float32),
        np.concatenate(actions).astype(np.float32),
        np.concatenate(deltas).astype(np.float32),
        robot_dim,
    )
