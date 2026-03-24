"""HDF5 dataset loading for model-based RL training."""

from typing import Tuple

import h5py  # type: ignore
import numpy as np


def load_transitions(hdf5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    with h5py.File(hdf5_path, "r") as file_handle:
        keys = sorted(file_handle["data"].keys())
        print(f"Loading {len(keys)} episodes...")
        for key in keys:
            ep = file_handle["data"][key]
            robot = ep["obs/robot_state"][:]
            env = ep["obs/env_state"][:]
            acts = ep["actions"][:]
            full = np.concatenate([robot, env], -1)
            delta = full[1:] - full[:-1]
            states.append(full[:-1])
            actions.append(acts[:-1])
            deltas.append(delta)
    return (
        np.concatenate(states).astype(np.float32),
        np.concatenate(actions).astype(np.float32),
        np.concatenate(deltas).astype(np.float32),
    )
