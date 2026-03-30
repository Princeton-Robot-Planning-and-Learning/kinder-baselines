"""Check which state-vector indices are constant across all demo pickle files.

Scans a directory of teleoperated demonstrations (each episode is a numbered
subdirectory containing a single .p pickle file) and reports:
  - Indices that never change across ALL timesteps and ALL episodes (globally
    constant — safe candidates for preserve_indices in the world model).
  - Indices that are constant within each individual episode but may differ
    across episodes (per-episode constant, e.g. randomised static fields).

Expected directory layout (same as produced by kinder teleoperation):
    <demo_dir>/
        0/
            <timestamp>.p
        1/
            <timestamp>.p
        ...

Each pickle file is expected to contain:
    {"observations": list[np.ndarray], "actions": list[np.ndarray], ...}

Usage:
    python scripts/check_index.py --demo_dir kindergarden/demos/Motion2D-p0
    python scripts/check_index.py --demo_dir kindergarden/demos/Motion2D-p0 --tol 1e-6
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def load_episodes(demo_dir: Path) -> list[np.ndarray]:
    """Load all episodes from a directory of pickle demos.

    Args:
        demo_dir: Path to the directory containing numbered episode subdirs.

    Returns:
        List of arrays, one per episode, each shaped (T, state_dim).
    """
    episode_dirs = sorted(
        [d for d in demo_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not episode_dirs:
        raise ValueError(f"No numbered episode directories found in {demo_dir}")

    episodes = []
    for ep_dir in episode_dirs:
        pickle_files = list(ep_dir.glob("*.p"))
        if not pickle_files:
            print(f"  Warning: no pickle file in {ep_dir}, skipping")
            continue
        with open(pickle_files[0], "rb") as f:
            ep_data = pickle.load(f)
        observations = np.array(ep_data["observations"], dtype=np.float32)
        episodes.append(observations)

    return episodes


def check_constant_indices(demo_dir: Path, tol: float = 1e-8) -> None:
    """Analyse which state indices are constant across all demo data.

    Args:
        demo_dir: Path to the directory containing numbered episode subdirs.
        tol: Maximum absolute range (max - min) for an index to be considered
             constant.
    """
    print(f"Loading episodes from {demo_dir} ...")
    episodes = load_episodes(demo_dir)
    if not episodes:
        raise RuntimeError("No episodes loaded.")

    state_dim = episodes[0].shape[1]
    total_steps = sum(ep.shape[0] for ep in episodes)
    print(
        f"Loaded {len(episodes)} episodes, "
        f"{total_steps} total timesteps, state_dim={state_dim}\n"
    )

    # Stack all data for global analysis.
    all_data = np.concatenate(episodes, axis=0)  # (N_total, state_dim)

    # --- Globally constant: same value everywhere ---
    global_range = all_data.max(axis=0) - all_data.min(axis=0)
    globally_constant = np.where(global_range <= tol)[0]

    print(f"=== Globally constant indices (tol={tol}) ===")
    print(f"Count: {len(globally_constant)} / {state_dim}")
    for idx in globally_constant:
        print(f"  [{idx:3d}]  value={all_data[0, idx]:.6g}")
    print()

    # --- Within-episode unchanged: max-min range within each episode is 0 ---
    # For each index, compute the range (max - min) across timesteps within every
    # episode. Take the worst-case (max) range across all episodes. If this is
    # <= tol, the index literally never moved during any episode.
    # Count is always >= globally_constant since globally constant=>within-ep unchanged.
    max_within_range = np.stack(
        [ep.max(axis=0) - ep.min(axis=0) for ep in episodes], axis=0
    ).max(
        axis=0
    )  # (state_dim,)  worst-case within-ep range per index
    within_ep_unchanged = np.where(max_within_range <= tol)[0]

    # Split into sub-categories for clarity.
    globally_only = globally_constant  # already computed above
    randomised_per_ep = np.setdiff1d(within_ep_unchanged, globally_constant)

    ep_first_vals = np.stack(
        [ep[0] for ep in episodes], axis=0
    )  # (num_episodes, state_dim)

    print(
        f"=== Within-episode unchanged (max within-ep range <= {tol})  "
        f"[count: {len(within_ep_unchanged)} / {state_dim}] ==="
    )
    print("  (use these as preserve_indices in the world model)\n")

    print(
        f"  -- Globally constant (same value in every ep)  "
        f"[{len(globally_only)}] --"
    )
    for idx in globally_only:
        print(f"    [{idx:3d}]  value={all_data[0, idx]:.6g}")

    print()
    print(
        f"  -- Randomised per episode, but unchanged within ep  "
        f"[{len(randomised_per_ep)}] --"
    )
    for idx in randomised_per_ep:
        vals = ep_first_vals[:, idx]
        print(
            f"    [{idx:3d}]  "
            f"across-ep min={vals.min():.6g}  max={vals.max():.6g}  "
            f"unique={len(np.unique(np.round(vals, 6)))}  "
            f"worst-ep-range={max_within_range[idx]:.2e}"
        )
    print()

    # --- Changing indices ---
    changing = np.setdiff1d(np.arange(state_dim), within_ep_unchanged)

    print("=== Summary ===")
    print(
        f"Within-ep unchanged (all)  [{len(within_ep_unchanged):3d}] : "
        f"{sorted(within_ep_unchanged.tolist())}"
    )
    print(
        f"  Globally constant        [{len(globally_only):3d}] : "
        f"{sorted(globally_only.tolist())}"
    )
    print(
        f"  Randomised per episode   [{len(randomised_per_ep):3d}] : "
        f"{sorted(randomised_per_ep.tolist())}"
    )
    print(
        f"Changing indices           [{len(changing):3d}] : "
        f"{sorted(changing.tolist())}"
    )


def main() -> None:
    """Main function to check the constant indices."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo_dir",
        required=True,
        help="Directory containing numbered episode subdirs with .p pickle files.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="Tolerance for treating an index as constant (default: 1e-8).",
    )
    args = parser.parse_args()
    check_constant_indices(Path(args.demo_dir), tol=args.tol)


if __name__ == "__main__":
    main()
