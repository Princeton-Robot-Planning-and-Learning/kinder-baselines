#!/usr/bin/env python3
"""Run experiments to evaluate bilevel planning under noisy observations/actions.

Sweeps observation noise and action noise independently across environments
and seeds. Prints progression tables showing solve rate, cumulative reward,
and planning time at each noise level.

Usage:
    python experiments/run_noisy_experiments.py

    python experiments/run_noisy_experiments.py \
        --envs motion2d-p0 stickbutton2d-b1 \
        --num_eval_episodes 50 \
        --num_seeds 5

    python experiments/run_noisy_experiments.py \
        --obs_noise_stds 0.0 0.01 0.05 0.1 \
        --action_noise_stds 0.0 0.01 0.05 0.1
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Default configuration.
DEFAULT_ENVS = ["motion2d-p0", "stickbutton2d-b1"]
DEFAULT_NUM_EVAL_EPISODES = 50
DEFAULT_NUM_SEEDS = 5
DEFAULT_OBS_NOISE_STDS = [0.0, 0.01, 0.1]
DEFAULT_ACTION_NOISE_STDS = [0.0, 0.01, 0.1]


def _config_dir_name(obs_noise_std: float, action_noise_std: float) -> str:
    """Generate directory name for a noise configuration."""
    return f"obs_{obs_noise_std:.3f}_act_{action_noise_std:.3f}"


def _run_single_config(
    env: str,
    seeds: list[int],
    num_eval_episodes: int,
    obs_noise_std: float,
    action_noise_std: float,
    log_dir: Path,
) -> None:
    """Run a single noise configuration across all seeds via Hydra sweep."""
    seeds_str = ",".join(str(s) for s in seeds)
    cmd = [
        sys.executable,
        "experiments/run_experiment.py",
        "-m",
        f"env={env}",
        f"seed={seeds_str}",
        f"num_eval_episodes={num_eval_episodes}",
        f"obs_noise_std={obs_noise_std}",
        f"action_noise_std={action_noise_std}",
        f"hydra.sweep.dir={log_dir}",
        "hydra.sweep.subdir=${hydra.job.num}",
        "hydra/launcher=joblib",
    ]
    # Run from the kinder-bilevel-planning root so imports resolve.
    project_root = Path(__file__).resolve().parent.parent
    print(f"    $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=project_root)


def _load_sweep_results(sweep_dir: Path) -> dict | None:
    """Load and aggregate results from a Hydra sweep directory.

    Computes per-seed metrics, then reports mean +/- std across seeds.
    """
    if not sweep_dir.exists():
        return None

    seed_dirs = sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name),
    )
    if not seed_dirs:
        return None

    seed_metrics = []
    for seed_dir in seed_dirs:
        results_path = seed_dir / "results.csv"
        if not results_path.exists():
            continue
        df = pd.read_csv(results_path)
        seed_metrics.append(
            {
                "success_rate": df["success"].mean(),
                "avg_reward": df["reward"].mean(),
                "avg_planning_time": df["planning_time"].mean(),
            }
        )

    if not seed_metrics:
        return None

    metrics_df = pd.DataFrame(seed_metrics)
    n = len(metrics_df)
    return {
        "success_rate_mean": metrics_df["success_rate"].mean(),
        "success_rate_std": metrics_df["success_rate"].std(ddof=1) if n > 1 else 0.0,
        "avg_reward_mean": metrics_df["avg_reward"].mean(),
        "avg_reward_std": metrics_df["avg_reward"].std(ddof=1) if n > 1 else 0.0,
        "avg_planning_time_mean": metrics_df["avg_planning_time"].mean(),
        "avg_planning_time_std": (
            metrics_df["avg_planning_time"].std(ddof=1) if n > 1 else 0.0
        ),
        "num_seeds": n,
    }


def _print_progression_table(
    title: str,
    noise_label: str,
    noise_levels: list[float],
    results_per_env: dict[str, list[dict | None]],
) -> None:
    """Print a formatted table showing metric progression across noise levels."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    for env_name, results_list in results_per_env.items():
        print(f"\n  Environment: {env_name}")
        print(f"  {'-' * 76}")
        print(
            f"  {noise_label:<15} {'Solve Rate':<22} {'Avg Reward':<22} "
            f"{'Planning Time (s)':<22}"
        )
        print(f"  {'-' * 76}")

        for noise_std, result in zip(noise_levels, results_list):
            if result is None:
                line = f"  {noise_std:<15.3f} {'N/A':<22} {'N/A':<22} {'N/A':<22}"
            else:
                sr = (
                    f"{result['success_rate_mean']:.3f} ± "
                    f"{result['success_rate_std']:.3f}"
                )
                rw = (
                    f"{result['avg_reward_mean']:.3f} ± "
                    f"{result['avg_reward_std']:.3f}"
                )
                pt = (
                    f"{result['avg_planning_time_mean']:.3f} ± "
                    f"{result['avg_planning_time_std']:.3f}"
                )
                line = f"  {noise_std:<15.3f} {sr:<22} {rw:<22} {pt:<22}"
            print(line)

    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate bilevel planning under noisy observations/actions."
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_ENVS,
        help=f"Environment configs (default: {DEFAULT_ENVS})",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=DEFAULT_NUM_EVAL_EPISODES,
        help=f"Episodes per seed (default: {DEFAULT_NUM_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"Number of seeds (default: {DEFAULT_NUM_SEEDS})",
    )
    parser.add_argument(
        "--obs_noise_stds",
        nargs="+",
        type=float,
        default=DEFAULT_OBS_NOISE_STDS,
        help=f"Observation noise levels (default: {DEFAULT_OBS_NOISE_STDS})",
    )
    parser.add_argument(
        "--action_noise_stds",
        nargs="+",
        type=float,
        default=DEFAULT_ACTION_NOISE_STDS,
        help=f"Action noise levels (default: {DEFAULT_ACTION_NOISE_STDS})",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Custom log directory (default: logs/noisy/<timestamp>)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Save aggregated progression results to CSV",
    )
    args = parser.parse_args()

    # Resolve log directory.
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        base_log_dir = Path(f"logs/noisy/{timestamp}")
    else:
        base_log_dir = Path(args.log_dir)

    seeds = list(range(args.num_seeds))

    # Build unique noise configurations (baseline shared between sweeps).
    obs_configs = [(obs_std, 0.0) for obs_std in args.obs_noise_stds]
    action_configs = [
        (0.0, act_std) for act_std in args.action_noise_stds if act_std > 0
    ]
    all_configs = list(dict.fromkeys(obs_configs + action_configs))

    print("Noisy Wrapper Experiment")
    print(f"  Environments:    {args.envs}")
    print(f"  Seeds:           {seeds}")
    print(f"  Episodes/seed:   {args.num_eval_episodes}")
    print(f"  Obs noise stds:  {args.obs_noise_stds}")
    print(f"  Act noise stds:  {args.action_noise_stds}")
    print(f"  Configs per env: {len(all_configs)}")
    print(f"  Log directory:   {base_log_dir}")

    # --- Run experiments ---
    for env in args.envs:
        print(f"\n[{env}]")
        for obs_std, act_std in all_configs:
            config_name = _config_dir_name(obs_std, act_std)
            config_log_dir = base_log_dir / env / config_name
            print(f"  obs_noise={obs_std:.3f}, action_noise={act_std:.3f}")
            _run_single_config(
                env=env,
                seeds=seeds,
                num_eval_episodes=args.num_eval_episodes,
                obs_noise_std=obs_std,
                action_noise_std=act_std,
                log_dir=config_log_dir,
            )

    # --- Analyze and print results ---
    print("\n\n" + "#" * 80)
    print("  RESULTS")
    print("#" * 80)

    # Observation noise progression.
    obs_results_per_env: dict[str, list[dict | None]] = {}
    for env in args.envs:
        obs_results_per_env[env] = [
            _load_sweep_results(
                base_log_dir / env / _config_dir_name(obs_std, 0.0)
            )
            for obs_std in args.obs_noise_stds
        ]

    _print_progression_table(
        "Observation Noise Sweep (action_noise_std = 0.0)",
        "obs_noise_std",
        args.obs_noise_stds,
        obs_results_per_env,
    )

    # Action noise progression.
    action_results_per_env: dict[str, list[dict | None]] = {}
    for env in args.envs:
        action_results_per_env[env] = [
            _load_sweep_results(
                base_log_dir / env / _config_dir_name(0.0, act_std)
            )
            for act_std in args.action_noise_stds
        ]

    _print_progression_table(
        "Action Noise Sweep (obs_noise_std = 0.0)",
        "action_noise_std",
        args.action_noise_stds,
        action_results_per_env,
    )

    # Optionally save CSV.
    if args.csv:
        rows = []
        for env in args.envs:
            for obs_std, act_std in all_configs:
                result = _load_sweep_results(
                    base_log_dir / env / _config_dir_name(obs_std, act_std)
                )
                if result:
                    rows.append(
                        {
                            "env": env,
                            "obs_noise_std": obs_std,
                            "action_noise_std": act_std,
                            **result,
                        }
                    )
        if rows:
            csv_path = Path(args.csv)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Results saved to: {csv_path}")

    print(f"\nFull logs: {base_log_dir}")
    print("Each noise config subdirectory is compatible with analyze_results.py")


if __name__ == "__main__":
    main()
