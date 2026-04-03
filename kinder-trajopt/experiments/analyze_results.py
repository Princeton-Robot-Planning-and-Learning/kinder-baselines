#!/usr/bin/env python3
"""Script to analyze experimental results from multi-run Hydra experiments.

Usage:
    python analyze_results.py <log_dir>

Example:
    python experiments/analyze_results.py logs/2026-03-26/01-29-47
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml  # type: ignore[import-untyped]


def load_run_data(run_dir: Path) -> Optional[Dict]:
    """Load configuration and results from a single run directory."""
    config_path = run_dir / "config.yaml"
    results_path = run_dir / "results.csv"

    if not config_path.exists() or not results_path.exists():
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_df = pd.read_csv(results_path)

    return {
        "env_name": config["env"]["env_id"],
        "seed": config["seed"],
        "horizon": config.get("horizon", "unknown"),
        "num_rollouts": config.get("num_rollouts", "unknown"),
        "noise_fraction": config.get("noise_fraction", "unknown"),
        "num_control_points": config.get("num_control_points", "unknown"),
        "warm_start": config.get("warm_start", "unknown"),
        "replan_interval": config["env"].get("replan_interval", "unknown"),
        "results": results_df,
    }


def analyze_results(log_dir: Path) -> pd.DataFrame:
    """Analyze all runs in the log directory and compute aggregated metrics."""
    all_data = []

    run_dirs = [d for d in log_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs.sort(key=lambda x: int(x.name))

    if not run_dirs:
        print(f"No run directories found in {log_dir}")
        return pd.DataFrame()

    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        data = load_run_data(run_dir)
        if data is None:
            print(f"Warning: Skipping {run_dir.name} - missing files")
            continue

        results = data["results"]
        for _, row in results.iterrows():
            row_data = {
                "env_name": data["env_name"],
                "horizon": data["horizon"],
                "num_rollouts": data["num_rollouts"],
                "noise_fraction": data["noise_fraction"],
                "num_control_points": data["num_control_points"],
                "warm_start": data["warm_start"],
                "replan_interval": data["replan_interval"],
                "seed": data["seed"],
                "eval_episode": row["eval_episode"],
                "success": row["success"],
                "planning_time": row["planning_time"],
                "execution_time": row["execution_time"],
                "steps": row["steps"],
            }
            if "reward" in row:
                row_data["reward"] = row["reward"]
            if row["planning_time"] != 0.0:
                all_data.append(row_data)

    if not all_data:
        print("No data loaded from runs")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    group_cols = ["env_name"]
    for col in [
        "horizon",
        "num_rollouts",
        "noise_fraction",
        "num_control_points",
        "warm_start",
        "replan_interval",
    ]:
        if df[col].nunique() > 1:
            group_cols.append(col)

    agg_dict = {
        "success": ["mean", "std", "count"],
        "planning_time": ["mean", "std"],
        "execution_time": ["mean", "std"],
        "steps": "mean",
    }

    has_reward = "reward" in df.columns
    if has_reward:
        agg_dict["reward"] = ["mean", "std"]

    grouped = df.groupby(group_cols).agg(agg_dict)

    col_names = [
        "solve_rate",
        "solve_rate_std",
        "num_runs",
        "avg_planning_time",
        "planning_time_std",
        "avg_execution_time",
        "execution_time_std",
        "avg_steps",
    ]
    if has_reward:
        col_names.extend(["avg_reward", "avg_reward_std"])

    grouped.columns = col_names
    grouped = grouped.reset_index()

    if has_reward:
        successful_reward = (
            df[df["success"] == True]  # pylint: disable=singleton-comparison
            .groupby(group_cols)["reward"]
            .agg(["mean", "std"])
            .reset_index()
        )
        successful_reward.columns = list(group_cols) + [
            "avg_reward_successful",
            "reward_successful_std",
        ]
        grouped = grouped.merge(successful_reward, on=group_cols, how="left")

    grouped = grouped.sort_values(group_cols)
    return grouped


def format_table(df: pd.DataFrame) -> str:
    """Format the results as a nice table with mean ± std for key metrics."""
    if df.empty:
        return "No results to display"

    has_reward = "avg_reward" in df.columns

    max_env_len = max(len(str(env)) for env in df["env_name"])
    env_col_width = max(max_env_len + 2, len("Environment"))

    lines = []
    header_parts = ["Environment"]
    config_cols = []
    for col in [
        "horizon",
        "num_rollouts",
        "noise_fraction",
        "num_control_points",
        "warm_start",
        "replan_interval",
    ]:
        if col in df.columns:
            config_cols.append(col)
            display_name = {
                "horizon": "Horizon",
                "num_rollouts": "Rollouts",
                "noise_fraction": "Noise",
                "num_control_points": "Ctrl Pts",
                "warm_start": "Warm Start",
                "replan_interval": "Replan",
            }.get(col, col)
            header_parts.append(display_name)

    header_parts.extend(
        [
            "Solve Rate (mean±std)",
            "Planning Time/s (mean±std)",
            "Exec Time/s (mean±std)",
            "Avg Steps",
        ]
    )
    if has_reward:
        header_parts.extend(["Avg Reward (mean±std)", "Avg Reward Success (mean±std)"])

    widths = [env_col_width] + [12] * len(config_cols) + [25, 30, 25, 12]
    if has_reward:
        widths.extend([25, 30])

    header_line_length = sum(widths) + len(widths)
    lines.append("=" * header_line_length)
    header = ""
    for part, width in zip(header_parts, widths):
        header += f"{part:<{width}} "
    lines.append(header.rstrip())
    lines.append("=" * header_line_length)

    for _, row in df.iterrows():
        parts = [row["env_name"]]
        for col in config_cols:
            parts.append(str(row[col]))
        parts.append(f"{row['solve_rate']:.1%} ± {row['solve_rate_std']:.1%}")
        parts.append(f"{row['avg_planning_time']:.2f} ± {row['planning_time_std']:.2f}")
        parts.append(
            f"{row['avg_execution_time']:.4f} ± {row['execution_time_std']:.4f}"
        )
        parts.append(f"{row['avg_steps']:.1f}")

        if has_reward:
            parts.append(f"{row['avg_reward']:.3f} ± {row['avg_reward_std']:.3f}")
            avg_reward_successful = (
                f"{row['avg_reward_successful']:.3f} ± {row['reward_successful_std']:.3f}" # pylint: disable=line-too-long
                if pd.notna(row["avg_reward_successful"])
                else "N/A"
            )
            parts.append(avg_reward_successful)

        row_str = ""
        for part, width in zip(parts, widths):
            row_str += f"{part:<{width}} "
        lines.append(row_str.rstrip())

    lines.append("=" * header_line_length)
    lines.append(f"\nTotal configurations: {len(df)}")
    lines.append(f"Runs per configuration: {int(df['num_runs'].iloc[0])}")
    return "\n".join(lines)


def main() -> None:
    """Main function to analyze experimental results."""
    parser = argparse.ArgumentParser(
        description="Analyze experimental results from Hydra multi-run experiments"
    )
    parser.add_argument(
        "log_dir", type=str, help="Path to the log directory containing run results"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Optional: Save results to CSV file"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        sys.exit(1)

    print(f"Analyzing results from: {log_dir}")
    print()

    results_df = analyze_results(log_dir)

    if results_df.empty:
        print("No results found")
        sys.exit(1)

    print(format_table(results_df))

    if args.csv:
        csv_path = Path(args.csv)
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
