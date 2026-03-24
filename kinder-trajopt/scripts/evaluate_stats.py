"""Summarize success rate (mean ± std across seeds) from multirun log dirs.

Usage:
    python scripts/evaluate_stats.py logs/2026-03-23/17-35-38
    python scripts/evaluate_stats.py logs/2026-03-23/17-35-38 logs/2026-03-23/17-58-19
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml  # type: ignore


def load_job(job_dir: Path) -> dict | None:
    """Return {env_id, seed, success_rate} for a single job subdirectory."""
    config_path = job_dir / "config.yaml"
    results_path = job_dir / "results.csv"
    if not config_path.exists() or not results_path.exists():
        return None

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_id = config["env"]["env_id"]
    seed = config["seed"]

    successes = []
    with open(results_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            successes.append(row["success"].strip().lower() == "true")

    if not successes:
        return None

    return {"env_id": env_id, "seed": seed, "success_rate": np.mean(successes)}


def summarize(log_dir: Path) -> dict[str, list[float]]:
    """Map env_id -> list of per-seed success rates for one multirun dir."""
    env_rates: dict[str, list[float]] = defaultdict(list)
    for job_dir in sorted(log_dir.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("."):
            continue
        job = load_job(job_dir)
        if job is not None:
            env_rates[job["env_id"]].append(job["success_rate"])
    return env_rates


def main(log_dirs: list[str]) -> None:
    """Main function to evaluate statistics from multiple log directories."""
    combined: dict[str, list[float]] = defaultdict(list)
    for log_dir_str in log_dirs:
        log_dir = Path(log_dir_str)
        for env_id, rates in summarize(log_dir).items():
            combined[env_id].extend(rates)

    print(f"{'Env':<45} {'Seeds':>5} {'Mean':>8} {'Std':>8}")
    print("-" * 70)
    for env_id in sorted(combined):
        rates = combined[env_id]
        mean = np.mean(rates)
        std = np.std(rates, ddof=1) if len(rates) > 1 else float("nan")
        print(f"{env_id:<45} {len(rates):>5} {mean:>8.3f} {std:>8.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
