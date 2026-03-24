"""Run trajopt experiments across all environments, 3 seeds each.

Usage:
    python scripts/run_all_envs.py
Run from the kinder-trajopt directory.
"""

import subprocess
import sys

ENVS = [
    "motion2d-p0-model",
    "stickbutton2d-b1-model",
    "dynobstruction2d-o1-model",
    "basemotion3d-model",
]

for env in ENVS:
    print(f"=== Running env: {env} ===", flush=True)
    cmd = [
        sys.executable,
        "experiments/run_experiment.py",
        "-m",
        "seed=range(0,5)",
        f"env={env}",
        "num_eval_episodes=50",
        "hydra/launcher=joblib",
    ]
    subprocess.run(cmd, check=True)
