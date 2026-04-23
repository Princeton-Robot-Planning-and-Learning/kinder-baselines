"""Run trajopt experiments across all environments, 5 seeds each.

Usage:
    python scripts/run_all_wm_envs.py
Run from the kinder-trajopt directory.
"""

import subprocess
import sys

# ENVS = [
#     "stickbutton2d-b1-model",
#     "dynobstruction2d-o1-model",
#     "basemotion3d-model",
#     "motion2d-p0-model",
# ]

# ENVS = [
#     "dynobstruction2d-o1-model",
# ]

ENVS = [
    "dynpushpullhook2d-o5-model",
    "shelf3d-o1-model",
    "transport3d-o2-model",
    "sweepintodrawer3d-o5-model",
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
