# Trajectory Optimization Baselines for KinDER

Predictive sampling-based trajectory optimization using MPC, applicable to any KinDER environment.

## Experiment

Running on a single environment with a single seed:
```bash
python experiments/run_experiment.py env=motion2d-p0 seed=0
```

Running with custom hyperparameters:
```bash
python experiments/run_experiment.py env=motion2d-p0 seed=0 \
    num_rollouts=200 horizon=50 noise_scale=0.5
```

Running on multiple environments and multiple seeds (sequential):
```bash
python experiments/run_experiment.py -m seed='range(0,3)' \
    env=motion2d-p0,motion2d-p2
```

Running on multiple environments and multiple seeds (parallel via joblib):
```bash
python experiments/run_experiment.py -m seed='range(0,3)' \
    env=motion2d-p0,motion2d-p2 hydra/launcher=joblib \
    hydra.launcher.n_jobs=-1
```

## Installation

We strongly recommend uv. The steps below assume that you have uv installed. If you do not, just remove uv from the commands and the installation should still work.

```bash
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
```
