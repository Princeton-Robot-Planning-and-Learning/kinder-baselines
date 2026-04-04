# kinder-mbrl

Model-based RL baselines for [KinDER](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden).

Provides:

- `**kinder_mbrl.models**` — MLP delta dynamics model (`MLPDynamics`) and per-feature normalizer.
- `**kinder_mbrl.data_utils**` — HDF5 dataset loading utilities.
- `**kinder_mbrl.planning**` — Random-shooting MPC planner that works with either the ground-truth simulator or a learned world model.

## Installation

```bash
uv pip install -e ".[develop]"
```

## Usage

### Train a world model

```bash
python experiments/train_world_model.py --mode train --hdf5_path /path/to/dataset.hdf5
```

The HDF5 files come from our collected demonstrations. You can download these demonstrations from our [Hugging Face](https://huggingface.co/datasets/kinder-bench/kinder-datasets).

### Evaluate open-loop rollout error

```bash
python experiments/train_world_model.py --mode eval --checkpoint output/wm.pt
```

### Run random-shooting MPC

```bash
# Simulator-based planning
python experiments/run_mpc.py

# World-model-based planning
python experiments/run_mpc.py --use_world_model --checkpoint output/wm.pt
```

## Running CI checks

```bash
./run_ci_checks.sh
```

