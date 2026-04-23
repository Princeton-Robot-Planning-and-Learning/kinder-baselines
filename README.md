# KinDER Baselines

Baseline methods for the [KinDER](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden) benchmark.

## Packages

| Package | Description |
|---------|-------------|
| `kinder-bilevel-planning` | Bilevel planning baselines |
| `kinder-ds-policies` | Domain-specifc policy baselines |
| `kinder-imitation-learning` | Imitation learning baselines |
| `kinder-models` | Model zoo (skills, operators, samplers) |
| `kinder-rl` | Reinforcement learning baselines |
| `kinder-vlm-planning` | Vision-language model planning baselines |
| `kinder-trajopt` | Model-predictive control baselines |
| `kinder-mbrl` | Model-based reinforcement learning (world model) baselines |

## Installation

1. Clone this repository.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Install all packages: `uv run python scripts/install_all.py`

To install a single package instead:
```bash
cd <package-dir>
uv pip install -e ".[develop]"
```

Some packages depend on others in this repo via `prpl_requirements.txt`. Install those first, or use `scripts/install_all.py` which handles the ordering automatically.

## Running CI Checks

From any package directory:
```bash
./run_ci_checks.sh
```

Or run checks across all packages:
```bash
./run_all_ci_checks.sh
```
