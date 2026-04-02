# KinDER Perception Planning

Bilevel planning with **VLM-based predicate grounding** for KinDER environments. Instead of using programmatic state inspection to determine predicate truth values, this package renders the scene as an image and queries a vision-language model (e.g., GPT-4o) to ground each predicate.

## Overview

The planning pipeline is:

1. **Render** the environment state as an RGB image
2. **Prompt** the VLM with predicate descriptions and candidate ground atoms
3. **Parse** the VLM response to build a `RelationalAbstractState`
4. **Plan** using the same SeSamE bilevel planner from `kinder-bilevel-planning`

Everything else (operators, skills, controllers, transition model) is identical to the standard bilevel planning baseline — only the **state abstractor** is swapped.

### Supported Environments

| Environment | Env ID | Predicates |
|---|---|---|
| Motion2D | `kinder/Motion2D-p{N}-v0` | AtTgt, NotAtTgt, AtPassage, NotAtPassage, NotAtAnyPassage |
| StickButton2D | `kinder/StickButton2D-b{N}-v0` | Grasped, HandEmpty, Pressed, RobotAboveButton, StickAboveButton, AboveNoButton |

## Installation

```bash
cd kinder-baselines/kinder-perception-planning
uv pip install -e ".[develop]"
```

Requires the sibling packages listed in `prpl_requirements.txt` (`kinder-models`, `kinder-bilevel-planning`). If not already installed:

```bash
cd kinder-baselines
uv run python scripts/install_all.py
```

## Configuration

Set your OpenAI API key (required for VLM queries):

```bash
export OPENAI_API_KEY="sk-..."
```

## Running Experiments

All experiments use [Hydra](https://hydra.cc/) for configuration. Run from the package root.

### Single run

```bash
# Motion2D with 2 passages, seed 0, GPT-4o
python experiments/run_experiment.py env=motion2d-p2 seed=0

# StickButton2D with 2 buttons
python experiments/run_experiment.py env=stickbutton2d-b2 seed=0

# Use a different VLM
python experiments/run_experiment.py env=motion2d-p2 seed=0 vlm_model_name=gpt-4.1
```

### Multi-seed sweep

```bash
python experiments/run_experiment.py -m env=motion2d-p2 seed='range(0,10)'
```

### Multiple environments (parallelized)

```bash
python experiments/run_experiment.py -m \
    env=motion2d-p2,stickbutton2d-b2 \
    seed='range(0,5)' \
    hydra/launcher=joblib
```

### Override planning hyperparameters

```bash
python experiments/run_experiment.py env=stickbutton2d-b2 seed=0 \
    max_abstract_plans=20 \
    samples_per_step=5 \
    planning_timeout=120
```

### Record videos

```bash
python experiments/run_experiment.py env=motion2d-p2 seed=0 make_videos=True
```

Results (CSV) and config (YAML) are saved to `experiments/logs/<date>/<time>/`.

## Testing

```bash
# Run all tests
pytest tests/

# Run tests for a specific environment
pytest tests/env_models/kinematic2d/test_motion2d.py -v
pytest tests/env_models/kinematic2d/test_stickbutton2d.py -v
```

Tests use a mock VLM (`OrderedResponseModel`) with ground-truth responses from the bilevel planning state abstractor, so no API key is needed.

## CI

```bash
./run_ci_checks.sh    # autoformat + mypy + pylint + pytest
./run_autoformat.sh   # black + docformatter + isort only
```

## Package Structure

```
kinder-perception-planning/
├── src/kinder_perception_planning/
│   ├── agent.py                 # Re-exports BilevelPlanningAgent
│   ├── vlm_utils.py             # VLM query + prompt construction + response parsing
│   └── env_models/
│       ├── __init__.py           # Dynamic model loader
│       └── kinematic2d/
│           ├── motion2d.py       # Motion2D with VLM state abstractor
│           └── stickbutton2d.py  # StickButton2D with VLM state abstractor
├── experiments/
│   ├── run_experiment.py         # Hydra entry point
│   └── conf/
│       ├── config.yaml           # Default config (includes vlm_model_name)
│       └── env/                  # Per-environment configs
├── tests/
│   └── env_models/kinematic2d/   # Unit tests with mock VLM
└── pyproject.toml
```
