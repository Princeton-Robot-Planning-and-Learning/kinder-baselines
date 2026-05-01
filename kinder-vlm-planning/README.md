# LLM/VLM Planning Baselines for KinDER

![workflow](https://github.com/yichao-liang/kinder-vlm-planning/actions/workflows/ci.yml/badge.svg)

## Usage

### Running Experiments

Experiments are configured using [Hydra](https://hydra.cc/). The key configuration options are:

| Option | Description | Default |
|--------|-------------|---------|
| `env` | KinDER environment ID (without `kinder/` prefix) | `Motion2D-p0-v0` |
| `seed` | Random seed or range (e.g., `'range(0,5)'`) | `0` |
| `vlm_model` | VLM model name (e.g., `gpt-5`, `gpt-5.2`) | `gpt-5` |
| `temperature` | Sampling temperature | `1` |
| `rgb_observation` | Use visual observations (VLM) vs text-only (LLM) | `false` |
| `prompt_type` | Prompt template: `basic` or `llmplanner` | `basic` |
| `max_eval_steps` | Maximum steps per episode | `300` |
| `num_eval_episodes` | Number of evaluation episodes | `1` |

**Note:** Set `rgb_observation=false` to test LLM planning (text-only, no image observations) instead of VLM planning.

### Prompt Types

- **`basic`**: Standard prompt template for task-based planning
- **`llmplanner`**: Structured prompt following the LLMPlanner format

### Quick Examples

**Single environment, single seed:**
```bash
python experiments/run_experiment.py env=Motion2D-p0-v0 seed=0 vlm_model=gpt-5 temperature=1
```

**Multiple environments and seeds (sequential):**
```bash
python experiments/run_experiment.py -m seed='range(0,3)' \
    env=Motion2D-p0-v0,StickButton2D-b1-v0 \
    vlm_model=gpt-5 rgb_observation=true,false temperature=1
```

**Multiple environments and seeds (parallel via joblib):**
```bash
python experiments/run_experiment.py -m seed='range(0,3)' \
    env=BaseMotion3D-v0,Transport3D-o2-v0,Shelf3D-o1-v0 \
    vlm_model=gpt-5 rgb_observation=true,false temperature=1 hydra/launcher=joblib \
    hydra.launcher.n_jobs=-1
```

### Environment-Specific Examples

#### Kinematic 2D Environments

**StickButton2D-b1-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=StickButton2D-b1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=StickButton2D-b1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=StickButton2D-b1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=StickButton2D-b1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

**Motion2D-p0-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=Motion2D-p0-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=Motion2D-p0-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=Motion2D-p0-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=Motion2D-p0-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

#### Dynamic 2D Environments

**DynObstruction2D-o1-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=DynObstruction2D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=DynObstruction2D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=DynObstruction2D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=DynObstruction2D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

**DynPushPullHook2D-o5-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=DynPushPullHook2D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=DynPushPullHook2D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=DynPushPullHook2D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=DynPushPullHook2D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

#### Kinematic 3D Environments

**BaseMotion3D-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=BaseMotion3D-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=BaseMotion3D-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=BaseMotion3D-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=BaseMotion3D-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

**Transport3D-o2-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=Transport3D-o2-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=Transport3D-o2-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=Transport3D-o2-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=Transport3D-o2-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

#### Dynamic 3D Environments

**Shelf3D-o1-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=Shelf3D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=Shelf3D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=Shelf3D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=Shelf3D-o1-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

**SweepIntoDrawer3D-o5-v0**
```bash
# GPT-5.2 with basic prompt (VLM)
python experiments/run_experiment.py -m env=SweepIntoDrawer3D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=basic

# GPT-5.2 with basic prompt (LLM)
python experiments/run_experiment.py -m env=SweepIntoDrawer3D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=basic

# GPT-5.2 with llmplanner prompt (VLM)
python experiments/run_experiment.py -m env=SweepIntoDrawer3D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=true prompt_type=llmplanner

# GPT-5.2 with llmplanner prompt (LLM)
python experiments/run_experiment.py -m env=SweepIntoDrawer3D-o5-v0 seed='range(0,5)' \
    vlm_model=gpt-5.2 temperature=1 rgb_observation=false prompt_type=llmplanner
```

### Output

After running an experiment, results are saved to `logs/<date>/<time>/`:
- `results.csv` - Episode-level metrics (success, steps, planning_time, execution_time, reward)
- `config.yaml` - Full experiment configuration

## Installation

We strongly recommend uv. The steps below assume that you have uv installed. If you do not, just remove uv from the commands and the installation should still work.

# Install PRPL dependencies.
uv pip install -r prpl_requirements.txt
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
