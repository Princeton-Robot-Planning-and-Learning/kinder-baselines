# LLM/VLM Planning Baselines for KinDER

![workflow](https://github.com/yichao-liang/kinder-vlm-planning/actions/workflows/ci.yml/badge.svg)

## Experiment

Set `rgb_observation=false` to test LLM planning (text-only, no image observations) instead of VLM planning.

Running on a single environment with a single seed:
```bash
python experiments/run_experiment.py env=Motion2D-p0-v0 seed=0 vlm_model=gpt-5 \
    temperature=1
python experiments/run_experiment.py -m env=StickButton2D-b1-v0 seed=0 \
    vlm_model=gpt-5 temperature=1
```

Running on multiple environments and multiple seeds (sequential):
```bash
python experiments/run_experiment.py -m seed='range(0,3)' \
    env=Motion2D-p0-v0,StickButton2D-b1-v0 \
    vlm_model=gpt-5 rgb_observation=true,false temperature=1
```

Running on multiple environments and multiple seeds (parallel via joblib):
```bash
python experiments/run_experiment.py -m seed='range(0,3)' \
    env=BaseMotion3D-v0,Transport3D-o2-v0,Shelf3D-o1-v0 \
    vlm_model=gpt-5 rgb_observation=true,false temperature=1 hydra/launcher=joblib \
    hydra.launcher.n_jobs=-1
```

## Installation

We strongly recommend uv. The steps below assume that you have uv installed. If you do not, just remove uv from the commands and the installation should still work.

# Install PRPL dependencies.
uv pip install -r prpl_requirements.txt
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
