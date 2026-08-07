# PDDLStream Planning Baselines for KinDER

Task and motion planning baselines built on [PDDLStream](https://github.com/caelan/pddlstream), covering the Motion2D and Packing3D environments.

## Installation

We strongly recommend uv. The steps below assume that you have uv installed. If you do not, just remove uv from the commands and the installation should still work.

```bash
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
```

That includes PDDLStream, which is pinned to a packaged fork at
[Princeton-Robot-Planning-and-Learning/pddlstream](https://github.com/Princeton-Robot-Planning-and-Learning/pddlstream).
FastDownward is compiled during the install, so no separate clone or build step is
needed - but `make` and a C++ compiler are **required**.

## Environments

| Environment | Domain | Actions |
|---|---|---|
| `motion2d` | 2D base navigation through narrow passages | `move` |
| `packing3d` | Pick and place parts into a rack | `move_base`, `pick`, `place` |

Each environment directory holds its `domain.pddl`, its `stream.pddl`, the stream implementations, and a `run_*.py` entry point. Both solve with the `adaptive` algorithm.

### Motion2D

```bash
python -m kinder_pddlstream_planning.motion2d.run --num-passages 3 --seed 0
```

### Packing3D

```bash
python -m kinder_pddlstream_planning.packing3d.run --num-parts 2 --seed 0
```

All run scripts take `--gif-path` to record the rollout, which is written even when planning or execution fails.

## Running CI Checks

```bash
./run_ci_checks.sh
```
