# PDDLStream Planning Baselines for KinDER

Task and motion planning baselines built on [PDDLStream](https://github.com/caelan/pddlstream), covering the Motion2D and Packing3D environments.

## Installation

We strongly recommend uv. The steps below assume that you have uv installed. If you do not, just remove uv from the commands and the installation should still work.

```bash
# Install this package and third-party dependencies.
uv pip install -e ".[develop]"
```

PDDLStream is not a pip package, so it is used in place from a git checkout with FastDownward built in-tree:

```bash
git clone --recursive https://github.com/caelan/pddlstream.git ~/pddlstream
cd ~/pddlstream && ./downward/build.py
```

The checkout is looked up at `~/pddlstream` by default. Set `PDDLSTREAM_PATH` to override it.

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
