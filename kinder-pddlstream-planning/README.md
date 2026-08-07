# PDDLStream Planning Baselines for KinDER

Task and motion planning baselines built on [PDDLStream](https://github.com/caelan/pddlstream), covering the Motion2D, Packing3D, and LimbRepositioning3D environments.

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
| `limbrepositioning3d` | Torque a welded human limb to a goal pose | `move_base`, `grasp`, `move_limb` |

Each environment directory holds its `domain.pddl`, its `stream.pddl`, the stream implementations, and a `run_*.py` entry point. All three solve with the `adaptive` algorithm.

### Motion2D

```bash
python -m kinder_pddlstream_planning.motion2d.run --num-passages 3 --seed 0
```

### Packing3D

```bash
python -m kinder_pddlstream_planning.packing3d.run --num-parts 2 --seed 0
```

### LimbRepositioning3D

```bash
python -m kinder_pddlstream_planning.limbrepositioning3d.run \
    --variant isolated-right-arm

# Run all sixteen variants and save a GIF for each.
python -m kinder_pddlstream_planning.limbrepositioning3d.run \
    --all-variants --max-time 600 --gif-dir outputs/limbrepositioning3d
```

Two flags exist for variants the defaults cannot solve. `--all-variants` applies them per variant on its own; a single-variant run sets them by hand. 
- `--no-check-base-collisions` admits base poses overlapping the furniture, which the `bed-*-leg` variants need because their goal is only reachable from inside the hospital bed. 
- `--robot-base-z` overrides the robot's world z, which recovers the two `wheelchair-*-arm` variants whose goal sits above the arm's vertical reach from any floor position. Use `0.4` for `wheelchair-left-arm` and `0.55` for `wheelchair-right-arm`, against their shipped 0.335 and 0.51. It translates the whole robot rigidly, so it is a reachability experiment rather than a model of a lift mechanism.

All run scripts take `--gif-path` to record the rollout, which is written even when planning or execution fails.

## Running CI Checks

```bash
./run_ci_checks.sh
```
