# PDDLStream Planning Baselines for KinDER

Task and motion planning baselines built on [PDDLStream](https://github.com/caelan/pddlstream), covering the Motion2D, Packing3D, and LimbRepositioning3D environments.

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
    --all-variants --max-time 600 --save-gif
```

Scene:

- `--variant` - which of the sixteen scene/limb combinations to solve (`wheelchair-left-arm`).
- `--all-variants` - run every variant in turn.
- `--standoff` - how far behind its grasp pose the base starts, in meters (1.0).
- `--robot-base-z` - world z to place the robot at, overriding the variant's own.
- `--seed` (0), `--max-time` - planning budget in seconds (600).

The person:

- `--human-torque-limit` - total N*m one of their joints may bear (a per-limb default).
- `--gravity` - world z gravity in m/s^2 (-9.81); 0 keeps the environment's own.
- `--muscle-tone` - `none` for limp, or a `spring` damper (`spring`).
- `--joint-limits-model` - `none` for unbounded, or per-joint `box` limits (`box`).
- `--range-of-motion-scale` - factor on every range-of-motion magnitude (1.0).
- `--limb-joint-damping` - viscous damping at the limb's joints, N*m*s/rad (0.5).
- `--goal-scale` - interpolate the goal toward the start; 0.5 asks half the move (1.0).

Search:

- `--check-base-collisions`, `--check-robot-collisions` - base and arm against the furniture and the person (both on).
- `--filter-saturated-bases` - defer, then past a cap drop, base poses the arm cannot hold the limb at (on).
- `--num-rollouts` (64), `--horizon` (12), `--commit-steps` (1), `--noise-scale` (0.12) - `move_limb`'s predictive-sampling MPC.

Output:

- `--save-gif`, `--gif-dir` - record the rollout, written even when the run fails (`outputs/`).
- `--use-gui` - show the PyBullet GUI (off).

## Running CI Checks

```bash
./run_ci_checks.sh
```
