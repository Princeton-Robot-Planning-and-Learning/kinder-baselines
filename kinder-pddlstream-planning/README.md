# PDDLStream Planning Baselines for KinDER

Task and motion planning baselines built on [PDDLStream](https://github.com/caelan/pddlstream), covering the Motion2D, Packing3D, LimbRepositioning3D, and MassTransport3D environments.

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
| `masstransport3d` | Carry items of unknown mass across in a tote, within the arm's torque limits | `move_base`, `travel`, `return`, `pick`, `place`, `load`, `unload`, `pick_tote`, `place_tote` |

Each environment directory holds its `domain.pddl`, its `stream.pddl`, the stream implementations, and a `run_*.py` entry point. All solve with the `adaptive` algorithm.

### Motion2D

```bash
python -m kinder_pddlstream_planning.motion2d.run --num-passages 3 --seed 0
```

### Packing3D

```bash
python -m kinder_pddlstream_planning.packing3d.run --num-parts 2 --seed 0
```

### MassTransport3D

```bash
# Two items, which must share one trip, so they go in the tote.
python -m kinder_pddlstream_planning.masstransport3d.run --num-items 2 --max-trips 1
```

Items move from a source table to a destination table, by hand or several at a time
in a tote with fixed slots. Item masses enter only through torque checks: on item
paths, and on the tote lift in the fluent test `test-carry-torque`, which sees what is
in the tote. A skeleton that overloads the tote therefore fails only there. The
planner uses the masses it is given (`item_masses`), by default the sim's true ones.

- `--max-trips` - bound on source-to-destination trips (one per item). A tight bound
  turns the task into bin packing under the torque limits.
- `--torque-limit-scale` - factor on the arm's torque limits; lower is harder (1.0).
- `--kinematic` - execute in the kinematic env, which checks torques statically,
  instead of the physics one.
- `--num-items` (2), `--seed` (0), `--max-time` (300).

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
