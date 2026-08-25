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

`move_limb` carries three constraints, all of them over the whole trajectory:
- `check-robot-torque-limits` holds the robot's torques inside its action space.
- `check-human-joint-limits` holds the limb inside the person's anatomical range of motion at every step, not only at the goal. The limb's joints are continuous, so PyBullet will happily bend them the wrong way.
- `check-human-torque-limits` holds the torque at each of the person's own joints under `--human-torque-limit`, which defaults to half of peak isometric strength: 40 N*m for an arm, 100 N*m for a leg. A second, tighter bound caps what the motion adds on top of the person's own static load at 5 N*m; it reads zero while the robot only holds the limb, so it bounds the motion rather than the posture. The environment models no such limit, so both are the baseline's own assumption; the torque is measured by inverse dynamics on the limb, which gives gravity plus its muscle tone plus everything the weld transmits into it.

`plan-limb-motion` enforces all three itself as well: the MPC penalizes joint-limit and human-torque violations, and a rollout that violates either is rejected rather than certified.

#### Gravity and muscle tone

The environment ships with both off. The baseline turns gravity on (`--gravity`, -9.81 by default), since a weightless limb costs the person nothing to hold and makes the torque check vacuous. Holding the limb up is then about 4.5 N*m at the shoulder and 18 to 39 N*m at the hip, against the 40 and 100 N*m limits.

Gravity needs one piece of support to be usable. The environment disables the joint motors and gives the arm a +-1 N*m action space, which is an order of magnitude short of holding the arm up, let alone the limb hanging off it - with raw gravity the arm collapses in under a second. So `advance_corrected` commands the robot a gravity-compensation torque - inverse dynamics for the arm's own weight, plus the coupling term that carries the limb through the grasp - and the commanded torque acts on top of it. The coupled system is then in equilibrium at zero command, which is where the gravity-off environment starts, so the action space and the MPC tuned against it still mean what they did. This models the person as carrying their own limb's weight, as they do when nothing supports it; the measurement reports that weight as a load on their joints. Note that `sim.step()` does no such compensation, so a plan replayed through the environment's own step function rather than `advance` will collapse.

`--muscle-tone spring` swaps the limp limb for the spring-damper model, which adds up to 5 N*m per joint of the person's own tone. Its parameters are hard-coded placeholders in kindergarden and its docstring marks it unvalidated. The tone is a torque the limb's own joints apply, so `limb_hold_torque` carries only what gravity leaves after it; carrying their sum instead leaves the coupled system accelerating under a zero command.

`--limb-joint-damping` (0.5 N*m*s/rad by default) gives the limb's joints viscous damping, which the environment zeroes along with their friction so that torques act directly. It is not cosmetic: the hand and the foot are a fraction of a kilogram at the end of the chain the robot grasps, and undamped they whip out of the range of motion under any torque the robot applies, so every MPC rollout is rejected and the planner's best move is to do nothing. Pass 0 for the environment's own frictionless joints.

Segment masses are the environment's own, from its `BodyMass` (Winter's fractions of a 78.4 kg male by default). `create_env` takes a `body_mass` to override the scene's, and the baseline no longer sets masses itself.

#### Joint limits

`--joint-limits-model` picks what `check-human-joint-limits` and the MPC enforce: `box` (the default) bounds each joint independently, `none` drops the constraint, and `realistic` intersects the box with assistive gym's learned shoulder-and-elbow region. The learned model covers arms only, and legs fall back to the box.

`--range-of-motion-scale` multiplies every range-of-motion magnitude, for a stiffer or looser person than the AAOS normal adult the environment ships. Scales below about 0.75 put the shipped initial configurations outside the limits, and the scene rejects them.

Two flags exist for variants the defaults cannot solve. `--all-variants` applies them per variant on its own; a single-variant run sets them by hand. 
- `--no-check-base-collisions` admits base poses overlapping the furniture, which the `bed-*-leg` variants need because their goal is only reachable from inside the hospital bed. 
- `--robot-base-z` overrides the robot's world z, which recovers the two `wheelchair-*-arm` variants whose goal sits above the arm's vertical reach from any floor position. Use `0.4` for `wheelchair-left-arm` and `0.55` for `wheelchair-right-arm`, against their shipped 0.335 and 0.51. It translates the whole robot rigidly, so it is a reachability experiment rather than a model of a lift mechanism.

All run scripts take `--save-gif` to record the rollout, which is written even when planning or execution fails. It goes to `--gif-dir`, which defaults to this package's `outputs/`, as `<environment>_<variant>.gif`.

`--trajectory-dir` writes `<variant>.npz` of per-step measurements and `<variant>.png` of their plots, for a failed run as well as a solved one - a failure replays its closest attempt, which is the rollout worth looking at. Recorded per step: the torque on each of the person's joints and its decomposition into gravity, muscle tone, and the robot's share; the force and moment the grasp transmits, read off the weld constraint; the limb's joint positions and velocities; the torque the robot is asked for, hold plus correction; and the distance left to the goal. The plots read each of those against the limit it is checked on. Redraw without re-planning with `python -m kinder_pddlstream_planning.limbrepositioning3d.plots <npz paths>`.

`--results-csv` appends one row per run - the settings above, whether a plan was found and the goal reached, the planning time, the trajectory length, the peak torque the person's joints bore, and why a failed run failed. It writes a header only when the file is new, so a sweep over settings accumulates in one file.

## Running CI Checks

```bash
./run_ci_checks.sh
```
