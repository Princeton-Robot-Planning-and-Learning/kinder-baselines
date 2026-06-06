# Visualizing when bilevel planning *must* resample

A teaching example for the
[bilevel planning visualizer](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/bilevel-planning):
a hand-designed `kinder/Obstruction2D-o0-v0` instance where refining the
abstract plan **requires more than one sample per step**.

Where the sibling `visualize_obstruction2d` example solves a random o1 instance,
this one is constructed so a single sample per step cannot succeed.

## The scenario

There are no obstructions -- just the robot, one target block, and the target
surface -- so the abstract plan is the single two-step sequence
`PickFromTable(target_block)`, `PlaceOnTarget(target_block)`. The geometry is
rigged:

- The **target surface touches the right wall** and is only slightly wider than
  the block, so the block's placement is essentially fixed -- it must sit on the
  narrow surface.
- The **target block is wide and starts all the way on the left**.

The robot has a circular body of radius `0.1` in a world `1.618` wide, and it
grasps the block at a sampled x-offset along the block. Because the placement is
fixed, the robot ends the place at `surface_x + grasp_offset`, so whether it
collides with the right wall is decided by the **grasp**, not by where the place
samples:

- A **right-side grasp** puts the body past `world_max_x - radius = 1.518` -- the
  place is impossible for *any* place sample, so the refiner must resample the
  grasp.
- A **far-left grasp** collides with the **left wall** at pick time (the block is
  against the left wall).
- Only a **middle-ish grasp** lets the robot place the block while keeping its
  body inside both walls.

So the first sampled grasp often fails, and the refiner must resample. With
`num_sampling_attempts_per_step = 1` the planner finds **no plan**; with `2` it
succeeds. To keep the demonstration about *sampling* rather than abstract
replanning, the abstract-plan budget is capped at `1` (`MAX_ABSTRACT_PLANS = 1`)
-- the planner commits to the single shortest abstract plan and can only succeed
by resampling its refinement.

In the graph you can see this directly: the search branches into two grasps --
a right-side grasp whose place dead-ends (the robot reaching into the right
wall), and a resampled middle grasp whose place succeeds. (Widening the surface
would instead make the *place* sample the deciding variable, which is a different
lesson.)

## Prerequisites

Use the kinder-baselines virtualenv, set up per the repo README
(`uv pip install -r prpl_requirements.txt` then `uv pip install -e ".[develop]"`).
It already has `kinder` (for the renderer) and the `bilevel_planning` visualizer.

If `python -m bilevel_planning.visualizer` fails to import `flask_cors`, install
the visualizer's server dependency into the venv: `uv pip install flask-cors`.

## 1. Generate the bundle

The bundle is not committed (it's a regenerable artifact under the gitignored
`data/`). Build it with a real planner run:

```bash
python examples/visualize_obstruction2d_resampling/generate_bundle.py
```

This solves the hand-designed instance with the `SesamePlanner`, exports the
`BilevelPlanningGraph` to `data/obstruction2d_o0_resampling.pkl`, and prints the
launch command.

## 2. View it

```bash
python -m bilevel_planning.visualizer \
    --bundle examples/visualize_obstruction2d_resampling/data/obstruction2d_o0_resampling.pkl \
    --renderer examples/visualize_obstruction2d_resampling/renderer.py
```

A browser opens to the graph. The concrete (lower) layer is the sequence of
world states the planner searched; click a concrete node to render that state --
including the failed branch where the robot reaches into the right wall.

## Files

- `generate_bundle.py` -- builds the hand-designed initial state, runs the
  planner, and exports the bundle. The geometry constants and the
  `MAX_ABSTRACT_PLANS` / `SAMPLES_PER_STEP` choices are documented at the top.
- `renderer.py` -- the `render_state(state) -> HxWx3 uint8` function the
  visualizer execs; draws an Obstruction2D state via kinder's `render_2dstate`.
