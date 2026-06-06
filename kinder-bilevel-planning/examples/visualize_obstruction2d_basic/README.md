# Visualizing the basic bilevel planning graph

The introductory example for the
[bilevel planning visualizer](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/bilevel-planning):
the simplest possible `kinder/Obstruction2D-o0-v0` solve, meant to show the basic
two-plane graph structure before anything more complicated.

## The scenario

There are no obstructions -- just the robot, one target block, and the target
surface -- so the abstract plan is the single two-step sequence
`PickFromTable(target_block)`, `PlaceOnTarget(target_block)`. The block starts
middle-left and the target surface sits middle-right, both well clear of the
walls, so any grasp and any placement work and the plan refines on the very
first sample.

With `MAX_ABSTRACT_PLANS = 1` and `SAMPLES_PER_STEP = 1` there is no abstract
replanning and no resampling, so the graph is a single path:

- the **concrete (lower) plane** is one pick-then-place trajectory, and
- the **abstract (upper) plane** is `HandEmpty -> Holding -> OnTarget`.

Once this structure is clear, see:

- `visualize_obstruction2d_resampling` -- an o0 case rigged so a single sample
  per step fails and the refiner must resample (the graph branches).
- `visualize_obstruction2d` -- a realistic o1 solve with search branches.

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
python examples/visualize_obstruction2d_basic/generate_bundle.py
```

This solves the instance with the `SesamePlanner`, exports the
`BilevelPlanningGraph` to `data/obstruction2d_o0_basic.pkl`, and prints the
launch command.

## 2. View it

```bash
python -m bilevel_planning.visualizer \
    --bundle examples/visualize_obstruction2d_basic/data/obstruction2d_o0_basic.pkl \
    --renderer examples/visualize_obstruction2d_basic/renderer.py
```

A browser opens to the graph. The concrete (lower) layer is the single pick-then-
place trajectory; click a concrete node to render that state.

## Files

- `generate_bundle.py` -- builds the hand-designed initial state, runs the
  planner, and exports the bundle. The geometry constants and the
  `MAX_ABSTRACT_PLANS` / `SAMPLES_PER_STEP` choices are documented at the top.
- `renderer.py` -- the `render_state(state) -> HxWx3 uint8` function the
  visualizer execs; draws an Obstruction2D state via kinder's `render_2dstate`.
