# Visualizing a large bilevel planning graph

A scaling stress test for the
[bilevel planning visualizer](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/bilevel-planning):
a real `kinder/Obstruction2D-o2-v0` solve with a generous search budget, to see
how the visualization holds up on a graph an order of magnitude larger than the
other examples.

## The scenario

Two obstructions must be cleared before the target block can be placed, and the
search runs with **5 abstract plans** and **5 samples per step**. The result is a
graph with a few thousand concrete states and substantial branching -- roughly
9x the size of the `visualize_obstruction2d` (o1) example.

This is a plain random instance (no hand-designed geometry). With two
obstructions and this budget the problem is hard: the chosen seed solves in about
ten seconds, but many other seeds instead blow up to ~15-19k states and time out
without finding a plan, so changing the seed may stop it from solving.

For the small, easy-to-read cases start with `visualize_obstruction2d_basic`,
then `visualize_obstruction2d_resampling`, then `visualize_obstruction2d`.

## Prerequisites

Use the kinder-baselines virtualenv, set up per the repo README
(`uv pip install -r prpl_requirements.txt` then `uv pip install -e ".[develop]"`).
It already has `kinder` (for the renderer) and the `bilevel_planning` visualizer.

If `python -m bilevel_planning.visualizer` fails to import `flask_cors`, install
the visualizer's server dependency into the venv: `uv pip install flask-cors`.

## 1. Generate the bundle

The bundle is not committed (it's a regenerable artifact under the gitignored
`data/`). Build it with a real planner run (takes ~10-15s):

```bash
python examples/visualize_obstruction2d_large/generate_bundle.py
```

This solves the instance with the `SesamePlanner`, exports the
`BilevelPlanningGraph` to `data/obstruction2d_o2_large.pkl`, and prints the
launch command.

## 2. View it

```bash
python -m bilevel_planning.visualizer \
    --bundle examples/visualize_obstruction2d_large/data/obstruction2d_o2_large.pkl \
    --renderer examples/visualize_obstruction2d_large/renderer.py
```

A browser opens to the graph. Expect a dense concrete (lower) plane; click a
concrete node to render that state.

## Files

- `generate_bundle.py` -- runs the planner and exports the bundle. The seed and
  the `MAX_ABSTRACT_PLANS` / `SAMPLES_PER_STEP` budget are documented at the top.
- `renderer.py` -- the `render_state(state) -> HxWx3 uint8` function the
  visualizer execs; draws an Obstruction2D state via kinder's `render_2dstate`.
