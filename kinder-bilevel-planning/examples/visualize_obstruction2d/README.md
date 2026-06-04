# Visualizing an Obstruction2D bilevel planning graph

A realistic example for the
[bilevel planning visualizer](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/bilevel-planning):
run a real bilevel-planning solve of `kinder/Obstruction2D-o1-v0`, then explore
the resulting graph in an interactive 3D viewer where clicking a node renders
the actual robot/blocks scene at that state.

Unlike the toy `simple_two_state` example in `bilevel-planning`, the states here
are real `ObjectCentricState`s and the renderer draws the real environment.

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
python examples/visualize_obstruction2d/generate_bundle.py
```

This solves obstruction2d-o1 with the `SesamePlanner`, exports the
`BilevelPlanningGraph` to `data/obstruction2d_o1.pkl`, and prints the launch
command.

## 2. View it

```bash
python -m bilevel_planning.visualizer \
    --bundle examples/visualize_obstruction2d/data/obstruction2d_o1.pkl \
    --renderer examples/visualize_obstruction2d/renderer.py
```

A browser opens to the graph. The concrete (lower) layer is the sequence of
world states the planner searched; the abstract (upper) layer is their symbolic
abstractions, with the found plan highlighted. Click a concrete node to render
that Obstruction2D state — the robot reaching past the obstruction to place the
target block on the target surface.

## Files

- `generate_bundle.py` — runs the planner and exports the bundle. Its planner
  construction mirrors `kinder_bilevel_planning.agent.BilevelPlanningAgent`, but
  keeps the graph the planner builds instead of discarding it.
- `renderer.py` — the `render_state(state) -> HxWx3 uint8` function the
  visualizer execs; draws an Obstruction2D state via kinder's `render_2dstate`.
