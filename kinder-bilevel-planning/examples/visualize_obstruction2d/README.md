# Visualizing a Bilevel Planning Graph for Obstruction2D

End-to-end example: run the bilevel planner on
`kinder/Obstruction2D-o1-v0`, export the resulting `BilevelPlanningGraph`
to a single pickle, and explore it in the
[bilevel-planning visualizer](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-mono/tree/main/bilevel-planning/src/bilevel_planning/visualizer)
with kinder rendering each node on demand.

## Prerequisites

- A Python environment with `kindergarden`, `kinder-bilevel-planning`,
  and `bilevel-planning` (≥ the visualizer release) installed.
- The bilevel-planning visualizer frontend has been built once. From
  your `prpl-mono` checkout:
  ```bash
  cd bilevel-planning/src/bilevel_planning/visualizer/frontend
  npm ci
  npm run build
  ```
  Re-run only when you pull frontend changes.

## Step 1 — Generate the visualizer bundle

```bash
python generate_bpg.py
```

Runs the SeSamE planner with a 30-second timeout, builds a
`BilevelPlanningGraph`, and writes a single pickle to `data/bpg.pkl`.
The pickle bundles both the graph topology (for the frontend) and the
original state objects (for rendering).

## Step 2 — Launch the visualizer

```bash
python -m bilevel_planning.visualizer
```

Open http://localhost:5001 and:

1. In the **Python renderer** pane (already expanded), replace the
   default template with the kinder renderer:

   ```python
   import kinder

   kinder.register_all_environments()
   env = kinder.make("kinder/Obstruction2D-o1-v0", render_mode="rgb_array")

   def render_state(state):
       env.reset(options={"init_state": state})
       return env.render()
   ```

   Click **Apply renderer**. The pane label flips to `(ready)`.
2. Click **Load pickle bundle** and pick `data/bpg.pkl` from this
   directory.
3. Click any concrete (non-purple) node in the 3D graph. A rendered
   image of that state appears in the side panel.

You can edit the renderer source and click **Apply renderer** again at
any time to swap it without restarting anything.

Stop the backend with `Ctrl-C` when done. The generated bundle stays at
`data/bpg.pkl` until you delete it.

## Files in this example

- `generate_bpg.py` — runs the planner and writes `data/bpg.pkl`.
- `README.md` — this file.
