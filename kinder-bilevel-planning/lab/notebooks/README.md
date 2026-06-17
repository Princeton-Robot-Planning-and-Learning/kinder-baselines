# Colab notebooks (Parts 1 & 2)

Zero-install versions of the guided lab parts. Students open these in Colab from
the badges in [`../README.md`](../README.md); the first cell clones this repo and
does the slim (2D-only) install.

| File | What it is |
|---|---|
| `lab_part1_stacking.ipynb` | Part 1, generated. Editable `# TODO` cells + inline checks + inline visualization. |
| `lab_part2_pyramid.ipynb` | Part 2, generated. |
| `colab_utils.py` | **Provided** (students don't edit): the inline visualizers (`render_state`, `show_storyboard`, `animate_states`) that replace the desktop `bilevel_planning.visualizer`. The clone + 2D-only install lives in each notebook's self-contained setup cell. |
| `build_notebooks.py` | Generator (source of truth for the `.ipynb` files). |

## Regenerating

The notebooks are generated — edit `build_notebooks.py` (cell sources live there as
string constants), then:

```bash
python build_notebooks.py
```

The exercise *holes* are factored into `*_HOLE` constants so the **worked
solutions** can be generated separately (privately) without solution text ever
living in this public repo. See the private solutions repo's `notebooks/`.
