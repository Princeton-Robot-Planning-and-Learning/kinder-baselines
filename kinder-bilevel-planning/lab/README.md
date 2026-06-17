# Bilevel planning lab

A ~90-minute hands-on lab on **bilevel (task-and-motion) planning**, built on
`kinder-bilevel-planning`. You'll extend the **Obstruction2D** world with your own
predicates, skills, and operators, and watch a planner use them — including a real
**motion planner** that routes the robot around obstacles.

## Two ways to do the lab

- **Colab (zero install) — Parts 1 & 2.** Open a notebook below; the first cell
  clones the repo and installs the 2D-only packages for you. No local setup.

  | Part | Notebook |
  |---|---|
  | **1** Stacking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Princeton-Robot-Planning-and-Learning/kinder-baselines/blob/main/kinder-bilevel-planning/lab/notebooks/lab_part1_stacking.ipynb) |
  | **2** Pyramid | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Princeton-Robot-Planning-and-Learning/kinder-baselines/blob/main/kinder-bilevel-planning/lab/notebooks/lab_part2_pyramid.ipynb) |

- **Local install — all three parts.** Part 3 is the open, advanced capstone and
  is local-only. Do this before the lab: follow [`SETUP.md`](SETUP.md) to install
  `uv`, clone the repo, and install the lab packages, then run everything from
  this `lab/` directory:

  ```bash
  source ../../.venv/bin/activate     # the venv created in SETUP.md
  cd kinder-bilevel-planning/lab
  python -m pytest part1_stacking -q
  ```

## How the lab works (local)

Each part is a directory with an `INSTRUCTIONS.md`, some code files with
`# TODO` markers, and tests. Your loop:

1. Open `partN_*/INSTRUCTIONS.md`.
2. Fill the `# TODO` holes in that part's files (your IDE, your choice).
3. Run that part's tests — the failures are your checklist:
   `python -m pytest partN_* -q`.
4. When it's green, run it and watch: `python -m partN_*.run`.

In the **Colab** notebooks (Parts 1 & 2) the loop is the same, but you fill the
`# TODO` *cells* and run the check + visualization cells inline instead of using
`pytest` and the desktop visualizer.

## The three parts

| Part | Task | Mode | AI? |
|---|---|---|---|
| **1** `part1_stacking` | Stack the obstruction on the target block | Instructor walks through it; you implement small holes | **No AI** |
| **2** `part2_pyramid` | Build a pyramid (target on two obstructions) | On your own | **No AI** |
| **3** `part3_open` | Invent your own task/environment | On your own (**local only**) | **AI allowed** |

> AI assistants are **not** allowed in Parts 1 and 2. They **are** allowed in
> Part 3. Parts 1 & 2 run in Colab or locally; Part 3 is open-ended and runs
> locally (full install, desktop visualizer).

## Layout

```
lab/
  README.md
  renderer.py            # shared visualizer renderer
  crv_skills.py          # PROVIDED motion-planning plumbing (you don't edit this)
  notebooks/             # Colab notebooks for Parts 1 & 2 (+ colab_utils.py, build_notebooks.py)
  part1_stacking/        # INSTRUCTIONS.md + models.py/skills.py (TODOs) + tests + run.py
  part2_pyramid/         # INSTRUCTIONS.md + models.py/skills.py (TODOs) + spec test + run.py
  part3_open/            # INSTRUCTIONS.md (open-ended, local; AI allowed)
```
