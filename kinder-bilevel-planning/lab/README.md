# Bilevel planning lab

A ~90-minute hands-on lab on **bilevel (task-and-motion) planning**, built on
`kinder-bilevel-planning`. You'll extend the **Obstruction2D** world with your own
predicates, skills, and operators, and watch a planner use them — including a real
**motion planner** that routes the robot around obstacles.

## Setup

Use the `kinder-bilevel-planning` virtualenv (activate it however you normally
do), and run everything from this `lab/` directory:

```bash
cd lab
python -m pytest part1_stacking -q
```

## How the lab works

Each part is a directory with an `INSTRUCTIONS.md`, some code files with
`# TODO` markers, and tests. Your loop:

1. Open `partN_*/INSTRUCTIONS.md`.
2. Fill the `# TODO` holes in that part's files (your IDE, your choice).
3. Run that part's tests — the failures are your checklist:
   `python -m pytest partN_* -q`.
4. When it's green, run it and watch: `python -m partN_*.run`.

## The three parts

| Part | Task | Mode | AI? |
|---|---|---|---|
| **1** `part1_stacking` | Stack the obstruction on the target block | Instructor walks through it; you implement small holes | **No AI** |
| **2** `part2_pyramid` | Build a pyramid (target on two obstructions) | On your own | **No AI** |
| **3** `part3_open` | Invent your own task/environment | On your own | **AI allowed** |

> AI assistants are **not** allowed in Parts 1 and 2. They **are** allowed in
> Part 3.

## Layout

```
lab/
  README.md
  renderer.py            # shared visualizer renderer
  crv_skills.py          # PROVIDED motion-planning plumbing (you don't edit this)
  part1_stacking/        # INSTRUCTIONS.md + models.py/skills.py (TODOs) + tests + run.py
  part2_pyramid/         # INSTRUCTIONS.md + models.py/skills.py (TODOs) + spec test + run.py
  part3_open/            # INSTRUCTIONS.md (open-ended; AI allowed)
```
