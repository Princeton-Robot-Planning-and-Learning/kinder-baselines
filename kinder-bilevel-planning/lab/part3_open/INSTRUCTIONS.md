# Part 3 — Make your own thing 🚀

**AI assistants are allowed for this part.** Go wild.

> **Local only.** Parts 1 & 2 have Colab notebooks; Part 3 is open-ended, so do
> it in your local install (full `SETUP.md` install + the desktop visualizer).
> Start from a copy of a Part 1/2 `run.py` and edit.

You've now built predicates, skills, operators, and goals, and watched a bilevel
planner use them. For the rest of the time, **invent your own task or
environment** and get the planner to solve it. There's no spec test and no right
answer — the goal is to make something *you* think is cool and see it run in the
visualizer.

## Some directions (pick one, or do your own)

- **A new goal in Obstruction2D.** Unstack, swap two blocks, line them up in
  order, build a taller tower, sort by size. (Closest to what you've done.)
- **A new scene from the same robot.** The CRV robot + rectangles can express a
  lot: a "clear the workspace" task, a wall the robot must build from blocks, a
  narrow corridor it must thread (motion planning shines here).
- **A different kinder environment.** `kinder` has more than the 2D kinematic
  worlds — browse `kinder.envs` (and the `examples/` in `kinder-bilevel-planning`,
  which set up planning models for several of them) and write models for one.
- **Something else entirely.** A brand-new environment, a different robot, a
  multi-step puzzle. Use the lab as a starting point.

## Three concrete paths (pick one)

Each is a short recipe — *goal → what to add → how to run*. All reuse the Part 1/2
machinery; copy a `run.py` as your starting point.

### A. Unstack — put a stacked block back on the table

- **Goal.** Start from a stacked state (obstruction on the target, like the end
  of Part 1) and ask for `OnTable(obstruction0)`.
- **What to add.** Nothing new conceptually: you already have `On`, `OnTable`,
  `Holding`, and a pick. Add an **`Unstack` operator** (precondition: `On(block,
  support)` + `HandEmpty`; effects: `Holding(block)`, not `On`) paired with a
  pick-style skill that grasps the *upper* block, and a **`PutOnTable`** operator
  + place skill (release on an empty patch of table → `OnTable`).
- **Run.** Copy `part1_stacking/run.py`, set the initial state stacked, change the
  goal, then `python -m your_run` and view the printed visualizer command.

### B. Line them up — order three blocks left to right

- **Goal.** `LeftOf(a, b)` ∧ `LeftOf(b, c)` with all three `OnTable`.
- **What to add.** A **`LeftOf` predicate** (classifier: `a`'s right edge < `b`'s
  left edge, both on the table — see `is_adjacent` in the Part 2 solution for the
  geometry pattern), and a **`PlaceLeftOf`/`PlaceRightOf`** place skill (subclass
  `MotionPlannedController`; choose a release `x` that lands the block on the
  correct side, like Part 2's `PlaceNextTo`).
- **Run.** Copy `part2_pyramid/run.py`; start the blocks out of order.

### C. Thread a corridor — motion planning shines

- **Goal.** Move one block from the far left to the far right; `On`/`OnTable` goal
  as in Part 1.
- **What to add.** Mostly *geometry*, not new operators: in your `run.py` place
  two tall barriers leaving a narrow gap, so the provided BiRRT motion planner has
  to route the carried block through it. Reuse Part 1's `Stack`/`place_on_block`
  (or a `PutOnTable`) unchanged and watch the path in the visualizer.

> Want a bigger jump? Browse `kinder.envs` and the `examples/` in
> `kinder-bilevel-planning` (several set up planning models for other envs) and
> write models for a non-Obstruction2D world.

## The toolkit you already have

- **Predicates**: classifiers over the state (Parts 1–2). Add them to the state
  abstractor.
- **Operators**: lifted preconditions/effects (`LiftedOperator`).
- **Skills**: controllers. Subclass `crv_skills.MotionPlannedController` to get a
  motion planner for free, or write your own.
- **Models**: assemble everything into `SesameModels` (see your `models.py`).
- **Run + visualize**: copy a `run.py`, point it at your models, and use the
  printed `python -m bilevel_planning.visualizer ...` command (with
  `../renderer.py`) to watch it.

## A good iteration loop

1. Decide on a task and the goal predicate that captures it.
2. Write (or sketch) the predicate(s)/operator(s)/skill(s), reusing the patterns
   from Parts 1–2.
3. Run the planner; when it fails, read the failure and iterate.
4. Visualize the solve and show a neighbor.

AI assistants are welcome here if you want them — e.g. to draft a predicate or
debug a failure — but they're entirely optional; do whatever you find most fun.
