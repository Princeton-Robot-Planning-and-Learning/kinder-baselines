# Part 3 — Make your own thing 🚀

**AI assistants are allowed for this part.** Go wild.

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
