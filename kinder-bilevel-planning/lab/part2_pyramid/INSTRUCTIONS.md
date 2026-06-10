# Part 2 — Build a pyramid 🔺

**On your own now. Still no AI assistants** (those come in Part 3).

Make the planner build a **pyramid**: the **target block resting on top of two
obstructions**, like this:

```
            ┌─────────────┐
            │ target_block│        <- the cap, on top of both
        ┌───┴───┐ ┌───────┴┐
        │  o0   │ │   o1   │       <- the base
   ═════╧═══════╧═╧════════╧═════  table
```

In `run.py` (and the spec test) the two obstructions start **apart** on the
table. That's the crux of this exercise: a single "put the target on top" won't
work — **think about what has to be true before the cap can go on, and add
whatever predicate / operator / skill makes that happen.**

## What to build

You design the domain. Reuse the patterns from Part 1 — the machinery is the
same, only the task is new. You'll likely add:

- one or more **predicates** (with classifier logic in the state abstractor) —
  `models.py` TODO(A), TODO(B);
- one or more **operators** (preconditions + effects) — `models.py` TODO(C);
- one or more **place skills** — `skills.py` (subclass `MotionPlannedController`
  like your Part 1 `place_on_block`);
- the **goal** — `models.py` TODO(D).

Geometry reminders: a block at `y` with `height` occupies `[y, y + height]` (`x`
is its left edge). `is_on(state, a, b, {})` is True when `a` rests on `b`;
`rectangle_object_to_geom(state, o, {})` (in `kinder.envs.kinematic2d.utils`)
gives a geom with `.vertices` and `.contains_point(px, py)` for corner checks.

## Done when

```
python -m pytest part2_pyramid/test_pyramid.py -q
```

passes. The test plans with your models and checks the **final geometry** is a
real pyramid (base side by side + target bridging it) — it doesn't care what you
named anything. Then watch it:

```
python -m part2_pyramid.run
```
