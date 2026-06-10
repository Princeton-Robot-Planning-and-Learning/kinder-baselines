# Part 1 — Stack the obstruction on the target block

**Follow along with the instructor. No AI assistants for this part.**

We'll make the planner achieve a new goal in the Obstruction2D world: **pick up
the obstruction and stack it on top of the target block.** To do that, the
planner needs three new pieces, and you'll fill a small hole in each:

| | piece | file | what it is |
|---|---|---|---|
| **TODO(1)** | `On` **predicate** | `models.py` | a *classifier*: is one block resting on another? |
| **TODO(2)** | `Stack` **operator** | `models.py` | the *abstract action*: its preconditions and effects |
| **TODO(3)** | `place_on_block` **skill** | `skills.py` | the *controller*: where the robot puts the block down |

Everything else is provided — the environment, the rest of the abstraction, the
other skills, and the **motion planner** that figures out a collision-free path
for the robot (you'll see why that matters in `run.py`, where a barrier sits
between the obstruction and the target).

## Do the TODOs

Open the files and find the `# TODO(n)` markers. In order:

1. **TODO(1)** in `models.py` → `find_support`: return the block that a given
   block is resting on (or `None`). A predicate is just a function of the state.
2. **TODO(2)** in `models.py` → the `Stack` operator: fill in its `preconditions`,
   `add_effects`, and `delete_effects`. Think about what must be true to stack,
   and what becomes true afterward.
3. **TODO(3)** in `skills.py` → `support_top_y`: the height at which the held
   block comes to rest (the top edge of the support block).

## Check your work

```
python -m pytest part1_stacking -q
```

- `test_predicate.py` checks TODO(1).
- `test_skill.py` checks TODO(2) and TODO(3) (it picks the obstruction and stacks it).

When both pass, watch the planner solve it — the robot routes the carried block
up and over the barrier:

```
python -m part1_stacking.run
# prints a `python -m bilevel_planning.visualizer ...` command; run that to watch it
```
