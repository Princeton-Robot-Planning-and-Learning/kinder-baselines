"""Generate the Colab notebooks for the bilevel-planning lab.

Run ``python build_notebooks.py`` to (re)write the student notebooks
``lab_part1_stacking.ipynb`` and ``lab_part2_pyramid.ipynb`` next to this file.

Design: the heavy *provided* machinery (env wiring, the motion-planned
controllers, the model assembler) lives in non-editable cells defined here as
string constants. The exercise *holes* are factored into the
``*_HOLE`` constants, and the two ``make_*_notebook`` builders take the hole
sources as arguments. The student notebooks pass the ``*_HOLE`` versions; the
private solutions repo imports these builders and passes the worked solutions
instead -- so solution text never lives in this public repo.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO_URL = (
    "https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines.git"
)

# --------------------------------------------------------------------------- #
# Shared setup cell: on Colab, clone the lab and install the 2D-only packages;
# locally, just put the lab on the path. Self-contained (no editable/local
# packages -- the lab vendors what it needs), so it's short and explicit.
# --------------------------------------------------------------------------- #
SETUP_CELL = f"""\
# === Setup: run me first ===
# On Colab: clones the lab and installs the 2D-only packages (no PyBullet;
# ~1-2 min the first time). Locally: just puts the lab on the path.
import os
import subprocess
import sys

REPO_URL = "{REPO_URL}"

if "google.colab" in sys.modules:
    if not os.path.exists("/content/kinder-baselines"):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, "/content/kinder-baselines"],
            check=True,
        )
    LAB_DIR = "/content/kinder-baselines/kinder-bilevel-planning/lab"
    try:
        import kinder  # already installed on a warm runtime?
    except ImportError:
        pip = [sys.executable, "-m", "pip", "install", "-q"]
        # Just the environments (no heavy sim backends) + the 2D deps, which
        # include the bilevel_planning planner.
        subprocess.run(pip + ["--no-deps", "kindergarden"], check=True)
        subprocess.run(pip + ["-r", LAB_DIR + "/requirements/lab2d.txt"], check=True)
else:
    LAB_DIR = os.getcwd()
    while LAB_DIR != "/" and not os.path.isdir(
        LAB_DIR + "/kinder-bilevel-planning/lab"
    ):
        LAB_DIR = os.path.dirname(LAB_DIR)
    LAB_DIR += "/kinder-bilevel-planning/lab"

for _p in (LAB_DIR, LAB_DIR + "/notebooks"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colab_utils  # provided visualization helpers
import kinder

kinder.register_all_environments()
print("\\u2705 setup complete")
"""

# --------------------------------------------------------------------------- #
# PART 1 -- stacking
# --------------------------------------------------------------------------- #
P1_INTRO_MD = """\
# Bilevel planning lab — Part 1: stack the obstruction on the target

This is the Colab version of Part 1 of the bilevel (task-and-motion) planning
lab. **Follow along with the instructor. No AI assistants for this part.**

We'll make a planner achieve a new goal in the **Obstruction2D** world: *pick up
the obstruction and stack it on top of the target block.* The planner needs three
new pieces, and you'll fill a small hole in each:

| | piece | what it is |
|---|---|---|
| **TODO(1)** | `On` **predicate** | a *classifier*: is one block resting on another? |
| **TODO(2)** | `Stack` **operator** | the *abstract action*: its preconditions and effects |
| **TODO(3)** | `place_on_block` **skill** | the *controller*: where the robot puts the block down |

Everything else — the environment, the rest of the abstraction, the other
skills, and the **motion planner** that routes the robot around obstacles — is
provided. Run the cells top to bottom; fill the three `# TODO` cells; then watch
the planner solve it.
"""

# This cell is provided -- students do not edit it. It defines the predicates and
# operator variables your TODO cells use, the provided controllers, and a
# ``build_stacking_models(env)`` that wires everything (including your TODO
# functions/values) into a planner-ready ``SesameModels``.
P1_PROVIDED = '''\
# === Provided machinery (you do NOT edit this cell) ===
import numpy as np
from bilevel_planning.sesame import run_sesame
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from kinder.envs.kinematic2d.obstruction2d import (
    ObjectCentricObstruction2DEnv,
    TargetBlockType,
    TargetSurfaceType,
)
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    is_on,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricStateSpace
from crv_skills import (
    MotionPlannedController,
    make_lifted_controller,
    make_lifted_pick_controller,
)
import colab_utils

# Predicates (``On`` is the new one) and operator variables. Your TODO cells
# reference these names.
Holding = Predicate("Holding", [CRVRobotType, RectangleType])
HandEmpty = Predicate("HandEmpty", [CRVRobotType])
OnTable = Predicate("OnTable", [RectangleType])
OnTarget = Predicate("OnTarget", [RectangleType])
On = Predicate("On", [RectangleType, RectangleType])

robot = Variable("?robot", CRVRobotType)
block = Variable("?block", RectangleType)
support = Variable("?support", RectangleType)


def get_robot_stack_position(held, support, state, robot_x, robot_arm_joint):
    """Robot (x, y) at which releasing ``held`` leaves it resting on ``support``."""
    robot_obj = state.get_objects(CRVRobotType)[0]
    ground = support_top_y(state, support)  # <-- your TODO(3)
    padding = 1e-4
    y = (
        ground
        + state.get(held, "height")
        + robot_arm_joint
        + state.get(robot_obj, "gripper_width") / 2
        + padding
    )
    return (robot_x, y)


class GroundPlaceOnBlockController(MotionPlannedController):
    """Stack a held block on a support block. Objects: [robot, held, support]."""

    def __init__(self, objects, action_space, init_constant_state=None):
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._support = objects[2]

    def sample_parameters(self, x, rng):
        support_x = x.get(self._support, "x")
        support_width = x.get(self._support, "width")
        block_x = x.get(self._block, "x")
        robot_x = x.get(self._robot, "x")
        offset_x = robot_x - block_x
        block_width = x.get(self._block, "width")
        lower_x = support_x + offset_x
        upper_x = lower_x + (support_width - block_width)
        if lower_x > upper_x:
            lower_x, upper_x = upper_x, lower_x
        return rng.uniform(lower_x, upper_x)

    def _retract_arm_in_transit(self):
        return False  # carrying a block: keep the arm out so it's carried + checked

    def _get_vacuum_actions(self):
        return 1.0, 0.0  # hold while moving, release at the end

    def _target_pose_and_arm(self, state):
        arm = state.get(self._robot, "arm_joint")
        placement_x = (
            self._current_params[0]
            if isinstance(self._current_params, (tuple, list))
            else self._current_params
        )
        tx, ty = get_robot_stack_position(
            self._block, self._support, state, placement_x, arm
        )
        return SE2Pose(tx, ty, state.get(self._robot, "theta")), arm


def build_stacking_models(env, num_obstructions=2):
    """Assemble SesameModels from the provided pieces + your TODO cells."""
    observation_space = env.observation_space
    action_space = env.action_space
    init_constant_state = getattr(
        env.unwrapped, "_object_centric_env"
    ).initial_constant_state
    sim = ObjectCentricObstruction2DEnv(num_obstructions=num_obstructions)

    def observation_to_state(o):
        return observation_space.devectorize(o)

    def transition_fn(x, u):
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {CRVRobotType, RectangleType, TargetBlockType, TargetSurfaceType}
    state_space = ObjectCentricStateSpace(types)
    predicates = {Holding, HandEmpty, OnTable, OnTarget, On}

    def state_abstractor(x):
        robot_o = Object("robot", CRVRobotType)
        target = Object("target_block", TargetBlockType)
        target_surface = Object("target_surface", TargetSurfaceType)
        obstructions = {
            Object(f"obstruction{i}", RectangleType) for i in range(num_obstructions)
        }
        blocks = obstructions | {target}
        atoms = set()
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot_o)}
        for obj in suctioned_objs & blocks:
            atoms.add(GroundAtom(Holding, [robot_o, obj]))
        if not suctioned_objs:
            atoms.add(GroundAtom(HandEmpty, [robot_o]))
        for blk in blocks:
            if blk in suctioned_objs:
                continue
            if is_on(x, blk, target_surface, {}):
                atoms.add(GroundAtom(OnTarget, [blk]))
                continue
            supp = find_support(x, blk, blocks - {blk})  # <-- your TODO(1)
            if supp is not None:
                atoms.add(GroundAtom(On, [blk, supp]))
            else:
                atoms.add(GroundAtom(OnTable, [blk]))
        objects = {robot_o, target, target_surface} | obstructions
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(x):
        del x
        obstruction = Object("obstruction0", RectangleType)
        target = Object("target_block", TargetBlockType)
        return RelationalAbstractGoal(
            {GroundAtom(On, [obstruction, target])}, state_abstractor
        )

    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
    )
    StackOperator = LiftedOperator(
        "Stack",
        [robot, block, support],
        preconditions=STACK_PRECONDITIONS,  # <-- your TODO(2)
        add_effects=STACK_ADD_EFFECTS,  # <-- your TODO(2)
        delete_effects=STACK_DELETE_EFFECTS,  # <-- your TODO(2)
    )

    controllers = {
        "pick": make_lifted_pick_controller(action_space, init_constant_state),
        "place_on_block": make_lifted_controller(
            [robot, block, support],
            GroundPlaceOnBlockController,
            action_space,
            init_constant_state,
        ),
    }
    skills = {
        LiftedSkill(PickFromTableOperator, controllers["pick"]),
        LiftedSkill(StackOperator, controllers["place_on_block"]),
    }
    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
'''

P1_TODO1_MD = """\
## TODO(1) — the `On` predicate's classifier

A predicate is just a function of the state. Return the block in `candidates`
that `block` is resting on, or `None`. `is_on(x, a, b, {})` is `True` exactly when
block `a` is resting on top of block `b`.
"""

P1_FIND_SUPPORT_HOLE = '''\
def find_support(x, block, candidates):
    """Return the block in `candidates` that `block` rests on, else None.

    `is_on(x, a, b, {})` is True exactly when block `a` rests on top of block `b`.
    `block` is never resting on itself.
    """
    # TODO(1): look through `candidates` and return the one that `block` is
    # resting on (use `is_on`), or None if it is on none of them.
    raise NotImplementedError("TODO(1): On predicate -- find the support block")
'''

P1_TODO2_MD = """\
## TODO(2) — the `Stack` operator's preconditions and effects

Build sets of `LiftedAtom`s from the predicates/variables defined above (e.g.
`LiftedAtom(Holding, [robot, block])`). To stack you must be **holding** the block
and the **support** must be on the table; afterward the block is **On** the support
and the hand is **empty** (no longer holding).
"""

P1_STACK_HOLE = """\
# TODO(2): fill in the three sets. Use Holding, OnTable, On, HandEmpty and the
# variables robot, block, support, e.g. LiftedAtom(Holding, [robot, block]).
STACK_PRECONDITIONS = set()  # TODO(2)
STACK_ADD_EFFECTS = set()  # TODO(2)
STACK_DELETE_EFFECTS = set()  # TODO(2)
"""

P1_TODO3_MD = """\
## TODO(3) — the resting height

The held block should come to rest **on top of** the support. A block at `y` with
`height` occupies `[y, y + height]` (`y` is its bottom edge), so its top edge is
`y + height`.
"""

P1_SUPPORT_TOP_Y_HOLE = '''\
def support_top_y(state, support):
    """The y-coordinate of the TOP edge of the `support` block."""
    # TODO(3): return the y of the support block's top edge, so the held block
    # comes to rest ON TOP of it. (Look at state.get(support, "y") / "height".)
    raise NotImplementedError("TODO(3): resting height -- top edge of the support")
'''

P1_CHECK_MD = """\
## Check your work

These cells check your TODOs in isolation, the same way the local lab's `pytest`
does. Re-run them after editing the cells above — a failure is your checklist.
"""

P1_CHECK_PREDICATE = """\
# Checks TODO(1): On fires when stacked, not when side by side.
_env1 = kinder.make("kinder/Obstruction2D-o1-v0")
_obs, _ = _env1.reset(seed=0)
_m = build_stacking_models(_env1, num_obstructions=1)
_state = _m.observation_to_state(_obs).copy()
_target = _state.get_object_from_name("target_block")
_obstruction = _state.get_object_from_name("obstruction0")

_state.set(_target, "x", 0.5); _state.set(_target, "width", 0.16)
_state.set(_target, "height", 0.09)
_state.set(_obstruction, "x", 0.2); _state.set(_obstruction, "width", 0.1)
_state.set(_obstruction, "height", 0.09)
_atoms = _m.state_abstractor(_state).atoms
assert GroundAtom(On, [_obstruction, _target]) not in _atoms
assert GroundAtom(OnTable, [_obstruction]) in _atoms

_top = _state.get(_target, "y") + _state.get(_target, "height")
_state.set(_obstruction, "x", 0.53); _state.set(_obstruction, "y", _top)
_atoms = _m.state_abstractor(_state).atoms
assert GroundAtom(On, [_obstruction, _target]) in _atoms
assert GroundAtom(OnTable, [_obstruction]) not in _atoms
print("\\u2705 TODO(1) On predicate looks correct")
"""

# Geometry constants for the cluttered instance (match part1_stacking/run.py).
P1_LAYOUT = '''\
ENV_NAME = "kinder/Obstruction2D-o2-v0"
SEED = 0
# Hand-designed clutter: obstruction0 (left) gets stacked on the target (right);
# obstruction1 is a tall barrier between them the robot must route around.
OBSTRUCTION_X, OBSTRUCTION_WIDTH, OBSTRUCTION_HEIGHT = 0.25, 0.1, 0.09
TARGET_X, TARGET_WIDTH, TARGET_HEIGHT = 1.15, 0.16, 0.09
BARRIER_X, BARRIER_WIDTH, BARRIER_HEIGHT = 0.75, 0.08, 0.28
ROBOT_X, ROBOT_Y = 0.25, 0.55
TARGET_SURFACE_X = 1.45


def make_cluttered_instance(env, env_models):
    """Reset to the cluttered layout and return (initial_state, constant_state)."""
    constant_state = getattr(
        env.unwrapped, "_object_centric_env"
    ).initial_constant_state
    obs, _ = env.reset(seed=SEED)
    state = env_models.observation_to_state(obs).copy()
    robot_o = state.get_object_from_name("robot")
    state.set(robot_o, "x", ROBOT_X); state.set(robot_o, "y", ROBOT_Y)
    state.set(robot_o, "theta", -np.pi / 2)
    layout = {
        "obstruction0": (OBSTRUCTION_X, OBSTRUCTION_WIDTH, OBSTRUCTION_HEIGHT),
        "target_block": (TARGET_X, TARGET_WIDTH, TARGET_HEIGHT),
        "obstruction1": (BARRIER_X, BARRIER_WIDTH, BARRIER_HEIGHT),
    }
    for name, (x, w, h) in layout.items():
        o = state.get_object_from_name(name)
        state.set(o, "x", x); state.set(o, "width", w); state.set(o, "height", h)
    state.set(state.get_object_from_name("target_surface"), "x", TARGET_SURFACE_X)
    return state, constant_state
'''

P1_CHECK_SKILL = """\
# Checks TODO(2) + TODO(3): pick the obstruction, then route it over the barrier
# and stack it; assert it ends up resting on the target.
_env = kinder.make(ENV_NAME)
_m = build_stacking_models(_env, num_obstructions=2)
assert _m  # build_stacking_models also needs TODO(2): a Stack op with effects
_stack_op = {s.operator.name: s for s in _m.skills}["Stack"].operator
assert _stack_op.add_effects, "TODO(2): the Stack operator has no effects yet"

_init, _const = make_cluttered_instance(_env, _m)
_obs, _ = _env.reset(options={"init_state": _init.copy()})
_skill = {s.operator.name: s for s in _m.skills}
_robot = _init.get_object_from_name("robot")
_obstruction = _init.get_object_from_name("obstruction0")
_target = _init.get_object_from_name("target_block")
_rng = np.random.default_rng(123)

for _ground in [
    _skill["PickFromTable"].ground((_robot, _obstruction)),
    _skill["Stack"].ground((_robot, _obstruction, _target)),
]:
    _state = _m.observation_to_state(_obs)
    _ctrl = _ground.controller
    _ctrl.reset(_state, _ctrl.sample_parameters(_state, _rng))
    for _ in range(400):
        _obs, _, _, _, _ = _env.step(_ctrl.step())
        _ctrl.observe(_m.observation_to_state(_obs))
        if _ctrl.terminated():
            break
_final = _m.observation_to_state(_obs)
_target_top = _final.get(_target, "y") + _final.get(_target, "height")
assert np.isclose(_final.get(_obstruction, "y"), _target_top, atol=1e-3)
print("\\u2705 TODO(2)+TODO(3): the obstruction ends up stacked on the target")
"""

P1_RUNVIZ_MD = """\
## Watch the planner solve it

Now run the full bilevel planner on the cluttered instance and visualize the
solve inline. The storyboard shows key frames; the animation plays the whole
trajectory — watch the robot route the carried block up and over the barrier.
"""

P1_RUNVIZ = """\
env = kinder.make(ENV_NAME)
env_models = build_stacking_models(env, num_obstructions=2)
initial_state, constant_state = make_cluttered_instance(env, env_models)

plan, _ = run_sesame(
    env_models, initial_state, seed=SEED, max_abstract_plans=1,
    samples_per_step=5, max_skill_horizon=200, timeout=120.0,
)
assert plan is not None, "planner found no plan -- check your TODOs above"
print(f"Plan found: {len(plan.actions)} actions.")
colab_utils.show_storyboard(plan.states, constant_state)
"""

P1_ANIM = """\
colab_utils.animate_states(plan.states, constant_state)
"""

P1_FOOTER_MD = """\
\U0001f389 **Done with Part 1!** You gave the planner a new predicate, operator, and
skill, and watched it route a carried block around an obstacle. Next: **Part 2**,
where you design a whole pyramid task yourself.
"""

# --------------------------------------------------------------------------- #
# PART 2 -- pyramid
# --------------------------------------------------------------------------- #
P2_INTRO_MD = """\
# Bilevel planning lab — Part 2: build a pyramid \U0001f53a

**On your own now. Still no AI assistants** (those come in Part 3, which is a
local exercise — see the lab README).

Make the planner build a **pyramid**: the **target block resting on top of two
obstructions**.

```
            ┌─────────────┐
            │ target_block│        <- the cap, on top of both
        ┌───┴───┐ ┌───────┴┐
        │  o0   │ │   o1   │       <- the base
   ═════╧═══════╧═╧════════╧═════  table
```

The two obstructions start **apart** on the table. That's the crux: a single "put
the target on top" won't work — think about the *order of operations*: what has
to be true about the obstructions before the cap can go on?

You design the domain, reusing the Part 1 toolkit:
- **Predicates** — classifiers over the state (TODO A) emitted in the
  state abstractor (TODO B);
- **Operators** — lifted preconditions/effects, and **skills** — controllers
  (subclass `MotionPlannedController` like Part 1's place skill) (TODO C);
- the **goal** (TODO D).

Fill the two editable cells (skills, then models). Geometry reminder: a block at
`y` with `height` occupies `[y, y + height]`; `x` is its left edge. `is_on(state,
a, b, {})` is True when `a` rests on `b`; `rectangle_object_to_geom(state, o, {})`
gives a geom with `.vertices` and `.contains_point(px, py)` for corner checks.
"""

P2_PROVIDED = """\
# === Provided imports (you do NOT edit this cell) ===
import numpy as np
from bilevel_planning.sesame import run_sesame
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from kinder.envs.kinematic2d.obstruction2d import (
    ObjectCentricObstruction2DEnv,
    TargetBlockType,
    TargetSurfaceType,
)
from kinder.envs.kinematic2d.structs import SE2Pose
from kinder.envs.kinematic2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    is_on,
    rectangle_object_to_geom,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricStateSpace
from crv_skills import (
    MotionPlannedController,
    get_robot_transfer_position,
    make_lifted_controller,
    make_lifted_pick_controller,
)
import colab_utils
"""

P2_SKILLS_MD = """\
## Your skills

Write the place skill(s) the pyramid needs. To make a place-style skill, subclass
`MotionPlannedController` and implement `_target_pose_and_arm(state)` (where to end
up + arm length), `_retract_arm_in_transit()` (`False` while carrying),
`_get_vacuum_actions()` (`(1.0, 0.0)` to hold then release), and
`sample_parameters(state, rng)` (where to release). Return the lifted controllers
from `create_pyramid_controllers`, keyed by name (`pick` is provided).
"""

P2_SKILLS_HOLE = '''\
# TODO: define your place controller class(es) here, e.g.
#
#   class GroundPlaceNextToController(MotionPlannedController):
#       def __init__(self, objects, action_space, init_constant_state=None):
#           super().__init__(objects, action_space, init_constant_state)
#           self._block = objects[1]
#           ...
#       def _retract_arm_in_transit(self): return False
#       def _get_vacuum_actions(self): return 1.0, 0.0
#       def sample_parameters(self, x, rng): ...
#       def _target_pose_and_arm(self, state): ...


def create_pyramid_controllers(action_space, init_constant_state=None):
    """Return the lifted controllers your skills need (keyed by name)."""
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    controllers = {
        "pick": make_lifted_pick_controller(action_space, init_constant_state),
    }
    # TODO: add your place controllers, e.g.
    #   controllers["place_next_to"] = make_lifted_controller(
    #       [robot, block, anchor], GroundPlaceNextToController, action_space,
    #       init_constant_state)
    return controllers
'''

P2_MODELS_MD = """\
## Your models

Define your predicate classifiers, build the `SesameModels` (predicates =
**TODO A**, abstractor logic = **TODO B**, operators + skills = **TODO C**, goal =
**TODO D**), reusing your skills from the cell above. `PickFromTable` is provided
as the pattern to copy.
"""

P2_MODELS_HOLE = '''\
# TODO(A): define any predicate classifier helpers you need (like Part 1's
# find_support), e.g. is_adjacent(x, left, right) / is_bridging(x, top, l, r).


def create_pyramid_models(env, num_obstructions=2):
    """Create the planning models for the pyramid task."""
    observation_space = env.observation_space
    action_space = env.action_space
    init_constant_state = getattr(
        env.unwrapped, "_object_centric_env"
    ).initial_constant_state
    sim = ObjectCentricObstruction2DEnv(num_obstructions=num_obstructions)

    def observation_to_state(o):
        return observation_space.devectorize(o)

    def transition_fn(x, u):
        state = x.copy()
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {CRVRobotType, RectangleType, TargetBlockType, TargetSurfaceType}
    state_space = ObjectCentricStateSpace(types)

    Holding = Predicate("Holding", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTable = Predicate("OnTable", [RectangleType])
    OnTarget = Predicate("OnTarget", [RectangleType])
    # TODO(A): define the predicate(s) the pyramid needs and add them to this set.
    predicates = {Holding, HandEmpty, OnTable, OnTarget}

    def state_abstractor(x):
        robot_o = Object("robot", CRVRobotType)
        target = Object("target_block", TargetBlockType)
        target_surface = Object("target_surface", TargetSurfaceType)
        obstructions = {
            Object(f"obstruction{i}", RectangleType) for i in range(num_obstructions)
        }
        blocks = obstructions | {target}
        atoms = set()
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot_o)}
        for obj in suctioned_objs & blocks:
            atoms.add(GroundAtom(Holding, [robot_o, obj]))
        if not suctioned_objs:
            atoms.add(GroundAtom(HandEmpty, [robot_o]))
        for blk in blocks:
            if blk in suctioned_objs:
                continue
            if is_on(x, blk, target_surface, {}):
                atoms.add(GroundAtom(OnTarget, [blk]))
                continue
            # TODO(B): emit your predicate atoms for `blk` (its relationship to the
            # OTHER blocks) instead of always calling it OnTable.
            atoms.add(GroundAtom(OnTable, [blk]))
        return RelationalAbstractState(atoms, {robot_o, target, target_surface} | obstructions)

    def goal_deriver(x):
        # TODO(D): return the goal for the finished pyramid (target supported by
        # both obstructions). Use a predicate you defined.
        del x
        raise NotImplementedError("TODO(D): the pyramid goal")

    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
    )
    # TODO(C): define the operator(s) the pyramid needs and pair each with a skill.

    controllers = create_pyramid_controllers(action_space, init_constant_state)
    skills = {
        LiftedSkill(PickFromTableOperator, controllers["pick"]),
        # TODO(C): add LiftedSkill(YourOperator, controllers["your_skill"]) entries.
    }

    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
'''

P2_LAYOUT = '''\
ENV_NAME = "kinder/Obstruction2D-o2-v0"
SEED = 0
# All three blocks on the table, none adjacent (matches the local part2 run.py).
LAYOUT = {
    "target_block": (0.15, 0.2, 0.09),
    "obstruction0": (0.4, 0.15, 0.09),
    "obstruction1": (1.1, 0.15, 0.09),
}
ROBOT_X, ROBOT_Y = 0.85, 0.85
TARGET_SURFACE_X = 1.45


def make_pyramid_instance(env, env_models):
    """Reset to the non-adjacent layout; return (initial_state, constant_state)."""
    constant_state = getattr(
        env.unwrapped, "_object_centric_env"
    ).initial_constant_state
    obs, _ = env.reset(seed=SEED)
    state = env_models.observation_to_state(obs).copy()
    robot_o = state.get_object_from_name("robot")
    state.set(robot_o, "x", ROBOT_X); state.set(robot_o, "y", ROBOT_Y)
    state.set(robot_o, "theta", -np.pi / 2)
    for name, (x, w, h) in LAYOUT.items():
        o = state.get_object_from_name(name)
        state.set(o, "x", x); state.set(o, "width", w); state.set(o, "height", h)
    state.set(state.get_object_from_name("target_surface"), "x", TARGET_SURFACE_X)
    return state, constant_state


def plan_pyramid():
    """Build models, plan on the instance; return (plan, constant_state)."""
    env = kinder.make(ENV_NAME)
    env_models = create_pyramid_models(env, num_obstructions=2)
    initial_state, constant_state = make_pyramid_instance(env, env_models)
    plan, _ = run_sesame(
        env_models, initial_state, seed=SEED, max_abstract_plans=5,
        samples_per_step=5, max_skill_horizon=200, timeout=120.0,
    )
    return plan, constant_state
'''

P2_CHECK_MD = """\
## Done when... (the spec)

This is what "done" means: plan with your models, then check the **geometry** of
the final state — the target rests on top of two obstructions that ended up side
by side as a base. It doesn't care what you named anything.
"""

P2_CHECK = """\
plan, constant_state = plan_pyramid()
assert plan is not None, "planner found no plan -- check your predicates/operators/skills/goal"

final = plan.states[-1]
tb = final.get_object_from_name("target_block")
o0 = final.get_object_from_name("obstruction0")
o1 = final.get_object_from_name("obstruction1")


def _span(o):
    return final.get(o, "x"), final.get(o, "x") + final.get(o, "width")


_left, _right = sorted([o0, o1], key=lambda o: final.get(o, "x"))
_left_lo, _left_hi = _span(_left)
_right_lo, _right_hi = _span(_right)
_gap = _right_lo - _left_hi
assert -1e-3 <= _gap <= 0.05, f"obstructions are not adjacent (gap={_gap:.3f})"

_obstruction_top = final.get(_left, "y") + final.get(_left, "height")
assert np.isclose(final.get(tb, "y"), _obstruction_top, atol=2e-3), "target not on top"

_tb_lo = final.get(tb, "x")
_tb_hi = _tb_lo + final.get(tb, "width")
assert _left_lo <= _tb_lo <= _left_hi, "target's left corner is not on the left base"
assert _right_lo <= _tb_hi <= _right_hi, "target's right corner is not on the right base"
print("\\u2705 It's a real pyramid!")
"""

P2_RUNVIZ_MD = """\
## Watch it build

Visualize the solve inline — storyboard then animation.
"""

P2_STORYBOARD = """\
colab_utils.show_storyboard(plan.states, constant_state)
"""

P2_ANIM = """\
colab_utils.animate_states(plan.states, constant_state)
"""


# --------------------------------------------------------------------------- #
# Notebook assembly
# --------------------------------------------------------------------------- #
def _notebook(cells):
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": []}
    return nb


def make_part1_notebook(find_support_src, stack_src, support_top_y_src):
    """Build the Part 1 notebook with the given (hole or solution) sources."""
    return _notebook(
        [
            new_markdown_cell(P1_INTRO_MD),
            new_code_cell(SETUP_CELL),
            new_code_cell(P1_PROVIDED),
            new_markdown_cell(P1_TODO1_MD),
            new_code_cell(find_support_src),
            new_markdown_cell(P1_TODO2_MD),
            new_code_cell(stack_src),
            new_markdown_cell(P1_TODO3_MD),
            new_code_cell(support_top_y_src),
            new_markdown_cell(P1_CHECK_MD),
            new_code_cell(P1_CHECK_PREDICATE),
            new_code_cell(P1_LAYOUT),
            new_code_cell(P1_CHECK_SKILL),
            new_markdown_cell(P1_RUNVIZ_MD),
            new_code_cell(P1_RUNVIZ),
            new_code_cell(P1_ANIM),
            new_markdown_cell(P1_FOOTER_MD),
        ]
    )


def make_part2_notebook(skills_src, models_src):
    """Build the Part 2 notebook with the given (hole or solution) sources."""
    return _notebook(
        [
            new_markdown_cell(P2_INTRO_MD),
            new_code_cell(SETUP_CELL),
            new_code_cell(P2_PROVIDED),
            new_markdown_cell(P2_SKILLS_MD),
            new_code_cell(skills_src),
            new_markdown_cell(P2_MODELS_MD),
            new_code_cell(models_src),
            new_code_cell(P2_LAYOUT),
            new_markdown_cell(P2_CHECK_MD),
            new_code_cell(P2_CHECK),
            new_markdown_cell(P2_RUNVIZ_MD),
            new_code_cell(P2_STORYBOARD),
            new_code_cell(P2_ANIM),
        ]
    )


def write_notebooks(notebooks_by_path):
    """Write a ``{path: notebook}`` mapping to disk.

    Shared by ``write_student_notebooks`` and the private solved-notebook
    generator, which imports this module and supplies the worked solutions.
    """
    for path, nb in notebooks_by_path.items():
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"wrote {path}")
    return list(notebooks_by_path)


def write_student_notebooks(out_dir):
    """Write the student (TODO-hole) notebooks into ``out_dir``."""
    out_dir = Path(out_dir)
    return write_notebooks(
        {
            out_dir
            / "lab_part1_stacking.ipynb": make_part1_notebook(
                P1_FIND_SUPPORT_HOLE, P1_STACK_HOLE, P1_SUPPORT_TOP_Y_HOLE
            ),
            out_dir
            / "lab_part2_pyramid.ipynb": make_part2_notebook(
                P2_SKILLS_HOLE, P2_MODELS_HOLE
            ),
        }
    )


if __name__ == "__main__":
    write_student_notebooks(Path(__file__).parent)  # type: ignore[no-untyped-call]
