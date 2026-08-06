# CLAUDE.md

Guidance for Claude Code (claude.ai/code) and for anyone else working in this repo.

`kinder-baselines` holds baseline methods for the
[KinDER](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden)
benchmark. See `README.md` for the package list and the install steps; this file covers
the conventions the code follows and the traps that are not visible from the config.

## Layout

Ten sibling packages, each self-contained: its own `pyproject.toml`, `.pylintrc`,
`run_autoformat.sh`, usually `run_ci_checks.sh`, `src/<distribution_name>/`, and
`tests/` mirroring `src/`. There is no root `pyproject.toml` and no shared settings
file — every package carries its own copy of the whole toolchain config, including its
own `pylint_plugins/` directory. A change to shared policy therefore has to be applied
ten times.

Packages depend on each other through `prpl_requirements.txt` (one relative path per
line). `scripts/generate_topological_order.py` reads those to order installs;
`scripts/install_all.py` uses that ordering, and `scripts/get_affected_packages.py`
uses the reverse graph to decide what CI runs. Adding an intra-repo dependency means
adding a line to `prpl_requirements.txt`, not just an import.

Two packages are exceptions worth knowing before you write a script that assumes
uniformity: `kinder-reward-grounding` has no `run_ci_checks.sh` (so
`./run_all_ci_checks.sh` fails there), and `kinder-rl` and `kinder-reward-grounding`
have no `pylint_plugins/` — `kinder-rl`'s `.pylintrc` has an empty `load-plugins=`.

## Running the checks

Run them from **inside the package directory**, not from the repo root:

```bash
cd <package-dir>
./run_ci_checks.sh
```

which is:

```bash
./run_autoformat.sh                                    # black, docformatter, isort
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc
pytest tests/
```

Two things about this are easy to get wrong.

**`run_autoformat.sh` reformats; it does not check.** It runs `black`, `docformatter -i`
and `isort` in place with no `--check`, and CI's `autoformat` job runs the same script.
So unformatted code does not fail CI — it is silently rewritten in the CI checkout and
thrown away, leaving `main` unformatted. Run it locally and commit the result.

**Pylint's plugin needs `pylint_plugins` on `sys.path`, and fails open when it is not.**
`load-plugins=pylint_plugins.no_np_random` is a plain import, resolved against
`sys.path`. Invoked the CI way — `pytest . --pylint` from inside the package — it
resolves, because pytest puts the invocation directory on `sys.path`. Invoke pylint
directly and it does not:

```bash
# from inside the package directory
PYTHONPATH=. pylint --rcfile=.pylintrc src/
# from the repo root
PYTHONPATH=kinder-models pylint --rcfile=kinder-models/.pylintrc kinder-models/src/
```

Without it pylint emits `E0013: Plugin 'pylint_plugins.no_np_random' is impossible to
load`, runs every *other* checker, and still prints `Your code has been rated at
10.00/10`. That is a pass you cannot trust. If you lint by hand, grep the output for
`E0013` before believing the score. (`kinder-mbrl/pylint_plugins/` also has no
`__init__.py`, so it is not an importable package there at all.)

## What CI runs

`.github/workflows/ci.yml` has four jobs — `autoformat`, `linting`,
`static-type-checking`, `unit-tests` — all on Python 3.10, installed with `uv`. Each
wraps `./run_in_affected_packages.sh` with `SEQUENTIAL: "true"`.

CI is **selective**. On a pull request `CI_BASE_SHA` is set and
`scripts/get_affected_packages.py` narrows the run to the changed packages plus
everything that transitively depends on them. It falls back to *all* packages when any
of the following is true: the change touches `.github/`, `scripts/`, or a root
`run_*.sh`; the change touches any file outside a package directory; the diff is empty;
or the dependency graph fails to build. On a push to `main`, `CI_BASE_SHA` is empty and
everything runs. Practical consequence: a `kinder-models` change also runs
`kinder-bilevel-planning`, `kinder-blockly`, `kinder-ds-policies` and
`kinder-vlm-planning`, because they list `../kinder-models` in `prpl_requirements.txt`.
A change confined to `kinder-bilevel-planning` runs it and `kinder-vlm-planning`.

Notebooks are **not** a CI job. `kinder-bilevel-planning/run_ci_checks.sh` runs
`pytest notebooks/ --nbmake --nbmake-timeout=120` locally, and passes `--ignore=notebooks`
to its pylint step, but CI's `linting` job does not pass that flag. Notebooks under
`kinder-bilevel-planning/` therefore get linted by CI and executed only locally.

`unit-tests` runs with `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=""`, and apt-installs
`liblapack-dev libblas-dev libosmesa6-dev libgl1 libglx-mesa0` for the `linting` and
`unit-tests` jobs.

## Style

Enforced by the tools, so mostly you just run them — but the settings that shape how
code *reads*:

- black, line length 88, target `py310`. docformatter at 88 for summaries and
  descriptions. Pylint's own `max-line-length=89`, which is why black's output passes.
- isort `profile = "black"`, `multi_line_output = 2`, `split_on_trailing_comma = true`.
  Let isort produce the import blocks; hand-formatted ones will be rewritten.
- mypy with `strict_equality`, `disallow_untyped_calls`, `warn_unreachable`. Silence an
  untyped third-party module by adding it to that package's
  `[[tool.mypy.overrides]]` `module` list in `pyproject.toml`, not with a blanket
  `# type: ignore` at the import site.
- **`np.random` is banned** by `pylint_plugins/no_np_random.py` — every attribute under
  it except `default_rng` and `Generator`. Thread an explicit
  `rng: np.random.Generator` through instead. Seeds in tests are fixed literals
  (`seed = 123`, `np.random.default_rng(123)`), never drawn.
- Docstrings are required on modules, classes and functions, including test functions
  and nested helpers, in imperative one-line-summary style.
- `invalid-name` is **disabled**, which is why CapWords locals such as
  `OpenDrawerOperator` and `GraspThreshold` are idiomatic here rather than a lint
  failure. So are `too-many-*`, `consider-using-with` and `raise-missing-from`; the
  `[DESIGN]` limits are all set to 1000. `duplicate-code` is disabled outright in
  `kinder-models`, and elsewhere has `min-similarity-lines=100`, so a new env model
  that is a near-copy of its sibling is the accepted pattern, not something to
  refactor into a shared helper.
- Suppress pylint narrowly and inline where it is genuinely needed, e.g.
  `# pylint: disable=protected-access` on `env.unwrapped._object_centric_env` or on
  `agent._current_plan`, and `# pylint: disable=global-statement` in the `conftest.py`
  files. `from conftest import MAKE_VIDEOS` needs no pragma in 25 of the 27 test files
  that use it; do not add one by default.

## Adding an environment: where the pieces go

A new KinDER environment usually needs two packages.

**`kinder-models`** — `src/kinder_models/<regime>/<env>/`, where `<regime>` is
`kinematic2d`, `dynamic2d`, `kinematic3d` or `dynamic3d`:

- `parameterized_skills.py` defines the `ParameterizedController` subclasses and a
  `create_lifted_controllers(action_space, ...) -> dict[str, LiftedParameterizedController]`
  factory keyed by the operator name the planner will use.
- `state_abstractions.py` defines module-level `Predicate(...)` constants in CapWords
  and one `<Env>StateAbstractor` class exposing `state_abstractor(state) ->
  RelationalAbstractState` and `goal_deriver(state) -> RelationalAbstractGoal`.

**`kinder-bilevel-planning`** —
`src/kinder_bilevel_planning/env_models/<regime>/<name>.py`, exposing exactly
`create_bilevel_planning_models(observation_space, action_space, **kwargs) -> SesameModels`.

`env_models/__init__.py` finds that file **by filename**, with
`importlib.util.spec_from_file_location`, searching `kinematic2d/`, `dynamic2d/`,
`kinematic3d/`, `dynamic3d/` in that order. There is no registry to edit and no
`__init__.py` to touch — but the string the caller passes must equal the module's
filename stem exactly, **including case**. Existing stems are inconsistent about it
(`tidybot3d_sweep3D.py`, `tidybot3d_shelf3D.py`), and the test filenames deliberately
do not match the modules they test (`test_tidybot3d_sweep3d.py`), so copy the stem from
the file, not from the test.

Follow the section order the existing models use, section comments included: assert on
the spaces → build the `sim` from a `task_config_path` under `Path(kinder.__file__).parent`
→ construct the abstractor → `sim.reset()` → `observation_to_state` → `transition_fn` →
`types` → `ObjectCentricStateSpace` → `predicates` → `Variable`s and `LiftedOperator`s →
`PyBulletSim` and `create_lifted_controllers` → `LiftedSkill` set → ground operators →
`SesameModels(...)`. Grounding is done by a module-private `_create_ground_operators`
that binds parameters to known object names, rather than by exhaustive grounding.

## Bilevel planning: three things that bite

- **`TrajectorySamplingFailure` subclasses `BaseException`, not `Exception`.** A bare
  `except Exception` around a call into `bilevel_planning` will not catch it, and the
  failure will escape a `try` block that looks like it handles everything.
- **Refinement succeeds only on `final_abstract_state == ns` — exact set equality of
  the whole abstract state** against what the abstract plan predicted, not "the
  operator's add effects hold". A predicate your abstractor emits after the skill runs,
  but your operator never adds, fails refinement just as surely as a missing add
  effect. When an operator refuses to refine, diff the two atom sets before touching
  the controller. Write operators so every predicate that *changes* appears in the add
  or delete effects — including ones incidental to the skill's purpose.
- **`RelationalControllerGenerator` grounds a fresh controller for every sampling
  attempt**, so anything a controller builds in `__init__` is built once per attempt,
  not once per plan. For `dynamic3d` that means a fresh `PyBulletSim` — see below.

## Working with `dynamic3d`

- `kinder.register_all_environments()` must be called before `kinder.make(...)`. When
  `DISPLAY` is unset it forces `MUJOCO_GL=osmesa`, and it silently skips any category
  whose backend imports fail — so a missing dependency shows up much later as
  `gymnasium.error.NameNotFound` from `kinder.make`, not as an ImportError. If you are
  running headless on a machine that has EGL, set `MUJOCO_GL=egl` and
  `PYOPENGL_PLATFORM=egl` *after* the registration call.
- Executing a skill in `dynamic3d` costs on the order of a hundred megabytes of
  PyBullet state. `PyBulletSim.__init__` registers a `weakref.finalize` that
  disconnects its client when the sim is garbage collected; the callback must not
  capture `self`, or the sim stays reachable and is never collected. Do not add an
  explicit `close()` on top of the finalizer — that double-disconnects. If you are
  running many skill executions in a loop, cap the process's memory rather than
  trusting it to stay flat.

## Tests

Tests here are **end-to-end against the real simulator**, not unit tests, and the
existing suite has no fixtures, no `parametrize`, and no mocking. One test file per
source file, at the mirrored path under `tests/`, with the module docstring
`"""Tests for <file>.py."""`. The three `dynamic3d` bilevel-planning tests are a single
test function each, around 60 lines.

The shape to copy:

```python
kinder.register_all_environments()
env = kinder.make(f"kinder/<Env>-o{num_objects}-v0", render_mode="rgb_array")
if MAKE_VIDEOS:
    env = RecordVideo(env, "unit_test_videos", name_prefix="<Env>-<what>")
obs, _ = env.reset(seed=123)
...
env.close()
```

`MAKE_VIDEOS` comes from each package's `tests/conftest.py`, which registers a
`--make-videos` flag (`kinder-models` also registers `--save-demos`). It is off by
default, so the recording branch is dead weight unless you pass the flag.

Roll a controller out with the for/else idiom, so a controller that never terminates
fails loudly instead of falling through:

```python
controller.reset(state, params)
for _ in range(300):
    action = controller.step()
    obs, _, _, _, _ = env.step(action)
    next_state = env.observation_space.devectorize(obs)
    controller.observe(next_state)
    state = next_state
    if controller.terminated():
        break
else:
    assert False, "Controller did not terminate"
```

**Abstract states are asserted as one exact string of the entire sorted atom set** —
`assert str(sorted(abstract_state.atoms)) == "[(HandEmpty robot), (OnTable cube_0), ...]"`
— which is why there are no unit tests of individual predicates anywhere in the repo.
The upside is that a predicate leaking in or out shows up immediately, which is exactly
the failure mode the refinement gate punishes; the cost is that the assertions are
brittle to renames, so update them from a real run rather than by hand.

Keep the assertion strong enough to mean something: a rollout whose only assertion is
that the loop terminated does not test the skill. Assert the abstract state, the reward,
or the goal, and assert that the plan actually reached the goal rather than merely
running out of steps.

## Commits and pull requests

- `main` is squash-merged, so the commit that lands is the PR title. Titles are
  `<package>: <lowercase imperative summary>` when the change sits in one package
  (`kinder-blockly: default pen colour to sky blue instead of red`), and a bare
  imperative summary for cross-package or infrastructure work.
- PR bodies use `## Summary`, then `## Test plan` listing the exact commands run as
  `- [x]` checkboxes, then `## Followup` if the change unblocks something elsewhere.
  A small, single-purpose PR — two files, source plus its test — is the norm and gets
  reviewed fastest.
- `gh pr edit` **fails on this repo**: it queries Projects (classic), which GitHub has
  sunset, and the whole command dies with a GraphQL `NOT_FOUND`. Edit a PR body with
  the REST API instead:

  ```bash
  gh api -X PATCH repos/Princeton-Robot-Planning-and-Learning/kinder-baselines/pulls/<n> \
    -F body=@body.md
  ```
