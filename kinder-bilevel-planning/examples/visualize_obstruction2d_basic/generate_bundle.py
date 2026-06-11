"""Generate a visualizer bundle for the simplest possible obstruction2d-o0 case.

This is the introductory example: it shows the basic two-plane graph structure
with nothing extra. The instance is hand-designed to be trivially solvable --
one target block and the target surface, both well away from the walls -- so the
single abstract plan (PickFromTable, PlaceOnTarget) refines on the very first
sample.

With ``MAX_ABSTRACT_PLANS = 1`` and ``SAMPLES_PER_STEP = 1`` there is no abstract
replanning and no resampling, so the graph is a single path:

  - the concrete (lower) plane is one pick-then-place trajectory, and
  - the abstract (upper) plane is ``HandEmpty -> Holding -> OnTarget``.

Start here to learn the visualizer, then see ``visualize_obstruction2d_resampling``
for a case that forces the refiner to resample, and ``visualize_obstruction2d``
for a realistic o1 solve with search branches.

Run from the kinder-bilevel-planning package root:

    python examples/visualize_obstruction2d_basic/generate_bundle.py
"""

from pathlib import Path

import kinder
import numpy as np
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.sesame import run_sesame
from relational_structs import ObjectCentricState

from kinder_bilevel_planning.env_models import create_bilevel_planning_models

ENV_NAME = "kinder/Obstruction2D-o0-v0"
NUM_OBSTRUCTIONS = 0
SEED = 0
# One abstract plan, one sample per step -- both succeed on the first try here.
MAX_ABSTRACT_PLANS = 1
SAMPLES_PER_STEP = 1
MAX_SKILL_HORIZON = 100
PLANNING_TIMEOUT = 30.0

# Hand-designed geometry (world x in [0, 1.618], y in [0, 1]; robot radius 0.1).
# Block middle-left, target surface middle-right, both well clear of the walls,
# so any grasp and any placement work -- the first sample refines the plan.
BLOCK_X, BLOCK_WIDTH, BLOCK_HEIGHT = 0.4, 0.1, 0.09
SURFACE_X, SURFACE_WIDTH = 1.0, 0.12
ROBOT_X, ROBOT_Y = 0.8, 0.85


def build_bilevel_planning_graph() -> (
    tuple[object, BilevelPlanningGraph, ObjectCentricState]
):
    """Solve the dead-simple obstruction2d-o0 instance.

    Returns ``(final_state, BilevelPlanningGraph, constant_state)``. The constant
    state holds the static objects (table, walls) that the env keeps separate
    from the dynamic per-step state; the caller bakes it into the pickled states
    so the renderer draws the full scene.
    """
    kinder.register_all_environments()
    env = kinder.make(ENV_NAME)
    env_models = create_bilevel_planning_models(
        "obstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=NUM_OBSTRUCTIONS,
    )
    object_centric_env = getattr(env.unwrapped, "_object_centric_env")
    constant_state = object_centric_env.initial_constant_state

    # Build the hand-designed initial state by overriding a reset state: block
    # middle-left, target surface middle-right, both clear of the walls.
    obs, _ = env.reset(seed=0)
    initial_state = env_models.observation_to_state(obs).copy()
    robot = initial_state.get_object_from_name("robot")
    target_block = initial_state.get_object_from_name("target_block")
    target_surface = initial_state.get_object_from_name("target_surface")
    initial_state.set(robot, "x", ROBOT_X)
    initial_state.set(robot, "y", ROBOT_Y)
    initial_state.set(robot, "theta", -np.pi / 2)
    initial_state.set(target_block, "x", BLOCK_X)
    initial_state.set(target_block, "width", BLOCK_WIDTH)
    initial_state.set(target_block, "height", BLOCK_HEIGHT)
    initial_state.set(target_surface, "x", SURFACE_X)
    initial_state.set(target_surface, "width", SURFACE_WIDTH)

    plan, bpg = run_sesame(
        env_models,
        initial_state,
        seed=SEED,
        max_abstract_plans=MAX_ABSTRACT_PLANS,
        samples_per_step=SAMPLES_PER_STEP,
        max_skill_horizon=MAX_SKILL_HORIZON,
        timeout=PLANNING_TIMEOUT,
    )
    if plan is None:
        raise RuntimeError("Planner found no plan for the basic instance.")
    return plan.states[-1], bpg, constant_state


def main() -> None:
    """Build the graph and export a visualizer bundle, then print how to view it."""
    final_state, bpg, constant_state = build_bilevel_planning_graph()

    out_path = Path(__file__).parent / "data" / "obstruction2d_o0_basic.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpg.export(out_path, final_state=final_state, constant_state=constant_state)
    print(f"Wrote visualizer bundle to {out_path}")

    renderer_path = Path(__file__).parent / "renderer.py"
    print(
        "\nView it with:\n"
        f"  python -m bilevel_planning.visualizer \\\n"
        f"      --bundle {out_path} \\\n"
        f"      --renderer {renderer_path}"
    )


if __name__ == "__main__":
    main()
