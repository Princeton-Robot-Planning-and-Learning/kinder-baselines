"""Generate a visualizer bundle from a real bilevel-planning solve of obstruction2d-o1.

Runs the SesamePlanner on ``kinder/Obstruction2D-o1-v0`` and exports the
``BilevelPlanningGraph`` it builds to ``data/obstruction2d_o1.pkl``, which the
``bilevel_planning.visualizer`` can load. The planner construction mirrors
``kinder_bilevel_planning.agent.BilevelPlanningAgent._run_planning`` (which
discards the graph); here we keep the graph so it can be visualized.

Run from the kinder-bilevel-planning package root:

    python examples/visualize_obstruction2d/generate_bundle.py
"""

from pathlib import Path

import kinder
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.sesame import run_sesame
from relational_structs import ObjectCentricState

from kinder_bilevel_planning.env_models import create_bilevel_planning_models

ENV_NAME = "kinder/Obstruction2D-o1-v0"
NUM_OBSTRUCTIONS = 1
SEED = 123
# Matches the obstruction2d-o1 case in the env_models obstruction2d tests.
MAX_ABSTRACT_PLANS = 10
SAMPLES_PER_STEP = 1
MAX_SKILL_HORIZON = 100
PLANNING_TIMEOUT = 30.0


def build_bilevel_planning_graph() -> (
    tuple[object, BilevelPlanningGraph, ObjectCentricState]
):
    """Solve obstruction2d-o1.

    Returns ``(final_state, BilevelPlanningGraph, constant_state)``. The
    constant state holds the static objects (table, walls) that the env keeps
    separate from the dynamic per-step state; the caller bakes it into the
    pickled states so the renderer draws the full scene.
    """
    kinder.register_all_environments()
    env = kinder.make(ENV_NAME)
    env_models = create_bilevel_planning_models(
        "obstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=NUM_OBSTRUCTIONS,
    )
    obs, _ = env.reset(seed=SEED)
    # The env merges these static objects into the state at render time; the
    # visualizer only has the pickled states, so we grab the constant state
    # here and bake it in below. getattr avoids a typed-attribute error on the
    # gym wrapper for this internal env handle.
    object_centric_env = getattr(env.unwrapped, "_object_centric_env")
    constant_state = object_centric_env.initial_constant_state

    # The pipeline below mirrors BilevelPlanningAgent._run_planning; the only
    # difference is that we keep the BilevelPlanningGraph that planner.run
    # returns alongside the plan.
    initial_state = env_models.observation_to_state(obs)
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
        raise RuntimeError("Planner found no plan for obstruction2d-o1.")
    return plan.states[-1], bpg, constant_state


def main() -> None:
    """Build the graph and export a visualizer bundle, then print how to view it."""
    final_state, bpg, constant_state = build_bilevel_planning_graph()

    out_path = Path(__file__).parent / "data" / "obstruction2d_o1.pkl"
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
