"""Generate a large visualizer bundle from an obstruction2d-o2 solve.

This example exists to see how the visualizer scales: it runs a real solve of a
two-obstruction instance with a generous search budget (5 abstract plans, 5
samples per step), which produces a graph an order of magnitude larger than the
other examples -- a few thousand concrete states with substantial branching.

Unlike the hand-designed o0 examples, this is a plain random instance (like
``visualize_obstruction2d``); the seed is chosen so the solve finishes and yields
a big-but-still-solvable graph. With two obstructions and this budget many seeds
instead blow up to ~15-19k states and time out without a plan, so don't be
surprised if changing the seed stops solving.

Run from the kinder-bilevel-planning package root:

    python examples/visualize_obstruction2d_large/generate_bundle.py
"""

import pickle
from pathlib import Path

import kinder
from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import PlanningProblem
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)

from kinder_bilevel_planning.env_models import create_bilevel_planning_models

ENV_NAME = "kinder/Obstruction2D-o2-v0"
NUM_OBSTRUCTIONS = 2
# Solves in ~11s with a ~2.6k-state graph; most other seeds blow up and time out.
SEED = 0
# Generous budget so the graph is large and branchy.
MAX_ABSTRACT_PLANS = 5
SAMPLES_PER_STEP = 5
MAX_SKILL_HORIZON = 100
HEURISTIC_NAME = "hff"
PLANNING_TIMEOUT = 60.0


def build_bilevel_planning_graph() -> tuple[object, BilevelPlanningGraph, object]:
    """Solve the obstruction2d-o2 instance.

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
    obs, _ = env.reset(seed=SEED)
    object_centric_env = getattr(env.unwrapped, "_object_centric_env")
    constant_state = object_centric_env.initial_constant_state

    initial_state = env_models.observation_to_state(obs)
    goal = env_models.goal_deriver(initial_state)
    problem = PlanningProblem(
        env_models.state_space,
        env_models.action_space,
        initial_state,
        env_models.transition_fn,
        goal,
    )
    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator(env_models.skills),
        transition_function=env_models.transition_fn,
        state_abstractor=env_models.state_abstractor,
        max_trajectory_steps=MAX_SKILL_HORIZON,
    )
    abstract_plan_generator: AbstractPlanGenerator = (
        RelationalHeuristicSearchAbstractPlanGenerator(
            env_models.types,
            env_models.predicates,
            env_models.operators,
            HEURISTIC_NAME,
            seed=SEED,
            precomputed_ground_operators=env_models.ground_operators,
        )
    )
    abstract_successor_fn = RelationalAbstractSuccessorGenerator(
        env_models.operators,
        precomputed_ground_operators=env_models.ground_operators,
    )
    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        MAX_ABSTRACT_PLANS,
        SAMPLES_PER_STEP,
        abstract_successor_fn,
        env_models.state_abstractor,
        seed=SEED,
    )

    plan, bpg = planner.run(problem, timeout=PLANNING_TIMEOUT)
    if plan is None:
        raise RuntimeError("Planner found no plan for the obstruction2d-o2 instance.")
    return plan.states[-1], bpg, constant_state


def _bake_constants_into_states(bundle_path: Path, constant_state: object) -> None:
    """Merge the constant (static) objects into each pickled state in place.

    The exported bundle's ``states`` map is what the visualizer renders. Each
    planner state holds only dynamic objects, so we copy in the static objects
    here. This touches only the rendered states, not the graph topology that
    ``export`` already wrote.
    """
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    baked = {}
    for node_id, state in bundle["states"].items():
        merged = state.copy()
        merged.data.update(constant_state.data)  # type: ignore[attr-defined]
        baked[node_id] = merged
    bundle["states"] = baked
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)


def main() -> None:
    """Build the graph and export a visualizer bundle, then print how to view it."""
    final_state, bpg, constant_state = build_bilevel_planning_graph()

    out_path = Path(__file__).parent / "data" / "obstruction2d_o2_large.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpg.export(out_path, final_state=final_state)
    _bake_constants_into_states(out_path, constant_state)
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
