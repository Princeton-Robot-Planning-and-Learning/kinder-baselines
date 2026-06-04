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

ENV_NAME = "kinder/Obstruction2D-o1-v0"
NUM_OBSTRUCTIONS = 1
SEED = 123
# Matches the obstruction2d-o1 case in the env_models obstruction2d tests.
MAX_ABSTRACT_PLANS = 10
SAMPLES_PER_STEP = 1
MAX_SKILL_HORIZON = 100
HEURISTIC_NAME = "hff"
PLANNING_TIMEOUT = 30.0


def build_bilevel_planning_graph() -> tuple[object, BilevelPlanningGraph]:
    """Solve obstruction2d-o1 and return ``(final_state, BilevelPlanningGraph)``."""
    kinder.register_all_environments()
    env = kinder.make(ENV_NAME)
    env_models = create_bilevel_planning_models(
        "obstruction2d",
        env.observation_space,
        env.action_space,
        num_obstructions=NUM_OBSTRUCTIONS,
    )
    obs, _ = env.reset(seed=SEED)

    # The pipeline below mirrors BilevelPlanningAgent._run_planning; the only
    # difference is that we keep the BilevelPlanningGraph that planner.run
    # returns alongside the plan.
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
        raise RuntimeError("Planner found no plan for obstruction2d-o1.")
    return plan.states[-1], bpg


def main() -> None:
    """Build the graph and export a visualizer bundle, then print how to view it."""
    final_state, bpg = build_bilevel_planning_graph()

    out_path = Path(__file__).parent / "data" / "obstruction2d_o1.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpg.export(out_path, final_state=final_state)
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
