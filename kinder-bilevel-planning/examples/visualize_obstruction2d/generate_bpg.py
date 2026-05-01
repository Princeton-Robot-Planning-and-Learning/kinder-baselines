"""Run bilevel planning on Obstruction2D-o1 and export a visualizer bundle.

Produces a single pickle at ``data/bpg.pkl`` (relative to this script) that
the bilevel-planning visualizer can load. See the README in this directory
for the rest of the workflow.
"""

from pathlib import Path

import kinder
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import PlanningProblem
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)

from kinder_bilevel_planning.env_models import create_bilevel_planning_models

SEED = 123
ENV_NAME = "kinder/Obstruction2D-o1-v0"
MODEL_NAME = "obstruction2d"
NUM_OBSTRUCTIONS = 1
PLAN_TIMEOUT_SECONDS = 30.0

OUTPUT_PATH = Path(__file__).parent / "data" / "bpg.pkl"


def main() -> None:
    """Build env + models, run the planner, export the bundle."""
    kinder.register_all_environments()
    env = kinder.make(ENV_NAME, render_mode="rgb_array")

    env_models = create_bilevel_planning_models(
        MODEL_NAME,
        env.observation_space,
        env.action_space,
        num_obstructions=NUM_OBSTRUCTIONS,
    )

    obs, _ = env.reset(seed=SEED)
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
        max_trajectory_steps=100,
    )

    abstract_plan_generator = RelationalHeuristicSearchAbstractPlanGenerator(
        env_models.types,
        env_models.predicates,
        env_models.operators,
        "hff",
        seed=SEED,
    )

    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        max_abstract_plans=10,
        num_sampling_attempts_per_step=1,
        abstract_successor_function=RelationalAbstractSuccessorGenerator(
            env_models.operators
        ),
        state_abstractor=env_models.state_abstractor,
        seed=SEED,
    )

    plan, bpg = planner.run(problem, timeout=PLAN_TIMEOUT_SECONDS)
    if plan is None:
        print("Planner did not find a plan within the timeout.")
    else:
        print(f"Plan found with {len(plan.actions)} actions.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_state = plan.states[-1] if plan is not None else None
    bpg.export(OUTPUT_PATH, final_state=final_state)
    print(f"Wrote visualizer bundle to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
