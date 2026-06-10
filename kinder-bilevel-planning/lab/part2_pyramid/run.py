"""Plan and export a visualizer bundle for the pyramid (run after it passes).

Hand-designed Obstruction2D-o2 instance: obstruction0, obstruction1, and the
target block all start on the table, none adjacent. Run from the lab/ directory:

    python -m part2_pyramid.run
"""

import pickle
from pathlib import Path

import kinder
import numpy as np
from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
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
from part2_pyramid.models import create_pyramid_models

ENV_NAME = "kinder/Obstruction2D-o2-v0"
NUM_OBSTRUCTIONS = 2
SEED = 0
MAX_ABSTRACT_PLANS = 5
SAMPLES_PER_STEP = 5
MAX_SKILL_HORIZON = 200
HEURISTIC_NAME = "hff"
PLANNING_TIMEOUT = 60.0

# All three blocks on the table, none adjacent.
LAYOUT = {
    "target_block": (0.15, 0.2, 0.09),
    "obstruction0": (0.4, 0.15, 0.09),
    "obstruction1": (1.1, 0.15, 0.09),
}
ROBOT_X, ROBOT_Y = 0.85, 0.85
TARGET_SURFACE_X = 1.45


def build_bilevel_planning_graph() -> tuple:
    """Solve the pyramid instance; return (final_state, bpg, constant_state)."""
    kinder.register_all_environments()
    env = kinder.make(ENV_NAME)
    constant_state = getattr(
        env.unwrapped, "_object_centric_env"
    ).initial_constant_state
    env_models = create_pyramid_models(
        env.observation_space,
        env.action_space,
        num_obstructions=NUM_OBSTRUCTIONS,
        init_constant_state=constant_state,
    )

    obs, _ = env.reset(seed=SEED)
    initial_state = env_models.observation_to_state(obs).copy()
    robot = initial_state.get_object_from_name("robot")
    initial_state.set(robot, "x", ROBOT_X)
    initial_state.set(robot, "y", ROBOT_Y)
    initial_state.set(robot, "theta", -np.pi / 2)
    for name, (x, w, h) in LAYOUT.items():
        obj = initial_state.get_object_from_name(name)
        initial_state.set(obj, "x", x)
        initial_state.set(obj, "width", w)
        initial_state.set(obj, "height", h)
    initial_state.set(
        initial_state.get_object_from_name("target_surface"), "x", TARGET_SURFACE_X
    )

    problem = PlanningProblem(
        env_models.state_space,
        env_models.action_space,
        initial_state,
        env_models.transition_fn,
        env_models.goal_deriver(initial_state),
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
        raise RuntimeError("Planner found no plan for the pyramid instance.")
    return plan.states[-1], bpg, constant_state


def _bake_constants_into_states(bundle_path: Path, constant_state) -> None:
    """Merge the constant (static) objects into each pickled state in place."""
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
    """Build the graph and export a visualizer bundle."""
    final_state, bpg, constant_state = build_bilevel_planning_graph()
    out_path = Path(__file__).parent / "data" / "pyramid.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpg.export(out_path, final_state=final_state)
    _bake_constants_into_states(out_path, constant_state)
    renderer_path = Path(__file__).parent.parent / "renderer.py"
    print(f"Wrote visualizer bundle to {out_path}")
    print(
        "\nView it with:\n"
        f"  python -m bilevel_planning.visualizer \\\n"
        f"      --bundle {out_path} \\\n"
        f"      --renderer {renderer_path}"
    )


if __name__ == "__main__":
    main()
