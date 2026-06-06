"""Generate a visualizer bundle for an obstruction2d-o0 case where the refiner
*must* sample more than once per step.

Unlike ``visualize_obstruction2d`` (which solves a randomly sampled o1 instance),
this example hand-designs an o0 instance -- no obstructions, just one target
block and the target surface -- chosen so that a single sample per step cannot
refine the (single) abstract plan:

  - The target surface touches the right wall.
  - The target block starts all the way on the left.

The robot has a circular body of radius 0.1 in a world that is 1.618 wide, and
it grasps the block at a sampled x-offset along the block. The target surface is
made only slightly wider than the block, so the block's placement is essentially
fixed (it must sit on the narrow surface) and the robot ends the place at
``surface_x + grasp_offset``. Whether that collides with the right wall is then
decided by the *grasp*, not by where the place samples:

  - A right-side grasp puts the body past ``world_max_x - radius = 1.518`` -- the
    entire place is impossible, for any place sample, so the refiner must
    resample the grasp.
  - A far-left grasp instead collides with the left wall at pick time (the block
    starts against the left wall).
  - Only a middle-ish grasp lets the robot place the block with its body inside
    both walls.

So the first sampled grasp often fails and the refiner must resample: the planner
finds no plan with ``num_sampling_attempts_per_step=1`` but succeeds with 2.
(Making the surface much wider would instead make the *place* sample the deciding
variable -- not what this example is about.)

To keep the demonstration about sampling (not abstract replanning), the abstract
plan budget is capped at 1: the planner commits to the single shortest abstract
plan (PickFromTable, PlaceOnTarget) and can only succeed by resampling its
refinement.

Run from the kinder-bilevel-planning package root:

    python examples/visualize_obstruction2d_resampling/generate_bundle.py
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

ENV_NAME = "kinder/Obstruction2D-o0-v0"
NUM_OBSTRUCTIONS = 0
# With this seed the first sampled grasp is on the right (place impossible) and a
# later one is middle (place succeeds), so the resampling is visible.
SEED = 5
# Single abstract plan, so the only way to succeed is resampling the refinement.
MAX_ABSTRACT_PLANS = 1
# 1 fails to refine this instance; 2 is the smallest value that succeeds.
SAMPLES_PER_STEP = 2
MAX_SKILL_HORIZON = 100
HEURISTIC_NAME = "hff"
PLANNING_TIMEOUT = 30.0

# Hand-designed geometry (world x in [0, 1.618], y in [0, 1]; robot radius 0.1).
WORLD_MAX_X = (1 + 5**0.5) / 2  # golden ratio, matches the env config
# Block is wide and starts all the way on the left, so a right-vs-middle grasp is
# a visibly different point on the block.
BLOCK_X, BLOCK_WIDTH, BLOCK_HEIGHT = 0.1, 0.16, 0.09
# Surface only slightly wider than the block (tiny place range) and flush against
# the right wall, so the place outcome is decided by the grasp, not the place sample.
SURFACE_WIDTH = BLOCK_WIDTH + 0.02
SURFACE_X = WORLD_MAX_X - SURFACE_WIDTH  # right edge flush with the right wall
ROBOT_X, ROBOT_Y = 0.8, 0.85


def build_bilevel_planning_graph() -> tuple[object, BilevelPlanningGraph, object]:
    """Solve the hand-designed obstruction2d-o0 resampling instance.

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
    # all the way left, target surface flush against the right wall.
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
        raise RuntimeError("Planner found no plan for the resampling instance.")
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

    out_path = Path(__file__).parent / "data" / "obstruction2d_o0_resampling.pkl"
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
