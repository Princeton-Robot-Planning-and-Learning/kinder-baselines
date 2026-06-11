"""Plan and export a visualizer bundle for the stacking walkthrough.

Hand-designed Obstruction2D-o2 instance with CLUTTER: obstruction0 (to be stacked)
is on the left, the target block is on the right, and obstruction1 is a tall
barrier between them. The two-step plan

    PickFromTable(obstruction0), Stack(obstruction0, target_block)

is the same as before, but now the motion-planned controllers must route the
robot -- and the carried block -- up and over the barrier (a straight path
collides). Run from the lab/ directory:

    python -m part1_stacking.run
"""

from pathlib import Path

import kinder
import numpy as np
from bilevel_planning.sesame import run_sesame
from part1_stacking.models import create_stacking_models

ENV_NAME = "kinder/Obstruction2D-o2-v0"
NUM_OBSTRUCTIONS = 2
SEED = 0
MAX_ABSTRACT_PLANS = 1
SAMPLES_PER_STEP = 5
MAX_SKILL_HORIZON = 200
PLANNING_TIMEOUT = 120.0

# Hand-designed geometry (world x in [0, 1.618]; blocks rest on the table top at
# y ~= 0.1). obstruction0 (left) gets stacked on the target (right); obstruction1
# is a tall barrier between them that the robot must route the carried block over.
OBSTRUCTION_X, OBSTRUCTION_WIDTH, OBSTRUCTION_HEIGHT = 0.25, 0.1, 0.09
TARGET_X, TARGET_WIDTH, TARGET_HEIGHT = 1.15, 0.16, 0.09
BARRIER_X, BARRIER_WIDTH, BARRIER_HEIGHT = 0.75, 0.08, 0.28
ROBOT_X, ROBOT_Y = 0.25, 0.55
# Park the (unused) target surface in a far corner so nothing sits on it.
TARGET_SURFACE_X = 1.45


def build_bilevel_planning_graph() -> tuple:
    """Solve the stacking instance; return (final_state, bpg, constant_state)."""
    kinder.register_all_environments()
    env = kinder.make(ENV_NAME)
    object_centric_env = getattr(env.unwrapped, "_object_centric_env")
    constant_state = object_centric_env.initial_constant_state
    env_models = create_stacking_models(
        env.observation_space,
        env.action_space,
        num_obstructions=NUM_OBSTRUCTIONS,
        init_constant_state=constant_state,
    )

    obs, _ = env.reset(seed=SEED)
    initial_state = env_models.observation_to_state(obs).copy()
    robot = initial_state.get_object_from_name("robot")
    target_block = initial_state.get_object_from_name("target_block")
    obstruction = initial_state.get_object_from_name("obstruction0")
    barrier = initial_state.get_object_from_name("obstruction1")
    initial_state.set(robot, "x", ROBOT_X)
    initial_state.set(robot, "y", ROBOT_Y)
    initial_state.set(robot, "theta", -np.pi / 2)
    initial_state.set(target_block, "x", TARGET_X)
    initial_state.set(target_block, "width", TARGET_WIDTH)
    initial_state.set(target_block, "height", TARGET_HEIGHT)
    initial_state.set(obstruction, "x", OBSTRUCTION_X)
    initial_state.set(obstruction, "width", OBSTRUCTION_WIDTH)
    initial_state.set(obstruction, "height", OBSTRUCTION_HEIGHT)
    initial_state.set(barrier, "x", BARRIER_X)
    initial_state.set(barrier, "width", BARRIER_WIDTH)
    initial_state.set(barrier, "height", BARRIER_HEIGHT)
    initial_state.set(
        initial_state.get_object_from_name("target_surface"), "x", TARGET_SURFACE_X
    )

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
        raise RuntimeError("Planner found no plan for the stacking instance.")
    return plan.states[-1], bpg, constant_state


def main() -> None:
    """Build the graph and export a visualizer bundle."""
    final_state, bpg, constant_state = build_bilevel_planning_graph()
    out_path = Path(__file__).parent / "data" / "stacking.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpg.export(out_path, final_state=final_state, constant_state=constant_state)
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
