"""Plan and export a visualizer bundle for the pyramid (run after it passes).

Hand-designed Obstruction2D-o2 instance: obstruction0, obstruction1, and the
target block all start on the table, none adjacent. Run from the lab/ directory:

    python -m part2_pyramid.run
"""

from pathlib import Path

import kinder
import numpy as np
from bilevel_planning.sesame import run_sesame
from part2_pyramid.models import create_pyramid_models

ENV_NAME = "kinder/Obstruction2D-o2-v0"
NUM_OBSTRUCTIONS = 2
SEED = 0
MAX_ABSTRACT_PLANS = 5
SAMPLES_PER_STEP = 5
MAX_SKILL_HORIZON = 200
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
        raise RuntimeError("Planner found no plan for the pyramid instance.")
    return plan.states[-1], bpg, constant_state


def main() -> None:
    """Build the graph and export a visualizer bundle."""
    final_state, bpg, constant_state = build_bilevel_planning_graph()
    out_path = Path(__file__).parent / "data" / "pyramid.pkl"
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
