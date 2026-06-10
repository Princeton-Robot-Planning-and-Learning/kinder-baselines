"""Spec for the pyramid -- this is what "done" means.

It plans with YOUR models (``part2_pyramid/models.py``) on a hand-designed
instance where the two obstructions start apart, then checks the *geometry* of
the final state: the target block must rest on top of the two obstructions, which
must themselves have ended up side by side as a base.

The check is about geometry, not predicate names -- you are free to design your
own predicates and operators, as long as the planner ends in a real pyramid:

    python -m pytest part2_pyramid/test_pyramid.py -q
"""

import kinder
import numpy as np
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

# Hand-designed, NON-adjacent layout (also used by run.py).
SEED = 0
MAX_ABSTRACT_PLANS = 5
SAMPLES_PER_STEP = 5
PLANNING_TIMEOUT = 60.0
LAYOUT = {
    "target_block": (0.15, 0.2, 0.09),
    "obstruction0": (0.4, 0.15, 0.09),
    "obstruction1": (1.1, 0.15, 0.09),
}


def _plan_final_state():
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o2-v0")
    constant_state = getattr(
        env.unwrapped, "_object_centric_env"
    ).initial_constant_state
    env_models = create_pyramid_models(
        env.observation_space,
        env.action_space,
        num_obstructions=2,
        init_constant_state=constant_state,
    )
    obs, _ = env.reset(seed=SEED)
    state = env_models.observation_to_state(obs).copy()
    robot = state.get_object_from_name("robot")
    state.set(robot, "x", 0.85)
    state.set(robot, "y", 0.85)
    state.set(robot, "theta", -np.pi / 2)
    for name, (x, w, h) in LAYOUT.items():
        obj = state.get_object_from_name(name)
        state.set(obj, "x", x)
        state.set(obj, "width", w)
        state.set(obj, "height", h)
    state.set(state.get_object_from_name("target_surface"), "x", 1.45)

    problem = PlanningProblem(
        env_models.state_space,
        env_models.action_space,
        state,
        env_models.transition_fn,
        env_models.goal_deriver(state),
    )
    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator(env_models.skills),
        transition_function=env_models.transition_fn,
        state_abstractor=env_models.state_abstractor,
        max_trajectory_steps=200,
    )
    abstract_plan_generator = RelationalHeuristicSearchAbstractPlanGenerator(
        env_models.types,
        env_models.predicates,
        env_models.operators,
        "hff",
        seed=SEED,
        precomputed_ground_operators=env_models.ground_operators,
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
    plan, _ = planner.run(problem, timeout=PLANNING_TIMEOUT)
    assert (
        plan is not None
    ), "planner found no plan -- check your predicates/operators/skills/goal"
    return plan.states[-1]


def test_pyramid_is_built():
    """The plan must end in a real pyramid: a base of two obstructions + a cap."""
    final = _plan_final_state()
    tb = final.get_object_from_name("target_block")
    o0 = final.get_object_from_name("obstruction0")
    o1 = final.get_object_from_name("obstruction1")

    def span(o):
        return final.get(o, "x"), final.get(o, "x") + final.get(o, "width")

    left, right = sorted([o0, o1], key=lambda o: final.get(o, "x"))
    left_lo, left_hi = span(left)
    right_lo, right_hi = span(right)

    # Base: the two obstructions ended up side by side (a small gap is fine).
    gap = right_lo - left_hi
    assert -1e-3 <= gap <= 0.05, f"obstructions are not adjacent (gap={gap:.3f})"

    # Cap: the target rests on top of the obstructions...
    obstruction_top = final.get(left, "y") + final.get(left, "height")
    assert np.isclose(
        final.get(tb, "y"), obstruction_top, atol=2e-3
    ), "target not on top"

    # ...with a corner supported by each obstruction (it bridges the seam).
    tb_lo = final.get(tb, "x")
    tb_hi = tb_lo + final.get(tb, "width")
    assert left_lo <= tb_lo <= left_hi, "target's left corner is not on the left base"
    assert (
        right_lo <= tb_hi <= right_hi
    ), "target's right corner is not on the right base"
