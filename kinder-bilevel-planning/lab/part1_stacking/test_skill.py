"""Unit test for the motion-planned pick + Stack skills, in isolation.

Rolls the controllers out in the real environment on a cluttered instance (a barrier
between the obstruction and the target) and checks that the held block ends up resting
on the support, and that the abstract state matches the operator's predicted effects at
each step.
"""

import kinder
import numpy as np
from part1_stacking.models import create_stacking_models


def _skill_test_helper(ground_skill, env_models, env, obs, params=None, max_steps=400):
    """Roll out a ground skill to termination; assert it matches its operator."""
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    operator = ground_skill.operator
    assert operator.preconditions.issubset(abstract_state.atoms)
    predicted_next_atoms = (
        abstract_state.atoms - operator.delete_effects
    ) | operator.add_effects
    controller = ground_skill.controller
    if params is None:
        params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    for _ in range(max_steps):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env_models.observation_to_state(obs)
        controller.observe(next_state)
        assert env_models.transition_fn(state, action) == next_state
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"
    assert env_models.state_abstractor(state).atoms == predicted_next_atoms
    return obs


def _make():
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o2-v0")
    constant_state = getattr(
        env.unwrapped, "_object_centric_env"
    ).initial_constant_state
    env_models = create_stacking_models(
        env.observation_space,
        env.action_space,
        num_obstructions=2,
        init_constant_state=constant_state,
    )
    return env, env_models


def _controlled_obs(env_models, env):
    """Reset to the cluttered layout (matches run.py)."""
    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs).copy()
    robot = state.get_object_from_name("robot")
    state.set(robot, "x", 0.25)
    state.set(robot, "y", 0.55)
    state.set(robot, "theta", -np.pi / 2)
    layout = {
        "obstruction0": (0.25, 0.1, 0.09),
        "target_block": (1.15, 0.16, 0.09),
        "obstruction1": (0.75, 0.08, 0.28),  # the barrier
    }
    for name, (x, w, h) in layout.items():
        obj = state.get_object_from_name(name)
        state.set(obj, "x", x)
        state.set(obj, "width", w)
        state.set(obj, "height", h)
    state.set(state.get_object_from_name("target_surface"), "x", 1.45)
    obs, _ = env.reset(options={"init_state": state})
    return obs


def test_motion_planned_stack_over_barrier():
    """Pick the obstruction, then route it over the barrier and stack it."""
    env, env_models = _make()
    skill = {s.operator.name: s for s in env_models.skills}
    # Friendly nudge if the Stack operator effects are still empty.
    stack_op = skill["Stack"].operator
    assert stack_op.add_effects, "TODO(2): the Stack operator has no effects yet"
    obs = _controlled_obs(env_models, env)
    state = env_models.observation_to_state(obs)
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("target_block")
    obstruction = state.get_object_from_name("obstruction0")
    barrier = state.get_object_from_name("obstruction1")

    obs = _skill_test_helper(
        skill["PickFromTable"].ground((robot, obstruction)), env_models, env, obs
    )
    obs = _skill_test_helper(
        skill["Stack"].ground((robot, obstruction, target)), env_models, env, obs
    )

    final = env_models.observation_to_state(obs)
    target_top = final.get(target, "y") + final.get(target, "height")
    assert np.isclose(final.get(obstruction, "y"), target_top, atol=1e-3)
    # The barrier was not disturbed.
    assert np.isclose(final.get(barrier, "x"), 0.75, atol=1e-6)
