"""Tests for dynpushpullhook2d bilevel planning models."""

import kinder
import numpy as np
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import PlanningProblem
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)

from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_dynpushpullhook2d_observation_to_state():
    """Tests for observation_to_state()."""
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )
    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert env_models.state_space.contains(state)
    assert env_models.observation_space == env.observation_space
    env.close()


def test_dynpushpullhook2d_transition_fn():
    """Tests for transition_fn()."""
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env.action_space.seed(0)
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )
    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs)

    for _ in range(10):
        action = env.action_space.sample()
        next_state = env_models.transition_fn(state, action)
        assert env_models.state_space.contains(next_state)
        assert isinstance(hash(next_state), int)
        state = next_state
    env.close()


def test_dynpushpullhook2d_goal_deriver():
    """Tests for goal_deriver()."""
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )
    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs)
    goal = env_models.goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(TargetAtGoal target_block)"


def test_dynpushpullhook2d_state_abstractor():
    """Tests for state_abstractor()."""
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    HandEmpty = pred_name_to_pred["HandEmpty"]
    HoldingHook = pred_name_to_pred["HoldingHook"]
    TargetAtGoal = pred_name_to_pred["TargetAtGoal"]

    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)

    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]

    # Initially the robot is not holding anything.
    assert HandEmpty([robot]) in abstract_state.atoms

    # Target should not be at goal initially.
    target_block = obj_name_to_obj["target_block"]
    assert TargetAtGoal([target_block]) not in abstract_state.atoms

    env.close()


def _skill_test_helper(ground_skill, env_models, env, obs, params=None, max_steps=500):
    """Execute a grounded skill and return the resulting observation."""
    rng = np.random.default_rng(123)
    state = env_models.observation_to_state(obs)

    controller = ground_skill.controller
    if params is None:
        params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    for _ in range(max_steps):
        try:
            action = controller.step()
            obs, _, terminated, _, _ = env.step(action)
            next_state = env_models.observation_to_state(obs)
            controller.observe(next_state)
            state = next_state
            if controller.terminated() or terminated:
                break
        except TrajectorySamplingFailure:
            break
    return obs, terminated


def test_dynpushpullhook2d_grasp_hook_skill():
    """Test the GraspHook skill via the bilevel model."""
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}

    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    hook = obj_name_to_obj["hook"]

    # Ground and execute GraspHook.
    grasp_skill = skill_name_to_skill["GraspHook"].ground((robot, hook))
    obs, _ = _skill_test_helper(grasp_skill, env_models, env, obs)

    # Verify: hook should be held.
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    assert pred_name_to_pred["HoldingHook"]([robot, hook]) in abstract_state.atoms
    assert pred_name_to_pred["HandEmpty"]([robot]) not in abstract_state.atoms

    env.close()


def test_dynpushpullhook2d_move_skill():
    """Test the Move skill via the bilevel model."""
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}

    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs)
    abstract_state = env_models.state_abstractor(state)
    obj_name_to_obj = {o.name: o for o in abstract_state.objects}
    robot = obj_name_to_obj["robot"]
    hook = obj_name_to_obj["hook"]

    init_hook_x = state.get(hook, "x")
    init_hook_y = state.get(hook, "y")

    # Execute several random moves — the hook should be displaced.
    rng = np.random.default_rng(42)
    move_skill = skill_name_to_skill["Move"]
    for _ in range(10):
        ground_move = move_skill.ground((robot,))
        state = env_models.observation_to_state(obs)
        params = ground_move.controller.sample_parameters(state, rng)
        obs, _ = _skill_test_helper(ground_move, env_models, env, obs, params=params)
        state = env_models.observation_to_state(obs)
        dx = state.get(hook, "x") - init_hook_x
        dy = state.get(hook, "y") - init_hook_y
        if np.sqrt(dx**2 + dy**2) > 0.01:
            break

    assert np.sqrt(dx**2 + dy**2) > 0.01, "Hook should be displaced by move"
    env.close()


def test_dynpushpullhook2d_full_pipeline():
    """Test the full pipeline: grasp → prehook → hookdown."""
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )
    skill_name_to_skill = {s.operator.name: s for s in env_models.skills}

    # Set up a state where the target block is near the hook's reach.
    init_obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(init_obs)
    obj_name_to_obj = {o.name: o for o in env_models.state_abstractor(state).objects}
    robot = obj_name_to_obj["robot"]
    hook = obj_name_to_obj["hook"]
    target_block = obj_name_to_obj["target_block"]

    # Adjust initial state so target is reachable by the hook.
    new_state = state.copy()
    new_state.set(target_block, "x", state.get(target_block, "x") + 2.3)
    new_state.set(target_block, "y", state.get(target_block, "y") - 0.5)
    new_state.set(hook, "x", state.get(hook, "x") - 0.2)
    obs, _ = env.reset(options={"init_state": new_state})

    # Phase 1: GraspHook.
    grasp = skill_name_to_skill["GraspHook"].ground((robot, hook))
    obs, _ = _skill_test_helper(grasp, env_models, env, obs)
    state = env_models.observation_to_state(obs)
    assert state.get(hook, "held"), "Hook should be held after GraspHook"

    # Phase 2: PreHook.
    prehook = skill_name_to_skill["PreHook"].ground((robot, hook, target_block))
    obs, _ = _skill_test_helper(prehook, env_models, env, obs, max_steps=2000)

    # Phase 3: HookDown.
    hookdown = skill_name_to_skill["HookDown"].ground((robot, hook, target_block))
    obs, terminated = _skill_test_helper(hookdown, env_models, env, obs, params=0.0, max_steps=2000)

    assert terminated, "HookDown should terminate when target is at goal"
    env.close()


def test_dynpushpullhook2d_auto_plan_and_execute():
    """Use the abstract planner to automatically discover the skill sequence,
    then execute each skill in the real env.

    This verifies end-to-end correctness: the STRIPS model produces a valid
    abstract plan (GraspHook → PreHook → HookDown), and executing the
    corresponding skills in the real environment solves the task.
    """
    env = kinder.make("kinder/DynPushPullHook2D-o0-v0")
    env_models = create_bilevel_planning_models(
        "dynpushpullhook2d",
        env.observation_space,
        env.action_space,
        num_obstructions=0,
    )

    # Set up initial state where the pipeline works.
    init_obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(init_obs)
    abstract = env_models.state_abstractor(state)
    obj = {o.name: o for o in abstract.objects}
    robot, hook, target_block = obj["robot"], obj["hook"], obj["target_block"]

    new_state = state.copy()
    new_state.set(target_block, "x", state.get(target_block, "x") + 2.3)
    new_state.set(target_block, "y", state.get(target_block, "y") - 0.5)
    new_state.set(hook, "x", state.get(hook, "x") - 0.2)
    obs, _ = env.reset(options={"init_state": new_state})
    state = env_models.observation_to_state(obs)

    # Use the abstract planner to find the skill sequence automatically.
    abstract_state = env_models.state_abstractor(state)
    goal = env_models.goal_deriver(state)

    plan_gen = RelationalHeuristicSearchAbstractPlanGenerator(
        env_models.types,
        env_models.predicates,
        env_models.operators,
        heuristic_name="hff",
        seed=123,
    )
    successor_fn = RelationalAbstractSuccessorGenerator(env_models.operators)

    # Dummy BPG needed by the plan generator interface.
    bpg: BilevelPlanningGraph = BilevelPlanningGraph()
    bpg.add_state_node(state)
    bpg.add_abstract_state_node(abstract_state)
    bpg.add_state_abstractor_edge(state, abstract_state)

    plan_iter = plan_gen(state, abstract_state, goal, timeout=30.0, bpg=bpg)
    abstract_states, abstract_actions = next(plan_iter)

    # Verify the abstract plan is GraspHook → PreHook → HookDown.
    action_names = [a.name for a in abstract_actions]
    assert action_names == ["GraspHook", "PreHook", "HookDown"], (
        f"Expected [GraspHook, PreHook, HookDown], got {action_names}"
    )

    # Build a controller generator to ground each abstract action.
    ctrl_gen = RelationalControllerGenerator(env_models.skills)
    pred_name = {p.name: p for p in env_models.predicates}
    rng = np.random.default_rng(123)

    # Execute each skill in the real env (not via transition_fn).
    for i, abstract_action in enumerate(abstract_actions):
        controller = ctrl_gen(abstract_action)
        params = controller.sample_parameters(state, rng)
        controller.reset(state, params)

        max_steps = 500 if i == 0 else 2000
        for _ in range(max_steps):
            try:
                action = controller.step()
                obs, _, terminated, _, _ = env.step(action)
                state = env_models.observation_to_state(obs)
                controller.observe(state)
                if controller.terminated() or terminated:
                    break
            except TrajectorySamplingFailure:
                break

    # Verify predicate transitions.
    abstract = env_models.state_abstractor(state)
    assert pred_name["TargetAtGoal"]([target_block]) in abstract.atoms, \
        "Auto-planned pipeline should achieve TargetAtGoal"

    env.close()
