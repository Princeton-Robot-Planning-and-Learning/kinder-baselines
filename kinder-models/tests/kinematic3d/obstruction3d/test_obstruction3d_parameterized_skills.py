"""Tests for Obstruction3D parameterized skills."""

from typing import Any

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.obstruction3d import ObjectCentricObstruction3DEnv
from kinder.envs.kinematic3d.save_utils import DEFAULT_DEMOS_DIR, save_demo
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.kinematic3d.obstruction3d.parameterized_skills import (
    create_lifted_controllers,
)

# Flag to enable trajectory saving
SAVE_TRAJECTORIES = MAKE_VIDEOS

kinder.register_all_environments()


def test_pick_controller():
    """Test pick controller in Obstruction3D environment."""

    env = kinder.make(
        "kinder/Obstruction3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Obstruction3D")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricObstruction3DEnv(
        num_obstructions=1, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    object_parameters = (robot, target_block)
    controller = lifted_controller.ground(object_parameters)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    env.close()


def test_pick_controller_recovers_from_rejected_step():
    """A step the environment rejects (reverts) must not silently desync the
    controller from the true robot pose.

    kindergarden#137 added a swept-path collision check that can reject/revert any
    step, including the one that empties a controller's plan -- absent the fix in
    this module, step() would already have set the phase-advance flag (e.g.
    self._pre_grasp) on that same call, so the controller would move on to the next
    phase even though the robot never reached the plan's final waypoint. This test
    forces exactly that scenario by making the real environment's collision check
    report a collision on the call that would otherwise accept the approach plan's
    final waypoint, and asserts the controller retries instead of desyncing.
    """

    env = kinder.make(
        "kinder/Obstruction3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricObstruction3DEnv(
        num_obstructions=1, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(env.action_space, sim)
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    controller = lifted_controller.ground((robot, target_block))

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)
    controller.reset(state, params)

    # The approach plan (self._current_plan) reaches length 0 exactly once, at the
    # env.step() call that pops its final waypoint. Force a collision on the first
    # such call to simulate the environment reverting that step.
    raw_env = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    orig_collision_check = (
        raw_env._robot_or_held_object_collision_exists  # pylint: disable=protected-access
    )
    rejected = {"done": False}

    def force_one_rejection() -> bool:
        if (
            not rejected["done"]
            and controller._current_plan is not None  # pylint: disable=protected-access
            and len(controller._current_plan) == 0  # pylint: disable=protected-access
        ):
            rejected["done"] = True
            return True
        return orig_collision_check()

    raw_env._robot_or_held_object_collision_exists = (  # pylint: disable=protected-access
        force_one_rejection
    )

    pre_grasp_flip_step: int | None = None
    reject_step: int | None = None
    plan_len_at_reject: tuple[int, int] | None = None
    for i in range(500):
        was_rejected_before = rejected["done"]
        pre_grasp_before = controller._pre_grasp  # pylint: disable=protected-access
        plan_before = controller._current_plan  # pylint: disable=protected-access
        plan_len_before = None if plan_before is None else len(plan_before)
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        pre_grasp_after = controller._pre_grasp  # pylint: disable=protected-access
        plan_after = controller._current_plan  # pylint: disable=protected-access
        plan_len_after = None if plan_after is None else len(plan_after)
        if not was_rejected_before and rejected["done"] and reject_step is None:
            reject_step = i
            assert plan_len_before is not None and plan_len_after is not None
            plan_len_at_reject = (plan_len_before, plan_len_after)
        if not pre_grasp_before and pre_grasp_after and pre_grasp_flip_step is None:
            pre_grasp_flip_step = i
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    assert rejected["done"], "test setup failed to force a rejected step"
    assert reject_step is not None and pre_grasp_flip_step is not None

    # The rejected call must not have shrunk the plan (the waypoint was popped, but
    # observe() must push it back on since the environment reverted the step).
    before, after = plan_len_at_reject  # type: ignore[misc]
    assert after >= before, (
        f"approach plan shrank from {before} to {after} on the rejected call -- the "
        "waypoint was lost instead of requeued"
    )
    # _pre_grasp must not flip True on the same call that got rejected -- only later,
    # once the retried waypoint is actually accepted.
    assert pre_grasp_flip_step > reject_step, (
        f"_pre_grasp flipped True at step {pre_grasp_flip_step}, which is not after "
        f"the rejected step {reject_step} -- the phase advanced on a step the "
        "environment reverted"
    )

    env.close()


def test_pick_controller_observe_requeues_on_mismatch():
    """Direct check of observe()'s rejection-detection logic.

    If observe() is given a state whose joints do not match what step() just
    commanded -- e.g. because the environment reverted the step -- it must push the
    popped waypoint back onto the plan it came from and undo any phase-advance flag
    that was set on that same pop, rather than silently accepting the mismatch and
    desyncing from the true robot pose.
    """

    env = kinder.make(
        "kinder/Obstruction3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricObstruction3DEnv(
        num_obstructions=1, use_gui=False, allow_state_access=True
    )
    controllers = create_lifted_controllers(env.action_space, sim)
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    controller = lifted_controller.ground((robot, target_block))

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)
    controller.reset(state, params)

    # Drive the plan down to its last waypoint via real steps, so the plan and the
    # simulator state stay mutually consistent, without forcing any rejection yet.
    for _ in range(500):
        if (
            controller._current_plan is not None  # pylint: disable=protected-access
            and len(controller._current_plan) == 1  # pylint: disable=protected-access
        ):
            break
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
    else:
        assert False, "Never reached the approach plan's final waypoint"
    pre_reject_state = state

    # Pop the final waypoint via step(), which optimistically sets _pre_grasp.
    controller.step()
    assert controller._pre_grasp is True  # pylint: disable=protected-access
    assert controller._current_plan == []  # pylint: disable=protected-access
    last_target = controller._last_commanded_target  # pylint: disable=protected-access
    assert last_target is not None

    # Feed observe() the *pre*-action state -- i.e. the robot did not move at all,
    # exactly as base_env.py leaves it after reverting a rejected step.
    controller.observe(pre_reject_state)

    assert (
        controller._pre_grasp is False  # pylint: disable=protected-access
    ), "rejected final waypoint falsely advanced the phase"
    assert controller._current_plan == [  # pylint: disable=protected-access
        last_target
    ], "rejected waypoint was not requeued"

    env.close()


def test_pick_place_controller():
    """Test pick and place controller in Obstruction3D environment."""

    seed = 123
    env = kinder.make(
        "kinder/Obstruction3D-o0-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="Obstruction3D")

    obs, _ = env.reset(seed=seed)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    # Initialize trajectory collection
    traj_observations: list[Any] = [obs.copy()]
    traj_actions: list[Any] = []
    traj_rewards: list[float] = []
    ep_terminated = False
    ep_truncated = False

    sim = ObjectCentricObstruction3DEnv(num_obstructions=0, allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target_block = state.get_object_from_name("target_block")
    object_parameters = (robot, target_block)
    controller = lifted_controller.ground(object_parameters)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target_region = state.get_object_from_name("target_region")
    object_parameters = (robot, target_region)
    controller = lifted_controller.ground(object_parameters)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)

    controller.reset(state, params)
    for _ in range(500):
        action = controller.step()
        obs, reward, terminated, truncated, _ = env.step(action)
        # Collect trajectory data
        traj_observations.append(obs.copy())
        traj_actions.append(action.copy())
        traj_rewards.append(float(reward))
        ep_terminated = ep_terminated or terminated
        ep_truncated = ep_truncated or truncated
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Save trajectory to pickle file
    if SAVE_TRAJECTORIES and len(traj_actions) > 0:
        demo_path = save_demo(
            demo_dir=DEFAULT_DEMOS_DIR,
            env_id="kinder/Obstruction3D-o0-v0",
            seed=seed,
            observations=traj_observations,
            actions=traj_actions,
            rewards=traj_rewards,
            terminated=ep_terminated,
            truncated=ep_truncated,
        )
        print(f"Trajectory saved to {demo_path}")
        print(f"  Observations: {len(traj_observations)}, Actions: {len(traj_actions)}")

    env.close()
