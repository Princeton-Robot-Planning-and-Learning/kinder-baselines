"""Tests for sweep3D parameterized skills."""

import time

import kinder
import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.dynamic3d.sweep3D.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.utils import PyBulletSim

kinder.register_all_environments()


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_open_drawer():
    """Test open drawer."""

    # Create the environment.
    num_cubes = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["open_drawer"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
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


def test_pick_wiper():
    """Test pick wiper."""

    # Create the environment.
    num_cubes = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["pick_wiper"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
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


def test_pick_wiper_efficiency():
    """Test env.reset and env.step timing for the pick_wiper controller.

    Thresholds (wall-clock, single process, CPU-only):
        - env.reset:  <= 10 s
        - env.step:   <= 0.5 s per call (mean over all steps executed)
    """
    num_cubes = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
    )

    # --- measure reset ---
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=123)
    reset_time = time.perf_counter() - t0

    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    assert state is not None

    pybullet_sim = PyBulletSim(state, rendering=False)
    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    lifted_controller = controllers["pick_wiper"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    controller.reset(state, params)

    # --- measure step ---
    step_times: list[float] = []
    terminated = False
    for _ in range(300):
        action = controller.step()
        t0 = time.perf_counter()
        obs, _, _, _, _ = env.step(action)
        step_times.append(time.perf_counter() - t0)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            terminated = True
            break

    env.close()

    mean_step = float(np.mean(step_times))
    max_step = float(np.max(step_times))

    print(f"\n[efficiency] env.reset time:      {reset_time:.3f}s")
    print(f"[efficiency] env.step mean time:   {mean_step:.3f}s  (over {len(step_times)} steps)")
    print(f"[efficiency] env.step max time:    {max_step:.3f}s")

    assert terminated, "pick_wiper controller did not terminate within 300 steps"
    assert reset_time <= 10.0, (
        f"env.reset took {reset_time:.2f}s, expected <= 10s"
    )
    assert mean_step <= 0.5, (
        f"Mean env.step time {mean_step:.3f}s over {len(step_times)} steps, "
        f"expected <= 0.5s (max was {max_step:.3f}s)"
    )


def test_open_drawer_pick_sweep_wiper():
    """Test open drawer, pick and sweep wiper."""

    # Create the environment.
    num_cubes = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array", scene_bg=True, scene_render_camera="agentview_1"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=133)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["open_drawer"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the pick ground controller.
    lifted_controller = controllers["pick_wiper"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the place ground controller.
    lifted_controller = controllers["sweep"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    cube3 = state.get_object_from_name("cube_3")
    cube4 = state.get_object_from_name("cube_4")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2, cube3, cube4)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.55, -np.pi])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
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

def test_open_drawer_pick_sweep_screw():
    """Test open drawer, pick and sweep screw."""

    # Create the environment.
    num_cubes = 3
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"TidyBot3D-cupboard-o{num_cubes}-real"
        )

    # Reset the environment and get the initial state.
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    assert state is not None
    pybullet_sim = PyBulletSim(state, rendering=False)

    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the pick ground controller.
    lifted_controller = controllers["open_drawer"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the pick ground controller.
    lifted_controller = controllers["pick_wiper"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # create the place ground controller.
    lifted_controller = controllers["sweep"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    cube0 = state.get_object_from_name("cube_0")
    cube1 = state.get_object_from_name("cube_1")
    cube2 = state.get_object_from_name("cube_2")
    object_parameters = (robot, wiper, drawer, cube0, cube1, cube2)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.55, -np.pi])

    # Reset and execute the controller until it terminates.
    controller.reset(state, params)
    for _ in range(200):
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
