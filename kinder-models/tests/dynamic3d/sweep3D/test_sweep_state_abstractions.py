"""Tests for sweep3D state_abstractions.py."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import ObjectCentricState

from kinder_models.dynamic3d.sweep3D.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.sweep3D.state_abstractions import (
    Sweep3DStateAbstractor,
)
from kinder_models.dynamic3d.utils import PyBulletSim


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_sweep3D_state_abstraction():
    """Tests for Sweep3DStateAbstractor()."""
    kinder.register_all_environments()
    num_objects = 5
    env = kinder.make(
        f"kinder/SweepIntoDrawer3D-o{num_objects}-v0", render_mode="rgb_array"
    )
    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix="SweepIntoDrawer3D-state-abstraction",
        )
    # from pathlib import Path
    # task_config_path = str(
    #     Path(kinder.__file__).parent
    #     / "envs/dynamic3d/tasks/SweepIntoDrawer3D"
    #     / f"SweepIntoDrawer3D-o{num_objects}.json"
    # )
    # sim = ObjectCentricTidyBot3DEnv(
    #     task_config_path=task_config_path,
    #     num_objects=num_objects,
    #     allow_state_access=True
    # )
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    abstractor = Sweep3DStateAbstractor(sim)

    # Check state abstraction in the initial state. The robot's hand should be empty
    # and the object should be on the ground.
    obs, _ = env.reset(seed=123)
    for _ in range(5):
        obs, _, _, _, _ = env.step(np.zeros(11))
    state = env.observation_space.devectorize(obs)
    assert isinstance(state, ObjectCentricState)
    abstract_state = abstractor.state_abstractor(state)
    robot = _get_robot_from_state(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(DrawerClosed kitchen_island_drawer_s1c1), "
        f"(HandEmpty {robot.name}), "
        f"(OnTable cube_0), "
        f"(OnTable cube_1), "
        f"(OnTable cube_2), "
        f"(OnTable cube_3), "
        f"(OnTable cube_4), "
        f"(OnTable wiper_0)]"
    )

    pybullet_sim = PyBulletSim(state, rendering=False)
    # Create controllers.
    controllers = create_lifted_controllers(env.action_space, pybullet_sim=pybullet_sim)

    # create the open drawer controller.
    lifted_controller = controllers["open_drawer"]
    robot = _get_robot_from_state(state)
    drawer = state.get_object_from_name("wiper_0")
    object_parameters = (robot, drawer)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.7, -np.pi])

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

    # Check updated state abstraction: the robot should be Holding the cube.
    abstract_state = abstractor.state_abstractor(state)
    robot = _get_robot_from_state(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(DrawerOpen kitchen_island_drawer_s1c1), "
        f"(HandEmpty {robot.name}), "
        f"(OnTable cube_0), "
        f"(OnTable cube_1), "
        f"(OnTable cube_2), "
        f"(OnTable cube_3), "
        f"(OnTable cube_4), "
        f"(OnTable wiper_0)]"
    )

    # create the pick ground controller.
    lifted_controller = controllers["pick_wiper"]
    robot = _get_robot_from_state(state)
    drawer = state.get_object_from_name("wiper_0")
    object_parameters = (robot, drawer)
    controller = lifted_controller.ground(object_parameters)
    params = controller.sample_parameters(state, np.random.default_rng(123))
    # params = np.array([0.7, -np.pi])

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

    abstract_state = abstractor.state_abstractor(state)
    robot = _get_robot_from_state(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(DrawerOpen kitchen_island_drawer_s1c1), "
        f"(Holding {robot.name} wiper_0), "
        f"(OnTable cube_0), "
        f"(OnTable cube_1), "
        f"(OnTable cube_2), "
        f"(OnTable cube_3), "
        f"(OnTable cube_4)]"
    )

    # create the place ground controller.
    lifted_controller = controllers["sweep"]
    robot = _get_robot_from_state(state)
    wiper = state.get_object_from_name("wiper_0")
    target_cube = state.get_object_from_name("cube_0")
    object_parameters = (robot, wiper, target_cube)
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

    abstract_state = abstractor.state_abstractor(state)
    robot = _get_robot_from_state(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(DrawerOpen kitchen_island_drawer_s1c1), "
        f"(Holding {robot.name} wiper_0), "
        f"(InDrawer cube_0 kitchen_island_drawer_s1c1), "
        f"(InDrawer cube_1 kitchen_island_drawer_s1c1), "
        f"(InDrawer cube_2 kitchen_island_drawer_s1c1), "
        f"(InDrawer cube_3 kitchen_island_drawer_s1c1), "
        f"(InDrawer cube_4 kitchen_island_drawer_s1c1)]"
    )

    env.close()
