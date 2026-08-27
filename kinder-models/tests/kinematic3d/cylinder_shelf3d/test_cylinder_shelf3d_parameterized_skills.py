"""Tests for CylinderShelf3D parameterized skills."""

import kinder
import numpy as np
import pytest
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.cylinder_shelf3d import ObjectCentricCylinderShelf3DEnv
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_models.kinematic3d.cylinder_shelf3d.parameterized_skills import (
    create_lifted_controllers,
)

kinder.register_all_environments()


def test_pick_and_place_controller():
    """Test pick and place controllers in the CylinderShelf3D environment."""
    env = kinder.make(
        "kinder/KinematicCylinderShelf3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="CylinderShelf3D")

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
    )
    lifted_controller = controllers["pick"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    object_parameters = (robot, target)
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

    # The cylinder should have been picked up: it is well above the ground.
    target = state.get_object_from_name("cylinder0")
    assert state.get(target, "pose_z") > 0.3, "Cylinder was not lifted"

    lifted_controller = controllers["place"]
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    target_shelf = state.get_object_from_name("shelf")
    object_parameters = (robot, target, target_shelf)

    # The place resamples on infeasible draws (e.g. a staging distance
    # whose reach or collision checks fail), just as the planner does.
    rng = np.random.default_rng(123)
    placed = False
    for _ in range(8):
        controller = lifted_controller.ground(object_parameters)
        params = controller.sample_parameters(state, rng)
        controller.reset(state, params)
        try:
            for _ in range(500):
                action = controller.step()
                obs, _, _, _, _ = env.step(action)
                next_state = env.observation_space.devectorize(obs)
                controller.observe(next_state)
                state = next_state
                if controller.terminated():
                    placed = True
                    break
            if placed:
                break
        except TrajectorySamplingFailure:
            continue
    assert placed, "Place controller never terminated"

    # The cylinder should be standing on a shelf board at resting height.
    config = sim.config
    target = state.get_object_from_name("cylinder0")
    half_height = state.get(target, "half_extent_z")
    resting_zs = [z + half_height for z in config.get_layer_surface_zs()]
    cylinder_z = state.get(target, "pose_z")
    assert any(
        abs(cylinder_z - resting_z) < 0.05 for resting_z in resting_zs
    ), f"Cylinder z {cylinder_z} is not resting on a shelf board"

    env.close()


def test_pick_controller_any_approach_angle():
    """The side grasp works from arbitrary approach angles around the cylinder."""
    env = kinder.make(
        "kinder/KinematicCylinderShelf3D-o1-v0",
        use_gui=False,
        realistic_bg=False,
    )
    obs, _ = env.reset(seed=456)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    init_state = env.observation_space.devectorize(obs)

    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim)
    lifted_controller = controllers["pick"]
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")

    for approach_rot in [-2.5, 0.0, 2.5]:
        obs, _ = env.reset(seed=456)
        state = env.observation_space.devectorize(obs)
        controller = lifted_controller.ground((robot, target))
        params = np.array([0.8, approach_rot])
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
            assert False, f"Controller did not terminate for rot={approach_rot}"
        target = state.get_object_from_name("cylinder0")
        assert state.get(target, "pose_z") > 0.3, f"No lift for rot={approach_rot}"

    env.close()


def test_top_down_pick_controller():
    """The top-down grasp mode picks the cylinder with a vertical planar descent; joints
    1/3/5/7 still never leave their retract values."""
    env = kinder.make(
        "kinder/KinematicCylinderShelf3D-o1-v0",
        use_gui=False,
        realistic_bg=False,
    )
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space, sim, grasp_mode="top_down"
    )
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")

    rng = np.random.default_rng(123)
    picked = False
    for _ in range(6):
        controller = controllers["pick"].ground((robot, target))
        params = controller.sample_parameters(state, rng)
        controller.reset(state, params)
        try:
            for _ in range(600):
                action = controller.step()
                obs, _, _, _, _ = env.step(action)
                state = env.observation_space.devectorize(obs)
                controller.observe(state)
                if controller.terminated():
                    picked = True
                    break
            if picked:
                break
        except TrajectorySamplingFailure:
            continue
    assert picked, "Top-down pick never terminated"
    target = state.get_object_from_name("cylinder0")
    assert state.get(target, "pose_z") > 0.3, "Cylinder was not lifted"
    env.close()


def test_top_down_place_infeasible_on_standard_shelf():
    """With a top-down grasp, no opening of the standard shelf can admit the gripper
    column above the held cylinder, so every place sample fails."""
    env = kinder.make(
        "kinder/KinematicCylinderShelf3D-o1-v0",
        use_gui=False,
        realistic_bg=False,
    )
    obs, _ = env.reset(seed=123)
    state = env.observation_space.devectorize(obs)

    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space, sim, grasp_mode="top_down"
    )
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    shelf = state.get_object_from_name("shelf")

    rng = np.random.default_rng(123)
    for _ in range(6):
        controller = controllers["pick"].ground((robot, target))
        controller.reset(state, controller.sample_parameters(state, rng))
        try:
            for _ in range(600):
                action = controller.step()
                obs, _, _, _, _ = env.step(action)
                state = env.observation_space.devectorize(obs)
                controller.observe(state)
                if controller.terminated():
                    break
            if controller.terminated():
                break
        except TrajectorySamplingFailure:
            continue

    for _ in range(3):
        controller = controllers["place"].ground((robot, target, shelf))
        controller.reset(state, controller.sample_parameters(state, rng))
        with pytest.raises(TrajectorySamplingFailure):
            for _ in range(600):
                action = controller.step()
                obs, _, _, _, _ = env.step(action)
                state = env.observation_space.devectorize(obs)
                controller.observe(state)
                if controller.terminated():
                    break
    env.close()
