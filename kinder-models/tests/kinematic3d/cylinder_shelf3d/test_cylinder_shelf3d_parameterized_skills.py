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
from kinder_models.magic import make_magic_lifted_controller
from kinder_models.structs import SkillCall

kinder.register_all_environments()


def _run_controller(env, controller, state, max_steps=600):
    """Step ``controller`` in ``env`` (an object-centric env) until it terminates."""
    for _ in range(max_steps):
        action = controller.step()
        state, _, _, _, _ = env.step(action)
        controller.observe(state)
        if controller.terminated():
            return state
    raise AssertionError("Controller did not terminate")


def test_pick_predict_outcome_matches_rollout():
    """The pick's outcome model agrees with actually running the pick controller."""
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    init_state, _ = env.reset(seed=456)
    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim)
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")
    params = np.array([0.8, 0.0])

    controller = controllers["pick"].ground((robot, target))
    predicted = controller.predict_outcome(init_state, params)

    controller = controllers["pick"].ground((robot, target))
    controller.reset(init_state, params)
    actual = _run_controller(env, controller, init_state)

    assert predicted.grasped_object == "cylinder0"
    assert actual.grasped_object == "cylinder0"
    assert predicted.get(target, "pose_z") > 0.3
    assert np.allclose(
        [predicted.base_pose.x, predicted.base_pose.y, predicted.base_pose.rot],
        [actual.base_pose.x, actual.base_pose.y, actual.base_pose.rot],
        atol=1e-2,
    )
    # Compare joints modulo 2*pi: the continuous joints may report an
    # equivalent angle on the other side of the wrap.
    joint_error = np.angle(
        np.exp(1j * (np.array(predicted.joint_positions) - actual.joint_positions))
    )
    assert np.allclose(joint_error, 0.0, atol=2e-2)
    assert np.isclose(predicted.finger_state, actual.finger_state, atol=1e-2)
    assert np.allclose(
        predicted.get_object_pose("cylinder0").position,
        actual.get_object_pose("cylinder0").position,
        atol=2e-2,
    )
    env.close()
    sim.close()


def test_pick_predict_outcome_rejects_infeasible_parameters():
    """Staging the base inside the shelf fails the prediction like a real sample."""
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    init_state, _ = env.reset(seed=456)
    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim)
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")
    controller = controllers["pick"].ground((robot, target))

    # Put the cylinder 0.8 m in front of the shelf and stage the base 0.8 m
    # behind the cylinder, i.e. at the shelf's own position.
    state = init_state.copy()
    shelf_pose = state.get_object_pose("shelf")
    half_height = state.get(target, "half_extent_z")
    state.set(target, "pose_x", shelf_pose.position[0])
    state.set(target, "pose_y", shelf_pose.position[1] - 0.8)
    state.set(target, "pose_z", half_height)
    with pytest.raises(TrajectorySamplingFailure):
        controller.predict_outcome(state, np.array([0.8, -np.pi / 2]))
    env.close()
    sim.close()


def test_magic_pick_then_real_place():
    """A magic pick's predicted state is a valid start for the real place skill."""
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    init_state, _ = env.reset(seed=456)
    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim)
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")
    shelf = init_state.get_object_from_name("shelf")

    magic_pick = make_magic_lifted_controller(controllers["pick"], "Pick")
    rng = np.random.default_rng(456)
    pick = magic_pick.ground((robot, target))
    pick.reset(init_state, np.array([0.8, 0.0]))
    call = pick.step()
    assert pick.terminated()
    assert isinstance(call, SkillCall)
    assert call.skill_name == "Pick"
    assert call.objects == (robot, target)
    state = call.predicted_state
    assert state.grasped_object == "cylinder0"

    # Execute the magic pick by teleporting the env to the predicted state.
    env.set_state(state)
    state = env.get_state()

    placed = False
    for _ in range(8):
        place = controllers["place"].ground((robot, target, shelf))
        place.reset(state, place.sample_parameters(state, rng))
        try:
            state = _run_controller(env, place, state)
            placed = True
            break
        except TrajectorySamplingFailure:
            continue
    assert placed, "Place controller never terminated from the predicted state"

    half_height = state.get(target, "half_extent_z")
    resting_zs = [z + half_height for z in sim.config.get_layer_surface_zs()]
    cylinder_z = state.get(target, "pose_z")
    assert any(abs(cylinder_z - z) < 0.05 for z in resting_zs)
    assert state.grasped_object is None
    env.close()
    sim.close()


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
