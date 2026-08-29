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
    PRE_GRASP_POSITION_TOL,
    create_lifted_controllers,
    get_grasp_positions,
    is_at_pre_grasp,
)
from kinder_models.magic import make_magic_lifted_controller
from kinder_models.structs import SkillCall

kinder.register_all_environments()

_STAGING_PARAMS = np.array([0.8, 0.0])


def _run_controller(env, controller, state, max_steps=600):
    """Step ``controller`` in ``env`` (an object-centric env) until it terminates."""
    for _ in range(max_steps):
        action = controller.step()
        state, _, _, _, _ = env.step(action)
        controller.observe(state)
        if controller.terminated():
            return state
    raise AssertionError("Controller did not terminate")


def _make_env_and_controllers(seed=456):
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    init_state, _ = env.reset(seed=seed)
    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim)
    return env, sim, controllers, init_state


def _move_to_pre_grasp(env, controllers, state, params=_STAGING_PARAMS):
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    controller = controllers["move_to_pre_grasp"].ground((robot, target))
    controller.reset(state, params)
    return _run_controller(env, controller, state)


def _grasp(env, controllers, state):
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    controller = controllers["grasp"].ground((robot, target))
    controller.reset(
        state, controller.sample_parameters(state, np.random.default_rng(0))
    )
    return _run_controller(env, controller, state)


def _place(env, controllers, state, rng):
    """Run the place, resampling on infeasible draws just as the planner does."""
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    shelf = state.get_object_from_name("shelf")
    for _ in range(8):
        controller = controllers["place"].ground((robot, target, shelf))
        controller.reset(state, controller.sample_parameters(state, rng))
        try:
            return _run_controller(env, controller, state)
        except TrajectorySamplingFailure:
            continue
    raise AssertionError("Place controller never terminated")


def _assert_on_shelf(sim, state):
    target = state.get_object_from_name("cylinder0")
    half_height = state.get(target, "half_extent_z")
    resting_zs = [z + half_height for z in sim.config.get_layer_surface_zs()]
    cylinder_z = state.get(target, "pose_z")
    assert any(
        abs(cylinder_z - resting_z) < 0.05 for resting_z in resting_zs
    ), f"Cylinder z {cylinder_z} is not resting on a shelf board"
    assert state.grasped_object is None


def test_move_to_pre_grasp_reaches_pre_grasp_pose():
    """MoveToPreGrasp ends with the empty gripper at the pre-grasp position."""
    env, sim, controllers, init_state = _make_env_and_controllers()
    assert not is_at_pre_grasp(sim, init_state, "cylinder0")
    state = _move_to_pre_grasp(env, controllers, init_state)
    assert state.grasped_object is None
    assert is_at_pre_grasp(sim, state, "cylinder0")
    sim.set_state(state)
    ee_position = np.array(sim.robot.arm.get_end_effector_pose().position)
    _, pre_grasp_position = get_grasp_positions(state, "cylinder0")
    assert np.linalg.norm(ee_position - pre_grasp_position) < PRE_GRASP_POSITION_TOL
    env.close()
    sim.close()


def test_move_to_pre_grasp_grasp_and_place():
    """The three skills in sequence pick the cylinder off the floor and place it on a
    shelf board."""
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
    controllers = create_lifted_controllers(env.action_space, sim)
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    shelf = state.get_object_from_name("shelf")
    rng = np.random.default_rng(123)

    def _run(controller, state):
        for _ in range(500):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
            if controller.terminated():
                return state
        raise AssertionError("Controller did not terminate")

    stage = controllers["move_to_pre_grasp"].ground((robot, target))
    stage.reset(state, stage.sample_parameters(state, rng))
    state = _run(stage, state)
    assert is_at_pre_grasp(sim, state, "cylinder0")

    grasp = controllers["grasp"].ground((robot, target))
    grasp.reset(state, grasp.sample_parameters(state, rng))
    state = _run(grasp, state)
    assert state.grasped_object == "cylinder0"
    assert state.get(target, "pose_z") > 0.3, "Cylinder was not lifted"
    assert not is_at_pre_grasp(sim, state, "cylinder0")

    # The place resamples on infeasible draws, just as the planner does.
    rng = np.random.default_rng(123)
    placed = False
    for _ in range(8):
        place = controllers["place"].ground((robot, target, shelf))
        place.reset(state, place.sample_parameters(state, rng))
        try:
            state = _run(place, state)
            placed = True
            break
        except TrajectorySamplingFailure:
            continue
    assert placed, "Place controller never terminated"
    _assert_on_shelf(sim, state)
    env.close()


def test_grasp_any_approach_angle():
    """The side grasp works from arbitrary approach angles around the cylinder."""
    env, sim, controllers, _ = _make_env_and_controllers()
    for approach_rot in [-2.5, 0.0, 2.5]:
        state, _ = env.reset(seed=456)
        state = _move_to_pre_grasp(
            env, controllers, state, params=np.array([0.8, approach_rot])
        )
        state = _grasp(env, controllers, state)
        target = state.get_object_from_name("cylinder0")
        assert state.get(target, "pose_z") > 0.3, f"No lift for rot={approach_rot}"
    env.close()
    sim.close()


def _assert_robot_close(predicted, actual, target):
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
        predicted.get_object_pose(target.name).position,
        actual.get_object_pose(target.name).position,
        atol=2e-2,
    )


def test_move_to_pre_grasp_predict_outcome_matches_rollout():
    """MoveToPreGrasp's outcome model agrees with actually running the controller."""
    env, sim, controllers, init_state = _make_env_and_controllers()
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")
    controller = controllers["move_to_pre_grasp"].ground((robot, target))
    predicted = controller.predict_outcome(init_state, _STAGING_PARAMS)
    actual = _move_to_pre_grasp(env, controllers, init_state)
    assert is_at_pre_grasp(sim, predicted, "cylinder0")
    _assert_robot_close(predicted, actual, target)
    env.close()
    sim.close()


def test_grasp_predict_outcome_matches_rollout():
    """Grasp's outcome model agrees with actually running the grasp from the pre-grasp
    pose."""
    env, sim, controllers, init_state = _make_env_and_controllers()
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")
    staged = _move_to_pre_grasp(env, controllers, init_state)
    controller = controllers["grasp"].ground((robot, target))
    predicted = controller.predict_outcome(staged, np.zeros(0))
    actual = _grasp(env, controllers, staged)
    assert predicted.grasped_object == "cylinder0"
    assert actual.grasped_object == "cylinder0"
    assert predicted.get(target, "pose_z") > 0.3
    _assert_robot_close(predicted, actual, target)
    env.close()
    sim.close()


def test_move_to_pre_grasp_predict_outcome_rejects_infeasible_parameters():
    """Staging the base inside the shelf fails the prediction like a real sample."""
    env, sim, controllers, init_state = _make_env_and_controllers()
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")
    controller = controllers["move_to_pre_grasp"].ground((robot, target))

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


def test_magic_grasp_then_real_place():
    """After a real MoveToPreGrasp, a magic Grasp's predicted state is a valid start for
    the real place skill."""
    env, sim, controllers, init_state = _make_env_and_controllers()
    robot = init_state.get_object_from_name("robot")
    target = init_state.get_object_from_name("cylinder0")
    staged = _move_to_pre_grasp(env, controllers, init_state)

    magic_grasp = make_magic_lifted_controller(controllers["grasp"], "Grasp")
    grasp = magic_grasp.ground((robot, target))
    grasp.reset(staged, grasp.sample_parameters(staged, np.random.default_rng(0)))
    call = grasp.step()
    assert grasp.terminated()
    assert isinstance(call, SkillCall)
    assert call.skill_name == "Grasp"
    assert call.objects == (robot, target)
    assert call.predicted_state.grasped_object == "cylinder0"

    # Execute the magic grasp by teleporting the env to the predicted state.
    env.set_state(call.predicted_state)
    state = env.get_state()
    state = _place(env, controllers, state, np.random.default_rng(456))
    _assert_on_shelf(sim, state)
    env.close()
    sim.close()


def test_place_with_fixed_offset():
    """A fixed place offset is used verbatim (only the base distance is sampled) and the
    cylinder ends up at that offset from the shelf centre."""
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    init_state, _ = env.reset(seed=456)
    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    offset = (0.10, -0.05)
    controllers = create_lifted_controllers(
        env.action_space, sim, place_offsets={"cylinder0": offset}
    )
    state = _move_to_pre_grasp(env, controllers, init_state)
    state = _grasp(env, controllers, state)

    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    shelf = state.get_object_from_name("shelf")
    controller = controllers["place"].ground((robot, target, shelf))
    for seed in range(3):
        params = controller.sample_parameters(state, np.random.default_rng(seed))
        assert tuple(params[:2]) == offset

    state = _place(env, controllers, state, np.random.default_rng(456))
    _assert_on_shelf(sim, state)
    shelf_pose = state.get_object_pose("shelf")
    expected_xy = (
        shelf_pose.position[0] + offset[0],
        shelf_pose.position[1] - 0.05 + offset[1],
    )
    placed_xy = (state.get(target, "pose_x"), state.get(target, "pose_y"))
    assert np.allclose(placed_xy, expected_xy, atol=0.02), (placed_xy, expected_xy)
    env.close()
    sim.close()
