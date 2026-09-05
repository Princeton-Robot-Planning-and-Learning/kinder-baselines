"""Tests for CylinderShelf3D parameterized skills."""

import kinder
import numpy as np
import pybullet as p
import pytest
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CylinderShelf3DEnvConfig,
    ObjectCentricCylinderShelf3DEnv,
)
from pybullet_helpers.geometry import Pose, SE2Pose
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
        env.action_space, sim, place_params={"cylinder0": offset}
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


def test_place_with_fixed_offset_and_base_distance():
    """With all three place parameters fixed the sampler is deterministic."""
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    state, _ = env.reset(seed=456)
    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space, sim, place_params={"cylinder0": (0.10, -0.05, 0.86)}
    )
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    shelf = state.get_object_from_name("shelf")
    controller = controllers["place"].ground((robot, target, shelf))
    for seed in range(3):
        params = controller.sample_parameters(state, np.random.default_rng(seed))
        assert tuple(params) == (0.10, -0.05, 0.86)
    with pytest.raises(ValueError, match="place_params"):
        create_lifted_controllers(
            env.action_space, sim, place_params={"cylinder0": (0.10,)}
        )["place"].ground((robot, target, shelf))
    env.close()
    sim.close()


def test_base_clearance_keeps_the_base_off_the_shelf_and_cylinder():
    """With base_clearance set, the base never comes closer than that to the shelf
    or the (still standing) cylinder while driving to the pre-grasp pose."""
    clearance = 0.08
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    state, _ = env.reset(seed=456)
    sim = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space, sim, base_clearance=clearance
    )
    robot = state.get_object_from_name("robot")
    target = state.get_object_from_name("cylinder0")
    controller = controllers["move_to_pre_grasp"].ground((robot, target))
    controller.reset(state, _STAGING_PARAMS)
    visited = [state]
    for _ in range(600):
        state, _, _, _, _ = env.step(controller.step())
        controller.observe(state)
        visited.append(state)
        if controller.terminated():
            break
    assert controller.terminated()

    base_id = env.robot.base.robot_id
    # pylint: disable=protected-access
    obstacles = {"shelf": env._shelf_id, "cylinder0": env._cylinders["cylinder0"]}
    # pylint: enable=protected-access
    for visited_state in visited:
        env.set_state(visited_state)
        for name, body in obstacles.items():
            points = p.getClosestPoints(
                base_id, body, 1.0, physicsClientId=env.physics_client_id
            )
            distance = min((point[8] for point in points), default=1.0)
            assert distance >= clearance - 0.03, (name, distance)
    env.close()
    sim.close()


def test_staging_pose_outside_the_base_box_is_rejected():
    """A sampled pre-grasp or place base pose outside the env's base pose box is a
    sampling failure (the planner resamples the angle), not a goal to drive to."""
    env = ObjectCentricCylinderShelf3DEnv(num_cylinders=1, allow_state_access=True)
    state, _ = env.reset(seed=456)
    target = state.get_object_from_name("cylinder0")
    cx = state.get(target, "pose_x")
    # A box whose +x edge is just past the cylinder: staging from +x (rot 0
    # puts the base at cx - 0.8, inside) is fine, staging from -x (rot pi puts
    # it at cx + 0.8) is out.
    config = CylinderShelf3DEnvConfig(
        robot_base_pose_lower_bound=SE2Pose(-10.0, -10.0, -np.pi),
        robot_base_pose_upper_bound=SE2Pose(cx + 0.3, 10.0, np.pi),
    )
    sim = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=1, config=config, allow_state_access=True
    )
    controllers = create_lifted_controllers(env.action_space, sim)
    robot = state.get_object_from_name("robot")
    controller = controllers["move_to_pre_grasp"].ground((robot, target))
    controller.reset(state, np.array([0.8, np.pi]))
    with pytest.raises(TrajectorySamplingFailure, match="outside the base pose box"):
        controller.step()
    with pytest.raises(TrajectorySamplingFailure, match="outside the base pose box"):
        controller.predict_outcome(state, np.array([0.8, np.pi]))
    env.close()
    sim.close()


def _real_restock_config():
    """The measured physical restock scene: non-uniform boards, mixed-radius
    cylinders staged at deterministic spots inside two open-top boxes."""
    board_half = 0.0127 / 2
    deep_center = (0.9075, 1.49)
    shallow_center, shallow_yaw = (0.40, 1.28), 0.25

    def zigzag(center, yaw, pitch, dy):
        out = []
        for lx, ly in [(-pitch, -dy), (0.0, dy), (pitch, -dy)]:
            c, s = np.cos(yaw), np.sin(yaw)
            out.append((center[0] + c * lx - s * ly, center[1] + s * lx + c * ly))
        return out

    spots = zigzag(deep_center, 0.0, 0.13, 0.06) + zigzag(
        shallow_center, shallow_yaw, 0.13, 0.07
    )
    return CylinderShelf3DEnvConfig(
        shelf_pose=Pose((1.63, 1.51, 0.0)),
        shelf_layer_zs=(
            0.100 - board_half,
            0.538 - board_half,
            0.800 - board_half,
        ),
        cylinder_heights=(0.29, 0.208, 0.233, 0.10, 0.10, 0.10),
        cylinder_radii=(0.0375, 0.0375, 0.0375, 0.0325, 0.0325, 0.0325),
        boxes=((0.71, 1.105, 1.34125, 1.63875, 0.215),),
        cylinder_init_regions=tuple((x, x, y, y) for x, y in spots),
        robot_base_home_pose=SE2Pose(1.48, 0.67, 1.54),
        robot_base_pose_lower_bound=SE2Pose(-0.2, -0.2, -np.pi),
        robot_base_pose_upper_bound=SE2Pose(2.0, 2.0, np.pi),
        x_lb=-0.2,
        x_ub=2.0,
        y_lb=-0.2,
        y_ub=2.0,
    )


def test_real_restock_boxed_scene_full_rollout():
    """All six real-dimension cylinders are picked out of their boxes with 45-degree
    grasps (per-cylinder depths, so every reach clears its box rim) and placed on
    layer-directed boards — talls into the big bottom opening, shorts onto the upper
    board — ending in the env's goal state."""
    config = _real_restock_config()
    env = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=6, config=config, allow_state_access=True
    )
    state, _ = env.reset(seed=0)
    sim = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=6, config=config, allow_state_access=True
    )
    pitch = np.deg2rad(45)
    side = np.deg2rad(15)
    grasp_params = {
        "cylinder0": (pitch, 0.03),
        "cylinder1": (pitch, 0.05),
        "cylinder2": (pitch, 0.03),
        "cylinder3": (side, 0.03),
        "cylinder4": (side, 0.03),
        "cylinder5": (side, 0.03),
    }
    place_params = {
        # (x offset, y offset, base distance, board layer): talls -> layer 0.
        # y offset -0.05 = the shallowest insertion the sampler allows: at a
        # 45-degree approach the wrist rises ~1:1 behind the gripper, so deep
        # insertions put the forearm at the compartment ceiling.
        "cylinder0": (-0.13, -0.05, 0.80, 0),
        "cylinder1": (0.0, -0.05, 0.80, 0),
        "cylinder2": (0.13, -0.05, 0.80, 0),
        "cylinder3": (-0.13, -0.05, 0.80, 1),
        "cylinder4": (0.0, -0.05, 0.80, 1),
        "cylinder5": (0.13, -0.05, 0.80, 1),
    }
    controllers = create_lifted_controllers(
        env.action_space,
        sim,
        place_params,
        0.0,
        grasp_params,
        # Deep-box rim is 0.215: carry every cylinder with its bottom above it.
        carry_lift_z=0.27,
    )
    robot = state.get_object_from_name("robot")
    shelf = state.get_object_from_name("shelf")
    # (distance, rot): rot pi/2 parks the base south, heading at the cylinder;
    # the shallow box is rotated 0.25, so its cylinders are approached along
    # the box's own normal and the chassis aligns with it.
    staging = {
        "cylinder0": (0.83, np.pi / 2),
        "cylinder1": (0.88, np.pi / 2),
        "cylinder2": (0.83, np.pi / 2),
        "cylinder3": (0.83, np.pi / 2),
        "cylinder4": (0.83, np.pi / 2),
        "cylinder5": (0.83, np.pi / 2),
    }
    rng = np.random.default_rng(0)
    # Shorts first (near row), then talls, mirroring a sensible restock order.
    for name in ("cylinder3", "cylinder4", "cylinder5",
                 "cylinder0", "cylinder1", "cylinder2"):
        target = state.get_object_from_name(name)
        move = controllers["move_to_pre_grasp"].ground((robot, target))
        move.reset(state, np.array(staging[name]))
        state = _run_controller(env, move, state, max_steps=800)
        grasp = controllers["grasp"].ground((robot, target))
        grasp.reset(state, grasp.sample_parameters(state, rng))
        state = _run_controller(env, grasp, state, max_steps=800)
        place = controllers["place"].ground((robot, target, shelf))
        place.reset(state, place.sample_parameters(state, rng))
        state = _run_controller(env, place, state, max_steps=800)

    # Talls rest on the bottom board (surface 0.100), shorts on the middle one
    # (surface 0.538).
    for idx, surface in ((0, 0.100), (1, 0.100), (2, 0.100),
                         (3, 0.538), (4, 0.538), (5, 0.538)):
        z = state.get(state.get_object_from_name(f"cylinder{idx}"), "pose_z")
        expected = surface + config.get_cylinder_height(idx) / 2
        assert abs(z - expected) < 0.03, f"cylinder{idx}: z={z:.3f} vs {expected:.3f}"
    assert env.goal_reached()
    env.close()
    sim.close()
