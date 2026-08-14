"""Tests for VegaPickPlace3D parameterized skills."""

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

# VegaPickPlace3D needs prpl_kinematics, which is an optional extra of both kindergarden
# and this package, and a kindergarden release that contains the environment. Skip the
# module rather than fail collection when either is missing.
vega_pickplace3d = pytest.importorskip("kinder.envs.kinematic3d_v2.vega_pickplace3d")
skills = pytest.importorskip(
    "kinder_models.kinematic3d_v2.vega_pickplace3d.parameterized_skills"
)
pytest.importorskip("prpl_kinematics.planning")

TrajectorySamplingFailure = pytest.importorskip(
    "bilevel_planning.trajectory_samplers.trajectory_sampler"
).TrajectorySamplingFailure

ObjectCentricVegaPickPlace3DEnv = vega_pickplace3d.ObjectCentricVegaPickPlace3DEnv
create_lifted_controllers = skills.create_lifted_controllers

kinder.register_all_environments()


def _make_env_and_state(seed, name_prefix=""):
    """A wrapped environment reset to ``seed`` and its devectorized state."""
    env = kinder.make("kinder/VegaPickPlace3D-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix=name_prefix)
    obs, _ = env.reset(seed=seed)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    return env, state


def _scene_objects(state):
    arms = {
        side: state.get_object_from_name(f"{side}_arm") for side in ("left", "right")
    }
    cube = state.get_object_from_name("cube")
    target = state.get_object_from_name("target")
    return arms, cube, target


def _run_controller(env, controller, state, max_steps=900):
    """Run a ground controller in ``env`` until it terminates."""
    for _ in range(max_steps):
        if controller.terminated():
            return state
        obs, _, _, _, _ = env.step(controller.step())
        state = env.observation_space.devectorize(obs)
        controller.observe(state)
    assert False, "Controller did not terminate"


@pytest.mark.parametrize("prefer_ompl", [True, False])
def test_pick_and_place_controllers(prefer_ompl):
    """Picking and then placing with one arm should reach the environment goal.

    Parameterized over both planners so the BiRRT fallback is covered even when ompl is
    installed.
    """
    # For seed 0 the cube and the target are both on the right side of the table.
    env, state = _make_env_and_state(
        0, name_prefix=f"VegaPickPlace3D-ompl{prefer_ompl}"
    )
    arms, cube, target = _scene_objects(state)
    sim = ObjectCentricVegaPickPlace3DEnv(allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space, sim, prefer_ompl=prefer_ompl
    )
    rng = np.random.default_rng(123)

    pick = controllers["pick"].ground((arms["right"], cube))
    params = pick.sample_parameters(state, rng)
    assert params.shape == (7,)
    pick.reset(state, params)
    state = _run_controller(env, pick, state)
    assert state.holder == "right"

    place = controllers["place"].ground((arms["right"], cube, target))
    params = place.sample_parameters(state, rng)
    assert params.shape == (7,)
    place.reset(state, params)
    state = _run_controller(env, place, state)
    assert state.holder is None

    sim.set_state(state)
    assert sim.goal_reached()

    sim.close()
    env.close()


def test_handover_controller():
    """Passing the cube between the arms should let the other arm place it."""
    # For seed 19 the cube is far on the left side and the target far on the right, so
    # the left arm picks, hands over, and the right arm places.
    env, state = _make_env_and_state(19, name_prefix="VegaPickPlace3D-handover")
    arms, cube, target = _scene_objects(state)
    sim = ObjectCentricVegaPickPlace3DEnv(allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim, prefer_ompl=False)
    rng = np.random.default_rng(123)

    pick = controllers["pick"].ground((arms["left"], cube))
    pick.reset(state, pick.sample_parameters(state, rng))
    state = _run_controller(env, pick, state)
    assert state.holder == "left"

    handover = controllers["handover"].ground((arms["left"], arms["right"], cube))
    params = handover.sample_parameters(state, rng)
    assert params.shape == (14,)
    handover.reset(state, params)
    state = _run_controller(env, handover, state)
    assert state.holder == "right"

    place = controllers["place"].ground((arms["right"], cube, target))
    place.reset(state, place.sample_parameters(state, rng))
    state = _run_controller(env, place, state)

    sim.set_state(state)
    assert sim.goal_reached()

    sim.close()
    env.close()


def test_sampled_pick_parameters_reach_the_cube():
    """Sampled pick parameters should put the end effector within grasp range."""
    env, state = _make_env_and_state(0)
    arms, cube, _ = _scene_objects(state)
    sim = ObjectCentricVegaPickPlace3DEnv(allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim, prefer_ompl=False)
    rng = np.random.default_rng(0)

    pick = controllers["pick"].ground((arms["right"], cube))
    params = pick.sample_parameters(state, rng)

    sim.set_state(state)
    sim.set_arm_joint_positions("right", params)
    distance = np.linalg.norm(
        sim.end_effector_pose("right").t - np.array(state.cube_position)
    )
    assert distance < sim.config.grasp_radius

    sim.close()
    env.close()


def test_pick_sampling_fails_for_out_of_reach_arm():
    """Sampling a grasp with an arm that cannot reach the cube should fail.

    Bilevel planning relies on this failure to discard abstract plans that use the wrong
    arm and fall back to the other arm or to a handover.
    """
    # For seed 19 the cube is far on the left side, out of the right arm's reach.
    env, state = _make_env_and_state(19)
    arms, cube, _ = _scene_objects(state)
    sim = ObjectCentricVegaPickPlace3DEnv(allow_state_access=True)
    controllers = create_lifted_controllers(env.action_space, sim, prefer_ompl=False)
    rng = np.random.default_rng(0)

    pick = controllers["pick"].ground((arms["right"], cube))
    with pytest.raises(TrajectorySamplingFailure):
        pick.sample_parameters(state, rng)

    sim.close()
    env.close()
