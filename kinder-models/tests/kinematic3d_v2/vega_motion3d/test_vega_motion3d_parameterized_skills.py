"""Tests for VegaMotion3D parameterized skills."""

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

# VegaMotion3D needs the optional prpl_kinematics backend, and is not in a kindergarden
# release yet. Skip the module rather than fail collection when either is missing.
vega_motion3d = pytest.importorskip("kinder.envs.kinematic3d_v2.vega_motion3d")
skills = pytest.importorskip(
    "kinder_models.kinematic3d_v2.vega_motion3d.parameterized_skills"
)
prpl_planning = pytest.importorskip("prpl_kinematics.planning")

ObjectCentricVegaMotion3DEnv = vega_motion3d.ObjectCentricVegaMotion3DEnv
create_lifted_controllers = skills.create_lifted_controllers
create_motion_planner = skills.create_motion_planner
ompl_is_available = skills.ompl_is_available

kinder.register_all_environments()


def _ground_controller(env, state, prefer_ompl=True):
    """Ground a move-to-target skill against a fresh internal sim."""
    sim = ObjectCentricVegaMotion3DEnv(allow_state_access=True)
    controllers = create_lifted_controllers(
        env.action_space, sim, prefer_ompl=prefer_ompl
    )
    controller = controllers["move_to_target"].ground(
        (state.get_object_from_name("robot"), state.get_object_from_name("target"))
    )
    return sim, controller


@pytest.mark.parametrize("prefer_ompl", [True, False])
def test_move_to_target_controller(prefer_ompl):
    """The controller should drive the arm to the target and then terminate.

    Parameterized over both planners so the BiRRT fallback is covered even when ompl is
    installed.
    """
    env = kinder.make("kinder/VegaMotion3D-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"VegaMotion3D-ompl{prefer_ompl}"
        )

    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    sim, controller = _ground_controller(env, state, prefer_ompl=prefer_ompl)

    rng = np.random.default_rng(123)
    params = controller.sample_parameters(state, rng)
    assert params.shape == (7,)

    controller.reset(state, params)
    terminated = False
    for _ in range(500):
        obs, _, terminated, _, _ = env.step(controller.step())
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Reaching the sampled IK solution should also reach the environment's goal.
    assert terminated

    sim.close()
    env.close()


def test_sampled_parameters_reach_the_target():
    """Sampled parameters should be joint positions whose end effector is on target."""
    env = kinder.make("kinder/VegaMotion3D-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=7)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    sim, controller = _ground_controller(env, state)

    rng = np.random.default_rng(0)
    params = controller.sample_parameters(state, rng)

    # Put the sim at the sampled configuration and check the end effector is in range.
    sim.set_state(state)
    sim.set_arm_joint_positions(params)
    distance = np.linalg.norm(sim.end_effector_pose.t - np.array(state.target_position))
    assert distance < sim.config.target_radius

    sim.close()
    env.close()


def test_ompl_is_preferred_when_available():
    """create_motion_planner should pick OMPL when ompl is importable."""
    sim = ObjectCentricVegaMotion3DEnv(allow_state_access=True)
    manipulator = sim.robot.manipulators[sim.config.manipulator]
    space = sim.robot.groups[manipulator.group]
    rng = np.random.default_rng(0)

    forced = create_motion_planner(space, lambda _: False, rng, prefer_ompl=False)
    assert isinstance(forced, prpl_planning.BiRRTPlanner)

    default = create_motion_planner(space, lambda _: False, rng)
    if ompl_is_available():
        assert not isinstance(default, prpl_planning.BiRRTPlanner)
        assert type(default).__name__ == "OMPLPlanner"
    else:
        assert isinstance(default, prpl_planning.BiRRTPlanner)

    sim.close()
