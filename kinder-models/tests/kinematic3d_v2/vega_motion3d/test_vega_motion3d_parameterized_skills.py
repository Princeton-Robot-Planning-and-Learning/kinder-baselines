"""Tests for VegaMotion3D parameterized skills."""

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from relational_structs.spaces import ObjectCentricBoxSpace

# VegaMotion3D needs prpl_kinematics, which is an optional extra of both kindergarden and
# this package. Skip the module rather than fail collection when it is not installed.
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

    # Reaching the sampled goal configuration should also reach the environment's goal.
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


def test_incident_target_plan_stays_near_start():
    """Goals and plans for the hardware-incident target must stay near the start.

    Regression test for issue #110 (fix direction 3 from kindergarden#150): for the
    target (0.5, -0.4, 0.8), the old fixed-orientation IK sampler returned goals on the
    far shoulder branch, 3.85 rad away on a single joint, and the planner faithfully
    produced a ~220 degree swing that was first caught on hardware. Goal sampling within
    a joint window of the current configuration bounds the net displacement by
    construction; this test pins that down in sim, for the sampled goal and for the
    executed plan.
    """
    env = kinder.make("kinder/VegaMotion3D-v0")
    obs, _ = env.reset(seed=123)
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    state = env.observation_space.devectorize(obs)
    target_obj = state.get_object_from_name("target")
    for feature, value in zip("xyz", (0.5, -0.4, 0.8), strict=True):
        state.set(target_obj, feature, value)
    sim, controller = _ground_controller(env, state, prefer_ompl=False)
    env.close()

    rng = np.random.default_rng(0)
    start_joints = np.asarray(state.arm_joint_positions)
    window = sim.config.target_witness_joint_delta

    # The sampled goal stays within the joint window of the start on every joint. The
    # incident goal violated this bound by nearly a factor of four.
    params = controller.sample_parameters(state, rng)
    assert np.max(np.abs(params - start_joints)) <= window

    # Execute the plan in a second sim and bound the trajectory itself: every visited
    # configuration stays within the window plus planner slack, so the plan cannot
    # detour through a far branch on the way to a nearby goal.
    executor = vega_motion3d.ObjectCentricVegaMotion3DEnv(allow_state_access=True)
    executor.reset(seed=0)
    executor.set_state(state)
    controller.reset(state, params)
    current = state
    max_excursion = 0.0
    for _ in range(500):
        current, _, _, _, _ = executor.step(controller.step())
        controller.observe(current)
        joints = np.asarray(current.arm_joint_positions)
        max_excursion = max(max_excursion, float(np.max(np.abs(joints - start_joints))))
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    assert executor.goal_reached()
    final_joints = np.asarray(current.arm_joint_positions)
    assert np.max(np.abs(final_joints - start_joints)) <= window
    assert max_excursion <= 2 * window

    executor.close()
    sim.close()


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
