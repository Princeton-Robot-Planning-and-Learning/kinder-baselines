"""Tests for vega_pickplace3d.py."""

import kinder
import numpy as np
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

# VegaPickPlace3D needs prpl_kinematics, which is an optional extra of both kindergarden
# and this package, and a kindergarden release that contains the environment. Skip the
# module rather than fail collection when either is missing.
vega_pickplace3d = pytest.importorskip("kinder.envs.kinematic3d_v2.vega_pickplace3d")
pytest.importorskip(
    "kinder_models.kinematic3d_v2.vega_pickplace3d.parameterized_skills"
)

kinder.register_all_environments()


def _run_bilevel_planning(env, seed):
    """Plan and execute one episode; the episode must reach the goal."""
    env_models = create_bilevel_planning_models(
        "vega_pickplace3d",
        env.observation_space,
        env.action_space,
        prefer_ompl=False,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=10,
        samples_per_step=2,
        planning_timeout=600.0,
        max_skill_horizon=1000,
    )
    obs, info = env.reset(seed=seed)
    agent.reset(obs, info)

    terminated = truncated = False
    for _ in range(3000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, reward, terminated or truncated, info)
        if terminated or truncated:
            break
    else:
        assert False, "Did not terminate successfully"

    assert terminated, "Episode truncated rather than reaching the goal"


def test_vega_pickplace3d_bilevel_planning():
    """Bilevel planning solves a single-arm VegaPickPlace3D episode."""
    env = kinder.make("kinder/VegaPickPlace3D-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="VegaPickPlace3D")
    # For seed 0 the cube and the target are both on the right side of the table.
    _run_bilevel_planning(env, seed=0)
    env.close()


def test_vega_pickplace3d_grasp_approach_max_tilt_plumbing():
    """The tilt constraint threads through model creation to the pick sampler."""
    max_tilt = 0.1
    env = kinder.make("kinder/VegaPickPlace3D-v0")
    env_models = create_bilevel_planning_models(
        "vega_pickplace3d",
        env.observation_space,
        env.action_space,
        prefer_ompl=False,
        grasp_approach_max_tilt=max_tilt,
    )
    # For seed 0 the cube is on the right side of the table.
    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs)
    pick = next(s for s in env_models.skills if s.operator.name == "Pick")
    arm = state.get_object_from_name("right_arm")
    cube = state.get_object_from_name("cube")
    params = pick.controller.ground((arm, cube)).sample_parameters(
        state, np.random.default_rng(0)
    )

    sim = vega_pickplace3d.ObjectCentricVegaPickPlace3DEnv(allow_state_access=True)
    sim.set_state(state)
    sim.set_arm_joint_positions("right", params)
    rotation = sim.end_effector_pose("right").R
    tilt = np.arccos(np.clip(-rotation[2, 2], -1.0, 1.0))
    assert tilt <= max_tilt

    sim.close()
    env.close()


def test_vega_pickplace3d_bilevel_planning_handover():
    """Bilevel planning solves an episode that requires a handover.

    The planner discovers the handover by backtracking: single-arm abstract plans are
    generated first and die during sampling, because no arm can reach both the cube
    and the target.
    """
    env = kinder.make("kinder/VegaPickPlace3D-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix="VegaPickPlace3D-handover"
        )
    # For seed 28 the cube is far on the left side and the target far on the right.
    _run_bilevel_planning(env, seed=28)
    env.close()
