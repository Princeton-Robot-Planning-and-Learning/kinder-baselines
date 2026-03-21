"""Tests for TrajOptAgent on Motion2D."""

import kinder
import pytest
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_trajopt.agent import TrajOptAgent

kinder.register_all_environments()


def test_trajopt_agent_single_step():
    """Agent can reset and produce a single action."""
    env = kinder.make("kinder/Motion2D-p0-v0", allow_state_access=True)
    agent = TrajOptAgent(
        env,
        seed=123,
        horizon=10,
        num_rollouts=5,
        num_control_points=5,
    )
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    action = agent.step()
    assert env.action_space.shape == action.shape
    env.close()


def test_trajopt_agent_multiple_steps():
    """Agent can run for multiple steps without errors."""
    env = kinder.make("kinder/Motion2D-p0-v0", allow_state_access=True)
    agent = TrajOptAgent(
        env,
        seed=123,
        horizon=10,
        num_rollouts=10,
        num_control_points=5,
    )
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    for _ in range(10):
        action = agent.step()
        obs, reward, done, _truncated, info = env.step(action)
        agent.update(obs, float(reward), done, info)
        if done:
            break
    env.close()


@pytest.mark.parametrize("num_passages", [0])
def test_trajopt_agent_motion2d(num_passages: int) -> None:
    """End-to-end test: agent should solve Motion2D-p0 within the step
    budget."""
    env_id = f"kinder/Motion2D-p{num_passages}-v0"
    env = kinder.make(env_id, allow_state_access=True, render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix=f"trajopt-Motion2D-p{num_passages}",
        )

    agent = TrajOptAgent(
        env,
        seed=123,
        horizon=20,
        num_rollouts=50,
        noise_fraction=0.5,
        num_control_points=5,
    )
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    total_reward = 0.0
    done = False
    for _ in range(200):
        action = agent.step()
        obs, reward, done, _truncated, info = env.step(action)
        total_reward += float(reward)
        agent.update(obs, float(reward), done, info)
        if done:
            break

    env.close()  # type: ignore[no-untyped-call]
    assert done, (
        f"Agent did not solve Motion2D-p{num_passages} "
        f"in 200 steps (reward={total_reward})"
    )
