"""Spec-driven reward MPC sanity rollout for Motion2D."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import kinder
import numpy as np
from PIL import Image as PILImage
from prpl_utils.trajopt.mpc_wrapper import MPCWrapper
from prpl_utils.trajopt.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)

from kinder_reward_grounding.adapters.kinder import KinDERObjectCentricAdapter
from kinder_reward_grounding.env_specs.motion2d import make_motion2d_reward_spec
from kinder_reward_grounding.evaluator import RewardEvaluator
from kinder_reward_grounding.safe_trajopt_problem import SafeKinderTrajOptProblem

ENV_ID = "kinder/Motion2D-p5-v0"
OUTPUT_PATH = Path("outputs/motion2d_spec_reward_mpc.gif")


def robot_goal_distance(
    env: Any,
    flat_state: np.ndarray,
    adapter: KinDERObjectCentricAdapter,
) -> float:
    """Return center distance from Motion2D robot to target region."""
    scene = adapter.decode(env, flat_state)
    robot = adapter.get_object_by_name(scene, "robot")
    goal = adapter.get_object_by_name(scene, "target_region")
    return adapter.distance(scene, robot, goal)


def main() -> None:
    """Run a minimal spec reward MPC rollout and save a GIF."""
    kinder.register_all_environments()

    eval_env = gym.make(ENV_ID, render_mode="rgb_array", allow_state_access=True)
    sim_env = gym.make(ENV_ID, allow_state_access=True)

    obs, _ = eval_env.reset(seed=42)

    spec = make_motion2d_reward_spec()
    adapter = KinDERObjectCentricAdapter()
    reward_fn = RewardEvaluator(spec, adapter)

    problem = SafeKinderTrajOptProblem(
        env=sim_env,
        initial_state=obs,
        horizon=50,
        reward_fn=reward_fn,
    )

    action_range = sim_env.action_space.high - sim_env.action_space.low
    config = PredictiveSamplingHyperparameters(
        num_rollouts=40,
        noise_scale=action_range * 1.0,
        num_control_points=5,
    )
    solver = PredictiveSamplingSolver(seed=42, config=config, warm_start=True)
    mpc = MPCWrapper(solver, replan_interval=1)
    mpc.reset(problem)

    frames = [eval_env.render()]
    distances = [robot_goal_distance(eval_env, obs, adapter)]
    print(f"Initial robot-goal distance: {distances[0]:.4f}")

    reached = False
    steps = 0
    max_steps = 200

    for step in range(max_steps):
        action = mpc.step(obs)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        frames.append(eval_env.render())

        distance = robot_goal_distance(eval_env, obs, adapter)
        distances.append(distance)
        steps = step + 1

        if step % 10 == 0:
            print(
                f"Step {steps:03d} | "
                f"distance={distance:.4f} | "
                f"terminated={terminated} | "
                f"truncated={truncated}"
            )

        if terminated:
            reached = True
            break
        if truncated:
            break

    print("\n=== Motion2D spec reward MPC summary ===")
    print(f"Initial robot-goal distance: {distances[0]:.4f}")
    print(f"Best robot-goal distance:    {min(distances):.4f}")
    print(f"Final robot-goal distance:   {distances[-1]:.4f}")
    print(f"Goal reached?                {reached}")
    print(f"Steps:                       {steps}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [PILImage.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        OUTPUT_PATH,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0,
    )
    print(f"Saved {len(frames)} frames to: {OUTPUT_PATH}")

    eval_env.close()
    sim_env.close()


if __name__ == "__main__":
    main()
