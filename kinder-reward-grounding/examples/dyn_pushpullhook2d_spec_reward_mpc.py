"""Spec-driven reward MPC rollout for DynPushPullHook2D."""

# Keeping the complete rollout in one function makes this example easy to run
# and modify independently.
# pylint: disable=too-many-locals

from __future__ import annotations

from pathlib import Path
from typing import Any

import kinder
import numpy as np
from gymnasium.spaces import Box
from PIL import Image as PILImage
from prpl_utils.trajopt.mpc_wrapper import MPCWrapper
from prpl_utils.trajopt.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)

from kinder_reward_grounding.adapters.kinder import KinDERObjectCentricAdapter
from kinder_reward_grounding.env_specs.dyn_pushpullhook2d import (
    make_dyn_pushpullhook2d_reward_spec,
)
from kinder_reward_grounding.evaluator import RewardEvaluator
from kinder_reward_grounding.safe_trajopt_problem import SafeKinderTrajOptProblem

ENV_ID = "kinder/DynPushPullHook2D-o0-v0"
OUTPUT_PATH = Path("outputs/dyn_pushpullhook2d_spec_reward_mpc.gif")


def render_frame(env: Any) -> np.ndarray:
    """Render one RGB frame or fail clearly if rendering is unavailable."""
    frame = env.render()
    if frame is None:
        raise RuntimeError("Environment returned no frame in rgb_array mode.")
    return np.asarray(frame)


def compute_metrics(
    env: Any,
    flat_state: np.ndarray,
    adapter: KinDERObjectCentricAdapter,
) -> dict[str, float]:
    """Compute compact diagnostics for the spec reward rollout."""
    scene = adapter.decode(env, flat_state)
    robot = adapter.get_object_by_name(scene, "robot")
    hook = adapter.get_object_by_name(scene, "hook")
    target = adapter.get_object_by_name(scene, "target_block")
    wall = adapter.get_object_by_name(scene, "middle_wall")
    return {
        "robot_hook_dist": adapter.distance(scene, robot, hook),
        "hook_keypoint_target_dist": adapter.min_keypoint_distance(
            scene,
            hook,
            target,
            keypoint_type="hook_heuristic",
        ),
        "target_wall_gap": adapter.vertical_gap(scene, target, wall),
        "hook_held": float(adapter.get(scene, hook, "held")),
    }


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    """Print compact task metrics."""
    print(
        f"{prefix} | "
        f"robot_hook={metrics['robot_hook_dist']:.3f} | "
        f"hook_keypoint_target={metrics['hook_keypoint_target_dist']:.3f} | "
        f"target_wall_gap={metrics['target_wall_gap']:.3f} | "
        f"hook_held={metrics['hook_held']:.0f}"
    )


def print_metric_summary(
    metric_history: list[dict[str, float]],
    problem: SafeKinderTrajOptProblem,
) -> None:
    """Print rollout summary metrics."""
    robot_hook = [m["robot_hook_dist"] for m in metric_history]
    hook_keypoint_target = [m["hook_keypoint_target_dist"] for m in metric_history]
    target_wall_gap = [m["target_wall_gap"] for m in metric_history]
    hook_held = [m["hook_held"] for m in metric_history]

    print("\n=== Spec reward MPC metric summary ===")
    print(f"Initial robot-hook distance:          {robot_hook[0]:.4f}")
    print(f"Best robot-hook distance:             {min(robot_hook):.4f}")
    print(f"Final robot-hook distance:            {robot_hook[-1]:.4f}")
    print(f"Initial hook-keypoint-target distance:{hook_keypoint_target[0]:.4f}")
    print(f"Best hook-keypoint-target distance:   {min(hook_keypoint_target):.4f}")
    print(f"Final hook-keypoint-target distance:  {hook_keypoint_target[-1]:.4f}")
    print(f"Initial target-wall gap:              {target_wall_gap[0]:.4f}")
    print(f"Best target-wall gap:                 {min(target_wall_gap):.4f}")
    print(f"Final target-wall gap:                {target_wall_gap[-1]:.4f}")
    print(f"Hook was ever held?                   {max(hook_held) > 0.5}")
    print(f"Invalid transitions caught:           {problem.num_invalid_transitions}")


def main() -> None:
    """Run the spec-driven reward MPC rollout and save a GIF."""
    kinder.register_all_environments()

    eval_env = kinder.make(ENV_ID, render_mode="rgb_array", allow_state_access=True)
    sim_env = kinder.make(ENV_ID, allow_state_access=True)

    obs, _ = eval_env.reset(seed=42)

    spec = make_dyn_pushpullhook2d_reward_spec()
    adapter = KinDERObjectCentricAdapter()
    reward_fn = RewardEvaluator(spec, adapter)

    problem = SafeKinderTrajOptProblem(
        env=sim_env,
        initial_state=obs,
        horizon=100,
        reward_fn=reward_fn,
        invalid_transition_reward=-1e6,
    )

    if not isinstance(sim_env.action_space, Box):
        raise TypeError("Expected a continuous Box action space.")
    action_range = sim_env.action_space.high - sim_env.action_space.low
    config = PredictiveSamplingHyperparameters(
        num_rollouts=20,
        noise_scale=action_range * 1.0,
        num_control_points=5,
    )
    solver = PredictiveSamplingSolver(seed=42, config=config, warm_start=True)
    mpc = MPCWrapper(solver, replan_interval=1)
    mpc.reset(problem)

    frames: list[np.ndarray] = [render_frame(eval_env)]
    metric_history = [compute_metrics(eval_env, obs, adapter)]
    print_metrics("Initial metrics", metric_history[-1])

    max_steps = 50
    print(
        f"Starting spec reward MPC rollout for {ENV_ID} with max_steps={max_steps}..."
    )

    for step in range(max_steps):
        action = mpc.step(obs)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        frames.append(render_frame(eval_env))

        metrics = compute_metrics(eval_env, obs, adapter)
        metric_history.append(metrics)

        if step % 10 == 0:
            print(
                f"Step {step + 1:03d} | "
                f"spec_reward={reward:.3f} | "
                f"terminated={terminated} | "
                f"truncated={truncated} | "
                f"invalid={problem.num_invalid_transitions} | "
                f"robot_hook={metrics['robot_hook_dist']:.3f} | "
                f"hook_keypoint_target={metrics['hook_keypoint_target_dist']:.3f} | "
                f"target_wall_gap={metrics['target_wall_gap']:.3f} | "
                f"hook_held={metrics['hook_held']:.0f}"
            )

        if terminated:
            print(f"Reached goal in {step + 1} steps.")
            break
        if truncated:
            print(f"Episode truncated after {step + 1} steps.")
            break
    else:
        print(f"Did not reach goal within {max_steps} steps.")

    print_metric_summary(metric_history, problem)

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
