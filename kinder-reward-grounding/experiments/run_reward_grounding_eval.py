"""Run reward-grounding MPC evaluations with trajopt-compatible logs.

This script writes a Hydra-like multirun directory that can be consumed by:

    ../kinder-trajopt/experiments/analyze_results.py

Each seed is written to one numbered subdirectory containing config.yaml and
results.csv.
"""

# Evaluation jobs deliberately keep their full serialized configuration and
# episode loop explicit for reproducibility.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-statements

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import kinder
import numpy as np
import yaml
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
from kinder_reward_grounding.env_specs.motion2d import make_motion2d_reward_spec
from kinder_reward_grounding.evaluator import RewardEvaluator
from kinder_reward_grounding.safe_trajopt_problem import SafeKinderTrajOptProblem
from kinder_reward_grounding.specs import RewardSpec

RewardFactory = Callable[[str], RewardSpec]


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for a supported reward-grounding evaluation env."""

    env_id: str
    spec_factory: RewardFactory
    default_horizon: int
    default_max_steps: int
    default_num_rollouts: int
    default_num_control_points: int
    diagnostic: str


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class EvalConfig:
    """Serializable evaluation configuration."""

    env_name: str
    env_id: str
    method: str
    seed: int
    num_eval_episodes: int
    horizon: int
    num_rollouts: int
    noise_fraction: float
    num_control_points: int
    warm_start: bool
    replan_interval: int
    max_steps: int
    save_gifs: bool
    output_dir: str


def parse_seeds(seed_args: list[str]) -> list[int]:
    """Parse seeds from repeated ints, comma lists, or Python-style ranges."""
    seeds: list[int] = []
    for seed_arg in seed_args:
        for part in seed_arg.split(","):
            text = part.strip()
            if not text:
                continue
            if text.startswith("range(") and text.endswith(")"):
                values = [int(v.strip()) for v in text[6:-1].split(",")]
                seeds.extend(list(range(*values)))
            elif "-" in text and not text.startswith("-"):
                start, stop = [int(v.strip()) for v in text.split("-", maxsplit=1)]
                seeds.extend(range(start, stop + 1))
            else:
                seeds.append(int(text))
    return seeds


def resolve_env(env_name: str) -> EnvConfig:
    """Resolve a user-facing env name into a KinDER env and diagnostics config."""
    normalized = env_name.lower().replace("_", "-")

    if normalized.startswith("motion2d"):
        variant = normalized.removeprefix("motion2d").lstrip("-") or "p5"
        env_id = f"kinder/Motion2D-{variant}-v0"
        return EnvConfig(
            env_id=env_id,
            spec_factory=make_motion2d_reward_spec,
            default_horizon=50,
            default_max_steps=200,
            default_num_rollouts=40,
            default_num_control_points=5,
            diagnostic="motion2d_distance",
        )

    if normalized in {"dyn-pushpullhook2d", "dynpushpullhook2d"}:
        normalized = "dynpushpullhook2d-o0"
    if normalized.startswith("dynpushpullhook2d"):
        variant = normalized.removeprefix("dynpushpullhook2d").lstrip("-") or "o0"
        env_id = f"kinder/DynPushPullHook2D-{variant}-v0"
        return EnvConfig(
            env_id=env_id,
            spec_factory=make_dyn_pushpullhook2d_reward_spec,
            default_horizon=100,
            default_max_steps=50,
            default_num_rollouts=20,
            default_num_control_points=5,
            diagnostic="dyn_pushpullhook2d_keypoint_distance",
        )

    raise ValueError(
        f"Unsupported env {env_name!r}. Try motion2d-p5 or dyn_pushpullhook2d."
    )


def build_reward_fn(
    method: str,
    env_id: str,
    adapter: KinDERObjectCentricAdapter,
    spec_factory: RewardFactory,
) -> tuple[RewardEvaluator | None, RewardSpec | None]:
    """Build the optional planner reward function for the requested method."""
    if method == "sparse":
        return None, None
    if method in {"oracle_dense", "spec_reward"}:
        spec = spec_factory(env_id)
        return RewardEvaluator(spec, adapter), spec
    raise ValueError("method must be one of: sparse, oracle_dense, spec_reward")


def compute_distance_diagnostic(
    diagnostic: str,
    env: Any,
    flat_state: np.ndarray,
    adapter: KinDERObjectCentricAdapter,
) -> float:
    """Compute the primary distance diagnostic for a supported environment."""
    scene = adapter.decode(env, flat_state)
    if diagnostic == "motion2d_distance":
        robot = adapter.get_object_by_name(scene, "robot")
        goal = adapter.get_object_by_name(scene, "target_region")
        return adapter.distance(scene, robot, goal)

    if diagnostic == "dyn_pushpullhook2d_keypoint_distance":
        hook = adapter.get_object_by_name(scene, "hook")
        target = adapter.get_object_by_name(scene, "target_block")
        return adapter.min_keypoint_distance(
            scene,
            hook,
            target,
            keypoint_type="hook_heuristic",
        )

    return float("nan")


def make_config_dict(
    cfg: EvalConfig,
    spec: RewardSpec | None,
) -> dict[str, Any]:
    """Build analyzer-compatible config.yaml content."""
    subgoals = [subgoal.name for subgoal in spec.subgoals] if spec is not None else []
    return {
        "env": {
            "env_id": cfg.env_id,
            "replan_interval": cfg.replan_interval,
        },
        "seed": cfg.seed,
        "horizon": cfg.horizon,
        "num_rollouts": cfg.num_rollouts,
        "noise_fraction": cfg.noise_fraction,
        "num_control_points": cfg.num_control_points,
        "warm_start": cfg.warm_start,
        "method": cfg.method,
        "num_eval_episodes": cfg.num_eval_episodes,
        "max_steps": cfg.max_steps,
        "save_gifs": cfg.save_gifs,
        "reward_grounding": {
            "grounder": "oracle_object" if spec is not None else "none",
            "subgoals": subgoals,
            "vlm_enabled": False,
        },
        "reward_grounding_eval": asdict(cfg),
    }


def run_episode(
    eval_env: Any,
    sim_env: Any,
    cfg: EvalConfig,
    env_cfg: EnvConfig,
    adapter: KinDERObjectCentricAdapter,
    reward_fn: RewardEvaluator | None,
    episode_seed: int,
    gif_path: Path | None,
) -> dict[str, Any]:
    """Run one MPC evaluation episode and return analyzer-compatible metrics."""
    obs, _ = eval_env.reset(seed=episode_seed)
    if not isinstance(sim_env.action_space, Box):
        raise TypeError("Expected a continuous Box action space.")
    action_range = sim_env.action_space.high - sim_env.action_space.low
    problem = SafeKinderTrajOptProblem(
        env=sim_env,
        initial_state=obs,
        horizon=cfg.horizon,
        reward_fn=reward_fn,
        invalid_transition_reward=-1e6,
    )
    solver_config = PredictiveSamplingHyperparameters(
        num_rollouts=cfg.num_rollouts,
        noise_scale=action_range * cfg.noise_fraction,
        num_control_points=cfg.num_control_points,
    )
    solver = PredictiveSamplingSolver(
        seed=episode_seed,
        config=solver_config,
        warm_start=cfg.warm_start,
    )
    mpc = MPCWrapper(solver, replan_interval=cfg.replan_interval)

    planning_time = 0.0
    execution_time = 0.0
    env_return = 0.0
    custom_return = 0.0
    steps = 0
    success = False
    success_reason = "max_steps"
    frames: list[np.ndarray] = []
    if gif_path is not None:
        frames.append(render_frame(eval_env))

    start = time.perf_counter()
    mpc.reset(problem)
    planning_time += time.perf_counter() - start

    initial_distance = compute_distance_diagnostic(
        env_cfg.diagnostic,
        eval_env,
        obs,
        adapter,
    )
    final_distance = initial_distance
    best_distance = initial_distance

    for step_idx in range(cfg.max_steps):
        prev_obs = obs

        start = time.perf_counter()
        action = mpc.step(obs)
        planning_time += time.perf_counter() - start

        start = time.perf_counter()
        obs, env_reward, terminated, truncated, _ = eval_env.step(action)
        execution_time += time.perf_counter() - start

        env_reward_float = float(env_reward)
        env_return += env_reward_float
        if reward_fn is not None:
            custom_return += reward_fn(
                prev_obs,
                action,
                obs,
                env_reward_float,
                terminated,
                eval_env,
            )
        else:
            custom_return += env_reward_float

        final_distance = compute_distance_diagnostic(
            env_cfg.diagnostic,
            eval_env,
            obs,
            adapter,
        )
        best_distance = min(best_distance, final_distance)
        steps = step_idx + 1

        if gif_path is not None:
            frames.append(render_frame(eval_env))

        if terminated:
            success = True
            success_reason = "terminated"
            break
        if truncated:
            success_reason = "truncated"
            break

    if gif_path is not None:
        save_gif(frames, gif_path)

    return {
        "success": success,
        "planning_time": planning_time,
        "execution_time": execution_time,
        "steps": steps,
        "reward": env_return,
        "method": cfg.method,
        "final_distance": final_distance,
        "best_distance": best_distance,
        "invalid_transitions": problem.num_invalid_transitions,
        "custom_reward": custom_return,
        "env_reward": env_return,
        "success_reason": success_reason,
    }


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    """Save rendered frames as a GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [PILImage.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0,
    )


def render_frame(env: Any) -> np.ndarray:
    """Render one RGB frame or fail clearly if rendering is unavailable."""
    frame = env.render()
    if frame is None:
        raise RuntimeError("Environment returned no frame in rgb_array mode.")
    return np.asarray(frame)


def run_seed(job_idx: int, cfg: EvalConfig, env_cfg: EnvConfig) -> None:
    """Run all evaluation episodes for one seed and write a numbered job dir."""
    job_dir = Path(cfg.output_dir) / str(job_idx)
    job_dir.mkdir(parents=True, exist_ok=True)

    adapter = KinDERObjectCentricAdapter()
    reward_fn, spec = build_reward_fn(
        cfg.method,
        cfg.env_id,
        adapter,
        env_cfg.spec_factory,
    )
    config_dict = make_config_dict(cfg, spec)
    with open(job_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    eval_env = kinder.make(cfg.env_id, render_mode="rgb_array")
    sim_env = kinder.make(cfg.env_id, allow_state_access=True)
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, Any]] = []

    try:
        for eval_episode in range(cfg.num_eval_episodes):
            episode_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            gif_path = (
                job_dir / "videos" / f"episode_{eval_episode:03d}.gif"
                if cfg.save_gifs
                else None
            )
            row = run_episode(
                eval_env,
                sim_env,
                cfg,
                env_cfg,
                adapter,
                reward_fn,
                episode_seed,
                gif_path,
            )
            row["eval_episode"] = eval_episode
            rows.append(row)
            print(
                f"seed={cfg.seed} episode={eval_episode} "
                f"success={row['success']} steps={row['steps']} "
                f"reward={row['reward']:.3f}",
                flush=True,
            )
    finally:
        eval_env.close()
        sim_env.close()

    fieldnames = [
        "eval_episode",
        "success",
        "planning_time",
        "execution_time",
        "steps",
        "reward",
        "method",
        "final_distance",
        "best_distance",
        "invalid_transitions",
        "custom_reward",
        "env_reward",
        "success_reason",
    ]
    with open(job_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run reward-grounding evaluations with trajopt-compatible logs."
    )
    parser.add_argument("--env", required=True, help="Env alias, e.g. motion2d-p5.")
    parser.add_argument(
        "--method",
        default="spec_reward",
        choices=["sparse", "oracle_dense", "spec_reward"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0"],
        help="Seeds as ints, comma lists, inclusive ranges (0-2), or range(0,3).",
    )
    parser.add_argument("--num-eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-gifs", action="store_true", help="Disable GIF saving.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--noise-fraction", type=float, default=1.0)
    parser.add_argument("--num-control-points", type=int, default=None)
    parser.add_argument("--replan-interval", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all requested seed jobs."""
    args = parse_args()
    env_cfg = resolve_env(args.env)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    kinder.register_all_environments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for job_idx, seed in enumerate(seeds):
        cfg = EvalConfig(
            env_name=args.env,
            env_id=env_cfg.env_id,
            method=args.method,
            seed=seed,
            num_eval_episodes=args.num_eval_episodes,
            horizon=args.horizon or env_cfg.default_horizon,
            num_rollouts=args.num_rollouts or env_cfg.default_num_rollouts,
            noise_fraction=args.noise_fraction,
            num_control_points=(
                args.num_control_points or env_cfg.default_num_control_points
            ),
            warm_start=not args.no_warm_start,
            replan_interval=args.replan_interval,
            max_steps=args.max_steps or env_cfg.default_max_steps,
            save_gifs=not args.no_gifs,
            output_dir=str(args.output_dir),
        )
        print(f"=== Running job {job_idx}: seed={seed}, env={cfg.env_id} ===")
        run_seed(job_idx, cfg, env_cfg)

    print(f"\nWrote trajopt-compatible logs to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
