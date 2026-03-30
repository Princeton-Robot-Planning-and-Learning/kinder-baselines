"""Main entry point for running trajectory optimization experiments.

Examples:
    python experiments/run_experiment.py env=motion2d-p0 seed=0

    python experiments/run_experiment.py -m env=motion2d-p0 seed='range(0,10)'

    python experiments/run_experiment.py -m env=motion2d-p0 seed=0 \
        num_rollouts=50,100,200

- Running on multiple environments and seeds (parallelized):
    python experiments/run_experiment.py -m seed='range(0,3)' \
        env=motion2d-p0,motion2d-p2 hydra/launcher=joblib
"""

import logging
import os
from pathlib import Path

import hydra
import kinder
import numpy as np
import pandas as pd
from gymnasium.core import Env
from gymnasium.wrappers import RecordVideo
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from prpl_utils.utils import sample_seed_from_rng, timer

from kinder_trajopt.agent import TrajOptAgent


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:

    logging.info(f"Running seed={cfg.seed}, env={cfg.env.env_id}")

    # Create the environments: one for evaluation, one for the agent's
    # internal simulation (so planning never mutates the real env).
    kinder.register_all_environments()
    env = kinder.make(cfg.env.env_id, render_mode="rgb_array")
    sim_env = kinder.make(cfg.env.env_id, allow_state_access=True)

    # Record videos.
    if cfg.make_videos:
        video_path = Path(cfg.video_folder)
        video_path.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, str(video_path), episode_trigger=lambda _: True)

    # Create the agent.
    agent = TrajOptAgent(
        sim_env,
        seed=cfg.seed,
        horizon=cfg.horizon,
        num_rollouts=cfg.num_rollouts,
        noise_fraction=cfg.noise_fraction,
        num_control_points=cfg.num_control_points,
        warm_start=cfg.warm_start,
        replan_interval=cfg.env.replan_interval,
        checkpoint=cfg.env.checkpoint if cfg.env.use_checkpoint else None,
        preserved_indices=cfg.env.preserved_indices if cfg.env.use_checkpoint else None,
    )

    # Evaluate.
    rng = np.random.default_rng(cfg.seed)
    metrics: list[dict[str, float]] = []
    for eval_episode in range(cfg.num_eval_episodes):
        logging.info(f"Starting evaluation episode {eval_episode}")
        try:
            episode_metrics = _run_single_episode_evaluation(
                agent,
                env,
                rng,
                max_eval_steps=cfg.env.max_eval_steps,
            )
        except Exception as e:
            logging.error(
                f"Episode {eval_episode} failed with error: {e}", exc_info=True
            )
            episode_metrics = {
                "success": False,
                "steps": 0,
                "planning_time": 0.0,
                "execution_time": 0.0,
                "reward": 0.0,
                "eval_episode": eval_episode,
                "error": str(e),
            }
            metrics.append(episode_metrics)
            continue
        episode_metrics["eval_episode"] = eval_episode
        metrics.append(episode_metrics)

    # Aggregate and save results.
    df = pd.DataFrame(metrics)

    # Save results and config.
    current_dir = HydraConfig.get().runtime.output_dir

    results_path = os.path.join(current_dir, "results.csv")
    df.to_csv(results_path, index=False)
    logging.info(f"Saved results to {results_path}")

    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
    logging.info(f"Saved config to {config_path}")

    env.close()  # type: ignore


def _run_single_episode_evaluation(
    agent: TrajOptAgent,
    env: Env,
    rng: np.random.Generator,
    max_eval_steps: int,
) -> dict[str, float]:
    steps = 0
    total_reward = 0.0
    seed = sample_seed_from_rng(rng)
    obs, info = env.reset(seed=seed)
    planning_time = 0.0
    execution_time = 0.0
    with timer() as result:
        agent.reset(obs, info)
    planning_time += result["time"]
    for _ in range(max_eval_steps):
        with timer() as result:
            action = agent.step()
        planning_time += result["time"]
        obs, rew, done, truncated, info = env.step(action)
        reward = float(rew)
        total_reward += reward
        assert not truncated
        with timer() as result:
            agent.update(obs, reward, done, info)
        execution_time += result["time"]
        if done:
            break
        steps += 1
    logging.info(f"Success: {done}, steps: {steps}")
    return {
        "success": done,
        "steps": steps,
        "planning_time": planning_time,
        "execution_time": execution_time,
        "reward": total_reward,
    }


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
