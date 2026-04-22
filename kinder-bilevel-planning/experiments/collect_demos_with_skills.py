"""Collect demonstrations with skill-level information using bilevel planning.

Retries planning until one successful demo is found.  Saves observations,
actions, rewards, *and* the ground-controller / parameter records for each
skill step in the plan.  The output is a single pickle file per demo.

Examples:
    python experiments/collect_demos_with_skills.py env=motion2d-p0 seed=0

    python experiments/collect_demos_with_skills.py env=dynpushpullhook2d-o0 \
        seed=0 max_attempts=50

    python experiments/collect_demos_with_skills.py env=motion2d-p2 seed=42 \
        num_demos=3 demo_dir=./skill_demos
"""

import logging
import sys
import time
from pathlib import Path

import dill as pkl  # type: ignore[import-untyped]
import hydra
import kinder
import numpy as np
from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import PlanningProblem, SesameModels
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from gymnasium.core import Env
from omegaconf import DictConfig
from prpl_utils.utils import sample_seed_from_rng

from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from kinder_bilevel_planning.logging_sampler import (
    LoggingParameterizedControllerTrajectorySampler,
    SkillExecutionRecord,
)


def _run_planning_with_logging(
    env_models: SesameModels,
    obs: np.ndarray,
    seed: int,
    max_abstract_plans: int,
    samples_per_step: int,
    max_skill_horizon: int,
    heuristic_name: str,
    planning_timeout: float,
) -> tuple[list, list[SkillExecutionRecord]] | None:
    """Run bilevel planning and return (plan_actions, skill_records) or None."""
    initial_state = env_models.observation_to_state(obs)
    goal = env_models.goal_deriver(initial_state)
    problem = PlanningProblem(
        env_models.state_space,
        env_models.action_space,
        initial_state,
        env_models.transition_fn,
        goal,
    )

    # Create trajectory sampler with logging.
    trajectory_sampler = LoggingParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator(env_models.skills),
        transition_function=env_models.transition_fn,
        state_abstractor=env_models.state_abstractor,
        max_trajectory_steps=max_skill_horizon,
    )

    abstract_plan_generator: AbstractPlanGenerator = (
        RelationalHeuristicSearchAbstractPlanGenerator(
            env_models.types,
            env_models.predicates,
            env_models.operators,
            heuristic_name,
            seed=seed,
        )
    )

    abstract_successor_fn = RelationalAbstractSuccessorGenerator(env_models.operators)

    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        max_abstract_plans,
        samples_per_step,
        abstract_successor_fn,
        env_models.state_abstractor,
        seed=seed,
    )

    trajectory_sampler.clear()
    plan, _ = planner.run(problem, timeout=planning_timeout)

    if plan is None:
        return None

    skill_records = trajectory_sampler.extract_plan_records(plan)
    return plan.actions, skill_records


def _collect_single_demo(
    env: Env,
    env_models: SesameModels,
    seed: int,
    max_abstract_plans: int,
    samples_per_step: int,
    max_skill_horizon: int,
    heuristic_name: str,
    planning_timeout: float,
    max_eval_steps: int,
) -> dict | None:
    """Attempt one demo collection.

    Returns demo dict or None on failure.
    """
    obs, _ = env.reset(seed=seed)

    result = _run_planning_with_logging(
        env_models,
        obs,
        seed,
        max_abstract_plans,
        samples_per_step,
        max_skill_horizon,
        heuristic_name,
        planning_timeout,
    )
    if result is None:
        logging.info("Planning failed for seed %d.", seed)
        return None

    plan_actions, skill_records = result

    # Execute the plan in the environment.
    observations: list = [obs]
    actions: list = []
    rewards: list[float] = []
    terminated = False
    truncated = False

    for step_i, action in enumerate(plan_actions):
        if step_i >= max_eval_steps:
            truncated = True
            break
        obs, rew, done, trunc, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(float(rew))
        if done:
            terminated = True
            break
        if trunc:
            truncated = True
            break

    if not terminated:
        logging.info("Plan execution did not reach goal for seed %d.", seed)
        return None

    # Serialise skill records.
    skill_info = []
    for rec in skill_records:
        skill_info.append(
            {
                "operator_name": rec.ground_operator.name,
                "operator_objects": [
                    (o.name, o.type.name) for o in rec.ground_operator.parameters
                ],
                "controller_class": type(rec.controller).__name__,
                "controller_objects": [
                    (o.name, o.type.name) for o in rec.controller.objects
                ],
                "params": rec.params,
                "num_actions": len(rec.actions),
            }
        )

    return {
        "env_id": env.spec.id if env.spec else "unknown",
        "seed": seed,
        "timestamp": int(time.time()),
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "skill_info": skill_info,
    }


@hydra.main(version_base=None, config_name="config", config_path="conf/")
def _main(cfg: DictConfig) -> None:
    logging.info("Collecting demos: seed=%d, env=%s", cfg.seed, cfg.env.env_name)

    demo_dir = Path(cfg.get("demo_dir", "./skill_demos"))
    demo_dir.mkdir(parents=True, exist_ok=True)
    num_demos = cfg.get("num_demos", 1)
    max_attempts = cfg.get("max_attempts", 100)

    kinder.register_all_environments()
    env = kinder.make(**cfg.env.make_kwargs, render_mode="rgb_array")

    env_models = create_bilevel_planning_models(
        cfg.env.env_name,
        env.observation_space,
        env.action_space,
        **cfg.env.env_model_kwargs,
    )

    def _resolve(key: str):
        return cfg.env.get(key, cfg[key])

    rng = np.random.default_rng(cfg.seed)
    successful = 0

    for attempt in range(max_attempts):
        if successful >= num_demos:
            break
        seed = sample_seed_from_rng(rng)
        logging.info(
            "Attempt %d/%d (seed=%d, collected=%d/%d)",
            attempt + 1,
            max_attempts,
            seed,
            successful,
            num_demos,
        )

        demo = _collect_single_demo(
            env,
            env_models,
            seed,
            max_abstract_plans=_resolve("max_abstract_plans"),
            samples_per_step=_resolve("samples_per_step"),
            max_skill_horizon=_resolve("max_skill_horizon"),
            heuristic_name=_resolve("heuristic_name"),
            planning_timeout=_resolve("planning_timeout"),
            max_eval_steps=cfg.max_eval_steps,
        )

        if demo is not None:
            env_id_clean = (
                cfg.env.make_kwargs["id"].replace("kinder/", "").replace("/", "_")
            )
            fname = f"{env_id_clean}_seed{seed}.pkl"
            path = demo_dir / fname
            with open(path, "wb") as f:
                pkl.dump(demo, f)
            logging.info("Saved demo to %s", path)
            successful += 1

    logging.info(
        "Done. Collected %d/%d demos in %d attempts.",
        successful,
        num_demos,
        attempt + 1,
    )

    if successful == 0:
        logging.error("No successful demos collected!")
        sys.exit(1)

    env.close()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
