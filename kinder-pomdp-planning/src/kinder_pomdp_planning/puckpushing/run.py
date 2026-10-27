"""Run POMCPOW on a PuckPushing env.

Each step plans from the current friction belief, executes the chosen push in the env
(2D PyMunk or 3D PyBullet), and conditions the belief on the measured release speed and
stop position.

Example:
    python -m kinder_pomdp_planning.puckpushing.run --env 2d --seed 0 --sims 2000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from kinder.envs.puckpushing_utils import SURFACE_NAMES, Push

from kinder_pomdp_planning.pomcpow import POMCPOW, ActionProposer, POMCPOWConfig
from kinder_pomdp_planning.puckpushing.belief import FrictionBelief, summarize
from kinder_pomdp_planning.puckpushing.model import PuckPushingModel, PuckRewards


@dataclass
class StepLog:
    """One executed push."""

    angle: float
    speed: float
    feasible: bool
    release_speed: float
    stop: list[float]
    fell_in_hole: bool
    goal_reached: bool
    drive_distance: float
    reward: float
    plan_seconds: float
    plan_sims: int
    root_actions: int
    ess: float
    belief: dict[str, list[float]]


@dataclass
class EpisodeLog:
    """One episode."""

    env: str
    map_name: str
    seed: int
    true_frictions: list[float]
    prior_belief: dict[str, list[float]]
    outcome: str = "timeout"
    total_reward: float = 0.0
    total_plan_seconds: float = 0.0
    steps: list[StepLog] = field(default_factory=list)


def make_env(env_name: str, map_name: str) -> Any:
    """The object-centric env; it reports the hidden frictions for evaluation."""
    if env_name == "2d":
        # pylint: disable-next=import-outside-toplevel
        from kinder.envs.dynamic2d.dyn_puckpushing2d import (
            ObjectCentricDynPuckPushing2DEnv,
        )

        return ObjectCentricDynPuckPushing2DEnv(map_name=map_name)
    if env_name == "3d":
        # pylint: disable-next=import-outside-toplevel
        from kinder.envs.dynamic3d.puckpushing3d import ObjectCentricPuckPushing3DEnv

        return ObjectCentricPuckPushing3DEnv(map_name=map_name)
    raise ValueError(f"Unknown env {env_name!r}; expected '2d' or '3d'")


def run_episode(
    env: Any,
    env_name: str,
    seed: int,
    config: POMCPOWConfig,
    rewards: PuckRewards = PuckRewards(),
    num_particles: int = 500,
    max_pushes: int = 15,
    frictions: tuple[float, ...] | None = None,
    proposer: ActionProposer | None = None,
    on_step: Callable[[StepLog], None] | None = None,
) -> EpisodeLog:
    """Plan and execute pushes until the goal, a hole, or `max_pushes`."""
    rng = np.random.default_rng(seed)
    options = {"frictions": frictions} if frictions is not None else None
    env.reset(seed=seed, options=options)
    world = env.world
    model = PuckPushingModel(
        world,
        rewards=rewards,
        friction_prior=env.config.friction_prior,
        max_pushes=max_pushes,
    )
    belief = FrictionBelief.from_prior(rng, num_particles, env.config.friction_prior)
    log = EpisodeLog(
        env=env_name,
        map_name=world.puck_map.name,
        seed=seed,
        true_frictions=list(env.frictions),
        prior_belief=summarize(belief, SURFACE_NAMES),
    )
    puck = np.asarray(world.puck_map.puck_start, dtype=np.float64)
    robot = np.asarray(world.puck_map.robot_start, dtype=np.float64)
    for pushes in range(max_pushes):
        model.rollout_frictions = tuple(float(f) for f in belief.mean)
        solver = POMCPOW(model, config, proposer=proposer, rng=rng)
        action, stats = solver.plan(belief.states(model, puck, robot, pushes))
        if action is None:
            log.outcome = "stuck"
            break
        _, _, terminated, _, info = env.step(np.array([action.angle, action.speed]))
        stop = np.asarray(info["stop_position"], dtype=np.float64)
        ess = float("nan")
        if info["push_feasible"]:
            ess = belief.update(
                world,
                puck,
                action.angle,
                float(info["release_speed"]),
                stop,
                bool(info["fell_in_hole"]),
                rng,
            )
            robot = world.push_station(puck, action.angle) + (
                world.physics.accel_distance * Push(action.angle, 0.0).direction
            )
        step_reward = _step_reward(info, rewards)
        step = StepLog(
            angle=action.angle,
            speed=action.speed,
            feasible=bool(info["push_feasible"]),
            release_speed=float(info["release_speed"]),
            stop=stop.tolist(),
            fell_in_hole=bool(info["fell_in_hole"]),
            goal_reached=bool(info["goal_reached"]),
            drive_distance=float(info["drive_distance"]),
            reward=step_reward,
            plan_seconds=stats.seconds,
            plan_sims=stats.num_sims,
            root_actions=stats.root_actions,
            ess=ess,
            belief=summarize(belief, SURFACE_NAMES),
        )
        log.steps.append(step)
        log.total_reward += step_reward
        log.total_plan_seconds += stats.seconds
        if on_step is not None:
            on_step(step)
        # A noisy measurement can land the puck slightly inside a wall.
        puck = world.nearest_free_puck_position(stop)
        if terminated:
            log.outcome = "goal" if info["goal_reached"] else "hole"
            break
    return log


def _step_reward(info: dict, rewards: PuckRewards) -> float:
    """The planner's reward for an executed push, from the env's report."""
    reward = -rewards.push_cost - rewards.drive_cost * float(info["drive_distance"])
    if info["fell_in_hole"]:
        reward -= rewards.hole_penalty
    elif info["goal_reached"]:
        reward += rewards.goal_reward
    return reward


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["2d", "3d"], default="2d")
    parser.add_argument("--map", default="threeroutes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-pushes", type=int, default=15)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--frictions", type=float, nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = POMCPOWConfig(
        num_sims=args.sims, max_time=args.max_time, max_depth=args.max_depth
    )
    env = make_env(args.env, args.map)
    start = time.perf_counter()

    def report(step: StepLog) -> None:
        means = {k: round(v[0], 3) for k, v in step.belief.items()}
        print(
            f"push angle={step.angle:+.2f} speed={step.speed:.2f} "
            f"stop=({step.stop[0]:.2f},{step.stop[1]:.2f}) reward={step.reward:.1f} "
            f"plan={step.plan_seconds:.1f}s belief={means}",
            flush=True,
        )

    frictions = tuple(args.frictions) if args.frictions else None
    log = run_episode(
        env,
        args.env,
        args.seed,
        config,
        num_particles=args.particles,
        max_pushes=args.max_pushes,
        frictions=frictions,
        on_step=report,
    )
    print(
        f"{log.outcome}: reward={log.total_reward:.1f} pushes={len(log.steps)} "
        f"plan={log.total_plan_seconds:.1f}s wall={time.perf_counter() - start:.1f}s "
        f"true={np.round(log.true_frictions, 3).tolist()}"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(asdict(log), indent=1))


if __name__ == "__main__":
    main()
