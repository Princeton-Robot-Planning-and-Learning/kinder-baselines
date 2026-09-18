"""Reproducible live pickup/toss coverage: python -m ...tossing.coverage --help.

Coverage is over an explicit parameter grid, not learned-policy performance.
Higher effort is simulation-only. Every trial starts with a fresh physical pickup.
"""

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import kinder
import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from pybullet_helpers.geometry import Pose

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import Tossing3DStateAbstractor
from kinder_models.dynamic3d.utils import PyBulletSim


def run_trial(
    *,
    seed: int,
    distance: float,
    speed: float,
    release_ms: float,
    max_effort: float,
    rotation: float = 0.0,
    step_limit: int = 400,
    bin_reset_region: dict | None = None,
    cube_reset_region: dict | None = None,
) -> dict:
    """Return physical outcomes and failures without dropping failed trials."""
    if step_limit <= 0 or not np.isfinite(max_effort) or max_effort <= 0:
        raise ValueError("step_limit and max_effort must be positive and finite")
    if not all(np.isfinite(v) for v in (distance, speed, release_ms, rotation)):
        raise ValueError("throw parameters must be finite")
    if distance <= 0 or speed <= 0 or release_ms < 0:
        raise ValueError("distance/speed must be positive and release_ms nonnegative")
    result = {
        "seed": seed,
        "distance": distance,
        "speed_deg_s": speed,
        "release_ms": release_ms,
        "max_effort": max_effort,
        "rotation": rotation,
        "status": "error",
        "scene_bg": True,
        "pickup_steps": 0,
        "toss_steps": 0,
        "bin_reset_region": bin_reset_region,
        "cube_reset_region": cube_reset_region,
    }
    started = time.monotonic()
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o1-v0", allow_state_access=True, scene_bg=True)
    sims = []
    phase = "pickup"
    stage = "reset"
    try:
        scene = env.unwrapped._object_centric_env  # pylint: disable=protected-access
        abstractor = Tossing3DStateAbstractor(scene)
        obs, _ = env.reset(seed=seed)
        state = env.observation_space.devectorize(obs)
        if bin_reset_region is not None or cube_reset_region is not None:
            # Use the production reset sampler, including room/furniture clearance.
            regions = {}
            placements = {"cube_0": "blocks_init_region"}
            if bin_reset_region is not None:
                regions["__coverage_bin_region"] = bin_reset_region
                placements["bin_0"] = "__coverage_bin_region"
            if cube_reset_region is not None:
                regions["__coverage_cube_region"] = cube_reset_region
                placements["cube_0"] = "__coverage_cube_region"
            scene.reset_ground_objects_to_regions(
                placements,
                region_configs=regions,
            )
            state = scene._get_current_state()  # pylint: disable=protected-access
        robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
        cube = state.get_object_from_name("cube_0")
        barrier = state.get_object_from_name("cuboid_barrier")
        bin_object = state.get_object_from_name("bin_0")
        result["initial_bin_xyz"] = [float(state.get(bin_object, a)) for a in "xyz"]
        result["initial_cube_xyz"] = [float(state.get(cube, a)) for a in "xyz"]
        for phase, key, params in [
            ("pickup", "pick_cube", None),
            (
                "toss",
                "move_to_toss_location_and_toss",
                np.array([distance, rotation, np.deg2rad(speed), release_ms]),
            ),
        ]:
            stage = "reset"
            sim = PyBulletSim(state)
            sims.append(sim)
            if phase == "pickup":
                geometry = scene.get_object("bin_0")
                sim.add_bin(
                    name="bin_0",
                    pose=Pose(
                        tuple(state.get(bin_object, a) for a in "xyz"),
                        tuple(
                            state.get(bin_object, a) for a in ("qx", "qy", "qz", "qw")
                        ),
                    ),
                    length=geometry.length,
                    width=geometry.width,
                    height=geometry.height,
                    wall_thickness=geometry.wall_thickness,
                )
            controller = create_lifted_controllers(
                env.action_space, init_constant_state=state, pybullet_sim=sim
            )[key].ground((robot, cube, barrier))
            if phase == "toss":
                # Instance override: paired trials never mutate global defaults.
                controller.MAX_SIMULATION_EFFORT = max_effort
            controller.reset(state, params)
            stage = "execution"
            for step in range(step_limit):
                obs, _, _, _, _ = env.step(controller.step())
                state = env.observation_space.devectorize(obs)
                result[f"{phase}_steps"] = step + 1
                controller.observe(state)
                if controller.terminated():
                    break
            else:
                result["status"] = f"{phase}_timeout"
                return result
            if phase == "pickup":
                held = abstractor.holding_evidence(state, robot, cube).contact_holding
                result["pickup_holding"] = bool(held)
                if not held:
                    result["status"] = "pickup_failed"
                    return result
        result["final_cube_xyz"] = [float(state.get(cube, a)) for a in "xyz"]
        result["status"] = (
            "success"
            if scene._check_goals()  # pylint: disable=protected-access
            else "toss_miss"
        )
    except TrajectorySamplingFailure as exc:
        result.update(status=f"{phase}_planning_failure", error=str(exc))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        result.update(
            phase=phase, stage=stage, elapsed_seconds=time.monotonic() - started
        )
        for sim in sims:
            sim.close()
        env.close()
    return result


def summarize(rows: list[dict]) -> dict:
    """Include planning failures, timeouts and errors in the denominator."""
    counts = Counter(row["status"] for row in rows)
    return {
        "trials": len(rows),
        "outcomes": dict(sorted(counts.items())),
        "success_fraction": counts["success"] / len(rows) if rows else None,
    }


def solution_coverage(
    rows: list[dict], *, seeds: list[int], efforts: list[float]
) -> list[dict]:
    """A successful witness for every requested scene, never an average-only check.

    Uncovered means no witness in this search, not proof of impossibility.
    """
    results = []
    for seed, effort in itertools.product(seeds, efforts):
        attempts = [r for r in rows if r["seed"] == seed and r["max_effort"] == effort]
        witness = next((r for r in attempts if r["status"] == "success"), None)
        results.append(
            {
                "seed": seed,
                "max_effort": effort,
                "covered": witness is not None,
                "attempts": len(attempts),
                "witness": witness,
            }
        )
    return results


def main() -> None:
    """Run the selected Cartesian grid and flush each trial to a new JSONL file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[10125, 10126, 10127])
    parser.add_argument("--distances", type=float, nargs="+", default=[2.5])
    parser.add_argument("--speeds", type=float, nargs="+", default=[358, 360, 380])
    parser.add_argument("--release-ms", type=float, nargs="+", default=[500])
    parser.add_argument("--efforts", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument("--rotation", type=float, default=0.0)
    parser.add_argument("--step-limit", type=int, default=400)
    parser.add_argument(
        "--bin-reset-region",
        type=Path,
        help="JSON region exported from the application's reset destination definition.",
    )
    parser.add_argument(
        "--find-solutions",
        action="store_true",
        help="Stop each seed/effort search at its first success; exit 2 if any has no witness.",
    )
    args = parser.parse_args()
    if args.find_solutions:
        controller = MoveToTossLocationAndTossController
        domains = (
            (args.distances, controller.TARGET_DISTANCE_BOUNDS),
            (args.speeds, np.rad2deg(controller.SPEED_BOUNDS)),
            (args.release_ms, controller.RELEASE_MS_BOUNDS),
            ([args.rotation], controller.TARGET_ROTATION_BOUNDS),
            (args.efforts, (0.0, controller.MAX_SIMULATION_EFFORT)),
        )
        if any(
            not np.isfinite(v) or not low <= v <= high
            for values, (low, high) in domains
            for v in values
        ):
            parser.error(
                "solution witnesses must use the controller's supported parameter bounds"
            )
    reset_region = (
        json.loads(args.bin_reset_region.read_text())
        if args.bin_reset_region is not None
        else None
    )
    rows = []
    solved = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "config",
                    "resolved_bin_reset_region": reset_region,
                    **{
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                }
            )
            + "\n"
        )
        for seed, distance, speed, release_ms, effort in itertools.product(
            args.seeds, args.distances, args.speeds, args.release_ms, args.efforts
        ):
            if args.find_solutions and (seed, effort) in solved:
                continue
            row = run_trial(
                seed=seed,
                distance=distance,
                speed=speed,
                release_ms=release_ms,
                max_effort=effort,
                rotation=args.rotation,
                step_limit=args.step_limit,
                bin_reset_region=reset_region,
            )
            rows.append(row)
            if row["status"] == "success":
                solved.add((seed, effort))
            stream.write(json.dumps({"kind": "trial", **row}) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)
        summary = {
            str(e): summarize([r for r in rows if r["max_effort"] == e])
            for e in args.efforts
        }
        coverage = solution_coverage(rows, seeds=args.seeds, efforts=args.efforts)
        stream.write(
            json.dumps(
                {"kind": "summary", "by_effort": summary, "solution_coverage": coverage}
            )
            + "\n"
        )
        print(json.dumps(summary), flush=True)
    if any(r["status"] == "error" for r in rows):
        raise SystemExit(1)
    if args.find_solutions and not all(c["covered"] for c in coverage):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
