"""Driver: plan one seed, execute the accepted plan, dump everything from both paths.

SCRATCH ONLY.

--repeat-exec N   execute the SAME accepted plan N times from a fresh reset of a fresh
                  env, to answer "is execution even deterministic?" before anything else.
"""

import argparse
import json
import os
import time

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv  # noqa: F401

import instrument

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent  # noqa: E402
from kinder_bilevel_planning.env_models import create_bilevel_planning_models  # noqa: E402

CUBE = "cube_0"


def cube_xyz(state) -> list[float]:  # noqa: ANN001
    o = state.get_object_from_name(CUBE)
    return [float(state.get(o, f)) for f in ("x", "y", "z")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--samples-per-step", type=int, default=5)
    ap.add_argument("--planning-timeout", type=float, default=300.0)
    ap.add_argument("--max-skill-horizon", type=int, default=400)
    ap.add_argument("--max-abstract-plans", type=int, default=1)
    ap.add_argument("--repeat-exec", type=int, default=1)
    ap.add_argument("--settle-steps", type=int, default=0)
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--summary", type=str, required=True)
    ap.add_argument("--no-verbose-predicates", action="store_true")
    args = ap.parse_args()

    instrument.open_log(args.log)
    instrument.install(verbose_predicates=not args.no_verbose_predicates)
    instrument.emit("run_begin", args=vars(args))

    seed = args.seed
    summary: dict = {"seed": seed}
    t0 = time.time()

    instrument.set_phase("setup")
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode=None)
    obs, info = env.reset(seed=seed)
    summary["obs_len"] = int(np.asarray(obs).size)
    summary["obs_dtype"] = str(np.asarray(obs).dtype)

    env_models = create_bilevel_planning_models(
        "tidybot3d_tossing3D", env.observation_space, env.action_space, num_objects=1
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=args.max_abstract_plans,
        samples_per_step=args.samples_per_step,
        planning_timeout=args.planning_timeout,
        max_skill_horizon=args.max_skill_horizon,
    )

    instrument.set_phase("planning")
    try:
        agent.reset(obs, info)
    except AgentFailure as e:
        summary["outcome"] = "plan_not_found"
        summary["detail"] = str(e)
        _write(args.summary, summary)
        instrument.emit("run_end", summary=summary)
        instrument.close_log()
        return

    planned_states = list(agent._planned_states)  # noqa: SLF001
    planned_actions = [np.asarray(u, dtype=np.float64).copy() for u in agent._planned_actions]  # noqa: SLF001
    summary["n_planned_states"] = len(planned_states)
    summary["n_planned_actions"] = len(planned_actions)
    summary["sim_cube_traj"] = [cube_xyz(s) for s in planned_states]
    summary["sim_final_cube"] = cube_xyz(planned_states[-1])

    instrument.emit(
        "plan_accepted",
        n_states=len(planned_states),
        n_actions=len(planned_actions),
        actions=[a.tolist() for a in planned_actions],
        sim_cube_traj=summary["sim_cube_traj"],
    )

    # ---- execute the accepted action sequence, possibly several times
    runs = []
    for rep in range(args.repeat_exec):
        instrument.set_phase(f"execution{rep}")
        if rep == 0:
            e = env
            o = obs
        else:
            e = kinder.make("kinder/Tossing3D-o1-v0", render_mode=None)
            o, _ = e.reset(seed=seed)
        r: dict = {"rep": rep}
        traj = [cube_xyz(env_models.observation_to_state(o))]
        term = trunc = False
        for i, u in enumerate(planned_actions):
            o, reward, term_i, trunc_i, _info = e.step(u)
            traj.append(cube_xyz(env_models.observation_to_state(o)))
            term = term or bool(term_i)
            trunc = trunc or bool(trunc_i)
            if term_i or trunc_i:
                r["stopped_at"] = i
                break
        sim = e.unwrapped._object_centric_env  # noqa: SLF001
        r["terminated"] = term
        r["truncated"] = trunc
        r["n_steps_executed"] = len(traj) - 1
        r["exec_cube_traj"] = traj
        r["exec_final_cube"] = traj[-1]
        r["check_goals"] = bool(sim._check_goals())  # noqa: SLF001
        if args.settle_steps:
            for _ in range(args.settle_steps):
                o, _, _, _, _ = e.step(planned_actions[-1])
            r["settled_cube"] = cube_xyz(env_models.observation_to_state(o))
            r["check_goals_after_settle"] = bool(sim._check_goals())  # noqa: SLF001
        runs.append(r)
        instrument.emit("execution_done", **r)
        if rep > 0:
            e.close()
    summary["runs"] = runs

    # ---- determinism across executions
    if len(runs) > 1:
        base = np.asarray(runs[0]["exec_cube_traj"])
        maxdiffs = []
        for r in runs[1:]:
            other = np.asarray(r["exec_cube_traj"])
            n = min(len(base), len(other))
            maxdiffs.append(float(np.max(np.abs(base[:n] - other[:n]))))
        summary["exec_determinism_max_diff"] = maxdiffs
        summary["exec_scored_agreement"] = [r["check_goals"] for r in runs]

    # ---- sim-vs-exec divergence, step by step
    s = np.asarray(summary["sim_cube_traj"])
    x = np.asarray(runs[0]["exec_cube_traj"])
    n = min(len(s), len(x))
    per_step = np.max(np.abs(s[:n] - x[:n]), axis=1)
    summary["sim_vs_exec_per_step_maxabs"] = per_step.tolist()
    summary["sim_vs_exec_final_offset"] = (
        np.asarray(summary["sim_final_cube"]) - np.asarray(runs[0]["exec_final_cube"])
    ).tolist()
    nz = np.nonzero(per_step > 1e-9)[0]
    summary["first_divergence_step"] = int(nz[0]) if len(nz) else None
    summary["wall_s"] = time.time() - t0

    _write(args.summary, summary)
    instrument.emit("run_end", summary={k: v for k, v in summary.items() if k != "runs"})
    instrument.close_log()

    print(
        json.dumps(
            {
                "seed": seed,
                "scored": [r["check_goals"] for r in runs],
                "sim_final_cube": summary["sim_final_cube"],
                "exec_final_cube": runs[0]["exec_final_cube"],
                "final_offset": summary["sim_vs_exec_final_offset"],
                "first_divergence_step": summary["first_divergence_step"],
                "n_planned_actions": summary["n_planned_actions"],
                "n_steps_executed": runs[0]["n_steps_executed"],
                "exec_determinism_max_diff": summary.get("exec_determinism_max_diff"),
            }
        )
    )


def _write(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    main()
