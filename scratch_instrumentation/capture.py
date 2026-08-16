"""Capture frames + full state/action metadata from BOTH paths, for one seed.

SCRATCH ONLY -- video instrumentation for the refinement-vs-execution demo.

Path A ("refinement"): the model's own persistent sim, whose transition function is
`sim.set_state(x); sim.step(u)`. Every sampling attempt the refiner makes runs through
it, including the ones it rejects.

Path B ("execution"): the real `kinder.make` env, stepped with the actions the agent
pops from the accepted plan.

Every `ObjectCentricTidyBot3DEnv.step` on either env renders a frame to
<out>/frames/<label>_<phase>_<step>.png and appends one JSONL record carrying the action
vector, the cube pose, all 39 qpos (the 8 Robotiq finger joints among them), and whether
a `set_state` fired immediately before that step and what it changed.
"""

import argparse
import json
import os
import time
from pathlib import Path

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from PIL import Image

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent  # noqa: E402
from kinder_bilevel_planning.env_models import create_bilevel_planning_models  # noqa: E402

CUBE = "cube_0"
FINGER_QPOS = list(range(31, 39))
FINGER_NAMES = [
    "r_driver",
    "r_coupler",
    "r_spring",
    "r_follow",
    "l_driver",
    "l_coupler",
    "l_spring",
    "l_follow",
]

STATE = {
    "phase": "init",
    "labels": {},
    "made": 0,
    "step_index": {},
    "pending_set_state": {},
    "attempt": 0,
    "fh": None,
    "outdir": None,
    "seq": 0,
}


def emit(kind: str, **kw) -> None:
    rec = {"seq": STATE["seq"], "kind": kind, "phase": STATE["phase"], "t": time.time()}
    STATE["seq"] += 1
    rec.update(kw)
    STATE["fh"].write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return repr(o)


def mj_snapshot(env) -> dict:
    re = env._robot_env  # noqa: SLF001
    md = re.sim.data.mj_data
    return {
        "qpos": np.asarray(md.qpos).copy().tolist(),
        "qvel": np.asarray(md.qvel).copy().tolist(),
        "time": float(md.time),
        "ncon": int(md.ncon),
    }


def cube_xyz(state) -> list:
    o = state.get_object_from_name(CUBE)
    return [float(state.get(o, f)) for f in ("x", "y", "z")]


def install(render: bool) -> None:
    """Patch __init__ (inject render_mode + label), step (render + log), set_state."""
    _orig_init = ObjectCentricTidyBot3DEnv.__init__

    def init(self, *a, **kw):
        if render:
            kw.setdefault("render_mode", "rgb_array")
        _orig_init(self, *a, **kw)
        if render:
            # kinder.make passes scene_render_camera="task_view"; a bare sim gets None
            # and render() then falls back to the FIRST camera ("overview"). Without
            # this the two panels would be different viewpoints of the same state.
            assert "task_view" in self.camera_names, self.camera_names
            self.set_render_camera("task_view")
        label = f"sim{STATE['made']}"
        STATE["made"] += 1
        STATE["labels"][id(self)] = label
        emit("env_created", env=label)

    ObjectCentricTidyBot3DEnv.__init__ = init

    _orig_set_state = ObjectCentricTidyBot3DEnv.set_state

    def set_state(self, state):
        label = STATE["labels"].get(id(self), "unknown")
        before = np.asarray(self._robot_env.sim.data.mj_data.qpos).copy()  # noqa: SLF001
        out = _orig_set_state(self, state)
        after = np.asarray(self._robot_env.sim.data.mj_data.qpos).copy()  # noqa: SLF001
        req = cube_xyz(state)
        STATE["pending_set_state"][label] = {
            "requested_cube": req,
            "fingers_before": before[FINGER_QPOS].tolist(),
            "fingers_after": after[FINGER_QPOS].tolist(),
            "max_abs_qpos_change": float(np.max(np.abs(after - before))),
            "max_abs_finger_change": float(
                np.max(np.abs(after[FINGER_QPOS] - before[FINGER_QPOS]))
            ),
        }
        return out

    ObjectCentricTidyBot3DEnv.set_state = set_state

    _orig_step = ObjectCentricTidyBot3DEnv.step

    def step(self, action):
        label = STATE["labels"].get(id(self), "unknown")
        key = f"{STATE['phase']}:{label}"
        i = STATE["step_index"].get(key, 0)
        STATE["step_index"][key] = i + 1
        a = np.asarray(action, dtype=np.float64)
        pre = mj_snapshot(self)
        ss = STATE["pending_set_state"].pop(label, None)
        out = _orig_step(self, action)
        post = mj_snapshot(self)
        obs = out[0]
        frame_name = None
        if render:
            img = np.asarray(self.render())
            frame_name = f"{label}_{STATE['phase']}_{i:04d}.png"
            Image.fromarray(img).save(str(STATE["outdir"] / "frames" / frame_name))
        emit(
            "step",
            env=label,
            step=i,
            attempt=STATE["attempt"],
            frame=frame_name,
            action=a.tolist(),
            action_shape=list(a.shape),
            set_state=ss,
            pre_fingers=[pre["qpos"][j] for j in FINGER_QPOS],
            post_fingers=[post["qpos"][j] for j in FINGER_QPOS],
            pre_qpos=pre["qpos"],
            post_qpos=post["qpos"],
            post_qvel=post["qvel"],
            cube=cube_xyz(obs),
            reward=float(out[1]),
            terminated=bool(out[2]),
            truncated=bool(out[3]),
            ncon=post["ncon"],
            mj_time=post["time"],
        )
        return out

    ObjectCentricTidyBot3DEnv.step = step

    # ---- sampler: mark attempt boundaries and accept/reject
    from bilevel_planning.trajectory_samplers import (  # noqa: PLC0415
        parameterized_controller_sampler as pcs,
    )

    S = pcs.ParameterizedControllerTrajectorySampler
    _orig_call = S.__call__

    def sampler_call(self, x, s, a, ns, bpg, rng):
        STATE["attempt"] += 1
        n = STATE["attempt"]
        first_step = STATE["step_index"].get(f"{STATE['phase']}:sim1", 0)
        emit("attempt_begin", attempt=n, abstract_action=str(a), first_step=first_step)
        try:
            res = _orig_call(self, x, s, a, ns, bpg, rng)
        except BaseException as e:  # TrajectorySamplingFailure is not an Exception
            last_step = STATE["step_index"].get(f"{STATE['phase']}:sim1", 0)
            emit(
                "attempt_end",
                attempt=n,
                accepted=False,
                err=type(e).__name__,
                first_step=first_step,
                last_step=last_step,
            )
            raise
        last_step = STATE["step_index"].get(f"{STATE['phase']}:sim1", 0)
        emit(
            "attempt_end",
            attempt=n,
            accepted=True,
            first_step=first_step,
            last_step=last_step,
            n_actions=len(res[1]),
        )
        return res

    S.__call__ = sampler_call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--samples-per-step", type=int, default=5)
    ap.add_argument("--max-abstract-plans", type=int, default=1)
    ap.add_argument("--planning-timeout", type=float, default=300.0)
    ap.add_argument("--max-skill-horizon", type=int, default=400)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    (outdir / "frames").mkdir(parents=True, exist_ok=True)
    STATE["outdir"] = outdir
    STATE["fh"] = open(outdir / "events.jsonl", "w")  # noqa: SIM115
    install(render=not args.no_render)

    t0 = time.time()
    summary: dict = {"seed": args.seed}

    STATE["phase"] = "setup"
    env = kinder.make(
        "kinder/Tossing3D-o1-v0", render_mode=None if args.no_render else "rgb_array"
    )
    obs, info = env.reset(seed=args.seed)

    env_models = create_bilevel_planning_models(
        "tidybot3d_tossing3D", env.observation_space, env.action_space, num_objects=1
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=args.seed,
        max_abstract_plans=args.max_abstract_plans,
        samples_per_step=args.samples_per_step,
        planning_timeout=args.planning_timeout,
        max_skill_horizon=args.max_skill_horizon,
    )

    STATE["phase"] = "planning"
    try:
        agent.reset(obs, info)
    except AgentFailure as e:
        summary["outcome"] = "plan_not_found"
        summary["detail"] = str(e)
        _write(outdir / "summary.json", summary)
        STATE["fh"].close()
        print(json.dumps(summary))
        return

    planned_states = list(agent._planned_states)  # noqa: SLF001
    planned_actions = [np.asarray(u, dtype=np.float64).copy() for u in agent._planned_actions]  # noqa: SLF001
    summary["n_planned_actions"] = len(planned_actions)
    summary["sim_cube_traj"] = [cube_xyz(s) for s in planned_states]
    summary["sim_final_cube"] = summary["sim_cube_traj"][-1]
    emit("plan_accepted", n_actions=len(planned_actions))

    STATE["phase"] = "execution"
    exec_traj = [cube_xyz(env_models.observation_to_state(obs))]
    executed_actions = []
    terminated = truncated = False
    for _ in range(4000):
        try:
            action = agent.step()
        except AgentFailure:
            break
        executed_actions.append(np.asarray(action, dtype=np.float64).copy())
        obs, reward, term, trunc, info = env.step(action)
        agent.update(obs, reward, term or trunc, info)
        exec_traj.append(cube_xyz(env_models.observation_to_state(obs)))
        terminated = terminated or bool(term)
        truncated = truncated or bool(trunc)
        if term or trunc or len(agent._current_plan) == 0:  # noqa: SLF001
            break
    summary["exec_cube_traj"] = exec_traj
    summary["exec_final_cube"] = exec_traj[-1]
    summary["n_executed_actions"] = len(executed_actions)
    summary["terminated"] = terminated
    summary["truncated"] = truncated

    n = min(len(executed_actions), len(planned_actions))
    summary["max_action_diff"] = (
        max(
            float(np.max(np.abs(executed_actions[i] - planned_actions[i])))
            for i in range(n)
        )
        if n
        else None
    )
    summary["action_count_match"] = len(executed_actions) == len(planned_actions)

    sim = env.unwrapped._object_centric_env  # noqa: SLF001
    summary["check_goals_at_end"] = bool(sim._check_goals())  # noqa: SLF001

    s = np.asarray(summary["sim_cube_traj"])
    x = np.asarray(exec_traj)
    m = min(len(s), len(x))
    summary["sim_vs_exec_per_step_maxabs"] = np.max(np.abs(s[:m] - x[:m]), axis=1).tolist()
    summary["sim_vs_exec_final_offset"] = (
        np.asarray(summary["sim_final_cube"]) - np.asarray(summary["exec_final_cube"])
    ).tolist()
    summary["final_offset_norm"] = float(
        np.linalg.norm(np.asarray(summary["sim_vs_exec_final_offset"]))
    )
    summary["wall_s"] = time.time() - t0

    _write(outdir / "summary.json", summary)
    emit("run_end")
    STATE["fh"].close()
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "seed",
                    "n_planned_actions",
                    "n_executed_actions",
                    "max_action_diff",
                    "sim_final_cube",
                    "exec_final_cube",
                    "final_offset_norm",
                    "check_goals_at_end",
                    "wall_s",
                )
            }
        )
    )


def _write(path, obj) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(obj, default=_json_default) + "\n")


if __name__ == "__main__":
    main()
