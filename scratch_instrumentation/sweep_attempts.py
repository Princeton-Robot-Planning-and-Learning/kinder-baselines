"""Decisive test: does a seed score iff the toss was accepted on sampling attempt #1?

SCRATCH ONLY.

Mechanism under test: `set_state` does not restore the gripper's 8 finger joints, so the
refiner's 2nd and later attempts at a skill start from the grasp the PREVIOUS rejected
attempt left behind (typically fully closed, because the cube fell out). The cube is then
teleported into a closed gripper, clamped by contact, and the toss succeeds in simulation
for a reason execution can never reproduce.

Prediction: accepted-on-attempt-1 => scores; accepted-on-attempt->=2 => planned_not_scored.
"""

import argparse
import json
import os
import sys
import time
import traceback

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv  # noqa: F401

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from bilevel_planning.trajectory_samplers import (  # noqa: E402
    parameterized_controller_sampler as pcs,
)
from kinder_bilevel_planning.agent import AgentFailure, BilevelPlanningAgent  # noqa: E402
from kinder_bilevel_planning.env_models import create_bilevel_planning_models  # noqa: E402

TRACE: list[dict] = []


def patch() -> None:
    S = pcs.ParameterizedControllerTrajectorySampler
    orig = S.__call__

    def call(self, x, s, a, ns, bpg, rng):  # noqa: ANN001, ANN202
        name = str(a).split("\n")[0].replace("(:action ", "").strip()
        # gripper joints at the moment this attempt starts
        grip = None
        try:
            env = self._transition_function.__globals__["sim"]  # noqa: SLF001
        except Exception:  # noqa: BLE001
            env = None
        rec = {"op": name, "grip_at_start": grip}
        # NB: TrajectorySamplingFailure is NOT an Exception subclass, so this must be
        # BaseException or every rejection is silently missed.
        try:
            out = orig(self, x, s, a, ns, bpg, rng)
            rec["accepted"] = True
            TRACE.append(rec)
            return out
        except BaseException:
            rec["accepted"] = False
            TRACE.append(rec)
            raise

    S.__call__ = call  # type: ignore[assignment]


def run_one(seed: int, samples_per_step: int, horizon: int) -> dict:
    TRACE.clear()
    t0 = time.time()
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode=None)
    r: dict = {"seed": seed}
    try:
        obs, info = env.reset(seed=seed)
        env_models = create_bilevel_planning_models(
            "tidybot3d_tossing3D", env.observation_space, env.action_space, num_objects=1
        )
        agent = BilevelPlanningAgent(
            env_models,
            seed=seed,
            max_abstract_plans=1,
            samples_per_step=samples_per_step,
            planning_timeout=300.0,
            max_skill_horizon=horizon,
        )
        try:
            agent.reset(obs, info)
        except AgentFailure as e:
            r["outcome"] = "plan_not_found"
            r["detail"] = str(e)
            r["trace"] = list(TRACE)
            return r
        # attempt bookkeeping, per operator
        per_op: dict[str, list[bool]] = {}
        for t in TRACE:
            per_op.setdefault(t["op"], []).append(t["accepted"])
        r["attempts_per_op"] = {k: len(v) for k, v in per_op.items()}
        r["accepted_attempt_per_op"] = {
            k: (v.index(True) + 1 if True in v else None) for k, v in per_op.items()
        }
        r["n_sampler_calls"] = len(TRACE)
        for _ in range(4000):
            action = agent.step()
            obs, reward, term, trunc, info = env.step(action)
            agent.update(obs, reward, term or trunc, info)
            if term or trunc or len(agent._current_plan) == 0:  # noqa: SLF001
                break
        sim = env.unwrapped._object_centric_env  # noqa: SLF001
        st = sim._get_current_state()  # noqa: SLF001
        cube = st.get_object_from_name("cube_0")
        r["final_cube"] = [float(st.get(cube, f)) for f in ("x", "y", "z")]
        r["outcome"] = (
            "planned_and_scored" if sim._check_goals() else "planned_not_scored"  # noqa: SLF001
        )
        return r
    except Exception as e:  # noqa: BLE001
        r["outcome"] = "exception"
        r["detail"] = f"{type(e).__name__}: {e}"
        r["traceback"] = traceback.format_exc()
        return r
    finally:
        r["wall_s"] = time.time() - t0
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, required=True)
    ap.add_argument("--samples-per-step", type=int, default=5)
    ap.add_argument("--max-skill-horizon", type=int, default=400)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    patch()
    lo, hi = args.seeds.split("-")
    with open(args.out, "w") as f:
        for seed in range(int(lo), int(hi) + 1):
            r = run_one(seed, args.samples_per_step, args.max_skill_horizon)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(
                f"seed={seed} {r['outcome']} attempts={r.get('attempts_per_op')} "
                f"accepted_on={r.get('accepted_attempt_per_op')}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
