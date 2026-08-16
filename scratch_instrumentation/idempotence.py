"""Is set_state(get_state()) a no-op? Every field that changes is hidden state lost.

SCRATCH ONLY.

This isolates set_state's fidelity from every rollout/divergence confound: run a rollout,
and at each step capture the object-centric state, immediately restore it, and diff the
raw MuJoCo buffers before vs after. Anything non-zero is state the object-centric
representation cannot carry -- which is exactly what refinement's transition function
silently drops on every single step.
"""

import argparse
import json
import os

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv  # noqa: F401

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from kinder_bilevel_planning.env_models import create_bilevel_planning_models  # noqa: E402

FIELDS = ("qpos", "qvel", "ctrl", "qacc_warmstart", "act", "qacc")


def snap(env) -> dict:  # noqa: ANN001
    md = env._robot_env.sim.data.mj_data  # noqa: SLF001
    out = {}
    for f in FIELDS:
        try:
            out[f] = np.array(getattr(md, f), dtype=float).copy()
        except Exception:  # noqa: BLE001, PERF203
            pass
    out["ncon"] = int(md.ncon)
    out["time"] = float(md.time)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--actions", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    actions = [np.asarray(a, dtype=np.float32) for a in json.load(open(args.actions))]

    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode=None)
    obs, _ = env.reset(seed=args.seed)
    models = create_bilevel_planning_models(
        "tidybot3d_tossing3D", env.observation_space, env.action_space, num_objects=1
    )
    # A sim we are allowed to set_state on, driven exactly like transition_fn drives it.
    x = models.observation_to_state(obs)

    # Reach into the model's own sim through the transition function's closure.
    sim = models.transition_fn.__globals__.get("sim")
    if sim is None:
        cells = models.transition_fn.__closure__ or ()
        for c in cells:
            if isinstance(c.cell_contents, ObjectCentricTidyBot3DEnv):
                sim = c.cell_contents
                break
    assert sim is not None, "could not reach the env model's sim"

    rows = []
    for i, u in enumerate(actions):
        # Advance the sim one step exactly as transition_fn would.
        sim.set_state(x.copy())
        obs_next, _, _, _, _ = sim.step(u)
        x = obs_next.copy()

        # Now the idempotence probe: capture, restore, diff.
        before = snap(sim)
        s = sim._get_current_state()  # noqa: SLF001
        sim.set_state(s.copy())
        after = snap(sim)

        row = {"step": i, "ncon_before": before["ncon"], "ncon_after": after["ncon"]}
        for f in FIELDS:
            if f in before and f in after and before[f].shape == after[f].shape:
                d = np.abs(before[f] - after[f])
                row[f"{f}_max"] = float(d.max()) if d.size else 0.0
                row[f"{f}_argmax"] = int(d.argmax()) if d.size else -1
        rows.append(row)

    out = {"label": args.label, "seed": args.seed, "rows": rows}
    with open(args.out, "w") as f:
        f.write(json.dumps(out) + "\n")

    print(f"=== set_state(get_state()) idempotence, label={args.label}")
    for f in FIELDS:
        key = f"{f}_max"
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        worst = max(vals)
        idx = [r["step"] for r in rows if r.get(key, 0) == worst]
        am = [r.get(f"{f}_argmax") for r in rows if r.get(key, 0) == worst]
        nnz = sum(1 for v in vals if v > 1e-9)
        print(
            f"  {f:16s} worst {worst:.6e} at step {idx[0]} index {am[0]} ; "
            f"steps with any change: {nnz}/{len(vals)}"
        )
    dn = sum(1 for r in rows if r["ncon_before"] != r["ncon_after"])
    print(f"  ncon changed on {dn}/{len(rows)} steps")


if __name__ == "__main__":
    main()
