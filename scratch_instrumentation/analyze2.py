"""Align the accepted refinement rollout with the executed rollout at the MuJoCo layer.

SCRATCH ONLY.

Planning stream (env sim1) is [pick 0..55][rejected toss 56..104][accepted toss 105..159].
Execution stream (env sim0) is [pick 0..55][accepted toss 56..110].
So planning step 105+k and execution step 56+k are the SAME action.
"""

import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]
plan_off = int(sys.argv[2]) if len(sys.argv) > 2 else 105
exec_off = int(sys.argv[3]) if len(sys.argv) > 3 else 56

pre = defaultdict(dict)
post = defaultdict(dict)
sets = defaultdict(dict)
sidx = defaultdict(int)

for line in open(path):
    r = json.loads(line)
    k = r["kind"]
    if k == "step_pre":
        pre[(r["phase"], r["env"])][r["step"]] = r
    elif k == "step_post":
        post[(r["phase"], r["env"])][r["step"]] = r
    elif k == "set_state":
        key = (r["phase"], r["env"])
        sets[key][sidx[key]] = r
        sidx[key] += 1

P = ("planning", "sim1")
E = ("execution0", "sim0")


def g(rec, *ks):
    o = rec
    for k in ks:
        o = o[k]
    return o


n = min(len(pre[P]) - plan_off, len(pre[E]) - exec_off)
print(f"comparing {n} aligned steps: planning[{plan_off}+k] vs execution[{exec_off}+k]\n")

hdr = (
    f"{'k':>4} {'act_d':>9} "
    f"{'cube_sim':>26} {'cube_exe':>26} {'cubeD':>9} "
    f"{'qpos_d':>9} {'qvel_d':>9} {'ctrl_d':>9} {'grip_sim':>9} {'grip_exe':>9} "
    f"{'ncon_s':>6} {'ncon_e':>6}"
)
print(hdr)
for k in range(n):
    ps, es = pre[P][plan_off + k], pre[E][exec_off + k]
    pp, ep = post[P][plan_off + k], post[E][exec_off + k]
    a1 = np.array(ps["action"], dtype=float)
    a2 = np.array(es["action"], dtype=float)
    ad = np.max(np.abs(a1 - a2))
    c1 = pp["state"]["cube_0"]
    c2 = ep["state"]["cube_0"]
    cd = max(abs(c1[f] - c2[f]) for f in ("x", "y", "z"))
    q1 = np.array(ps["mj"]["qpos"], dtype=float)
    q2 = np.array(es["mj"]["qpos"], dtype=float)
    v1 = np.array(ps["mj"]["qvel"], dtype=float)
    v2 = np.array(es["mj"]["qvel"], dtype=float)
    ct1 = np.array(ps["mj"]["ctrl"], dtype=float)
    ct2 = np.array(es["mj"]["ctrl"], dtype=float)
    g1 = np.array(ps["mj"].get("qpos_gripper", []), dtype=float)
    g2 = np.array(es["mj"].get("qpos_gripper", []), dtype=float)
    if k < 30 or k % 5 == 0 or k > n - 5:
        print(
            f"{k:4d} {ad:9.2e} "
            f"({c1['x']:7.4f},{c1['y']:7.4f},{c1['z']:7.4f}) "
            f"({c2['x']:7.4f},{c2['y']:7.4f},{c2['z']:7.4f}) {cd:9.2e} "
            f"{np.max(np.abs(q1-q2)):9.2e} {np.max(np.abs(v1-v2)):9.2e} "
            f"{np.max(np.abs(ct1-ct2)):9.2e} "
            f"{(g1[0] if g1.size else float('nan')):9.4f} "
            f"{(g2[0] if g2.size else float('nan')):9.4f} "
            f"{ps['mj']['ncon']:6d} {es['mj']['ncon']:6d}"
        )

# Which qpos indices diverge first, and by how much?
print("\n=== first step where max|qpos diff| > 1e-6, per-index breakdown")
for k in range(n):
    ps, es = pre[P][plan_off + k], pre[E][exec_off + k]
    q1 = np.array(ps["mj"]["qpos"], dtype=float)
    q2 = np.array(es["mj"]["qpos"], dtype=float)
    d = np.abs(q1 - q2)
    if d.max() > 1e-6:
        print(f"k={k} max={d.max():.3e}")
        for i in np.argsort(-d)[:12]:
            print(f"   qpos[{i}] sim={q1[i]:+.6f} exec={q2[i]:+.6f} d={d[i]:.3e}")
        break

print("\n=== set_state efficacy in planning (what it failed to carry)")
ss = sets[P]
for k in range(min(6, n)):
    r = ss.get(plan_off + k)
    if r is None:
        continue
    print(f"k={k} max_abs_change={r['max_abs_change']}")
