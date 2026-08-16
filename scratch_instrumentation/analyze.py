"""Pull aligned per-step streams out of the instrumentation log. SCRATCH ONLY."""

import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]

steps = defaultdict(list)  # (phase, env) -> list of dicts
setstates = defaultdict(list)
other = []
for line in open(path):
    r = json.loads(line)
    k = r["kind"]
    if k == "step_post":
        steps[(r["phase"], r["env"])].append(r)
    elif k == "step_pre":
        steps[(r["phase"], r["env"], "pre")].append(r)
    elif k == "set_state":
        setstates[(r["phase"], r["env"])].append(r)
    elif k in (
        "sampler_begin",
        "sampler_params",
        "sampler_final",
        "sampler_accept",
        "sampler_reject",
        "refine_enter",
        "refine_exit",
        "controller_terminated",
        "transition_failure",
        "env_created",
        "plan_accepted",
    ):
        other.append(r)

print("=== step streams")
for k, v in sorted(steps.items(), key=lambda kv: str(kv[0])):
    print(k, len(v))
print("=== set_state streams")
for k, v in sorted(setstates.items(), key=lambda kv: str(kv[0])):
    print(k, len(v))

print("\n=== events")
for r in other:
    k = r["kind"]
    if k == "sampler_params":
        print(f"[{r['seq']}] {k} call={r['call']} params={r['params']}")
    elif k == "sampler_final":
        print(
            f"[{r['seq']}] {k} call={r['call']} n_steps={r['n_steps']} equal={r['equal']}"
        )
        print(f"      only_in_final={r['only_in_final']}")
        print(f"      only_in_target={r['only_in_target']}")
        cube = r["final_state"].get("cube_0", {})
        print(
            f"      final cube xyz=({cube.get('x'):.4f},{cube.get('y'):.4f},{cube.get('z'):.4f})"
        )
    elif k == "sampler_begin":
        print(f"[{r['seq']}] {k} call={r['call']} a={r['abstract_action']}")
    elif k in ("refine_enter", "refine_exit"):
        print(f"[{r['seq']}] {k} index={r['index']} " + str(r.get("success", "")))
    elif k == "controller_terminated":
        print(f"[{r['seq']}] {k} call={r['call']} step={r['step']}")
    elif k == "env_created":
        print(f"[{r['seq']}] {k} env={r['env']}")
    elif k == "plan_accepted":
        print(f"[{r['seq']}] plan_accepted n_actions={r['n_actions']}")
