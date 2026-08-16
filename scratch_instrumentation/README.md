# Scratch instrumentation: Tossing3D refinement-vs-execution gap

**DO NOT MERGE.** This directory is parked diagnostic tooling, not shipping code. It
carries logging that must never appear in a real fix.

## What it found

`planned_not_scored` on 19/40 seeds was caused by `ObjectCentricRobotEnv.set_state`
not restoring the gripper's finger joints. The Robotiq 2F-85 has **eight** joints
(`{right,left}_{driver,coupler,spring_link,follower}`, qpos 31..38); only the two
driver joints are exposed through `_robot_env.qpos["gripper"]`, and stock `set_state`
restored **none** of them.

Bilevel refinement's transition function is `sim.set_state(x); sim.step(u)`, so on every
backtrack the sim keeps the finger pose the **previous rejected attempt** ended in —
typically fully closed, because the cube had just fallen out. The cube is then teleported
into a closed gripper, clamped by contact penetration, and the toss "succeeds" for a
reason execution can never reproduce. 33/34 refined seeds accepted the toss on attempt
>= 2, so nearly every accepted plan was validated under this artefact.

Measured on `Tossing3D-o1`, seeds 100-139, `samples_per_step=5`, `max_abstract_plans=1`:

| gripper joints restored by `set_state` | scored | planned_not_scored | plan_not_found | honest |
| --- | --- | --- | --- | --- |
| none (stock)  | 15/40 | 19/40 | 6/40  | 15/34 |
| 2 driver only | 15/40 | 20/40 | 5/40  | 15/35 |
| all 8         | 15/40 | **0/40** | 25/40 | **15/15** |

"honest" = of the plans refinement claimed to have solved, how many actually scored.

## Files

- `instrument.py` — monkeypatches everything (env step/set_state/reset/`_check_goals`,
  `check_in_region`, the abstractor and every predicate, the trajectory sampler, the
  backtracking refiner) and emits one flat JSONL record per event.
- `probe.py` — plans one seed, replays the accepted action sequence, dumps both the
  refinement-simulated and the executed rollout. `--repeat-exec N` re-executes to test
  determinism (measured: bit-identical, `0.0`).
- `analyze.py` / `analyze2.py` — pull aligned per-step streams out of the log and diff
  the two paths at the MuJoCo layer.
- `minrepro.py` — no planner: replays one fixed action sequence continuously vs
  reset-per-step, which is exactly refinement's transition function.
- `idempotence.py` — is `set_state(get_state())` a no-op? Stock: fails on 110/111 steps.
  With all 8 joints restored: bit-idempotent on qpos/qvel/ctrl/qacc_warmstart.
- `sweep_attempts.py` / `tally.py` — 40-seed sweep recording which sampling attempt the
  toss was accepted on, cross-tabulated against outcome. NB `TrajectorySamplingFailure`
  is not an `Exception` subclass, so rejections must be caught as `BaseException`.
- `joints.py` — dumps the MuJoCo joint/qpos/qvel address map.
- `envx.sh` — selects which kindergarden tree to run against.

## Companion branch

`kindergarden` `scratch/instrument-tossing3d-gripper-state` carries the all-8-joint
`set_state` patch these numbers were measured with.
