# Long-range toss coverage demo

Run from an environment with this branch of `kinder_models` and the current
KINDER scene/static-collider implementation installed. This is a **simulation-only**
coverage experiment; effort 3 is not a hardware-safe setting.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m kinder_models.dynamic3d.tossing.coverage \
  --output /tmp/toss-coverage.jsonl
```

The default demo executes 18 independent trials: seeds 10125/10126/10127,
standoff 2.5 m, speeds 358/360/380 deg/s, release 500 ms, and effort ceilings 1/3.
Every speed is tested on every seed under both ceilings. These seeds and parameter
values were used in calibration: **this is not an unbiased evaluation estimate**.

Observed smoke-grid result (2026-09-18): original effort 0/9 successes;
extended effort 5/9 successes. All other trials were completed tosses that missed.
All 18 pickups passed the simulator-contact Holding check. This is deliberately
broader than choosing one successful parameter setting per seed.

For a broader, previously uncalibrated grid:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m kinder_models.dynamic3d.tossing.coverage \
  --output /tmp/toss-held-out.jsonl \
  --seeds 10200 10201 10202 \
  --distances 2.0 2.3 2.5 2.6 \
  --speeds 280 320 360 400 \
  --release-ms 450 500 550 600 \
  --efforts 1 3
```

That command runs **384 trials**, serially; it is not part of the quick demo and
has not been run as validation. Change the output path for each run: existing
files are never overwritten. Each completed trial is flushed to JSONL so partial
results survive interruption.

## Reading results

## Per-configuration solution checks

Add `--find-solutions` to stop each seed/effort search at its first successful
physical trajectory. The summary records that witness and its parameters for
each requested seed. Exit status 2 means at least one requested scene has no
witness in the tested grid; it does not prove that scene is impossible. Witness
parameters must stay inside the composed controller's supported bounds.

For practice placements, pass `--bin-reset-region region.json`, containing the
application's reset-region definition (`target`, `ranges`, `yaw_ranges`). The
runner invokes the environment's real feasible-placement reset method for the
cube and bin, rather than teleporting them to unchecked poses. Run each reset
destination separately; do not pool them with initial evaluation scenes.

Executed on 2026-09-18, using seeds 10200 through 10209 and effort ceiling 3:

| Placement family | Scenes with a physical solution |
| --- | ---: |
| Initial opposite-side evaluation | 10/10 |
| Robot-side practice reset | 10/10 |
| Opposite-side practice reset | 10/10 |

Evaluation searched distance 2.5, speeds 360/380/358/400/340 deg/s and release
500/480/520 ms. Reset searches used distances 1.35/1.4/1.3/1.45, speeds
140/130/120 deg/s and release 780/760/800/720 ms. All used rotation 0.
Reset regions came from the application's `BIN_RESET_REGION_BY_SIDE` definition.

These are existence checks, **not learned-policy success rates or a proof over
the continuous configuration space**. Resets here start from the initial robot
pose; arbitrary robot poses after practice are not certified by this corpus.

## Trial outcomes

- `success`: the actual simulator goal reports the cube in the bin.
- `toss_miss`: the toss controller completed but the goal is false.
- `pickup_failed`: pickup completed without the contact-based Holding predicate.
- `pickup_timeout` / `toss_timeout`: controller did not complete within the step cap.
- `pickup_planning_failure` / `toss_planning_failure`: the planner raised a sampling
  failure; the error text distinguishes base planning from arm planning. **Failure
  to find a plan is not itself proof that the target is unreachable.**
- `error`: an unexpected exception; retained in the log and causes a nonzero CLI exit.

Records include configuration, physical step counts, elapsed time, initial bin
position and completed-toss cube position. Summaries count every trial, including
failures, in the denominator. Trials start from a fresh seeded environment and
execute a real pickup; there is no restored approximate grasp state. No bin,
barrier, torque limit, or physics manipulation is used.

## Regression tests

```bash
pytest tests/dynamic3d/tossing/test_long_range_toss_e2e.py \
       tests/dynamic3d/tossing/test_toss_coverage_cli.py \
       tests/dynamic3d/tossing/test_extended_toss_effort.py \
       tests/dynamic3d/tossing/test_toss_target_coverage.py
```

The paired end-to-end tests call the same runner as the demo. Additional tests
exercise failed navigation, pickup timeout, grid enumeration, JSONL output,
summary accounting, refusing overwrites and nonzero exit on unexpected errors.
