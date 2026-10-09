# Pickup recovery stress coverage

`test_pick_recovery_coverage.py` extends the initial-placement tests with 109 cases:

- Eight recorded pre-reset landings, replayed in Tossing3D and FragileTossing3D (16).
- A 3 × 3 grid inside the robot-side bin, with two cube yaws and both environments (36).
- Floor landings outside five bin edges/corners, two yaws, both environments (20).
- Four rim placements, both environments (8).
- Five room-floor locations, two yaws, both environments (20).
- Two locations beside the barrier, two yaws, both environments (8).
- A regression for waiting until the measured fingers actually stop closing (1).

Each physical case requires a stable lifted cube, live MuJoCo gripper contact, the
native Holding predicate, and less than 2 cm of bin displacement. Initial placement
uses state restoration; execution uses only robot actions. The controller does not
reset the environment, move the cube directly, or disable physical collisions.
Rotated interior cubes are placed with at least 3 mm of initial wall clearance,
so the tests do not start with an overlapping cube and bin.

Run from `kinder-models` with the corresponding Kinder environment available:

```sh
pytest tests/dynamic3d/tossing/test_pick_recovery_coverage.py -vv
pytest tests/dynamic3d/tossing/test_pick_cube_controller_e2e.py -q
```

The recorded fixture contains movable-object, robot, and barrier state from the
fragile seed-0 POMDP run with human cost 100 and damage cost 10. Static collider IDs
are intentionally omitted: the environment reconstructs its own room colliders.
`practice_step` identifies the source landing immediately before a human reset.

## Current limitation

This is a target coverage suite, not a claim that every case passes. The expanded
controller recovers recorded near-bin failures, but several tight interior placements
still fail. Keep those failures visible while developing repositioning/recovery;
do not silently exclude them or turn planning feasibility into the success criterion.
The draft is not ready to merge until that coverage or its explicitly agreed contract
is resolved. No experiment result should be attributed to this controller until it
has been rerun with a pinned revision.
