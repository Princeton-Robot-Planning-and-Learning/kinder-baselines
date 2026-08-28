# Tossing3D bin retrieval

[Watch the continuous 33.4-second recording](retrieval.mp4) ·
[Measured controller results](results.json)

The robot picks a cube from the floor, throws it into an open bin, and retrieves
it with `pick_cube_from_bin`. There is no reset between these actions. The test
checks `Holding` after each pickup and `MovableInGoalRegion` after the throw;
the final cube height is 0.58768 m.

This is a scripted controller regression for one seed and landing pose, not an
autonomous planner or a grasp-success-rate benchmark.

## Reproduce

From `kinder-models/`, with the package and KINDER installed:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl DISPLAY=:0 pytest -q tests/dynamic3d/tossing/test_tossing_parameterized_skills.py::test_pick_cube_from_bin_after_toss --make-videos
```

The existing test video recorder writes `unit_test_videos/tossing-bin-retrieval-episode-0.mp4`
and the test writes its measured step counts and heights alongside it. The test
constructs a temporary scene JSON with the bin on the robot's side and seed 125;
it does not modify KINDER's default task.

Recorded from clean kinder-baselines revision
`4b710549696b21209125ccf479242062792c2638`, rebased on
`4fa4b28cca794b1f09eb06e9552fd86013c08ef6`, with KINDER revision
`86c55a4e9a19f4f37cd7103d425ee88465042173`.
The H.264 recording contains 668 frames at 20 fps, 640 × 480 pixels.

## Controller contract

Provide a `PyBulletSim` containing the named bin, added with `add_bin` using
the live pose, dimensions, and wall thickness. Ground `pick_cube_from_bin` with
`(robot, cube, bin)`; there are no continuous parameters. Missing bin geometry
is rejected before planning.

The controller reuses the floor pick's phases, tries four grasp orientations,
uses 3 mm base-position tolerance, and lifts above the rim before retracting.
The bin remains a collision body throughout; held-object motion permits at most
1 mm of contact penetration. The stock floor controller remains unchanged.
