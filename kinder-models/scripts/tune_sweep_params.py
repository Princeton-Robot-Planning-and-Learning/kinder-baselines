"""Sweep parameter search for SweepIntoDrawer3D.

Tries a grid of WIPER_SWEEP_TRANSFORM / WIPER_SWEEP_TRANSFORM_END
x-offsets, runs the full open_drawer → pick_wiper → sweep sequence for
each combination, and reports how many cubes end up below the table surface.

The two constants control the start and end end-effector poses of the sweep
motion, expressed relative to the target cube:
  WIPER_SWEEP_TRANSFORM     -- where the wiper approaches the cube from
  WIPER_SWEEP_TRANSFORM_END -- where the wiper finishes the sweeping stroke

Usage:
    python scripts/tune_sweep_params.py
"""

import itertools
from unittest.mock import patch

import kinder
import numpy as np
from kinder.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoTidyBotRobotObjectType,
)
from pybullet_helpers.geometry import Pose
from relational_structs.spaces import ObjectCentricBoxSpace

import kinder_models.dynamic3d.ground.parameterized_skills as ps_module
from kinder_models.dynamic3d.ground.parameterized_skills import (
    PyBulletSim,
    create_lifted_controllers,
)

kinder.register_all_environments()

# ── Success threshold ────────────────────────────────────────────────────────
# Cubes whose z drops below this value are considered off the table / in the
# drawer. Print cube z values on a first dry run to calibrate this number.
TABLE_Z_THRESHOLD = 0.65

# ── Parameter grid ───────────────────────────────────────────────────────────
# x-offset for the sweep START pose (negative = behind the cube).
# Original value: -0.05
START_X_OFFSETS = [-0.08, -0.05, -0.02]

# x-offset for the sweep END pose (positive = past the cube / into drawer).
# Original value: 0.18
END_X_OFFSETS = [0.14, 0.18, 0.22, 0.26]


def _get_robot(state):
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    return list(robots)[0]


def _run_controller(controller, env, state, max_steps: int, label: str):
    """Step a controller until termination or max_steps.

    Returns:
        (final_state, success) where success is True iff the controller
        terminated within max_steps.
    """
    for _ in range(max_steps):
        action = controller.step()
        obs, _, terminated_env, truncated_env, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            return state, True
        if terminated_env or truncated_env:
            break
    print(f"  WARNING: {label} did not terminate within {max_steps} steps")
    return state, False


def run_trial(start_x: float, end_x: float, seed: int = 123) -> dict:
    """Run the full open_drawer → pick_wiper → sweep sequence.

    Patches WIPER_SWEEP_TRANSFORM and WIPER_SWEEP_TRANSFORM_END with the
    given x-offsets (all other components kept at their original values),
    then executes the three controllers in sequence and checks how many
    cube z-positions have dropped below TABLE_Z_THRESHOLD.

    Args:
        start_x: x-offset for WIPER_SWEEP_TRANSFORM (sweep approach pose).
        end_x: x-offset for WIPER_SWEEP_TRANSFORM_END (sweep finish pose).
        seed: Environment reset seed.

    Returns:
        Dict with keys:
            success         - True if all three controllers terminated.
            cubes_off_table - Number of cubes with z < TABLE_Z_THRESHOLD.
            cube_z_values   - List of final z values for all cubes.
    """
    sweep_start = Pose.from_rpy((start_x, 0, 0.04), (-np.pi, 0, -np.pi / 2))
    sweep_end   = Pose.from_rpy((end_x,   0, 0.02), (-np.pi, 0, -np.pi / 2))

    with (
        patch.object(ps_module, "WIPER_SWEEP_TRANSFORM",     sweep_start),
        patch.object(ps_module, "WIPER_SWEEP_TRANSFORM_END", sweep_end),
    ):
        num_cubes = 5
        env = kinder.make(
            f"kinder/SweepIntoDrawer3D-o{num_cubes}-v0", render_mode="rgb_array"
        )
        obs, _ = env.reset(seed=seed)
        for _ in range(5):
            obs, _, _, _, _ = env.step(np.zeros(11))
        assert isinstance(env.observation_space, ObjectCentricBoxSpace)
        state = env.observation_space.devectorize(obs)

        pybullet_sim = PyBulletSim(state, rendering=False)
        controllers  = create_lifted_controllers(
            env.action_space, pybullet_sim=pybullet_sim
        )

        # ── open_drawer ──────────────────────────────────────────────────
        robot = _get_robot(state)
        wiper = state.get_object_from_name("wiper_0")
        ctrl  = controllers["open_drawer"].ground((robot, wiper))
        ctrl.reset(state, np.array([0.7, -np.pi]))
        state, ok = _run_controller(ctrl, env, state, 300, "open_drawer")
        if not ok:
            env.close()
            return {"success": False, "cubes_off_table": 0, "cube_z_values": []}

        # ── pick_wiper ───────────────────────────────────────────────────
        robot = _get_robot(state)
        wiper = state.get_object_from_name("wiper_0")
        ctrl  = controllers["pick_wiper"].ground((robot, wiper))
        ctrl.reset(state, np.array([0.7, -np.pi]))
        state, ok = _run_controller(ctrl, env, state, 300, "pick_wiper")
        if not ok:
            env.close()
            return {"success": False, "cubes_off_table": 0, "cube_z_values": []}

        # ── sweep ────────────────────────────────────────────────────────
        robot  = _get_robot(state)
        wiper  = state.get_object_from_name("wiper_0")
        target = state.get_object_from_name("cube_0")
        ctrl   = controllers["sweep"].ground((robot, wiper, target))
        ctrl.reset(state, np.array([0.55, -np.pi]))
        state, ok = _run_controller(ctrl, env, state, 200, "sweep")

        # ── measure cube z positions ─────────────────────────────────────
        cubes = sorted(
            [
                obj for obj in state.get_objects(MujocoMovableObjectType)
                if obj.name.startswith("cube")
            ],
            key=lambda o: o.name,
        )
        z_values  = [state.get(c, "z") for c in cubes]
        off_table = sum(z < TABLE_Z_THRESHOLD for z in z_values)

        env.close()
        return {
            "success":         ok,
            "cubes_off_table": off_table,
            "cube_z_values":   z_values,
        }


def main() -> None:
    """Run grid search and print a summary table."""
    header = f"{'start_x':>10} {'end_x':>8} {'success':>9} {'off_table':>10}  cube_z_values"
    print(header)
    print("-" * len(header))

    results = []
    for start_x, end_x in itertools.product(START_X_OFFSETS, END_X_OFFSETS):
        print(f"Running start_x={start_x:.3f}  end_x={end_x:.3f} ...", flush=True)
        result = run_trial(start_x, end_x)
        z_str  = "  ".join(f"{z:.3f}" for z in result["cube_z_values"])
        print(
            f"{start_x:>10.3f} {end_x:>8.3f}"
            f"  {str(result['success']):>7}"
            f"  {result['cubes_off_table']:>9}"
            f"  [{z_str}]"
        )
        results.append({"start_x": start_x, "end_x": end_x, **result})

    best = max(results, key=lambda r: (r["cubes_off_table"], r["success"]))
    print("\n── Best parameters ─────────────────────────────────────────────────")
    print(f"  WIPER_SWEEP_TRANSFORM     x = {best['start_x']:.3f}   (original: -0.050)")
    print(f"  WIPER_SWEEP_TRANSFORM_END x = {best['end_x']:.3f}   (original:  0.180)")
    print(f"  Cubes off table : {best['cubes_off_table']}")
    print(f"  Controller success: {best['success']}")
    print(f"  Cube z values   : {best['cube_z_values']}")


if __name__ == "__main__":
    main()
