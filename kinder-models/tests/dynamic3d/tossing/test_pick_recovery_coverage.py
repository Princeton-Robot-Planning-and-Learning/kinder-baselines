"""Physical recovery coverage beyond the reset distribution; no reset or bin pushing.

The eight replay fixtures are pre-reset landings from the fragile seed-0 POMDP run
(human cost 100, damage cost 10). Static room colliders are recreated by the environment;
only movable-object, robot, and barrier features are restored from the recording.
All placements require actual MuJoCo contact, a stable lift, and <2 cm bin displacement.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_pick_cube_controller_e2e import _make_env, _run_pick, _set_values

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    PickCubeController,
    create_lifted_controllers,
)

_RECORDED = json.loads(
    Path(__file__)
    .with_name("near_bin_recovery_states.json")
    .read_text(encoding="utf-8")
)


@pytest.mark.parametrize("environment_name", ["Tossing3D", "FragileTossing3D"])
@pytest.mark.parametrize("record", _RECORDED, ids=lambda r: str(r["practice_step"]))
def test_pick_recovers_recorded_near_bin_landings(record, environment_name):
    """Replay stranded practice states and require a stable, contact-backed grasp."""
    env = _make_env(environment_name)
    try:
        obs, _ = env.reset(seed=125)
        state = env.observation_space.devectorize(obs)
        for name, values in record["state"].items():
            obj = state.get_object_from_name(name)
            _set_values(state, obj, state.type_features[obj.type], values)
        _run_pick(env, state)
        final = env.observation_space.devectorize(env.unwrapped.get_state())
        bin_obj = state.get_object_from_name("bin_0")
        displacement = np.linalg.norm(
            [final.get(bin_obj, axis) - state.get(bin_obj, axis) for axis in ("x", "y")]
        )
        assert displacement < 0.02, f"Pickup pushed bin by {displacement:.3f} m"
    finally:
        env.close()


# Deliberately include walls/corners of the bin interior, its rim, missed throws
# outside each wall, ordinary floor recovery, and the robot side of the barrier.
# Coordinates remain within the physical room; across-wall retrieval is excluded.
_COVERAGE = (
    [
        ("bin", dx, dy, yaw)
        for dx, dy, yaw in itertools.product(
            (-0.10, 0.0, 0.10), (-0.10, 0.0, 0.10), (0, 45)
        )
    ]
    + [
        ("outside", dx, dy, yaw)
        for (dx, dy), yaw in itertools.product(
            ((0.20, 0), (-0.20, 0), (0, 0.20), (0, -0.20), (0.25, 0.25)), (0, 45)
        )
    ]
    + [("rim", dx, dy, 0) for dx, dy in ((0.14, 0), (-0.14, 0), (0, 0.14), (0, -0.14))]
    + [
        ("floor", x, y, yaw)
        for (x, y), yaw in itertools.product(
            ((0.55, 0.0), (-0.5, 0.8), (0.9, 1.0), (0.0, 2.5), (-1.5, 0.0)), (0, 45)
        )
    ]
    + [("barrier", 1.22, y, yaw) for y, yaw in itertools.product((-0.5, 0.5), (0, 45))]
)


@pytest.mark.parametrize("environment_name", ["Tossing3D", "FragileTossing3D"])
@pytest.mark.parametrize(
    "category,x,y,yaw",
    _COVERAGE,
    ids=[f"{c}-{x:g}-{y:g}-{yaw}" for c, x, y, yaw in _COVERAGE],
)
def test_pick_recovery_workspace_coverage(category, x, y, yaw, environment_name):
    """Stress retrieval around the bin and room without changing contact physics."""
    env = _make_env(environment_name)
    try:
        obs, _ = env.reset(seed=125)
        state = env.observation_space.devectorize(obs)
        cube = state.get_object_from_name("cube_0")
        bin_obj = state.get_object_from_name("bin_0")
        _set_values(
            state,
            bin_obj,
            ("x", "y", "z", "qx", "qy", "qz", "qw"),
            (0.1, -1.5, 0.0, 0.0, 0.0, 0.0, 1.0),
        )
        z = 0.045 if category == "bin" else 0.225 if category == "rim" else 0.025
        if category == "bin":
            # Three mm of physical clearance: a rotated cube has a wider footprint.
            half_footprint = 0.025 * (
                abs(np.cos(np.deg2rad(yaw))) + abs(np.sin(np.deg2rad(yaw)))
            )
            limit = 0.15 - 0.02 - half_footprint - 0.003
            x, y = np.clip(x, -limit, limit), np.clip(y, -limit, limit)
        if category in ("bin", "outside", "rim"):
            x, y = x + 0.1, y - 1.5
        quat = Rotation.from_euler("z", yaw, degrees=True).as_quat()
        _set_values(
            state, cube, ("x", "y", "z", "qx", "qy", "qz", "qw"), (x, y, z, *quat)
        )
        _run_pick(env, state)
        final = env.observation_space.devectorize(env.unwrapped.get_state())
        displacement = np.linalg.norm(
            [final.get(bin_obj, "x") - 0.1, final.get(bin_obj, "y") + 1.5]
        )
        assert displacement < 0.02, f"Pickup pushed bin by {displacement:.3f} m"
    finally:
        env.close()


def test_pick_waits_for_actual_finger_motion_to_stop():
    """Crossing the binary 'closed' threshold does not mean the grasp has settled."""
    env = _make_env()
    try:
        observation, _ = env.reset(seed=125)
        state = env.observation_space.devectorize(observation)
        robot = state.get_object_from_name("robot")
        controller = create_lifted_controllers(env.action_space, state)[
            "pick_cube"
        ].ground(
            (
                robot,
                state.get_object_from_name("cube_0"),
                state.get_object_from_name("cuboid_barrier"),
            )
        )
        assert isinstance(controller, PickCubeController)
        phase = controller.PickCubeControllerPhase
        controller.current_phase = phase.CLOSE_GRIPPER_TO_GRASP_CUBE
        for reading in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85):
            state.set(robot, "pos_gripper", reading)
            controller.observe(state)
            assert controller.step()[-1] == 1
            assert controller.current_phase == phase.CLOSE_GRIPPER_TO_GRASP_CUBE
        for _ in range(3):
            controller.step()
        assert controller.current_phase == phase.LIFT_CUBE_TO_HOME
    finally:
        env.close()
