"""Live pickup and toss regression for the farther KINDER receiver distribution."""

import kinder
import numpy as np
import pytest
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from pybullet_helpers.geometry import Pose

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.utils import PyBulletSim


@pytest.mark.parametrize("extended", [False, True])
@pytest.mark.parametrize(
    "seed, speed", [(10125, 360.0), (10126, 380.0), (10127, 358.0)]
)
def test_far_receiver_requires_extended_effort(
    monkeypatch, extended: bool, seed: int, speed: float
) -> None:
    """Same state and requested throw: old ceiling misses, extended effort scores.

    Execute the pickup each time rather than restoring an approximate grasp.
    No scene, actuator, gravity, or collision modifications are used.
    """
    if not extended:
        monkeypatch.setattr(
            MoveToTossLocationAndTossController, "MAX_SIMULATION_EFFORT", 1.0
        )
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o1-v0", allow_state_access=True)
    try:
        obs, _ = env.reset(seed=seed)
        state = env.observation_space.devectorize(obs)
        robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
        cube = state.get_object_from_name("cube_0")
        barrier = state.get_object_from_name("cuboid_barrier")
        for key, params in [
            ("pick_cube", None),
            (
                "move_to_toss_location_and_toss",
                np.array([2.5, 0.0, np.deg2rad(speed), 500.0]),
            ),
        ]:
            controller = create_lifted_controllers(
                env.action_space,
                init_constant_state=state,
                pybullet_sim=_bin_aware_sim(env, state) if key == "pick_cube" else None,
            )[key].ground((robot, cube, barrier))
            controller.reset(state, params)
            for _ in range(400):
                obs, _, _, _, _ = env.step(controller.step())
                state = env.observation_space.devectorize(obs)
                controller.observe(state)
                if controller.terminated():
                    break
            assert controller.terminated(), key
            if key == "pick_cube":
                assert state.get(cube, "z") > 0.5
        scene = env.unwrapped._object_centric_env  # pylint: disable=protected-access
        assert (
            bool(scene._check_goals()) == extended
        )  # pylint: disable=protected-access
    finally:
        env.close()


def _bin_aware_sim(env, state):
    """Use the real bin's open-top geometry for pickup collision checks."""
    sim = PyBulletSim(state)
    bin_object = state.get_object_from_name("bin_0")
    scene = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    bin_geometry = scene.get_object("bin_0")
    sim.add_bin(
        name="bin_0",
        pose=Pose(
            tuple(state.get(bin_object, f) for f in ("x", "y", "z")),
            tuple(state.get(bin_object, f) for f in ("qx", "qy", "qz", "qw")),
        ),
        length=bin_geometry.length,
        width=bin_geometry.width,
        height=bin_geometry.height,
        wall_thickness=bin_geometry.wall_thickness,
    )
    return sim
