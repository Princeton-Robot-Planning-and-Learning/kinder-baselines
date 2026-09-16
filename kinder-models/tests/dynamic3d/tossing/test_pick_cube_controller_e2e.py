"""End-to-end robustness tests for Tossing3D's physical PickCube controller."""

import itertools
import os

# Force the workstation's headless GPU backend before importing/registering Kinder;
# registration otherwise selects OSMesa and contaminates other Dynamic3D tests collected
# in the same process.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import kinder
import numpy as np
import pytest
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    Holding,
    Tossing3DStateAbstractor,
)
from kinder_models.dynamic3d.utils import (
    END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE,
    GRIPPER_GRASPING_THRESHOLD,
    MINIMUM_HOLDING_HEIGHT,
    PyBulletSim,
)
from pybullet_helpers.geometry import Pose
from relational_structs import GroundAtom, ObjectCentricState
from scipy.spatial.transform import Rotation

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

_ROBOT_FEATURES = (
    "pos_base_x",
    "pos_base_y",
    "pos_base_rot",
    "pos_arm_joint1",
    "pos_arm_joint2",
    "pos_arm_joint3",
    "pos_arm_joint4",
    "pos_arm_joint5",
    "pos_arm_joint6",
    "pos_arm_joint7",
    "pos_gripper",
)
_CUBE_POSE_FEATURES = ("x", "y", "z", "qx", "qy", "qz", "qw")

# Outside the intended grasp coverage: missed tosses against the bin wall may
# require a human reset. Collision checking must reject these before execution.
_INFEASIBLE_WALL_GRASPS = (
    (
        (
            -0.698426995,
            -1.889215958,
            3.150197574,
            0.001627074,
            -0.330695789,
            3.139703063,
            -2.535009853,
            0.000244449,
            -0.882603461,
            1.568244506,
            0.0,
        ),
        (
            -1.898246760,
            -1.905228951,
            0.024892245,
            -0.004186846,
            0.707094386,
            0.004186846,
            0.707094386,
        ),
        (-2.091620684, -1.901647091, -0.000107755),
    ),
    (
        (
            -0.749110930,
            -1.205785164,
            3.146857610,
            -0.000288908,
            -0.330315597,
            3.142389715,
            -2.536415654,
            -0.000527563,
            -0.880430267,
            1.573213837,
            0.0,
        ),
        (-2.000954782, -1.218251737, 0.024892245, 0.0, 0.999378470, -0.035251578, 0.0),
        (-2.178002577, -1.213747918, -0.000107755),
    ),
)

# Feasible controller-entry states from the September 15 and 16 planning runs.
# Before the fixes, execution finished with the cube still on the floor.
_OBSERVED_FAILURES = (
    # Controller-entry states of failed picks 2 and 9 in the verified-pick
    # seed-0 rerun (2026-09-16). The reset cube has not yet fallen to the floor.
    (
        (
            -0.0279723215,
            0.0288276784,
            -0.0777655246,
            0.0,
            -0.3490659055,
            3.1415926554,
            -2.5481806005,
            0.0,
            -0.8726646315,
            1.5707963268,
            0.0,
        ),
        (
            0.5677592158,
            -0.2490022033,
            0.0661099702,
            0.9788499685,
            0.0,
            0.0,
            0.2045794200,
        ),
        (-2.0916206837, -1.9016470909, 0.0025526281),
    ),
    (
        (
            0.0636342324,
            -0.4683418699,
            0.4120710306,
            -0.0014911572,
            -0.3196624506,
            3.1436012656,
            -2.5338299766,
            -0.0249035012,
            -0.8611035094,
            1.5943182739,
            0.0,
        ),
        (
            0.7399140000,
            -0.2429455370,
            0.0731230602,
            0.9722070101,
            0.0,
            0.0,
            0.2341228939,
        ),
        (-1.8231077194, 0.8255638480, 0.0241436101),
    ),
    (
        (
            -0.7409365849,
            -1.8901957309,
            3.1499751584,
            -0.0003058289,
            -0.3527653489,
            3.1413261595,
            -2.5312961315,
            0.0004760278,
            -0.8635641132,
            1.5700105171,
            0.0,
        ),
        (
            -1.8836931258,
            -1.9069861934,
            0.0248922446,
            0.0,
            0.9999959649,
            0.0028408145,
            0.0,
        ),
        (-2.0916207, -1.9016471, -0.0001078),
    ),
    (
        (
            -0.2340553863,
            1.5058140333,
            3.0820454690,
            0.0000662466,
            -0.3487624999,
            3.1426103913,
            -2.5272174745,
            0.0012931055,
            -0.8617003356,
            1.5709340609,
            0.0,
        ),
        (
            0.7399140000,
            -0.2429455370,
            0.0239092952,
            0.9722070101,
            0.0,
            0.0,
            0.2341228939,
        ),
        (2.1794789, 0.0380994, -0.0001078),
    ),
)


def _make_env():
    return kinder.make(
        "kinder/Tossing3D-o1-v0",
        render_mode="rgb_array",
        num_objects=1,
        allow_state_access=True,
    )


def _set_values(state: ObjectCentricState, obj, names, values) -> None:
    for name, value in zip(names, values, strict=True):
        state.set(obj, name, float(value))


def _create_bin_aware_sim(state: ObjectCentricState, scene) -> PyBulletSim:
    sim = PyBulletSim(state)
    bin_obj = state.get_object_from_name("bin_0")
    geometry = scene.get_object("bin_0")
    sim.add_bin(
        name="bin_0",
        pose=Pose(
            tuple(state.get(bin_obj, key) for key in ("x", "y", "z")),
            tuple(state.get(bin_obj, key) for key in ("qx", "qy", "qz", "qw")),
        ),
        length=geometry.length,
        width=geometry.width,
        height=geometry.height,
        wall_thickness=geometry.wall_thickness,
    )
    return sim


def _run_pick(env, state: ObjectCentricState, *, max_steps: int = 400) -> None:
    """Execute a real MuJoCo pick and require physical, predicate-level success."""
    scene = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    # Construction initializes the abstraction simulator and resets the scene. Do it
    # before restoring the test state, never after executing the physical grasp.
    abstractor = Tossing3DStateAbstractor(scene)
    env.unwrapped.set_state(env.observation_space.vectorize(state))
    state = env.observation_space.devectorize(env.unwrapped.get_state())
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    barrier = state.get_object_from_name("cuboid_barrier")
    scene = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    sim = _create_bin_aware_sim(state, scene)
    try:
        controller = create_lifted_controllers(
            env.action_space, state, pybullet_sim=sim
        )["pick_cube"].ground((robot, cube, barrier))
        controller.reset(state, None)
        for _ in range(max_steps):
            obs, _, _, _, _ = env.step(controller.step())
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
            if controller.terminated():
                break
        assert controller.terminated(), f"stuck in {controller.current_phase.name}"
        # Termination is not enough: hold the terminal command long enough to expose a
        # cube that was merely knocked upward or only momentarily pinched.
        for _ in range(10):
            obs, _, _, _, _ = env.step(controller.step())
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
        sim.set_state(state)
        ee_position = sim.get_ee_pose().position
        cube_position = tuple(state.get(cube, axis) for axis in ("x", "y", "z"))
        ee_delta = tuple(abs(a - b) for a, b in zip(ee_position, cube_position))
        assert state.get(robot, "pos_gripper") > GRIPPER_GRASPING_THRESHOLD
        assert (
            state.get(cube, "z") > MINIMUM_HOLDING_HEIGHT
        ), f"cube returned to z={state.get(cube, 'z'):.4f} after the stability hold"
        assert all(
            delta < END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE for delta in ee_delta
        ), f"lifted cube escaped the gripper after the stability hold: ee_delta={ee_delta}"
        live_contacts = scene.get_gripper_object_contacts(cube.name)
        assert live_contacts, "Lifted cube has no live MuJoCo gripper contact"
        replay_contacts = scene.get_gripper_object_contacts(cube.name, state=state)
        assert replay_contacts, "Holding contact disappeared when evaluating state"
        assert (
            GroundAtom(Holding, [robot, cube])
            in abstractor.state_abstractor(state).atoms
        )
    finally:
        sim.close()
        abstractor._pybullet_sim.close()  # pylint: disable=protected-access


def _run_recorded_pick(robot_values, cube_values, bin_xyz):
    """Restore a recorded placement and execute the real controller."""
    env = _make_env()
    try:
        obs, _ = env.reset(seed=125)
        state = env.observation_space.devectorize(obs)
        robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
        cube = state.get_object_from_name("cube_0")
        bin_obj = state.get_object_from_name("bin_0")
        _set_values(state, robot, _ROBOT_FEATURES, robot_values)
        _set_values(state, cube, _CUBE_POSE_FEATURES, cube_values)
        _set_values(state, bin_obj, ("x", "y", "z"), bin_xyz)
        _run_pick(env, state)
    finally:
        env.close()


@pytest.mark.parametrize("robot_values,cube_values,bin_xyz", _OBSERVED_FAILURES)
def test_pick_cube_recovers_observed_false_successes(
    robot_values, cube_values, bin_xyz
):
    """Regress feasible controller failures observed in the planning experiment."""
    _run_recorded_pick(robot_values, cube_values, bin_xyz)


@pytest.mark.parametrize("robot_values,cube_values,bin_xyz", _INFEASIBLE_WALL_GRASPS)
def test_pick_cube_rejects_wall_obstructed_grasps(robot_values, cube_values, bin_xyz):
    """Outside coverage, preserve collision rejection so a human can reset."""
    with pytest.raises(ValueError, match="No collision-free cube grasp"):
        _run_recorded_pick(robot_values, cube_values, bin_xyz)


def test_pick_cube_dense_declared_initial_state_lattice():
    """Cover every combination of five x, y, and yaw levels in the task prior.

    This 5x5x5 lattice includes all faces, edges, corners, midplanes, and the centre
    of Tossing3D-o1's declared cube reset region; it is deterministic and substantially
    finer than a seed sweep, whose random samples need not exercise any boundary.
    """
    env = _make_env()
    failures = []
    try:
        obs, _ = env.reset(seed=125)
        baseline = env.observation_space.devectorize(obs)
        for x, y, yaw_degrees in itertools.product(
            np.linspace(0.5, 0.75, 5),
            np.linspace(-0.25, 0.25, 5),
            np.linspace(-45.0, 45.0, 5),
        ):
            state = baseline.copy()
            cube = state.get_object_from_name("cube_0")
            quaternion = Rotation.from_euler("z", yaw_degrees, degrees=True).as_quat()
            _set_values(
                state,
                cube,
                _CUBE_POSE_FEATURES,
                (x, y, 0.0248922446, *quaternion),
            )
            try:
                _run_pick(env, state)
            except (AssertionError, ValueError) as error:
                failures.append((float(x), float(y), float(yaw_degrees), str(error)))
        assert not failures, f"{len(failures)}/125 failed: {failures[:10]}"
    finally:
        env.close()


@pytest.mark.parametrize("height", np.linspace(0.025, 0.105, 9))
def test_pick_cube_after_unsettled_reset(height):
    """Grasp a reset cube that falls before descent, across placement and cube faces.

    Nine release heights at 1 cm spacing, each at nine x/y placements. Alternate
    resting faces and yaw to exercise orientation-equivalent grasps as well.
    """
    env = _make_env()
    try:
        obs, _ = env.reset(seed=125)
        baseline = env.observation_space.devectorize(obs)
        for i, (x, y) in enumerate(
            itertools.product((0.5, 0.625, 0.75), (-0.25, 0.0, 0.25))
        ):
            state = baseline.copy()
            cube = state.get_object_from_name("cube_0")
            quaternion = Rotation.from_euler(
                "xyz", (90 * (i % 4), 0, -45 + 45 * (i % 3)), degrees=True
            ).as_quat()
            _set_values(state, cube, _CUBE_POSE_FEATURES, (x, y, height, *quaternion))
            _run_pick(env, state)
    finally:
        env.close()
