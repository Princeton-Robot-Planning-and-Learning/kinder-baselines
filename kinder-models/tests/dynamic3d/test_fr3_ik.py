"""Tests for the Franka FR3 inverse kinematics solver (FR3IKSolver)."""

import mujoco
import numpy as np
from kinder.envs.dynamic3d.robots.fr3_robot_env import FR3RobotEnv

from kinder_models.dynamic3d.fr3_ik_solver import FR3IKSolver


def _rotation_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a 3x3 rotation matrix."""
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat[[3, 0, 1, 2]])  # pylint: disable=no-member
    return mat.reshape(3, 3)


def test_fr3_fk_home_pose():
    """FK at the home configuration matches the measured MJCF geometry.

    The TCP (2F-85 pinch point) sits in front of the mount with the approach axis
    pointing straight down, which is what the top-down pick-and-place skills rely on.
    """
    ik = FR3IKSolver()
    pos, quat = ik.get_site_pose(FR3RobotEnv.HOME_QPOS)
    # Flange at (0.5545, 0, 0.6245) in the base frame, minus the 0.125 m TCP
    # extension along the downward approach axis.
    assert np.allclose(pos, [0.5545, 0.0, 0.4995], atol=1e-2)
    approach_axis = _rotation_from_quat_xyzw(quat)[:, 2]
    assert np.allclose(approach_axis, [0.0, 0.0, -1.0], atol=1e-6)


def test_fr3_ik_solver_basic():
    """Solving for the home pose from the home configuration is a fixed point."""
    ik = FR3IKSolver()
    target_pos, target_quat = ik.get_site_pose(FR3RobotEnv.HOME_QPOS)
    result = ik.solve(target_pos, target_quat, FR3RobotEnv.HOME_QPOS)
    assert result.shape == (7,)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, FR3RobotEnv.HOME_QPOS, atol=1e-3)


def test_fr3_ik_fk_roundtrip():
    """IK followed by FK reproduces top-down targets across the desk workspace.

    Targets are in the arm base frame and bracket the poses the pick-and-place
    skills request: grasp/pre-grasp near the cube region and place/pre-place
    near the goal region of FrankaPickPlace3D-o1.
    """
    ik = FR3IKSolver()
    home = FR3RobotEnv.HOME_QPOS
    _, down_quat = ik.get_site_pose(home)
    targets = [
        (0.56, -0.03, 0.03),  # grasp height near the cube init region
        (0.56, -0.03, 0.15),  # pre-grasp above it
        (0.40, 0.18, 0.05),  # place height in the goal region
        (0.40, 0.18, 0.17),  # pre-place above it
        (0.45, -0.20, 0.10),  # opposite side of the desk
    ]
    for target in targets:
        target_pos = np.array(target)
        qpos = ik.solve(target_pos, down_quat, home)
        assert qpos.shape == (7,)
        assert np.all(np.isfinite(qpos))
        fk_pos, fk_quat = ik.get_site_pose(qpos)
        assert np.allclose(
            fk_pos, target_pos, atol=1e-3
        ), f"Position mismatch at {target}: {fk_pos}"
        quat_dist = min(
            float(np.linalg.norm(fk_quat - down_quat)),
            float(np.linalg.norm(fk_quat + down_quat)),
        )
        assert quat_dist < 1e-2, f"Orientation mismatch at {target}: {fk_quat}"


def test_fr3_ik_unreachable_target():
    """An unreachable target yields a large FK error rather than a failure signal.

    The solver is best-effort; callers are responsible for verifying the solution with
    FK, as the pick-and-place skills do.
    """
    ik = FR3IKSolver()
    home = FR3RobotEnv.HOME_QPOS
    _, down_quat = ik.get_site_pose(home)
    # Well beyond the FR3's ~0.855 m reach.
    target_pos = np.array([1.5, 0.0, 0.1])
    qpos = ik.solve(target_pos, down_quat, home)
    assert np.all(np.isfinite(qpos))
    fk_pos, _ = ik.get_site_pose(qpos)
    assert np.linalg.norm(fk_pos - target_pos) > 0.3
