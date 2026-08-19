"""Inverse kinematics solver for the Franka FR3 arm.

This module provides an inverse kinematics solver that uses the MuJoCo physics engine on
the FR3's own MJCF model (``franka_fr3/fr3.xml`` from the kindergarden package), so the
kinematics exactly match the simulated robot — no cross-model approximation is needed.
The solver implements a damped least squares approach with nullspace optimization toward
the home configuration.

Adapted from TidybotIKSolver in ik_solver.py.
"""

import re
from pathlib import Path

import kinder
import mujoco
import numpy as np
from kinder.envs.dynamic3d.robots.fr3_robot_env import FR3RobotEnv

# Distance from the flange attachment site to the pinch point between the
# Robotiq 2F-85 finger pads, along the approach axis. Measured in the MJCF at
# the home configuration (pad centers sit ~0.112 m beyond the flange; the
# pinch point is slightly past the pad centers, toward the tips).
FR3_TCP_OFFSET = 0.125


class FR3IKSolver:
    """Inverse kinematics solver for the Franka FR3 arm.

    Targets are expressed in the arm base (mount) frame, at the pinch point of the
    Robotiq 2F-85 gripper. Only the 7 arm joints are solved for; the gripper joints are
    held at zero.
    """

    def __init__(
        self,
        ee_offset: float = 0.0,
        damping_coeff: float = 1e-12,
        max_angle_change: float = np.deg2rad(45),
    ) -> None:
        # Load the FR3 model (arm + gripper). The MJCF's meshdir assumes the
        # kindergarden scene-merging rewrite, so point it at the real assets
        # directory for standalone loading.
        models_dir = Path(kinder.__file__).parent / "envs" / "dynamic3d" / "models"
        xml = (models_dir / "franka_fr3" / "fr3.xml").read_text(encoding="utf-8")
        xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{models_dir / "assets"}"', xml)
        self.model = mujoco.MjModel.from_xml_string(xml)  # pylint: disable=no-member
        self.data = mujoco.MjData(self.model)  # pylint: disable=no-member
        self.model.body_gravcomp[:] = 1.0

        # The 7 arm joints must come first so we can slice qpos and the
        # Jacobian by [:7].
        arm_joint_names = [self.model.joint(i).name for i in range(7)]
        assert arm_joint_names == [f"fr3_joint{i}" for i in range(1, 8)]

        # Cache references
        self.qpos0 = FR3RobotEnv.HOME_QPOS.copy()
        self.site_id = self.model.site("attachment_site").id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Extend the site from the flange to the gripper pinch point.
        self.model.site(self.site_id).pos[2] += FR3_TCP_OFFSET + ee_offset

        # Preallocate arrays
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = damping_coeff * np.eye(6)
        self.eye = np.eye(7)
        self.max_angle_change = max_angle_change

    def get_site_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the pinch-point pose for the given 7 arm joint positions.

        Returns (position, quaternion) in the arm base frame, with the quaternion in (x,
        y, z, w) order.
        """
        self.data.qpos[:7] = qpos
        self.data.qpos[7:] = 0.0
        mujoco.mj_kinematics(self.model, self.data)  # pylint: disable=no-member
        mujoco.mju_mat2Quat(self.site_quat, self.site_mat)  # pylint: disable=no-member
        return self.site_pos.copy(), self.site_quat[[1, 2, 3, 0]].copy()

    def solve(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        curr_qpos: np.ndarray,
        max_iters: int = 50,
        err_thresh: float = 1e-4,
    ) -> np.ndarray:
        """Solve inverse kinematics to achieve the desired pinch-point pose.

        Args:
            pos: Target position (x, y, z) in the arm base frame, in meters
            quat: Target orientation as quaternion (x, y, z, w)
            curr_qpos: Current 7 arm joint positions
            max_iters: Maximum number of iterations (default: 50)
            err_thresh: Error threshold for convergence (default: 1e-4)

        Returns:
            The 7 arm joint positions that achieve the target pose
        """
        quat = quat[[3, 0, 1, 2]]  # (x, y, z, w) -> (w, x, y, z)

        # Set arm to initial joint configuration; keep the gripper at zero.
        self.data.qpos[:7] = curr_qpos
        self.data.qpos[7:] = 0.0

        for _ in range(max_iters):
            # Update site pose
            mujoco.mj_kinematics(self.model, self.data)  # pylint: disable=no-member
            mujoco.mj_comPos(self.model, self.data)  # pylint: disable=no-member

            # Translational error
            self.err_pos[:] = pos - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(  # pylint: disable=no-member
                self.site_quat, self.site_mat
            )
            mujoco.mju_negQuat(  # pylint: disable=no-member
                self.site_quat_inv, self.site_quat
            )
            mujoco.mju_mulQuat(  # pylint: disable=no-member
                self.err_quat, quat, self.site_quat_inv
            )
            mujoco.mju_quat2Vel(  # pylint: disable=no-member
                self.err_rot, self.err_quat, 1.0
            )

            # Check if target pose reached
            if np.linalg.norm(self.err) < err_thresh:
                break

            # Calculate update, restricted to the 7 arm dofs
            mujoco.mj_jacSite(  # pylint: disable=no-member
                self.model, self.data, self.jac_pos, self.jac_rot, self.site_id
            )
            jac_arm = self.jac[:, :7]
            update = jac_arm.T @ np.linalg.solve(
                jac_arm @ jac_arm.T + self.damping, self.err
            )
            qpos0_err = (
                np.mod(self.qpos0 - self.data.qpos[:7] + np.pi, 2 * np.pi) - np.pi
            )
            update += (
                self.eye
                - (
                    jac_arm.T
                    @ np.linalg.pinv(jac_arm @ jac_arm.T + self.damping)
                    @ jac_arm
                )
            ) @ qpos0_err

            # Enforce max angle change
            update_max = np.abs(update).max()
            if update_max > self.max_angle_change:
                update *= self.max_angle_change / update_max

            # Apply update
            self.data.qpos[:7] += update

        return self.data.qpos[:7].copy()
