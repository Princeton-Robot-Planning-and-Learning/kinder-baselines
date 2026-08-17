"""The rotations that map a cube onto itself, and what follows from them.

A cube resting on any of its faces is the same cube in the same place, so its roll and
pitch carry no information and only its yaw does. Anything deriving a grasp or a resting
test from a cube's measured rotation has to say so explicitly, or it ends up
distinguishing poses that are physically identical.

Rotations cross this module's boundary as `Quaternion`, which is pybullet_helpers' own
alias for an (x, y, z, w) tuple -- the order the dynamic3d state features are in
(`qx`, `qy`, `qz`, `qw`) and the order `Pose.orientation` is in, so a caller never
converts. Note that Mujoco's native qpos order is (w, x, y, z), which this is not.
`Rotation` is the computation type only, constructed and discarded inside a function.
"""

import numpy as np
from pybullet_helpers.geometry import Quaternion
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

# The 24 rotations mapping a cube onto itself: 6 faces down, each at 4 yaws.
CUBE_ROTATION_SYMMETRIES: tuple[Quaternion, ...] = tuple(
    (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    for q in Rotation.create_group("O").as_quat()
)
_CUBE_ROTATION_SYMMETRY_GROUP = Rotation.from_quat(np.array(CUBE_ROTATION_SYMMETRIES))
_CUBE_ROTATION_SYMMETRY_ANGLES = _CUBE_ROTATION_SYMMETRY_GROUP.magnitude()


def cube_tilt_from_upright(rotation: Quaternion) -> float:
    """How far a cube is from resting flat on one of its faces.

    Zero for any face-down rest at any yaw, and larger the closer the cube is to
    balancing on an edge or a corner.
    """
    composed = (Rotation.from_quat(rotation) * _CUBE_ROTATION_SYMMETRY_GROUP).as_quat()
    return float(np.min(composed[:, 0] ** 2 + composed[:, 1] ** 2))


def upright_grasp_rotations(rotation: Quaternion) -> tuple[Quaternion, ...]:
    """Every upright rotation the cube could equally be grasped at, nearest yaw first.

    Deriving a grasp from the measured rotation asks the gripper to approach along
    whichever face happens to be up, from underneath the floor for a cube on its top.
    Resting on a face a cube is also four-fold symmetric about the vertical, so all four
    yaws are the same grasp and a caller can fall through when the arm cannot reach one.
    """
    composed = (Rotation.from_quat(rotation) * _CUBE_ROTATION_SYMMETRY_GROUP).as_quat()
    x, y, z, w = composed[:, 0], composed[:, 1], composed[:, 2], composed[:, 3]
    tilt = x * x + y * y
    # Every minimal-tilt family is at least a tied 180-degree pair, so the tilt alone
    # never picks a unique winner. Break the tie toward the symmetry closest to
    # identity, so an already-near-upright cube keeps its own measured yaw instead of
    # jumping to an arbitrary tied alternative -- which the real robot cannot always
    # reach from the same standoff.
    tied = np.flatnonzero(tilt < tilt.min() + 1e-12)
    nearest = int(tied[np.argmin(_CUBE_ROTATION_SYMMETRY_ANGLES[tied])])
    yaw = float(
        np.arctan2(
            2 * (w[nearest] * z[nearest] + x[nearest] * y[nearest]),
            1 - 2 * (y[nearest] ** 2 + z[nearest] ** 2),
        )
    )
    return tuple(
        (0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2)))
        for angle in (yaw, yaw + np.pi / 2, yaw - np.pi / 2, yaw + np.pi)
    )
