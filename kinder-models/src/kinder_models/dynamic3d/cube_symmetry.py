"""The rotations that map a cube onto itself, and what follows from them.

A cube resting on any of its faces is the same cube in the same place, so its roll and
pitch carry no information and only its yaw does. Anything deriving a grasp or a
resting test from a cube's measured rotation has to say so explicitly, or it ends up
distinguishing poses that are physically identical.
"""

import numpy as np


def cube_tilt_from_upright(rotation: tuple[float, float, float, float]) -> float:
    """How far a cube is from resting flat on one of its faces.

    Zero for any face-down rest at any yaw, and larger the closer the cube is to
    balancing on an edge or a corner.
    """
    return float(
        min(
            x * x + y * y
            for x, y, _, _ in (
                _quaternion_product(rotation, symmetry)
                for symmetry in CUBE_ROTATION_SYMMETRIES
            )
        )
    )


def upright_grasp_rotations(
    rotation: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float, float], ...]:
    """Every upright rotation the cube could equally be grasped at, nearest yaw first.

    Deriving a grasp from the measured rotation asks the gripper to approach along
    whichever face happens to be up, from underneath the floor for a cube on its top.
    Resting on a face a cube is also four-fold symmetric about the vertical, so all four
    yaws are the same grasp and a caller can fall through when the arm cannot reach one.
    """
    best_tilt = np.inf
    yaw = 0.0
    for symmetry in CUBE_ROTATION_SYMMETRIES:
        x, y, z, w = _quaternion_product(rotation, symmetry)
        tilt = x * x + y * y
        if tilt < best_tilt - 1e-12:
            best_tilt = tilt
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return tuple(
        (0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2)))
        for angle in (yaw, yaw + np.pi / 2, yaw - np.pi / 2, yaw + np.pi)
    )


def canonical_upright_rotation(
    rotation: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """The nearest upright rotation, the first of upright_grasp_rotations."""
    return upright_grasp_rotations(rotation)[0]


def _quaternion_product(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _cube_rotation_symmetries() -> tuple[tuple[float, float, float, float], ...]:
    """The 24 rotations mapping a cube onto itself: 6 faces down, each at 4 yaws.

    Closed under composition from the three quarter-turns, which is the definition
    rather than a listing that could be mistyped.
    """
    half = np.sqrt(0.5)
    generators = [
        (half, 0.0, 0.0, half),
        (0.0, half, 0.0, half),
        (0.0, 0.0, half, half),
    ]

    def canonical(q: tuple[float, float, float, float]) -> tuple[float, ...]:
        # q and -q are the same rotation; pick one so the set dedupes.
        rounded = tuple(0.0 + round(v, 6) for v in q)
        for value in rounded:
            if value > 1e-9:
                return rounded
            if value < -1e-9:
                return tuple(-v + 0.0 for v in rounded)
        return rounded

    found = {canonical((0.0, 0.0, 0.0, 1.0)): (0.0, 0.0, 0.0, 1.0)}
    frontier = [(0.0, 0.0, 0.0, 1.0)]
    while frontier:
        current = frontier.pop()
        for generator in generators:
            product = _quaternion_product(current, generator)
            key = canonical(product)
            if key not in found:
                found[key] = (
                    float(key[0]),
                    float(key[1]),
                    float(key[2]),
                    float(key[3]),
                )
                frontier.append(product)
    return tuple(found.values())


CUBE_ROTATION_SYMMETRIES = _cube_rotation_symmetries()
