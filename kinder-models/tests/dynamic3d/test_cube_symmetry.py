"""Tests for cube symmetry helpers in utils.py."""

import numpy as np

from kinder_models.dynamic3d.utils import (
    CUBE_ROTATION_SYMMETRIES,
    upright_grasp_rotations,
)


def test_there_are_twenty_four_cube_rotation_symmetries():
    """A cube has 6 faces it can rest on, each at 4 yaws."""
    assert len(CUBE_ROTATION_SYMMETRIES) == 24
    seen = {tuple(np.round(q, 6)) for q in CUBE_ROTATION_SYMMETRIES}
    assert len(seen) == 24


def test_every_cube_symmetry_is_a_rotation():
    """Unit quaternions, so each maps the cube onto itself without scaling it."""
    for q in CUBE_ROTATION_SYMMETRIES:
        assert np.isclose(np.linalg.norm(q), 1.0)


def test_the_nearest_upright_rotation_flattens_every_face_down_rest():
    """Each of the six face-down rests is the same cube, so each must canonicalise to a
    pure yaw -- which is what makes a top-down grasp derivable from it."""

    def quat(axis, deg):
        a = np.deg2rad(deg) / 2
        v = np.array(axis, dtype=float)
        x, y, z = np.sin(a) * v
        return (float(x), float(y), float(z), float(np.cos(a)))

    for axis, deg in [
        ([0, 0, 1], 0),
        ([0, 0, 1], 90),
        ([1, 0, 0], 90),
        ([1, 0, 0], 180),
        ([1, 0, 0], -90),
        ([0, 1, 0], 90),
        ([0, 1, 0], -90),
        ([0, 1, 0], 180),
    ]:
        x, y, _, _ = upright_grasp_rotations(quat(axis, deg))[0]
        assert abs(x) < 1e-6, (axis, deg, x)
        assert abs(y) < 1e-6, (axis, deg, y)


def test_the_nearest_upright_rotation_keeps_the_yaw():
    """Yaw is the only real information in a cube's resting pose, so it must survive."""
    for deg in (0.0, 30.0, 90.0, 200.0):
        a = np.deg2rad(deg) / 2
        q = (0.0, 0.0, float(np.sin(a)), float(np.cos(a)))
        _, _, z, w = upright_grasp_rotations(q)[0]
        got = np.rad2deg(2 * np.arctan2(z, w)) % 90.0
        assert np.isclose(got, deg % 90.0, atol=1e-4) or np.isclose(
            got, deg % 90.0 - 90.0, atol=1e-4
        ), (deg, got)


def test_upright_grasp_rotations_offers_all_four_equivalent_yaws():
    """Resting on a face, a cube is four-fold symmetric about the vertical, so all four
    yaws are the same grasp -- and the arm cannot reach every one of them."""
    a = np.deg2rad(90) / 2
    pitched = (0.0, float(np.sin(a)), 0.0, float(np.cos(a)))
    rotations = upright_grasp_rotations(pitched)
    assert len(rotations) == 4
    for x, y, _, _ in rotations:
        assert abs(x) < 1e-6 and abs(y) < 1e-6
    yaws = sorted(
        round(np.rad2deg(2 * np.arctan2(z, w)) % 360.0, 3) for _, _, z, w in rotations
    )
    gaps = {round((b - a_) % 360.0, 3) for a_, b in zip(yaws, yaws[1:])}
    assert gaps == {90.0}, yaws
