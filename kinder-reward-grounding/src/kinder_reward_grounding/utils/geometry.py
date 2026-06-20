"""Geometry helpers for reward grounding."""

from __future__ import annotations

import numpy as np


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """Return a 2D rotation matrix."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def hook_heuristic_keypoints(
    center: np.ndarray,
    theta: float,
    width: float,
    length_side1: float,
    length_side2: float,
) -> tuple[np.ndarray, ...]:
    """Return heuristic keypoints for an L-shaped 2D hook."""
    rot = rotation_matrix_2d(theta)
    long_half = length_side1 / 2.0
    short_half = length_side2 / 2.0
    width_half = width / 2.0

    local_points = (
        np.array([0.0, 0.0], dtype=float),
        np.array([long_half, 0.0], dtype=float),
        np.array([-long_half, 0.0], dtype=float),
        np.array([0.0, short_half], dtype=float),
        np.array([0.0, -short_half], dtype=float),
        np.array([long_half, width_half], dtype=float),
        np.array([long_half, -width_half], dtype=float),
        np.array([-long_half, width_half], dtype=float),
        np.array([-long_half, -width_half], dtype=float),
        np.array([width_half, short_half], dtype=float),
        np.array([-width_half, short_half], dtype=float),
        np.array([width_half, -short_half], dtype=float),
        np.array([-width_half, -short_half], dtype=float),
        np.array([long_half, short_half], dtype=float),
        np.array([long_half, -short_half], dtype=float),
        np.array([-long_half, short_half], dtype=float),
        np.array([-long_half, -short_half], dtype=float),
    )
    return tuple(center + rot @ point for point in local_points)


def point_to_oriented_rectangle_distance(
    point: np.ndarray,
    center: np.ndarray,
    theta: float,
    width: float,
    height: float,
) -> float:
    """Return the distance from a point to an oriented rectangle."""
    local = rotation_matrix_2d(-theta) @ (point - center)
    dx = max(abs(float(local[0])) - width / 2.0, 0.0)
    dy = max(abs(float(local[1])) - height / 2.0, 0.0)
    return float(np.sqrt(dx * dx + dy * dy))
