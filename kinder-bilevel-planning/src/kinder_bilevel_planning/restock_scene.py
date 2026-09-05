"""The measured real-lab restock scene and its calibrated skill parameters.

This is the single source of truth for the boxed CylinderShelf3D scene that mirrors
the physical lab (map frame: shelf, staging boxes, six cans at their logged spots) and
for the per-cylinder execution calibration that makes the rigid skills succeed there.
The in-repo tests and the robot-side consumer (prpl-tidybot) both build from here;
the emitting planner (alphatamp) stages the same layout in its own frame.

Scene facts (2026-09): board surfaces 0.100/0.538/0.800 m; deep box (talls) inner
0.395 x 0.2975 m, 0.215 tall, axis-aligned; shallow box (shorts) inner 0.40 x 0.32 m,
0.115 tall, set down rotated by 0.25 rad; cans staged in zigzag rows inside the boxes.

Calibration facts: every grasp pitches 45 degrees down (the box walls rule out side
grasps); cans whose tops sit at or below their box rim need the shallow 0.015 pinch
depth; staging rots follow each box's own normal so the chassis aligns with it; carries
lift to 0.27 before tucking.
"""

import numpy as np
from kinder.envs.kinematic3d.cylinder_shelf3d import CylinderShelf3DEnvConfig
from pybullet_helpers.geometry import Pose, SE2Pose

_BOARD_HALF = 0.0127 / 2
_DEEP_CENTER = (0.9075, 1.49)
_SHALLOW_CENTER = (0.40, 1.28)
SHALLOW_BOX_YAW = 0.25

#: Fixed place-parameter calibration for this robot (see place_params_from_ir).
PLACE_Y_OFFSET = -0.05
PLACE_BASE_DISTANCE = 0.80
CARRY_LIFT_Z = 0.27
#: Per-cylinder height (m) the bottom rides above the board during the level
#: insertion and at release. The compliant arm droops under load at the
#: placement extension, so the heavy peanut-butter jar (c4) rides higher:
#: droop brings it down to roughly the light cans' margin before release.
PLACE_RELEASE_HEIGHTS = (0.016, 0.016, 0.016, 0.016, 0.045, 0.016)


def _zigzag(
    center: tuple[float, float], yaw: float, pitch: float, dy: float
) -> list[tuple[float, float]]:
    out = []
    for lx, ly in [(-pitch, -dy), (0.0, dy), (pitch, -dy)]:
        c, s = np.cos(yaw), np.sin(yaw)
        out.append((center[0] + c * lx - s * ly, center[1] + s * lx + c * ly))
    return out


def real_restock_config() -> CylinderShelf3DEnvConfig:
    """The boxed real-lab scene (map frame), cans at their logged staging spots."""
    spots = _zigzag(_DEEP_CENTER, 0.0, 0.13, 0.06) + _zigzag(
        _SHALLOW_CENTER, SHALLOW_BOX_YAW, 0.13, 0.07
    )
    return CylinderShelf3DEnvConfig(
        shelf_pose=Pose((1.63, 1.51, 0.0)),
        shelf_layer_zs=(
            0.100 - _BOARD_HALF,
            0.538 - _BOARD_HALF,
            0.800 - _BOARD_HALF,
        ),
        cylinder_heights=(0.29, 0.208, 0.233, 0.12, 0.125, 0.10),
        cylinder_radii=(0.0375, 0.0375, 0.0375, 0.0375, 0.035, 0.0325),
        boxes=(
            (0.71, 1.105, 1.34125, 1.63875, 0.215),
            (0.20, 0.60, 1.12, 1.44, 0.115, SHALLOW_BOX_YAW),
        ),
        cylinder_init_regions=tuple((x, x, y, y) for x, y in spots),
        robot_base_home_pose=SE2Pose(1.48, 0.67, 1.54),
        # The placement-registration distance must cover the largest
        # PLACE_RELEASE_HEIGHTS entry, or the sim refuses that release (the
        # gripper would open with the can "too far" above the board). Only
        # the place skill ever opens the gripper, so the loose threshold has
        # no other effect.
        min_placement_dist=0.05,
        robot_base_pose_lower_bound=SE2Pose(-0.2, -0.2, -np.pi),
        robot_base_pose_upper_bound=SE2Pose(2.0, 2.0, np.pi),
        x_lb=-0.2,
        x_ub=2.0,
        y_lb=-0.2,
        y_ub=2.0,
    )


def real_restock_grasp_params() -> list[tuple[float, float]]:
    """Per-cylinder (pitch, depth_below_top).

    Depths sit 1 cm deeper than the original sweep (the real gripper rode
    too high on the cans, 2026-09-04) with two exceptions. Campbell's: the
    shortest can's top is below the shallow box rim, and in sim any grasp
    deeper than 0.015 fouls the wrist on the rim during the reach or carry.
    Skippy: the heavy jar slips down through the compliant grip during the
    carry until its lid ridge catches the fingers, so it is gripped just
    UNDER the ridge on purpose — slip is then arrested within millimetres
    and the hang distance the placement assumes stays true."""
    pitch45 = np.deg2rad(45)
    return [
        (pitch45, 0.04),
        (pitch45, 0.06),
        (pitch45, 0.04),
        (pitch45, 0.025),
        (pitch45, 0.03),
        (pitch45, 0.015),
    ]


def real_restock_move_params() -> list[tuple[float, float]]:
    """Per-cylinder staging (distance, rot). Rot pi/2 parks the base south of the
    cylinder, heading at it; the shallow box's cylinders are approached along the
    box's own normal so the chassis aligns with it."""
    return [
        (0.83, np.pi / 2),
        (0.88, np.pi / 2),
        (0.83, np.pi / 2),
        (0.72, np.pi / 2 + SHALLOW_BOX_YAW),
        (0.78, np.pi / 2 + SHALLOW_BOX_YAW),
        (0.78, np.pi / 2 + SHALLOW_BOX_YAW),
    ]
