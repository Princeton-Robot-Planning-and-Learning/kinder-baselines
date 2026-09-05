"""The measured real-lab restock scene and its calibrated skill parameters.

This is the single source of truth for the boxed CylinderShelf3D scene that mirrors
the physical lab (map frame: shelf, staging boxes, six cans at their logged spots) and
for the per-cylinder execution calibration that makes the rigid skills succeed there.
The in-repo tests and the robot-side consumer (prpl-tidybot) both build from here;
the emitting planner (alphatamp) stages the same layout in its own frame.

Scene facts (2026-09): board surfaces 0.100/0.538/0.800 m; deep box (talls) inner
0.395 x 0.2975 m, 0.215 tall, axis-aligned; the three Campbell's-size shorts stand on
the open floor (their old shallow box is retired) in the same zigzag spots.

Calibration facts: the talls pitch 45 degrees down (the deep box's walls rule out side
grasps — and that steep wrist is what justifies them not fitting the upper openings);
the floor shorts take the plain 15-degree side grasp, which also keeps the wrist low
under the top opening at their place; carries lift to 0.27 before tucking.
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
#: insertion and at release. The real side grasp lands ~3 cm high on the
#: shorts (observed 2026-09-05), so they hang lower than modelled; the
#: compensation splits 2 cm into a deeper commanded grasp (see
#: real_restock_grasp_params) and 1 cm here — the level 15-degree wrist
#: runs out of workspace above ~0.03 of ride on the upper board.
PLACE_RELEASE_HEIGHTS = (0.016, 0.016, 0.016, 0.026, 0.026, 0.026)


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
        # DELIBERATE HACK (2026-09-05): the modelled boards sit 5 cm above
        # the measured surfaces (0.100/0.538/0.800 by tape) so every place
        # lands 5 cm higher on the real, unmoved shelf — blunt compensation
        # for a persistent real-vs-model placement lowness that resisted
        # joint-space fixes. The ceilings shift with the boards, so opening
        # clearances are unchanged.
        shelf_layer_zs=(
            # Bottom board back to the measured surface (no +5 cm hack): the
            # model-place talls were dropping from too high, and net +2 cm was
            # still 2 cm high. The demo-replayed shorts ignore these heights,
            # so only the talls are affected.
            0.100 + 0.00 - _BOARD_HALF,
            0.538 + 0.05 - _BOARD_HALF,
            0.800 + 0.05 - _BOARD_HALF,
        ),
        # The three shorts are all Campbell's-size cans (2026-09-05: the
        # taller/heavier shorts made shelf clearance and grip too tight).
        cylinder_heights=(0.29, 0.208, 0.233, 0.10, 0.10, 0.10),
        cylinder_radii=(0.0375, 0.0375, 0.0375, 0.0325, 0.0325, 0.0325),
        boxes=((0.71, 1.105, 1.34125, 1.63875, 0.215),),
        cylinder_init_regions=tuple((x, x, y, y) for x, y in spots),
        robot_base_home_pose=SE2Pose(1.48, 0.67, 1.54),
        # The placement-registration distance must cover the largest
        # PLACE_RELEASE_HEIGHTS entry, or the sim refuses that release (the
        # gripper would open with the can "too far" above the board). Only
        # the place skill ever opens the gripper, so a loose threshold has
        # no other effect.
        min_placement_dist=0.03,
        robot_base_pose_lower_bound=SE2Pose(-0.2, -0.2, -np.pi),
        robot_base_pose_upper_bound=SE2Pose(2.0, 2.0, np.pi),
        x_lb=-0.2,
        x_ub=2.0,
        y_lb=-0.2,
        y_ub=2.0,
    )


def real_restock_grasp_params() -> list[tuple[float, float]]:
    """Per-cylinder (pitch, depth_below_top).

    Tall depths sit 1 cm deeper than the original sweep (the real gripper
    rode too high on the cans, 2026-09-04). The floor shorts command a
    deep 0.065: closes deliberately fire at the executor's cruising
    tolerance (releases alone get the convergence integrator), so the
    real grip rides the deadband high to roughly mid-can — the pick
    regime that worked; the ride height absorbs the hang mismatch at the
    place. Their staging distance must be 0.83 —
    closer stagings cannot reach the low grasp height (swept 2026-09-05)."""
    pitch45 = np.deg2rad(45)
    pitch15 = np.deg2rad(15)
    return [
        (pitch45, 0.04),
        (pitch45, 0.06),
        (pitch45, 0.04),
        (pitch15, 0.065),
        (pitch15, 0.065),
        (pitch15, 0.065),
    ]


def real_restock_move_params() -> list[tuple[float, float]]:
    """Per-cylinder staging (distance, rot). Rot pi/2 parks the base south of
    the cylinder, heading at it. The floor shorts stage at 0.83 like the
    talls: the low side grasp is unreachable from closer in."""
    return [
        (0.83, np.pi / 2),
        (0.88, np.pi / 2),
        (0.83, np.pi / 2),
        (0.83, np.pi / 2),
        (0.83, np.pi / 2),
        (0.83, np.pi / 2),
    ]
