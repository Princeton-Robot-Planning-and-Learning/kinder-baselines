"""Tests for dynamic3d predicate_checks.py."""

import subprocess
import sys

import numpy as np

from kinder_models.dynamic3d.predicate_checks import (
    END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE,
    GRIPPER_GRASPING_THRESHOLD,
    GRIPPER_OPEN_COMMAND_TOLERANCE,
    MINIMUM_HOLDING_HEIGHT,
    ON_GROUND_TOLERANCE,
    check_end_effector_at_object,
    check_grasped_and_lifted,
    check_gripper_open,
    check_is_down_x,
    check_on_ground,
)


def test_importing_this_module_pulls_in_no_simulator():
    """The point of the module: arithmetic a consumer can import without KINDER.

    Checked in a subprocess because the rest of this file's imports would otherwise
    have loaded the simulator into this interpreter already.
    """
    program = (
        "import sys\n"
        "import kinder_models.dynamic3d.predicate_checks\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in "
        "('kinder', 'pybullet', 'mujoco', 'relational_structs', 'bilevel_planning')]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_gripper_open_reads_the_command_at_upstream_tolerance():
    """Open is a commanded zero, not a finger pose, so the tolerance is tiny."""
    assert check_gripper_open(0.0)
    assert check_gripper_open(GRIPPER_OPEN_COMMAND_TOLERANCE / 2)
    assert not check_gripper_open(10 * GRIPPER_OPEN_COMMAND_TOLERANCE)


def test_on_ground_needs_floor_height_and_flatness_together():
    """Flatness is what keeps the bottom-face arithmetic valid, so it is required."""
    assert check_on_ground(0.025, 0.05, 0.0, 0.0)
    assert not check_on_ground(0.5, 0.05, 0.0, 0.0)
    assert not check_on_ground(0.025, 0.05, 0.5, 0.0)
    assert not check_on_ground(0.025, 0.05, 0.0, 0.5)


def test_on_ground_is_inclusive_out_to_the_tolerance():
    """The bottom face may sit a tolerance off the floor and still count."""
    height = 0.05
    assert check_on_ground(height / 2 + ON_GROUND_TOLERANCE / 2, height, 0.0, 0.0)
    assert not check_on_ground(height / 2 + 2 * ON_GROUND_TOLERANCE, height, 0.0, 0.0)


def test_grasped_and_lifted_needs_both_conjuncts():
    """Neither a closed gripper nor a raised object is enough on its own."""
    closed = 2 * GRIPPER_GRASPING_THRESHOLD
    lifted = 2 * MINIMUM_HOLDING_HEIGHT
    assert check_grasped_and_lifted(closed, lifted)
    assert not check_grasped_and_lifted(GRIPPER_GRASPING_THRESHOLD / 2, lifted)
    assert not check_grasped_and_lifted(closed, MINIMUM_HOLDING_HEIGHT / 2)


def test_grasped_and_lifted_excludes_its_own_thresholds():
    """Strict comparisons, so a value sitting exactly on a threshold does not hold."""
    assert not check_grasped_and_lifted(
        GRIPPER_GRASPING_THRESHOLD, 2 * MINIMUM_HOLDING_HEIGHT
    )
    assert not check_grasped_and_lifted(
        2 * GRIPPER_GRASPING_THRESHOLD, MINIMUM_HOLDING_HEIGHT
    )


def test_end_effector_at_object_is_a_per_axis_box_not_a_sphere():
    """Three independent axis tests, which admits the box's corners."""
    offset = 0.9 * END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE
    assert check_end_effector_at_object(np.array([offset, offset, offset]), np.zeros(3))
    assert not check_end_effector_at_object(
        np.array([2 * END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE, 0.0, 0.0]), np.zeros(3)
    )


def test_is_down_x_is_strict_so_equal_positions_do_not_hold():
    """A cube exactly at the barrier is not yet past it, and not yet short of it."""
    assert check_is_down_x(0.5, 1.3)
    assert not check_is_down_x(1.3, 1.3)
    assert not check_is_down_x(2.0, 1.3)
