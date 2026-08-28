"""Tests for kinder_models.structs."""

import numpy as np
from relational_structs import Object, Type

from kinder_models.structs import SkillCall


def test_skill_call_repr_is_deterministic_and_complete():
    """Two calls with equal contents share a repr; changing any field changes it."""
    robot_type = Type("robot")
    robot = Object("robot", robot_type)
    target = Object("cylinder0", robot_type)
    call = SkillCall("Pick", (robot, target), np.array([0.8, 0.1]), 7)
    same = SkillCall("Pick", (robot, target), np.array([0.8, 0.1]), 7)
    assert repr(call) == repr(same)
    assert "Pick" in repr(call) and "cylinder0" in repr(call)
    assert str(call) == "Pick(robot, cylinder0)"

    other_params = SkillCall("Pick", (robot, target), np.array([0.8, 0.2]), 7)
    other_state = SkillCall("Pick", (robot, target), np.array([0.8, 0.1]), 8)
    other_name = SkillCall("Place", (robot, target), np.array([0.8, 0.1]), 7)
    other_objects = SkillCall("Pick", (robot,), np.array([0.8, 0.1]), 7)
    reprs = {
        repr(call),
        repr(other_params),
        repr(other_state),
        repr(other_name),
        repr(other_objects),
    }
    assert len(reprs) == 5


def test_skill_call_accepts_non_array_params():
    """Scalar and tuple parameters are rendered without a tolist() conversion."""
    obj = Object("robot", Type("robot"))
    assert "params=0.5" in repr(SkillCall("Push", (obj,), 0.5, None))
    assert "params=(1, 2)" in repr(SkillCall("Push", (obj,), (1, 2), None))
