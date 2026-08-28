"""Tests for kinder_models.magic."""

from typing import Any, Sequence

import numpy as np
import pytest
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from gymnasium.spaces import Box
from relational_structs import Object, Type, Variable

from kinder_models.magic import (
    MagicController,
    OutcomePredictor,
    make_magic_lifted_controller,
)
from kinder_models.structs import SkillCall

_COUNTER_TYPE = Type("counter")


class _IncrementController(
    GroundParameterizedController[int, int], OutcomePredictor[int]
):
    """Adds ``params`` to the state one unit per step; predicts the sum outright."""

    def __init__(self, objects: Sequence[Object]) -> None:
        super().__init__(objects)
        self._remaining = 0
        self._state = 0

    def sample_parameters(self, x: int, rng: np.random.Generator) -> Any:
        del x
        return int(rng.integers(1, 4))

    def reset(self, x: int, params: Any) -> None:
        self._state = x
        self._remaining = int(params)

    def terminated(self) -> bool:
        return self._remaining == 0

    def step(self) -> int:
        self._remaining -= 1
        return 1

    def observe(self, x: int) -> None:
        self._state = x

    def predict_outcome(self, x: int, params: Any) -> int:
        return x + int(params)


class _PlainController(GroundParameterizedController[int, int]):
    """A controller with no outcome model."""

    def sample_parameters(self, x: int, rng: np.random.Generator) -> Any:
        return 0

    def reset(self, x: int, params: Any) -> None:
        pass

    def terminated(self) -> bool:
        return True

    def step(self) -> int:
        return 0

    def observe(self, x: int) -> None:
        pass


def test_magic_controller_emits_one_skill_call():
    """The magic controller samples through the inner controller and terminates after a
    single SkillCall carrying the inner controller's prediction."""
    obj = Object("c", _COUNTER_TYPE)
    inner = _IncrementController([obj])
    controller = MagicController([obj], inner, "Increment")

    rng = np.random.default_rng(0)
    params = controller.sample_parameters(5, rng)
    assert 1 <= params <= 3

    controller.reset(5, params)
    assert not controller.terminated()
    action = controller.step()
    assert isinstance(action, SkillCall)
    assert action.skill_name == "Increment"
    assert action.objects == (obj,)
    assert action.params == params
    assert action.predicted_state == 5 + params
    assert controller.terminated()

    # Resetting clears the termination.
    controller.reset(10, 2)
    assert not controller.terminated()
    assert controller.step().predicted_state == 12


def test_magic_controller_rejects_inner_without_outcome_model():
    """Wrapping a controller that cannot predict its outcome is a TypeError."""
    obj = Object("c", _COUNTER_TYPE)
    with pytest.raises(TypeError, match="OutcomePredictor"):
        MagicController([obj], _PlainController([obj]), "Plain")


def test_make_magic_lifted_controller():
    """The wrapped lifted controller keeps the variables and parameter space and grounds
    to a MagicController around the original ground controller."""
    var = Variable("?c", _COUNTER_TYPE)
    params_space = Box(low=np.array([1.0]), high=np.array([3.0]))
    lifted: LiftedParameterizedController = LiftedParameterizedController(
        [var], _IncrementController, params_space
    )
    magic = make_magic_lifted_controller(lifted, "Increment")
    assert magic.variables == lifted.variables
    assert magic.params_space is params_space
    assert magic.name == "Magic_IncrementController"

    obj = Object("c", _COUNTER_TYPE)
    ground = magic.ground([obj])
    assert isinstance(ground, MagicController)
    ground.reset(1, 3)
    assert ground.step().predicted_state == 4

    plain: LiftedParameterizedController = LiftedParameterizedController(
        [var], _PlainController
    )
    with pytest.raises(TypeError, match="OutcomePredictor"):
        make_magic_lifted_controller(plain, "Plain")
