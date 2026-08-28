"""Magic skills: controllers that replace a skill's execution with one SkillCall.

A regular parameterized controller produces a low-level action sequence that
the planner simulates step by step. A magic controller instead asks the
underlying controller to *predict* the state it would end in
(:class:`OutcomePredictor`) and emits a single :class:`SkillCall` carrying
that prediction. Planning then treats the skill as one transition, and the
executor decides how the skill is actually performed.

Use :func:`make_magic_lifted_controller` to turn an existing lifted controller
into its magic counterpart without touching the skill's operator or sampler.
"""

from __future__ import annotations

import abc
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from relational_structs import Object

from kinder_models.structs import SkillCall

_X = TypeVar("_X")


class OutcomePredictor(abc.ABC, Generic[_X]):
    """A controller that can predict its terminal state without running.

    Mix this into a :class:`GroundParameterizedController` to make the skill
    eligible for :class:`MagicController`. The prediction is the skill's
    option model: given the start state and the sampled parameters, return
    the state the skill is expected to end in. Implementations may raise
    ``TrajectorySamplingFailure`` when the parameters are infeasible, just
    as ``step`` would.
    """

    @abc.abstractmethod
    def predict_outcome(self, x: _X, params: Any) -> _X:
        """Predict the state after executing the skill from ``x`` with ``params``."""


class MagicController(GroundParameterizedController[_X, SkillCall[_X]]):
    """A one-step controller that emits a SkillCall for an inner controller.

    Parameters are sampled by the inner controller, so the magic skill
    explores the same parameter space as the real one. The single action is
    a :class:`SkillCall` whose ``predicted_state`` comes from the inner
    controller's :meth:`OutcomePredictor.predict_outcome`.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        inner: GroundParameterizedController[_X, Any],
        skill_name: str,
    ) -> None:
        super().__init__(objects)
        if not isinstance(inner, OutcomePredictor):
            raise TypeError(
                f"{type(inner).__name__} cannot be made magic: it does not "
                "implement OutcomePredictor.predict_outcome"
            )
        self._inner = inner
        self._skill_name = skill_name
        self._current_state: _X | None = None
        self._current_params: Any = None
        self._called = False

    def sample_parameters(self, x: _X, rng: np.random.Generator) -> Any:
        return self._inner.sample_parameters(x, rng)

    def reset(self, x: _X, params: Any) -> None:
        self._current_state = x
        self._current_params = params
        self._called = False

    def terminated(self) -> bool:
        return self._called

    def step(self) -> SkillCall[_X]:
        assert self._current_state is not None
        assert isinstance(self._inner, OutcomePredictor)
        predicted = self._inner.predict_outcome(
            self._current_state, self._current_params
        )
        self._called = True
        return SkillCall(
            skill_name=self._skill_name,
            objects=tuple(self.objects),
            params=self._current_params,
            predicted_state=predicted,
        )

    def observe(self, x: _X) -> None:
        self._current_state = x


def make_magic_lifted_controller(
    lifted: LiftedParameterizedController[_X, Any], skill_name: str
) -> LiftedParameterizedController[_X, SkillCall[_X]]:
    """Wrap a lifted controller so grounding it yields a MagicController.

    The returned lifted controller shares the original's variables and
    parameter space; only the ground controller class changes. The inner
    ground controller class must implement :class:`OutcomePredictor`.
    """
    inner_cls = lifted.controller_cls
    if not issubclass(inner_cls, OutcomePredictor):
        raise TypeError(
            f"{inner_cls.__name__} cannot be made magic: it does not implement "
            "OutcomePredictor.predict_outcome"
        )

    class _Magic(MagicController[_X]):
        def __init__(self, objects: Sequence[Object]) -> None:
            super().__init__(objects, inner_cls(objects), skill_name)

    _Magic.__name__ = f"Magic{inner_cls.__name__}"
    _Magic.__qualname__ = _Magic.__name__
    _Magic.__doc__ = f"Magic version of {inner_cls.__name__}."
    return LiftedParameterizedController(
        lifted.variables, _Magic, params_space=lifted.params_space
    )
