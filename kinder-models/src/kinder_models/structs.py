"""Common data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from relational_structs import Object

_X = TypeVar("_X")


@dataclass(frozen=True, eq=False)
class SkillCall(Generic[_X]):
    """One action that stands in for an entire skill execution.

    A planner that treats a skill as magic emits a single ``SkillCall`` where
    the skill's low-level action sequence would otherwise go. The transition
    function maps the call straight to ``predicted_state`` (the skill's option
    model), and whatever executes the plan decides how the skill is really
    carried out: teleporting a simulator, handing control to a human
    teleoperator, or running a learned policy.

    Equality is by identity and the class is hashed through ``repr`` (see
    ``prpl_utils.utils.consistent_hash``), so ``__repr__`` must be
    deterministic and must pin down every field. ``params`` may be a numpy
    array; it is rendered through ``repr`` of its tuple form for that reason.
    """

    skill_name: str
    objects: tuple[Object, ...]
    params: Any
    predicted_state: _X

    def __repr__(self) -> str:
        params = self.params
        if hasattr(params, "tolist"):
            params = tuple(params.tolist())
        object_names = tuple(o.name for o in self.objects)
        return (
            f"SkillCall(skill_name={self.skill_name!r}, objects={object_names!r}, "
            f"params={params!r}, predicted_state_hash={hash(self.predicted_state)})"
        )

    def __str__(self) -> str:
        object_names = ", ".join(o.name for o in self.objects)
        return f"{self.skill_name}({object_names})"
