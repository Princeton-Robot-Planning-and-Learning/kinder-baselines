"""A trajectory sampler wrapper that logs controller and parameter info.

Because the BacktrackingRefiner may explore dead-end branches, the log
can accumulate records that are NOT part of the final plan.  After planning
succeeds, call ``extract_plan_records(plan)`` to get only the records that
correspond to the returned Plan.
"""

from dataclasses import dataclass
from typing import Any, Callable, Hashable, TypeVar

import numpy as np
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    GroundParameterizedController,
    ParameterizedController,
    Plan,
    TransitionFailure,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySampler,
    TrajectorySamplingFailure,
)
from relational_structs import GroundOperator

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


@dataclass
class SkillExecutionRecord:
    """Records one skill execution step during planning."""

    ground_operator: GroundOperator
    controller: GroundParameterizedController
    params: Any
    states: list  # states visited during this skill
    actions: list  # actions taken during this skill


class LoggingParameterizedControllerTrajectorySampler(
    TrajectorySampler[_X, _U, _S, _A]
):
    """Wraps ParameterizedControllerTrajectorySampler to log controller info.

    After a successful plan, call ``extract_plan_records(plan)`` to get the
    SkillExecutionRecords that match the final plan.
    """

    def __init__(
        self,
        controller_generator: Callable[[_A], ParameterizedController[_X, _U]],
        transition_function: Callable[[_X, _U], _X],
        state_abstractor: Callable[[_X], _S],
        max_trajectory_steps: int,
    ) -> None:
        self._controller_generator = controller_generator
        self._transition_function = transition_function
        self._state_abstractor = state_abstractor
        self._max_trajectory_steps = max_trajectory_steps
        self._all_records: list[SkillExecutionRecord] = []

    def clear(self) -> None:
        """Clear all accumulated records (call before each planning run)."""
        self._all_records.clear()

    def extract_plan_records(
        self, plan: Plan,
    ) -> list[SkillExecutionRecord]:
        """Match logged records to the final successful plan.

        Walks through Plan.states and finds the unique sequence of records
        whose states/actions tiles cover the plan exactly.
        """
        matched: list[SkillExecutionRecord] = []
        plan_idx = 0  # current position in plan.states
        search_start = 0  # only look at records created after previous match

        while plan_idx < len(plan.states) - 1:
            found = False
            for rec_i in range(search_start, len(self._all_records)):
                rec = self._all_records[rec_i]
                n_actions = len(rec.actions)
                if n_actions == 0:
                    continue
                # Check start and end state match.
                if (
                    rec.states[0] == plan.states[plan_idx]
                    and rec.states[-1] == plan.states[plan_idx + n_actions]
                ):
                    matched.append(rec)
                    plan_idx += n_actions
                    search_start = rec_i + 1
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    "Could not match all plan segments to logged records."
                )
        return matched

    def __call__(
        self,
        x: _X,
        s: _S,
        a: _A,
        ns: _S,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
        rng: np.random.Generator,
    ) -> tuple[list[_X], list[_U]]:
        """Sample a trajectory, logging controller info on success."""
        controller = self._controller_generator(a)

        x_traj: list[_X] = [x]
        u_traj: list[_U] = []

        params = controller.sample_parameters(x, rng)
        controller.reset(x, params)

        for _ in range(self._max_trajectory_steps):
            if controller.terminated():
                break
            u = controller.step()
            try:
                nx = self._transition_function(x, u)
            except TransitionFailure:
                break
            controller.observe(nx)
            x_traj.append(nx)
            u_traj.append(u)
            bpg.add_state_node(nx)
            bpg.add_action_edge(x, u, nx)
            x = nx

        final_state = x_traj[-1]
        final_abstract_state = self._state_abstractor(final_state)
        bpg.add_abstract_state_node(final_abstract_state)
        bpg.add_state_abstractor_edge(final_state, final_abstract_state)
        if final_abstract_state == ns:
            assert isinstance(a, GroundOperator)
            assert isinstance(controller, GroundParameterizedController)
            self._all_records.append(
                SkillExecutionRecord(
                    ground_operator=a,
                    controller=controller,
                    params=params,
                    states=list(x_traj),
                    actions=list(u_traj),
                )
            )
            return x_traj, u_traj

        raise TrajectorySamplingFailure()
