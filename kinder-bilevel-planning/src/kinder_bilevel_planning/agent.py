"""A general interface for an agent that runs bilevel planning."""

from collections.abc import Hashable
from typing import Any, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import (
    PlanningProblem,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from prpl_utils.planning_agent import PlanningAgent

_O = TypeVar("_O", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)
_X = TypeVar("_X", bound=Hashable)


class AgentFailure(BaseException):
    """Raised when the agent fails to find a plan."""


class BilevelPlanningAgent(PlanningAgent[_O, _U, _X]):
    """A general interface for an agent that runs bilevel planning.

    The full state-action trajectory is computed once in :meth:`reset` and
    cached. Callers using the per-action :meth:`Agent.step` API pop actions one
    at a time; callers using the per-trajectory :meth:`plan` API receive the
    whole (state, action) sequence in one call. Mixing the two within an
    episode is unsupported.
    """

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        planning_timeout: float = 30.0,
    ) -> None:
        self._env_models = env_models
        # `_current_plan` is the remaining action queue consumed by `_get_action`.
        # `_planned_states` / `_planned_actions` are the immutable record of what
        # the planner returned in the last `reset`, used by `plan` to expose the
        # full state-action trajectory.
        self._current_plan: list[_U] | None = None
        self._planned_states: list[_X] = []
        self._planned_actions: list[_U] = []
        self._plan_consumed: bool = False
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._planning_timeout = planning_timeout
        self._seed = seed
        super().__init__(seed)

    def reset(
        self,
        obs: _O,
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        self._planned_states, self._planned_actions = self._run_planning()
        self._current_plan = list(self._planned_actions)
        self._plan_consumed = False

    def _get_action(self) -> _U:
        if not self._current_plan:
            raise AgentFailure("Ran out of planning steps, failure!")
        return self._current_plan.pop(0)

    def plan(self) -> list[tuple[_X, _U]]:
        """Return the full state-action trajectory from the last reset.

        The bilevel planner produces a single plan in :meth:`reset` and does not
        replan, so the second call within an episode raises :class:`AgentFailure`
        — matching the per-action exhaust behaviour of :meth:`Agent.step`.
        """
        if self._plan_consumed:
            raise AgentFailure("Ran out of planning steps, failure!")
        self._plan_consumed = True
        return list(zip(self._planned_states[:-1], self._planned_actions))

    def _run_planning(self) -> tuple[list[_X], list[_U]]:
        # Create planning problem.
        initial_state = self._env_models.observation_to_state(self._last_observation)
        goal = self._env_models.goal_deriver(initial_state)
        problem = PlanningProblem(
            self._env_models.state_space,
            self._env_models.action_space,
            initial_state,
            self._env_models.transition_fn,
            goal,
        )

        # Create the sampler.
        trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # Create the abstract plan generator.
        abstract_plan_generator: AbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                self._env_models.types,
                self._env_models.predicates,
                self._env_models.operators,
                self._heuristic_name,
                seed=self._seed,
                precomputed_ground_operators=self._env_models.ground_operators,
            )
        )

        # Create the abstract successor function (not really used).
        abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators,
            precomputed_ground_operators=self._env_models.ground_operators,
        )

        # Finish the planner.
        planner = SesamePlanner(
            abstract_plan_generator,
            trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

        # Run the planner.
        plan, _ = planner.run(problem, timeout=self._planning_timeout)
        if plan is None:
            raise AgentFailure("No plan found")

        return plan.states, plan.actions
