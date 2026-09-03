"""Run the SESAME refiner on an externally supplied abstract plan (a "skeleton").

A higher-level planner (e.g. one reasoning about which objects go to which shelf
section) can decide the operator sequence and the key continuous parameters, and hand
just that intermediate representation over: the skeleton is injected in place of the
abstract search, and refinement fills in the motions. Continuous parameters ride along
separately, through the env models' fixed per-object parameters (``place_params``,
``move_params``, ``grasp_params``), so with everything fixed a single sample per step
suffices and planning is fast and deterministic.

The skeleton is specified in a plain, serializable form — ``(operator name, object
names)`` pairs — so it can cross a process/repo boundary as JSON.
"""

from typing import Hashable, Sequence, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    Plan,
    PlanningProblem,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from relational_structs import GroundOperator

_O = TypeVar("_O", bound=Hashable)
_X = TypeVar("_X", bound=Hashable)
_U = TypeVar("_U", bound=Hashable)

#: A serializable skeleton: ordered (operator name, argument object names) pairs.
SkeletonSpec = Sequence[tuple[str, Sequence[str]]]


def ground_skeleton(
    env_models: SesameModels,
    initial_state: Hashable,
    skeleton_spec: SkeletonSpec,
) -> list[GroundOperator]:
    """Ground a serializable skeleton against the env models and initial state.

    Operator names must match the env models' operators; object names must exist in
    the initial (object-centric) state.
    """
    operators_by_name = {op.name: op for op in env_models.operators}
    ground_ops: list[GroundOperator] = []
    for op_name, object_names in skeleton_spec:
        if op_name not in operators_by_name:
            raise ValueError(
                f"Unknown operator {op_name!r}; expected one of "
                f"{sorted(operators_by_name)}"
            )
        objects = tuple(
            initial_state.get_object_from_name(name)  # type: ignore[attr-defined]
            for name in object_names
        )
        ground_ops.append(operators_by_name[op_name].ground(objects))
    return ground_ops


class InjectedAbstractPlanGenerator(AbstractPlanGenerator):
    """Yields exactly one abstract plan: the injected skeleton.

    The abstract state chain is derived by applying each ground operator's
    add/delete effects from the initial abstract state. Preconditions are checked
    along the way so a malformed skeleton fails loudly at injection time rather
    than as an unexplainable refinement failure.
    """

    def __init__(self, ground_ops: Sequence[GroundOperator], seed: int = 0) -> None:
        super().__init__(lambda s: (), seed)
        self._ground_ops = list(ground_ops)

    def __call__(self, x0, s0, goal, timeout, bpg):
        del x0, timeout, bpg
        states = [s0]
        atoms = set(s0.atoms)
        for ground_op in self._ground_ops:
            missing = set(ground_op.preconditions) - atoms
            if missing:
                raise ValueError(
                    f"Injected skeleton is inconsistent: {ground_op.short_str} is "
                    f"missing preconditions {sorted(map(str, missing))}"
                )
            atoms = (atoms - set(ground_op.delete_effects)) | set(
                ground_op.add_effects
            )
            states.append(RelationalAbstractState(set(atoms), set(s0.objects)))
        if not goal.check_abstract_state(states[-1]):
            raise ValueError("Injected skeleton does not reach the goal")
        yield states, list(self._ground_ops)


def run_injected_sesame(
    env_models: SesameModels,
    initial_state,
    skeleton_spec: SkeletonSpec,
    *,
    seed: int = 0,
    samples_per_step: int = 1,
    max_skill_horizon: int = 1000,
    timeout: float = 120.0,
) -> tuple[Plan | None, BilevelPlanningGraph]:
    """Refine the injected skeleton with SESAME (no abstract search).

    Mirrors ``bilevel_planning.sesame.run_sesame`` but replaces the heuristic-search
    abstract plan generator with the injected skeleton. With every skill's continuous
    parameters fixed through the env models, ``samples_per_step=1`` is enough.
    """
    problem = PlanningProblem(
        env_models.state_space,
        env_models.action_space,
        initial_state,
        env_models.transition_fn,
        env_models.goal_deriver(initial_state),
    )
    trajectory_sampler = ParameterizedControllerTrajectorySampler(
        controller_generator=RelationalControllerGenerator(env_models.skills),
        transition_function=env_models.transition_fn,
        state_abstractor=env_models.state_abstractor,
        max_trajectory_steps=max_skill_horizon,
    )
    ground_ops = ground_skeleton(env_models, initial_state, skeleton_spec)
    abstract_plan_generator = InjectedAbstractPlanGenerator(ground_ops, seed=seed)
    abstract_successor_fn = RelationalAbstractSuccessorGenerator(
        env_models.operators,
        precomputed_ground_operators=env_models.ground_operators,
    )
    planner = SesamePlanner(
        abstract_plan_generator,
        trajectory_sampler,
        1,  # max_abstract_plans: the skeleton is the plan
        samples_per_step,
        abstract_successor_fn,
        env_models.state_abstractor,
        seed=seed,
    )
    return planner.run(problem, timeout=timeout)
