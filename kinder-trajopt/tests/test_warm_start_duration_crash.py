"""Regression test for warm-start crash when trajectory duration ≈ 1.

After many MPC warm-start iterations the trajectory duration shrinks toward 1.
Floating-point arithmetic can produce duration = 1 + ε where ε < 1e-10.  The
warm-start guard (`duration > 1`) passes, but `get_sub_trajectory(1, 1+ε)`
filters every segment (the 1e-10 tolerance in _ConcatTrajectory) and returns an
empty trajectory with duration 0.  Calling that trajectory at t=0.0 then raises
``ValueError: Time 0.0 exceeds duration 0``.
"""

import numpy as np
from gymnasium.spaces import Box
from prpl_utils.structs import Image
from prpl_utils.trajopt.predictive_sampling import (
    PredictiveSamplingHyperparameters,
    PredictiveSamplingSolver,
)
from prpl_utils.trajopt.trajectory import point_sequence_to_trajectory
from prpl_utils.trajopt.trajopt_problem import (
    TrajOptAction,
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)


class _TrivialProblem(TrajOptProblem):
    """Minimal problem: identity dynamics, zero cost."""

    def __init__(self, horizon: int, state_dim: int = 2, action_dim: int = 2):
        self._horizon = horizon
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def state_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=(self._state_dim,), dtype=np.float64)

    @property
    def action_space(self) -> Box:
        return Box(-1.0, 1.0, shape=(self._action_dim,), dtype=np.float32)

    @property
    def initial_state(self) -> TrajOptState:
        return np.zeros(self._state_dim)

    def get_next_state(
        self, state: TrajOptState, action: TrajOptAction
    ) -> TrajOptState:
        return state + action.astype(state.dtype)

    def get_traj_cost(self, traj: TrajOptTraj) -> float:
        return 0.0

    def render_state(self, state: TrajOptState) -> Image:
        return np.zeros((1, 1, 3), dtype=np.uint8)


def test_warm_start_crash_duration_near_one():
    """Injecting a last_solution with duration = 1 + ε triggers the crash.

    The solver's warm-start guard passes (duration > 1), but get_sub_trajectory(1, 1+ε)
    returns an empty trajectory (duration 0) because every segment is smaller than the
    1e-10 tolerance.
    """
    horizon = 100
    num_cp = 10
    action_dim = 2
    problem = _TrivialProblem(horizon=horizon, action_dim=action_dim)

    config = PredictiveSamplingHyperparameters(
        num_rollouts=5,
        noise_scale=0.1,
        num_control_points=num_cp,
    )
    solver = PredictiveSamplingSolver(seed=0, config=config, warm_start=True)
    solver.reset(problem)

    # Build a trajectory whose duration is just barely above 1.
    # With num_cp=10, dt = duration/(10-1). Each segment has duration dt.
    # get_sub_trajectory(1, duration) will produce segments that are each
    # ~epsilon/9 wide — all below the 1e-10 filter threshold.
    epsilon = 1e-14
    duration = 1.0 + epsilon
    dt = duration / (num_cp - 1)
    points = [np.zeros(action_dim) for _ in range(num_cp)]
    fake_solution = point_sequence_to_trajectory(points, dt=dt)
    assert fake_solution.duration > 1

    # Inject the crafted trajectory as the solver's last solution.
    solver._last_solution = fake_solution  # pylint: disable=protected-access

    state = problem.initial_state
    # This call warm-starts from the injected solution and should crash
    # with: ValueError: Time 0.0 exceeds duration 0
    solver.solve(initial_state=state, horizon=horizon)
