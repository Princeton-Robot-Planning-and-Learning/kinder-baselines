"""Tests for the PuckPushing POMDP model, belief and runner."""

import math

import numpy as np
from kinder.envs.puckpushing_utils import THREE_ROUTES, PuckWorld, Push

from kinder_pomdp_planning.pomcpow import POMCPOWConfig
from kinder_pomdp_planning.puckpushing.belief import FrictionBelief
from kinder_pomdp_planning.puckpushing.model import (
    GOAL,
    HOLE,
    ONGOING,
    PuckPushingModel,
)
from kinder_pomdp_planning.puckpushing.run import make_env, run_episode

FRICTIONS = (0.03, 0.07, 0.15)


def _model() -> PuckPushingModel:
    return PuckPushingModel(PuckWorld(THREE_ROUTES))


def test_transition_rewards_and_outcomes():
    """Pushes cost a push plus driving; the goal pays and a hole costs."""
    model = _model()
    world = model.world
    rewards = model.rewards
    start = model.initial_state(
        np.array(THREE_ROUTES.puck_start), np.array(THREE_ROUTES.robot_start), FRICTIONS
    )
    push = Push(math.pi / 2, 0.5)
    nxt = model.transition(start, push, push.angle, push.speed)
    expected = world.slide(
        world.release_point(start.puck_array, push.angle), 0.5, push.angle, FRICTIONS
    )
    assert np.allclose(nxt.puck, expected.stop, atol=1e-3)
    assert nxt.outcome == ONGOING and nxt.pushes == 1
    assert nxt.last_reward < -rewards.push_cost  # it also drove
    # Into the felt hole on slippery felt.
    near_hole = model.initial_state(
        np.array([1.6, 1.1]), np.array([1.0, 1.1]), (0.03, 0.07, 0.05)
    )
    fell = model.transition(near_hole, Push(0.0, 1.0), 0.0, 1.0)
    assert fell.outcome == HOLE and model.is_terminal(fell)
    assert fell.last_reward < -rewards.hole_penalty
    # Up into the goal from just below it.
    below_goal = model.initial_state(
        np.array([3.9, 1.8]), np.array([3.9, 1.2]), FRICTIONS
    )
    speed = math.sqrt(2 * FRICTIONS[1] * 9.81 * 0.1)
    won = model.transition(below_goal, Push(math.pi / 2, speed), math.pi / 2, speed)
    assert won.outcome == GOAL and won.last_reward > 0


def test_observation_likelihood_prefers_the_true_state():
    """The measurement density peaks at the state that produced it."""
    model = _model()
    rng = np.random.default_rng(0)
    start = model.initial_state(
        np.array(THREE_ROUTES.puck_start), np.array(THREE_ROUTES.robot_start), FRICTIONS
    )
    push = Push(math.pi / 2, 0.6)
    nxt = model.transition(start, push, push.angle, push.speed)
    other = model.transition(start, push, push.angle, push.speed * 1.3)
    obs = model.observe(nxt, push, rng)
    assert model.likelihood(obs, nxt, push) > model.likelihood(obs, other, push)
    assert model.likelihood((*obs[:3], True), nxt, push) == 0.0


def test_proposals_are_feasible():
    """The default proposer only returns pushes the robot can make."""
    model = _model()
    rng = np.random.default_rng(0)
    state = model.initial_state(
        np.array(THREE_ROUTES.puck_start), np.array(THREE_ROUTES.robot_start), FRICTIONS
    )
    for _ in range(20):
        push = model.propose(state, [], rng)
        assert push is not None
        assert model.world.push_feasible(state.puck_array, push.angle)


def test_greedy_rollout_reaches_the_goal_with_known_frictions():
    """With the true frictions, the rollout policy gets the puck to the goal."""
    model = _model()
    model.rollout_frictions = FRICTIONS
    state = model.initial_state(
        np.array(THREE_ROUTES.puck_start), np.array(THREE_ROUTES.robot_start), FRICTIONS
    )
    assert model.value(state) > 0
    path = model.path_to_goal(state.puck_array)
    assert np.allclose(path[-1], model.world.goal)
    # The rollout paths avoid the slot, which the robot cannot push from.
    assert not any(2.1 < p[0] < 3.1 and 2.0 < p[1] < 2.4 for p in path)


def test_belief_concentrates_on_the_true_frictions():
    """Pushes on laminate and wood narrow those two coefficients."""
    model = _model()
    world = model.world
    rng = np.random.default_rng(0)
    belief = FrictionBelief.from_prior(rng, 400)
    prior_std = belief.std.copy()
    true = (0.04, 0.06, 0.15)
    puck = np.array([1.0, 2.0])  # wood
    for angle, speed in [(math.pi / 2, 1.0), (-math.pi / 2, 0.4)]:
        result = world.slide(world.release_point(puck, angle), speed, angle, true)
        belief.update(
            world, puck, angle, speed, result.stop, result.hole is not None, rng
        )
        puck = result.stop
    assert belief.std[0] < prior_std[0] and belief.std[1] < 0.5 * prior_std[1]
    assert abs(belief.mean[1] - true[1]) < 0.01


def test_run_episode_smoke():
    """A short episode runs and logs every push."""
    env = make_env("2d", "threeroutes")
    log = run_episode(
        env,
        "2d",
        seed=0,
        config=POMCPOWConfig(num_sims=20, max_depth=3),
        num_particles=50,
        max_pushes=2,
    )
    assert 1 <= len(log.steps) <= 2
    assert log.outcome in ("goal", "hole", "timeout", "stuck")
    assert len(log.true_frictions) == 3
