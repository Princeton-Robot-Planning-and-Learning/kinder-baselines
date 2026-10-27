"""Tests for pomcpow.py on a small continuous POMDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from kinder_pomdp_planning.pomcpow import POMCPOW, POMCPOWConfig


@dataclass(frozen=True)
class _State:
    position: float
    target: float  # hidden
    steps: int = 0
    last_reward: float = 0.0


class _GuessModel:
    """Move to a hidden target in [0, 1]; the observation is the noisy target."""

    def step(self, state, action, rng):
        """Move to `action`; the reward is minus the distance to the target."""
        del rng
        reward = -abs(action - state.target)
        nxt = _State(action, state.target, state.steps + 1, reward)
        return nxt, reward

    def reward(self, state, action, next_state):
        """The reward stored on the next state."""
        del state, action
        return next_state.last_reward

    def observe(self, next_state, action, rng):
        """The target, with noise."""
        del action
        return round(next_state.target + float(rng.normal(0.0, 0.05)), 3)

    def likelihood(self, observation, next_state, action):
        """Gaussian density of the observed target."""
        del action
        return float(np.exp(-0.5 * ((observation - next_state.target) / 0.05) ** 2))

    def is_terminal(self, state):
        """Two moves end the episode."""
        return state.steps >= 2

    def propose(self, state, tried: Sequence[float], rng):
        """A uniform random position."""
        del state, tried
        return round(float(rng.uniform(0.0, 1.0)), 4)

    def value(self, state):
        """No leaf estimate."""
        del state
        return 0.0


def test_pomcpow_finds_the_hidden_target():
    """With a known target, the best first action lands near it."""
    particles = [_State(0.0, 0.7)] * 20
    solver = POMCPOW(
        _GuessModel(), POMCPOWConfig(num_sims=2000, max_depth=2, exploration=0.5)
    )
    action, stats = solver.plan(particles)
    assert action is not None and abs(action - 0.7) < 0.1
    assert stats.num_sims == 2000
    assert stats.root_actions == len(solver.root.actions)
    assert stats.root_values[0][1] >= stats.root_values[-1][1]


def test_pomcpow_proposer_controls_the_actions():
    """A custom proposer replaces the model's sampler."""
    calls = []

    def proposer(state, states, tried, rng):
        del rng
        calls.append((state, len(states), len(tried)))
        return 0.5 if not tried else None

    particles = [_State(0.0, 0.3), _State(0.0, 0.9)]
    solver = POMCPOW(_GuessModel(), POMCPOWConfig(num_sims=50, max_depth=2), proposer)
    action, stats = solver.plan(particles)
    assert action == 0.5 and stats.root_actions == 1
    assert calls and calls[0][1] == 2


def test_pomcpow_time_budget_and_dead_ends():
    """The time budget stops the search, and a root with no actions returns None."""
    solver = POMCPOW(_GuessModel(), POMCPOWConfig(num_sims=10**9, max_time=0.2))
    action, stats = solver.plan([_State(0.0, 0.5)])
    assert action is not None and stats.seconds < 2.0
    stuck = POMCPOW(_GuessModel(), proposer=lambda *args: None)
    action, stats = stuck.plan([_State(0.0, 0.5)])
    assert action is None and stats.root_actions == 0
