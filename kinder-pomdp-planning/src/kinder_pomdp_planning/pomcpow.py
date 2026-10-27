"""POMCPOW: online tree search for POMDPs with continuous actions and observations.

Sunberg & Kochenderfer, "Online Algorithms for POMDPs with Continuous State, Action, and
Observation Spaces" (ICAPS 2018). Progressive widening limits the actions tried at each
observation node and the observations kept under each action node, and every observation
node holds the weighted states that reached it, so it carries a belief.

The solver is domain-free. A domain supplies a `POMDPModel`. New actions at a node come
from an `ActionProposer`, which defaults to the model's own sampler; passing a different
proposer (for example a learned one) changes only where the search looks.
"""

from __future__ import annotations

import bisect
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, Protocol, Sequence, TypeVar

import numpy as np

S = TypeVar("S")
A = TypeVar("A", bound=Hashable)
O = TypeVar("O", bound=Hashable)


class POMDPModel(Protocol[S, A, O]):
    """What POMCPOW needs from a domain."""

    def step(self, state: S, action: A, rng: np.random.Generator) -> tuple[S, float]:
        """Sample a next state and the reward of the transition."""

    def reward(self, state: S, action: A, next_state: S) -> float:
        """Reward of reaching `next_state` from `state` with `action`.

        Used when a simulation continues from a state drawn from a node's belief.
        """

    def observe(self, next_state: S, action: A, rng: np.random.Generator) -> O:
        """Sample an observation of `next_state`."""

    def likelihood(self, observation: O, next_state: S, action: A) -> float:
        """Density (up to a constant) of `observation` given `next_state`."""

    def is_terminal(self, state: S) -> bool:
        """Whether the episode is over in `state`."""

    def propose(
        self, state: S, tried: Sequence[A], rng: np.random.Generator
    ) -> A | None:
        """A new action to try from `state`, or None if there is none."""

    def value(self, state: S) -> float:
        """Estimated value of a leaf state."""


# (sampled state, all states at the node, actions already tried, rng) -> action or None.
ActionProposer = Callable[
    [S, Sequence[S], Sequence[A], np.random.Generator], "A | None"
]


@dataclass(frozen=True)
class POMCPOWConfig:
    """Search parameters."""

    num_sims: int = 1000
    # Stop early once this many seconds have passed (None: run all simulations).
    max_time: float | None = None
    max_depth: int = 10
    discount: float = 1.0
    exploration: float = 20.0
    # Progressive widening: a node with N visits may have up to k * N^alpha children.
    action_widening: tuple[float, float] = (2.0, 0.5)
    observation_widening: tuple[float, float] = (1.0, 0.25)
    # Random rollout steps before the model's leaf value takes over.
    rollout_depth: int = 0


@dataclass
class _ObservationNode(Generic[S, A]):
    visits: int = 0
    actions: dict = field(default_factory=dict)
    # The states that reached this node, with a running total of their weights.
    states: list = field(default_factory=list)
    cumulative: list = field(default_factory=list)
    exhausted: bool = False


@dataclass
class _ActionNode:
    visits: int = 0
    value: float = 0.0
    children: dict = field(default_factory=dict)
    # Times each child observation was generated.
    counts: dict = field(default_factory=dict)


@dataclass
class SearchStats:
    """What one call to `plan` did."""

    num_sims: int = 0
    seconds: float = 0.0
    root_actions: int = 0
    max_depth_reached: int = 0
    # Visits and value of each root action, best first.
    root_values: list = field(default_factory=list)


class POMCPOW(Generic[S, A, O]):
    """`plan(particles)` returns the best root action and search statistics."""

    def __init__(
        self,
        model: POMDPModel[S, A, O],
        config: POMCPOWConfig = POMCPOWConfig(),
        proposer: ActionProposer | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.proposer = proposer
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.root: _ObservationNode | None = None
        self._depth_reached = 0

    def plan(self, particles: Sequence[S]) -> tuple[A | None, SearchStats]:
        """Search from a belief given as equally weighted state particles.

        Returns None if no action could be proposed at the root.
        """
        start = time.perf_counter()
        config = self.config
        particles = list(particles)
        self.root = _ObservationNode(
            states=particles, cumulative=list(range(1, len(particles) + 1))
        )
        self._depth_reached = 0
        sims = 0
        while sims < config.num_sims:
            if config.max_time is not None and sims > 0:
                if time.perf_counter() - start > config.max_time:
                    break
            state = particles[int(self.rng.integers(len(particles)))]
            self._simulate(state, self.root, config.max_depth)
            sims += 1
        stats = SearchStats(
            num_sims=sims,
            seconds=time.perf_counter() - start,
            root_actions=len(self.root.actions),
            max_depth_reached=self._depth_reached,
            root_values=sorted(
                ((a, n.visits, n.value) for a, n in self.root.actions.items()),
                key=lambda item: (item[1], item[2]),
                reverse=True,
            ),
        )
        if not self.root.actions:
            return None, stats
        # The most visited action: values of rarely tried continuous ones are noisy.
        best = max(
            self.root.actions.items(), key=lambda kv: (kv[1].visits, kv[1].value)
        )
        return best[0], stats

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _widened(visits: int, size: int, widening: tuple[float, float]) -> bool:
        k, alpha = widening
        return size <= k * max(visits, 1) ** alpha

    def _propose(self, state: S, node: _ObservationNode) -> A | None:
        tried = list(node.actions)
        if self.proposer is not None:
            return self.proposer(state, node.states, tried, self.rng)
        return self.model.propose(state, tried, self.rng)

    def _select_action(self, state: S, node: _ObservationNode):
        if not node.exhausted and self._widened(
            node.visits, len(node.actions), self.config.action_widening
        ):
            action = self._propose(state, node)
            if action is None:
                node.exhausted = True
            elif action not in node.actions:
                node.actions[action] = _ActionNode()
        if not node.actions:
            return None
        log_visits = math.log(max(node.visits, 1))

        def ucb(item) -> float:
            child = item[1]
            if child.visits == 0:
                return math.inf
            bonus = math.sqrt(log_visits / child.visits)
            return child.value + self.config.exploration * bonus

        return max(node.actions.items(), key=ucb)

    def _rollout(self, state: S, depth: int) -> float:
        model, config = self.model, self.config
        total, scale = 0.0, 1.0
        steps = min(depth, config.rollout_depth)
        for _ in range(steps):
            if model.is_terminal(state):
                return total
            action = model.propose(state, [], self.rng)
            if action is None:
                break
            state, reward = model.step(state, action, self.rng)
            total += scale * reward
            scale *= config.discount
        if model.is_terminal(state):
            return total
        return total + scale * model.value(state)

    def _simulate(self, state: S, node: _ObservationNode, depth: int) -> float:
        model, config = self.model, self.config
        if model.is_terminal(state):
            return 0.0
        if depth == 0:
            return model.value(state)
        self._depth_reached = max(self._depth_reached, config.max_depth - depth + 1)
        selected = self._select_action(state, node)
        if selected is None:
            # No action is possible here: the state is a dead end.
            return model.value(state)
        action, child = selected
        next_state, reward = model.step(state, action, self.rng)
        observation = model.observe(next_state, action, self.rng)
        if self._widened(
            child.visits, len(child.children), config.observation_widening
        ):
            child.counts[observation] = child.counts.get(observation, 0) + 1
        else:
            # Reuse an observation, picked by how often it was generated.
            seen = list(child.counts)
            counts = np.array([child.counts[o] for o in seen], dtype=float)
            observation = seen[self.rng.choice(len(seen), p=counts / counts.sum())]
        new = observation not in child.children
        if new:
            child.children[observation] = _ObservationNode()
        below = child.children[observation]
        weight = model.likelihood(observation, next_state, action)
        below.states.append(next_state)
        below.cumulative.append(
            (below.cumulative[-1] if below.cumulative else 0.0) + weight
        )
        if new:
            total = reward + config.discount * self._rollout(next_state, depth - 1)
        else:
            # Continue from a state drawn from the node's belief.
            total_weight = below.cumulative[-1]
            if total_weight > 0:
                pick = bisect.bisect_right(
                    below.cumulative, self.rng.random() * total_weight
                )
            else:
                pick = int(self.rng.integers(len(below.states)))
            next_state = below.states[min(pick, len(below.states) - 1)]
            reward = model.reward(state, action, next_state)
            total = reward + config.discount * self._simulate(
                next_state, below, depth - 1
            )
        node.visits += 1
        child.visits += 1
        child.value += (total - child.value) / child.visits
        return total
