"""PuckPushing as a POMDP for POMCPOW.

The hidden part of the state is one friction coefficient per floor surface. Everything
else (puck and robot positions, the map, the strike physics) is known. A push is a
continuous (heading, speed) action. Transitions use the analytic slide model of
`kinder.envs.puckpushing_utils`, with the same execution noise as the envs, and the
observation is the noisy release speed and stop position.

Rewards: each push costs `push_cost` plus `drive_cost` per meter the robot drives to its
push station. Stopping in the goal earns `goal_reward`; passing over a hole costs
`hole_penalty` and ends the episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Sequence

import numpy as np
from kinder.envs.puckpushing_utils import DEFAULT_FRICTION_PRIOR, PuckWorld, Push
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ONGOING, GOAL, HOLE, STUCK = 0, 1, 2, 3


@dataclass(frozen=True)
class PuckState:
    """A POMDP state.

    Positions are rounded so states hash stably.
    """

    puck: tuple[float, float]
    robot: tuple[float, float]
    frictions: tuple[float, ...]
    outcome: int = ONGOING
    pushes: int = 0
    # Speed the puck actually left the paddle with, on the push that led here.
    release_speed: float = 0.0
    # Reward of the transition that led here.
    last_reward: float = 0.0

    @property
    def puck_array(self) -> NDArray[np.float64]:
        """The puck position as an array."""
        return np.asarray(self.puck, dtype=np.float64)

    @property
    def robot_array(self) -> NDArray[np.float64]:
        """The robot position as an array."""
        return np.asarray(self.robot, dtype=np.float64)


# (release speed, stop x, stop y, fell in a hole), rounded to keep them hashable.
PuckObservation = tuple[float, float, float, bool]


@dataclass(frozen=True)
class PuckRewards:
    """Reward parameters."""

    push_cost: float = 1.0
    drive_cost: float = 0.5
    goal_reward: float = 20.0
    hole_penalty: float = 50.0
    # Value of a state from which no push is possible (e.g. stopped inside the slot).
    stuck_penalty: float = 30.0


@dataclass
class PuckPushingModel:
    """The POMCPOW model of PuckPushing."""

    world: PuckWorld
    rewards: PuckRewards = field(default_factory=PuckRewards)
    friction_prior: tuple[tuple[float, float], ...] = DEFAULT_FRICTION_PRIOR
    max_pushes: int = 15
    # Proposal attempts before declaring that no push is possible.
    max_proposal_tries: int = 64
    # Grid resolution of the leaf-value distance field.
    value_resolution: float = 0.1
    # Greedy rollouts keep this far from holes, beyond their radius.
    hole_margin: float = 0.05
    rollout_pushes: int = 6
    # Path cells ahead used to check that a rollout can push on from where it stops.
    onward_lookahead: int = 4
    # Frictions the rollout policy assumes when choosing speeds (None: prior mean).
    rollout_frictions: tuple[float, ...] | None = None

    # ------------------------------------------------------------------ states

    def initial_state(
        self,
        puck: NDArray[np.float64],
        robot: NDArray[np.float64],
        frictions: Sequence[float],
        pushes: int = 0,
    ) -> PuckState:
        """A state with the puck and robot at rest."""
        return PuckState(
            puck=_round_point(puck),
            robot=_round_point(robot),
            frictions=tuple(float(f) for f in frictions),
            pushes=pushes,
        )

    def is_terminal(self, state: PuckState) -> bool:
        """Goal, hole, stuck, or out of pushes."""
        return state.outcome != ONGOING or state.pushes >= self.max_pushes

    # ------------------------------------------------------------------ dynamics

    def step(
        self, state: PuckState, action: Push, rng: np.random.Generator
    ) -> tuple[PuckState, float]:
        """Sample the push's execution noise and slide the puck."""
        phys = self.world.physics
        angle = action.angle + float(rng.normal(0.0, phys.angle_noise))
        speed = max(
            action.speed * (1.0 + float(rng.normal(0.0, phys.speed_noise))), 0.0
        )
        next_state = self.transition(state, action, angle, speed)
        return next_state, next_state.last_reward

    def transition(
        self, state: PuckState, action: Push, angle: float, speed: float
    ) -> PuckState:
        """The deterministic result of executing `action` as (angle, speed)."""
        world, rewards = self.world, self.rewards
        puck = state.puck_array
        if not world.push_feasible(puck, action.angle):
            # The robot cannot make this push; it costs a push and changes nothing.
            return replace(
                state,
                pushes=state.pushes + 1,
                release_speed=0.0,
                last_reward=-rewards.push_cost,
            )
        station = world.push_station(puck, angle)
        drive = world.drive_distance(world.nav_node_of(state.robot_array), station)
        if not math.isfinite(drive):
            drive = 0.0
        result = world.slide(
            world.release_point(puck, angle), speed, angle, state.frictions
        )
        direction = np.array([math.cos(angle), math.sin(angle)])
        robot = station + world.physics.accel_distance * direction
        reward = -rewards.push_cost - rewards.drive_cost * drive
        if result.hole is not None:
            outcome = HOLE
            reward -= rewards.hole_penalty
        elif world.in_goal(result.stop):
            outcome = GOAL
            reward += rewards.goal_reward
        else:
            outcome = ONGOING
        return PuckState(
            puck=_round_point(result.stop),
            robot=_round_point(robot),
            frictions=state.frictions,
            outcome=outcome,
            pushes=state.pushes + 1,
            release_speed=round(speed, 4),
            last_reward=reward,
        )

    def reward(self, state: PuckState, action: Push, next_state: PuckState) -> float:
        """The reward stored on `next_state` by the transition that made it."""
        del state, action
        return next_state.last_reward

    # ------------------------------------------------------------------ observations

    def observe(
        self, next_state: PuckState, action: Push, rng: np.random.Generator
    ) -> PuckObservation:
        """Noisy release speed and stop position."""
        del action
        phys = self.world.physics
        speed = next_state.release_speed + float(rng.normal(0.0, phys.speed_obs_noise))
        stop = next_state.puck_array + rng.normal(0.0, phys.position_obs_noise, size=2)
        return (
            round(max(speed, 0.0), 4),
            round(float(stop[0]), 4),
            round(float(stop[1]), 4),
            next_state.outcome == HOLE,
        )

    def likelihood(
        self, observation: PuckObservation, next_state: PuckState, action: Push
    ) -> float:
        """Gaussian density of the measurements; the hole flag must match."""
        del action
        phys = self.world.physics
        if observation[3] != (next_state.outcome == HOLE):
            return 0.0
        speed_err = (observation[0] - next_state.release_speed) / phys.speed_obs_noise
        dx = (observation[1] - next_state.puck[0]) / phys.position_obs_noise
        dy = (observation[2] - next_state.puck[1]) / phys.position_obs_noise
        return math.exp(-0.5 * (speed_err**2 + dx**2 + dy**2))

    # ------------------------------------------------------------------ actions

    def propose(
        self, state: PuckState, tried: Sequence[Push], rng: np.random.Generator
    ) -> Push | None:
        """A uniformly random feasible push (the plain POMCPOW proposal)."""
        del tried
        phys = self.world.physics
        puck = state.puck_array
        for _ in range(self.max_proposal_tries):
            angle = float(rng.uniform(-math.pi, math.pi))
            if self.world.push_feasible(puck, angle):
                speed = float(rng.uniform(phys.min_speed, phys.max_speed))
                return Push(round(angle, 4), round(speed, 4))
        return None

    # ------------------------------------------------------------------ leaf values

    @cached_property
    def _goal_paths(self) -> tuple[NDArray[np.float64], NDArray[np.int64], int, int]:
        """Shortest puck paths to the goal on a grid around walls and holes, through
        cells the robot can push from (so not through the slot).

        Returns the distance to the goal per cell, each cell's next cell toward the
        goal, and the grid size (nx, ny). Cells are indexed j * nx + i.
        """
        world, res = self.world, self.value_resolution
        margin = world.physics.hole_radius + self.hole_margin
        nx = int(round(world.puck_map.width / res))
        ny = int(round(world.puck_map.height / res))
        free = np.zeros((ny, nx), dtype=bool)
        for j in range(ny):
            for i in range(nx):
                center = np.array([(i + 0.5) * res, (j + 0.5) * res])
                near_hole = len(world.holes) > 0 and bool(
                    np.any(np.linalg.norm(world.holes - center, axis=1) <= margin)
                )
                free[j, i] = (
                    world.puck_position_free(center)
                    and not near_hole
                    and self._pushable(center)
                )
        rows, cols, lengths = [], [], []
        for j, i in zip(*np.nonzero(free)):
            for dj, di in ((0, 1), (1, 0), (1, 1), (1, -1)):
                jj, ii = j + dj, i + di
                if not (0 <= jj < ny and 0 <= ii < nx and free[jj, ii]):
                    continue
                if dj and di and not (free[j, ii] and free[jj, i]):
                    continue
                length = res * math.hypot(dj, di)
                rows += [j * nx + i, jj * nx + ii]
                cols += [jj * nx + ii, j * nx + i]
                lengths += [length, length]
        goal_i = min(int(world.goal[0] // res), nx - 1)
        goal_j = min(int(world.goal[1] // res), ny - 1)
        graph = csr_matrix((lengths, (rows, cols)), shape=(nx * ny, nx * ny))
        dist, pred = dijkstra(  # type: ignore[call-overload]
            graph, indices=goal_j * nx + goal_i, return_predecessors=True
        )
        return dist, pred, nx, ny

    def _pushable(self, point: NDArray[np.float64], num_angles: int = 16) -> bool:
        """Whether some push is feasible from `point`, on evenly spaced headings."""
        angles = np.linspace(-math.pi, math.pi, num_angles, endpoint=False)
        return any(self.world.push_feasible(point, float(a)) for a in angles)

    def path_to_goal(self, point: NDArray[np.float64]) -> list[NDArray[np.float64]]:
        """Grid points of a shortest puck path from `point` to the goal (empty if none),
        ending at the goal itself."""
        dist, pred, nx, ny = self._goal_paths
        res = self.value_resolution
        i = min(max(int(point[0] // res), 0), nx - 1)
        j = min(max(int(point[1] // res), 0), ny - 1)
        cell = j * nx + i
        if not math.isfinite(dist[cell]):
            return []
        path = []
        while cell >= 0:
            path.append(np.array([(cell % nx + 0.5) * res, (cell // nx + 0.5) * res]))
            cell = int(pred[cell])
        path[-1] = self.world.goal.copy()
        return path

    def speed_to_travel(
        self,
        puck: NDArray[np.float64],
        angle: float,
        distance: float,
        frictions: Sequence[float],
        iterations: int = 14,
    ) -> float | None:
        """Push speed that moves the puck `distance` along a clear straight line, or
        None if even the top speed falls short."""
        phys, world = self.world.physics, self.world
        release = world.release_point(puck, angle)
        target = distance - phys.accel_distance
        if target <= 0:
            return phys.min_speed
        if world.slide(release, phys.max_speed, angle, frictions).distance < target:
            return None
        low, high = phys.min_speed, phys.max_speed
        for _ in range(iterations):
            mid = (low + high) / 2
            if world.slide(release, mid, angle, frictions).distance < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def greedy_push(
        self, puck: NDArray[np.float64], frictions: Sequence[float]
    ) -> Push | None:
        """Push toward the farthest point on the shortest path to the goal that the puck
        can reach in a straight line, at the speed that stops it there.

        Unless the point is the goal, the robot must be able to push on from it along
        the path, so the rollout does not park the puck in a dead end.
        """
        world = self.world
        path = self.path_to_goal(puck)
        clearance = world.physics.hole_radius + self.hole_margin
        for k in range(len(path) - 1, -1, -1):
            target = path[k]
            offset = target - puck
            distance = float(np.linalg.norm(offset))
            if distance < 1e-6:
                continue
            if not world.puck_path_clear(puck, target, clearance):
                continue
            angle = math.atan2(offset[1], offset[0])
            if not world.push_feasible(puck, angle):
                continue
            if k < len(path) - 1:
                onward = path[min(k + self.onward_lookahead, len(path) - 1)] - target
                if not world.push_feasible(target, math.atan2(onward[1], onward[0])):
                    continue
            speed = self.speed_to_travel(puck, angle, distance, frictions)
            if speed is not None:
                return Push(angle, speed)
        return None

    def value(self, state: PuckState) -> float:
        """Return of a greedy rollout from `state`.

        Each push heads for the farthest point on the shortest path to the goal that the
        puck can reach in a straight line. Speeds are chosen with `rollout_frictions`
        (the belief mean), while the slides use the state's own frictions, so the
        estimate pays for misjudging them. A rollout that has not reached the goal after
        `rollout_pushes` gets no goal reward.
        """
        rewards = self.rewards
        if state.outcome in (GOAL, HOLE):
            return 0.0
        if state.outcome == STUCK:
            return -rewards.stuck_penalty
        frictions = self.rollout_frictions or tuple(
            (lo + hi) / 2 for lo, hi in self.friction_prior
        )
        total = 0.0
        current = state
        steps = min(self.rollout_pushes, self.max_pushes - state.pushes)
        for _ in range(steps):
            push = self.greedy_push(current.puck_array, frictions)
            if push is None:
                return total - rewards.stuck_penalty
            current = self.transition(current, push, push.angle, push.speed)
            total += current.last_reward
            if current.outcome != ONGOING:
                return total
        return total


def _round_point(point: NDArray[np.float64]) -> tuple[float, float]:
    return (round(float(point[0]), 4), round(float(point[1]), 4))
