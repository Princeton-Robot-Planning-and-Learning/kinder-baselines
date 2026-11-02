"""Particle belief over the surface friction coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from kinder.envs.puckpushing_utils import (
    DEFAULT_FRICTION_PRIOR,
    BoxFrictionPrior,
    FrictionPrior,
    PuckWorld,
)
from numpy.typing import NDArray

from kinder_pomdp_planning.puckpushing.model import PuckPushingModel, PuckState

# Weight factor for a particle whose predicted hole outcome disagrees with the
# observation: unlikely, but heading noise can make it happen.
_HOLE_MISMATCH = 0.05


@dataclass
class FrictionBelief:
    """Weighted samples of the friction vector, one coefficient per surface."""

    particles: NDArray[np.float64]  # (N, num_surfaces)
    prior: FrictionPrior | BoxFrictionPrior = DEFAULT_FRICTION_PRIOR
    # Relative jitter (of each prior range) added when resampling.
    jitter: float = 0.02

    @classmethod
    def from_prior(
        cls,
        rng: np.random.Generator,
        num_particles: int = 500,
        prior: FrictionPrior | BoxFrictionPrior = DEFAULT_FRICTION_PRIOR,
    ) -> FrictionBelief:
        """Samples from the prior."""
        return cls(particles=prior.sample(rng, num_particles), prior=prior)

    @property
    def mean(self) -> NDArray[np.float64]:
        """Posterior mean per surface."""
        return self.particles.mean(axis=0)

    @property
    def std(self) -> NDArray[np.float64]:
        """Posterior standard deviation per surface."""
        return self.particles.std(axis=0)

    def states(
        self,
        model: PuckPushingModel,
        puck: NDArray[np.float64],
        robot: NDArray[np.float64],
        pushes: int = 0,
    ) -> list[PuckState]:
        """One POMDP state per particle, at the known puck and robot positions."""
        return [model.initial_state(puck, robot, f, pushes) for f in self.particles]

    def likelihoods(
        self,
        world: PuckWorld,
        puck: NDArray[np.float64],
        angle: float,
        release_speed: float,
        stop: NDArray[np.float64],
        fell: bool,
        stance: float | None = None,
    ) -> NDArray[np.float64]:
        """How well each particle explains one push's measurements.

        Each particle predicts where the puck would stop if it left at the measured
        speed along the commanded heading. The error is split along and across the
        predicted final direction of travel: speed noise moves the stop along it,
        heading noise mostly across it, and position noise both ways.
        """
        phys = world.physics
        release = world.release_point(puck, angle)
        robot = world.strike_end(world.push_station(puck, angle, stance), puck, angle)
        out = np.empty(len(self.particles))
        speed = max(release_speed, 1e-3)
        for k, frictions in enumerate(self.particles):
            result = world.slide(release, release_speed, angle, frictions, robot=robot)
            length = result.distance
            heading = angle
            if len(result.path) >= 2:
                last = result.path[-1] - result.path[-2]
                if np.linalg.norm(last) > 1e-9:
                    heading = math.atan2(last[1], last[0])
            along_dir = np.array([math.cos(heading), math.sin(heading)])
            error = stop - result.stop
            along = float(error @ along_dir)
            across = float(error[0] * along_dir[1] - error[1] * along_dir[0])
            var_along = (
                phys.position_obs_noise**2
                + (2.0 * length * phys.speed_obs_noise / speed) ** 2
            )
            var_across = phys.position_obs_noise**2 + (length * phys.angle_noise) ** 2
            weight = math.exp(
                -0.5 * (along**2 / var_along + across**2 / var_across)
            ) / math.sqrt(var_along * var_across)
            if (result.hole is not None) != fell:
                weight *= _HOLE_MISMATCH
            out[k] = weight
        return out

    def update(
        self,
        world: PuckWorld,
        puck: NDArray[np.float64],
        angle: float,
        release_speed: float,
        stop: NDArray[np.float64],
        fell: bool,
        rng: np.random.Generator,
        stance: float | None = None,
    ) -> float:
        """Condition on one push and resample.

        Returns the effective sample size before resampling. If no particle explains the
        push at all, the belief is left unchanged and 0 is returned.
        """
        weights = self.likelihoods(
            world, puck, angle, release_speed, stop, fell, stance
        )
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return 0.0
        weights = weights / total
        ess = float(1.0 / np.sum(weights**2))
        self.particles = self._resample(weights, rng)
        return ess

    def _resample(
        self, weights: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Systematic resampling with a little jitter, kept inside the prior box."""
        n = len(weights)
        positions = (rng.random() + np.arange(n)) / n
        indices = np.minimum(np.searchsorted(np.cumsum(weights), positions), n - 1)
        lows, highs = self.prior.bounds
        noise = rng.normal(0.0, self.jitter, size=(n, len(lows))) * (highs - lows)
        return np.clip(self.particles[indices] + noise, lows, highs)


def summarize(belief: FrictionBelief, names: Sequence[str]) -> dict[str, list[float]]:
    """Mean and std per named surface, for logs."""
    return {
        name: [float(m), float(s)] for name, m, s in zip(names, belief.mean, belief.std)
    }
