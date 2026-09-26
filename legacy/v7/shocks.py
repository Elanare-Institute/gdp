"""Stochastic shock processes for the Monte Carlo runs.

Three processes, all seeded and drawn independently of the modality being
tested. That last point is what makes common random numbers work: the same
seed produces the same shock path whichever transfer is in place, so a
difference between two runs is the policy and not the draw.

Standard library only, per the project's constraint on ``model/``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ShockConfig:
    """Parameters of the three shock processes."""

    # Reserve-bloc interest rate: AR(1) with occasional jumps.
    rate_rho: float = 0.85
    rate_sigma: float = 0.0015
    rate_jump_prob: float = 0.04        # per year
    rate_jump_low: float = 0.02         # +200bp
    rate_jump_high: float = 0.05        # +500bp
    rate_jump_decay: float = 0.6        # per year, once a jump has landed

    # Imported food price: log AR(1), feeding pF directly.
    food_rho: float = 0.80
    food_sigma: float = 0.06

    # Sudden stops: Poisson arrivals, outflow proportional to openness.
    stop_rate: float = 0.05             # expected arrivals per year
    stop_quarters: int = 4
    stop_severity: float = 0.08         # share of GDP per quarter at openness 1

    enabled: bool = True


@dataclass(frozen=True)
class ShockPath:
    """One realised shock path, sampled quarterly.

    Attributes:
        rate_add: Additive shock to the reserve bloc's policy rate.
        food_factor: Multiplicative factor on the imported food price.
        sudden_stop: Outflow intensity, before scaling by a bloc's openness.
    """

    rate_add: tuple[float, ...]
    food_factor: tuple[float, ...]
    sudden_stop: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.rate_add)


def zero_path(steps: int) -> ShockPath:
    """A path with no shocks, for the deterministic baseline."""
    zeros = tuple(0.0 for _ in range(steps))
    return ShockPath(rate_add=zeros, food_factor=tuple(1.0 for _ in range(steps)),
                     sudden_stop=zeros)


def draw_path(seed: int, steps: int, dt: float = 0.25,
              config: ShockConfig | None = None) -> ShockPath:
    """Draw one shock path.

    The path depends only on `seed`, so the same seed can be reused across
    modalities to difference out the draw.
    """
    config = config or ShockConfig()
    if not config.enabled:
        return zero_path(steps)

    rng = random.Random(seed)
    rate_add: list[float] = []
    food: list[float] = []
    stops: list[float] = []

    rate_state = 0.0
    jump_state = 0.0
    food_state = 0.0
    stop_remaining = 0
    stop_size = 0.0

    # Per-period probabilities from the annual rates.
    jump_p = 1.0 - (1.0 - config.rate_jump_prob) ** dt
    stop_p = 1.0 - math.exp(-config.stop_rate * dt)

    for _ in range(steps):
        # AR(1) plus a decaying jump component.
        rate_state = (config.rate_rho ** dt) * rate_state + rng.gauss(0.0, config.rate_sigma)
        if rng.random() < jump_p:
            jump_state += rng.uniform(config.rate_jump_low, config.rate_jump_high)
        jump_state *= config.rate_jump_decay ** dt
        rate_add.append(rate_state + jump_state)

        # Log AR(1) on the food price.
        food_state = (config.food_rho ** dt) * food_state + rng.gauss(0.0, config.food_sigma)
        food.append(math.exp(food_state))

        # Poisson arrivals; an arrival runs for a few quarters.
        if stop_remaining <= 0 and rng.random() < stop_p:
            stop_remaining = config.stop_quarters
            stop_size = config.stop_severity
        if stop_remaining > 0:
            stops.append(stop_size)
            stop_remaining -= 1
        else:
            stops.append(0.0)

    return ShockPath(rate_add=tuple(rate_add), food_factor=tuple(food),
                     sudden_stop=tuple(stops))


def summarise(path: ShockPath) -> dict[str, float]:
    """Descriptive statistics of a drawn path, for diagnostics."""
    n = max(1, len(path))
    return {
        "rate_mean": sum(path.rate_add) / n,
        "rate_max": max(path.rate_add) if path.rate_add else 0.0,
        "food_max": max(path.food_factor) if path.food_factor else 1.0,
        "food_min": min(path.food_factor) if path.food_factor else 1.0,
        "stop_quarters": sum(1 for s in path.sudden_stop if s > 0),
    }
