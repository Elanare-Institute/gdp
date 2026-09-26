"""Channel decomposition under three conventions.

The specification asks for the "one channel at a time" decomposition, which
yields over-counting. That result is convention-dependent, and the dependence
is the substantive finding: on the same model and the same parameters,

    A  one-at-a-time   sum > joint   (over-counting)
    B  leave-one-out   sum < joint   (under-counting)
    C  Shapley         sum = joint   (exact by construction)

The reason is algebraic. The specified functional forms make the labor share
multiplicatively separable,

    theta_L(phi) = (1 - phi*beta)/mu(phi) * F(alpha(phi), s_H(phi))

so it is additive in logs and the interaction terms are second-order. Reporting
only convention A would select the convention that produces the desired answer.

This bears directly on the Grossman-Oberfield puzzle: the five existing
explanations are each estimated under convention A — switching on one's own
channel and holding the rest fixed — so the convention is itself a source of
the over-counting they diagnose.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import factorial
from typing import Sequence

from .config import CHANNELS, ModelParams
from .model import ChannelPhis, labor_share


@dataclass(frozen=True)
class Decomposition:
    """Per-channel contributions under one convention."""

    convention: str
    contributions: dict[str, float]
    total: float
    joint: float

    @property
    def ratio(self) -> float:
        """Sum of individual contributions divided by the joint effect."""
        return self.total / self.joint if self.joint != 0.0 else float("nan")


def _theta(params: ModelParams, active: Sequence[str], phi: float, baseline: float) -> float:
    """Labor share with `active` channels at `phi`, the rest at `baseline`."""
    return labor_share(params, ChannelPhis.selective(active, phi, baseline))


def joint_effect(params: ModelParams, phi_low: float, phi_high: float) -> float:
    """Total decline when all channels move together."""
    return _theta(params, CHANNELS, phi_low, phi_low) - _theta(params, CHANNELS, phi_high, phi_low)


def decompose_one_at_a_time(params: ModelParams, phi_low: float, phi_high: float) -> Decomposition:
    """Convention A: switch on one channel, hold the others at baseline.

    This is the specification's convention and the one implicitly used by each
    of the five literatures.
    """
    base = _theta(params, CHANNELS, phi_low, phi_low)
    contributions = {c: base - _theta(params, [c], phi_high, phi_low) for c in CHANNELS}
    return Decomposition(
        convention="one-at-a-time",
        contributions=contributions,
        total=sum(contributions.values()),
        joint=joint_effect(params, phi_low, phi_high),
    )


def decompose_leave_one_out(params: ModelParams, phi_low: float, phi_high: float) -> Decomposition:
    """Convention B: switch off one channel with all others active."""
    full = _theta(params, CHANNELS, phi_high, phi_low)
    contributions = {
        c: _theta(params, [x for x in CHANNELS if x != c], phi_high, phi_low) - full
        for c in CHANNELS
    }
    return Decomposition(
        convention="leave-one-out",
        contributions=contributions,
        total=sum(contributions.values()),
        joint=joint_effect(params, phi_low, phi_high),
    )


def decompose_shapley(params: ModelParams, phi_low: float, phi_high: float) -> Decomposition:
    """Convention C: Shapley value, averaging marginal effects over all orders.

    Sums to the joint effect by construction, which makes the residual
    interaction term explicit rather than hidden in a convention choice.
    """
    contributions = {c: 0.0 for c in CHANNELS}
    n_orders = factorial(len(CHANNELS))

    # Cache: the marginal effect of adding a channel depends only on the set.
    cache: dict[frozenset[str], float] = {}

    def theta_for(active: frozenset[str]) -> float:
        if active not in cache:
            cache[active] = _theta(params, sorted(active), phi_high, phi_low)
        return cache[active]

    for order in permutations(CHANNELS):
        current: frozenset[str] = frozenset()
        for channel in order:
            before = theta_for(current)
            current = current | {channel}
            contributions[channel] += before - theta_for(current)

    contributions = {c: v / n_orders for c, v in contributions.items()}
    return Decomposition(
        convention="shapley",
        contributions=contributions,
        total=sum(contributions.values()),
        joint=joint_effect(params, phi_low, phi_high),
    )


def decompose_all(params: ModelParams, phi_low: float, phi_high: float) -> list[Decomposition]:
    """All three conventions, in reporting order."""
    return [
        decompose_one_at_a_time(params, phi_low, phi_high),
        decompose_leave_one_out(params, phi_low, phi_high),
        decompose_shapley(params, phi_low, phi_high),
    ]
