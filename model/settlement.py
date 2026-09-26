"""Who can pay for their imports, and in what.

The paper's third claim is that the reserve issuer holds up not because it
escapes inflation but because it settles its imports in money it issues. v7
asserted that advantage through three parameters — safe-haven inflow, a higher
debt threshold, a damped exchange rate — which is the form of assumption the
editor objected to. Those are off by default in v8. What is left is one
structural difference: everyone else has to find foreign exchange, and the
reserve bloc does not.

A bloc that cannot find it does not simply run a larger deficit. Its imports
are rationed to what it can pay for, and since imports are food as much as
machinery, the rationing lands on what households eat.

The reserve issuer's exemption is not unconditional either. It holds only
while other blocs want to hold its money, and that want falls as the money
loses purchasing power. Where it stops is an output of the model, not a
parameter of it.

See `specs/V8_D.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .params import Bloc


@dataclass(frozen=True)
class SettlementOutcome:
    """What one bloc could actually pay for this period."""

    capacity: float
    """Foreign exchange available: exports plus reserves plus new borrowing."""

    desired: float
    """Imports the bloc would buy if it could pay for them."""

    realised: float
    """Imports it can pay for."""

    @property
    def rationed(self) -> float:
        """Imports it wanted and could not have."""
        return max(0.0, self.desired - self.realised)

    @property
    def rationing_share(self) -> float:
        """The share of desired imports that was cut, in [0, 1]."""
        return self.rationed / self.desired if self.desired > 0 else 0.0

    @property
    def bound(self) -> bool:
        """Whether the constraint actually bit this period."""
        return self.rationed > 1e-12


def borrowing_capacity(gdp: float, limit: float, risk_premium: float,
                       sensitivity: float) -> float:
    """New foreign borrowing a bloc can raise this period.

    Falls as lenders charge more: a bloc already in trouble finds the door
    closing, which is what turns a deficit into a rationing episode rather
    than into more debt. This is part of the settlement constraint, not a
    privilege — the same formula applies to the reserve bloc on the occasions
    when it faces the constraint at all.
    """
    return max(0.0, gdp * limit * max(0.0, 1.0 - sensitivity * risk_premium))


def settle(desired: float, exports: float, reserves: float,
           borrowing: float, exempt: bool = False) -> SettlementOutcome:
    """Ration one bloc's imports to what it can pay for.

    Args:
        exempt: True for a bloc that settles in its own money, which is not a
            claim it has to acquire from anyone. Its imports are realised in
            full and the capacity figure is reported as the desired amount.
    """
    if exempt:
        return SettlementOutcome(capacity=desired, desired=desired, realised=desired)
    capacity = max(0.0, exports + reserves + borrowing)
    return SettlementOutcome(capacity=capacity, desired=desired,
                             realised=min(desired, capacity))


def reserve_currency_demand(world_price: float, base_demand: float,
                            elasticity: float) -> float:
    """How much of the reserve currency other blocs want to hold.

    Reserves are held to settle future imports, so what makes them worth
    holding is what they will buy. As the world price rises the reserve
    currency buys less, and the reason to hold it weakens. `elasticity` says
    how sharply; it has no empirical basis here and is swept rather than
    chosen.
    """
    if world_price <= 0:
        return base_demand
    purchasing_power = 1.0 / world_price
    return base_demand * purchasing_power ** elasticity


def reserve_is_constrained(demand: float, base_demand: float,
                           threshold: float) -> bool:
    """Whether the reserve issuer has lost its exemption.

    The exemption rests on other blocs wanting the currency. Below a share of
    the original demand for it, that has stopped being true and the issuer has
    to find foreign exchange like everyone else. The threshold is a reporting
    cut-off: the model is asked where the exemption goes, not told.
    """
    if base_demand <= 0:
        return True
    return demand < threshold * base_demand


def update_reserves(reserves: float, exports: float, realised_imports: float,
                    net_capital: float, floor: float = 0.0) -> float:
    """Carry foreign exchange reserves to the next period.

    Reserves are what is left after paying for the imports that were realised:
    export earnings in, import payments out, net capital flows either way. The
    floor keeps the stock from going negative, which would mean the bloc paid
    with foreign exchange it never had.
    """
    return max(floor, reserves + exports - realised_imports + net_capital)


def rationing_summary(outcomes: Sequence[SettlementOutcome]) -> dict[str, float]:
    """How hard the constraint bit across the world this period."""
    bound = [o for o in outcomes if o.bound]
    return {
        "bound_blocs": float(len(bound)),
        "total_rationed": sum(o.rationed for o in outcomes),
        "max_rationing_share": max((o.rationing_share for o in outcomes), default=0.0),
    }


def axis_distance(a: "Bloc", b: "Bloc") -> float:
    """How alike two blocs look to a lender, on the two axes.

    Euclidean distance in (credit standing, import dependence), each
    normalised to roughly [0, 1]. Lenders do not reassess each borrower from
    scratch after a default; they mark down everything that resembles the one
    that failed, and resemblance here means the same two things that decide
    whether a bloc can pay for its imports.
    """
    credit_gap = abs(a.credit_standing - b.credit_standing) / 1.6
    import_gap = abs(a.m_F - b.m_F) / 0.25
    return (credit_gap ** 2 + import_gap ** 2) ** 0.5


def contagion_weight(distance: float, reach: float) -> float:
    """How much of a default's withdrawal lands on a bloc at `distance`.

    Falls off exponentially with distance, so the blocs most like the one that
    failed are hit hardest and distant ones barely at all. `reach` sets how
    far the reassessment travels; it is swept rather than chosen.
    """
    if reach <= 0:
        return 0.0
    return 2.718281828459045 ** (-distance / reach)


def decay_contagion(level: float, years: float, dt: float) -> float:
    """Wind a contagion shock down over `years`.

    Lenders come back, but not at once. A shock decays geometrically so that
    roughly `years` after the default the withdrawal has largely unwound.
    """
    if years <= 0:
        return 0.0
    return level * (1.0 - dt / years) if dt < years else 0.0
