"""Whether the budget reaches the subsistence bundle.

CLAUDE.md sets the standard for sizing a transfer: `u` should follow from what
a constrained household needs in order to live, not from what it takes to move
aggregate demand.

**This model does not represent an allocation below subsistence.** The demand
system clips the supernumerary income at zero —

    x[k] = sub[k] + beta[k] * max(supern, 0) / p[k]

— so the quantities secured are never less than the floor, whatever prices do.
That is an implementation measure, not a claim: Stone-Geary utility is
undefined below the floor.

So what is measured here is the **budget**, not the allocation:

    budget_ratio = (income + transfer) / cost of the subsistence bundle

Below 1.0 the household cannot pay for the bundle at current prices. What
happens to a household in that position — which of the three goods gives way,
and what follows — is outside this model.

**Two definitions of the floor coexist, deliberately.**

  household behaviour   the floor is a share of income (`sub_F * Ic`), as it
                        has been since v6. Phase 2's welfare calibration and
                        everything built on it rest on this.
  this indicator        the floor is a **quantity**, fixed at what the bundle
                        was at the programme's start, and re-priced each
                        period.

They disagree, and the disagreement is the point. A floor proportional to
income cannot fall short: if income halves, so does the floor, and the ratio
never moves. What a person needs to eat does not halve because their income
did. Keeping the behavioural floor as it was preserves the calibration; making
the *measured* floor a quantity gives the indicator the property the question
requires. The inconsistency is recorded in `specs/V8_U.md` and belongs in the
paper's limitations, not hidden in a helper.

The word "meets" is deliberately avoided throughout. The ratio says whether
the money reaches the bundle, not whether the household is fed.

See `specs/V8_U.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .params import Bloc, G, HouseholdType, income_quantiles


@dataclass(frozen=True)
class BudgetPosition:
    """One household type's budget against the subsistence bundle."""

    name: str
    income: float
    transfer: float
    bundle_cost: float

    @property
    def budget_ratio(self) -> float:
        """Resources over what the bundle costs. Below 1.0 is a shortfall."""
        if self.bundle_cost <= 0:
            return float("inf")
        return (self.income + self.transfer) / self.bundle_cost

    @property
    def shortfall(self) -> float:
        """How far below 1.0, or zero when the budget reaches the bundle."""
        return max(0.0, 1.0 - self.budget_ratio)


def subsistence_quantities(g: G, income: float, sub_scale: float = 1.0
                           ) -> dict[str, float]:
    """The floor as physical quantities, at a given income level.

    Called once, at the reference point, and the result is then held fixed:
    what a household needs to eat does not shrink because its income did.
    """
    scale = g.subsistence_scale * sub_scale
    return {"F": g.sub_F * income * scale,
            "G": g.sub_G * income * scale,
            "H": g.sub_H * income * scale}


def bundle_cost(prices: dict[str, float], quantities: dict[str, float]) -> float:
    """What a fixed bundle costs at today's prices.

    The quantities come from the reference point and do not move; only the
    prices do. So a bloc whose currency has fallen, or whose imports are being
    rationed, faces a bundle that costs more than it did — which is how the
    indicator registers a squeeze that the allocation itself cannot show.
    """
    return sum(prices[k] * quantities[k] for k in quantities)


def positions(b: Bloc, g: G, output: float, prices: dict[str, float],
              transfer: float = 0.0,
              types: Sequence[HouseholdType] | None = None,
              reference: dict[str, dict[str, float]] | None = None,
              income_factor: float = 1.0,
              ) -> list[BudgetPosition]:
    """Budget position of each income quantile in the constrained block.

    The transfer is split by population share, as the household block splits
    it. The poorest quantile carries the largest `sub_scale`, so its bundle
    costs more relative to its income and its ratio falls first.

    Args:
        reference: Fixed bundle quantities per type, from
            `reference_bundles`. Without it the bundle is computed at today's
            income, which makes the ratio insensitive to income — that is the
            behavioural definition, kept only for comparison.
        income_factor: Carries the linkage rule, so that income which no
            longer tracks output shows up in the numerator.
    """
    types = types or income_quantiles(g.n_types, g.income_spread)
    total_income = b.c_income_share * output * income_factor
    out: list[BudgetPosition] = []
    for t in types:
        income = total_income * t.income_share
        quantities = (reference[t.name] if reference
                      else subsistence_quantities(g, income, t.sub_scale))
        out.append(BudgetPosition(name=t.name, income=income,
                                  transfer=transfer * t.pop_share,
                                  bundle_cost=bundle_cost(prices, quantities)))
    return out


def reference_bundles(b: Bloc, g: G, output: float,
                      types: Sequence[HouseholdType] | None = None,
                      ) -> dict[str, dict[str, float]]:
    """Fix each quantile's subsistence bundle at a reference output level.

    Computed once and carried forward, so later periods re-price the same
    quantities rather than redefining what subsistence is.
    """
    types = types or income_quantiles(g.n_types, g.income_spread)
    total_income = b.c_income_share * output
    return {t.name: subsistence_quantities(g, total_income * t.income_share,
                                           t.sub_scale)
            for t in types}


def poorest(positions_: Sequence[BudgetPosition]) -> BudgetPosition | None:
    """The quantile with the lowest budget ratio."""
    return min(positions_, key=lambda p: p.budget_ratio, default=None)


def summarise(positions_: Sequence[BudgetPosition]) -> dict[str, float]:
    """Budget ratios across the quantiles, for reporting."""
    if not positions_:
        return {}
    ratios = [p.budget_ratio for p in positions_]
    return {
        "min": min(ratios),
        "median": sorted(ratios)[len(ratios) // 2],
        "share_below_one": sum(1 for r in ratios if r < 1.0) / len(ratios),
    }


def required_transfer(b: Bloc, g: G, output: float, prices: dict[str, float],
                      types: Sequence[HouseholdType] | None = None,
                      reference: dict[str, dict[str, float]] | None = None,
                      income_factor: float = 1.0,
                      erosion: float = 1.0) -> float:
    """Transfer that would hold the poorest quantile's budget at the bundle.

    Solved directly rather than searched: the ratio is linear in the transfer.
    Returns zero when the budget already reaches the bundle.

    This is a *time series*, not a constant. As the currency falls and world
    prices rise, the bundle costs more and the transfer needed to pay for it
    grows — which is the structure the phase exists to expose, since the blocs
    where it grows fastest are the ones that can least afford to pay it.

    Args:
        erosion: What a unit of transfer handed out at the start is worth now,
            under the indexation rule in force. Below 1.0 the rule has let the
            transfer lose value, so a larger initial transfer is needed to
            deliver the same purchasing power today.

            Without this the requirement was computed as though every rule
            delivered its face value, which made an unindexed transfer look
            cheaper than it is: the comparison of rules was not on the same
            footing. See `reports/PHASE_V8_POSTEMP.md` §3.
    """
    worst = 0.0
    for position in positions(b, g, output, prices, transfer=0.0, types=types,
                              reference=reference, income_factor=income_factor):
        gap = position.bundle_cost - position.income
        share = next((t.pop_share for t in
                      (types or income_quantiles(g.n_types, g.income_spread))
                      if t.name == position.name), 0.0)
        if gap > 0 and share > 0:
            worst = max(worst, gap / share)
    # A transfer that has lost half its value must have been twice as large to
    # begin with.
    return worst / max(0.05, erosion)
