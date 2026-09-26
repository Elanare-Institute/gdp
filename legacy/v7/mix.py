"""Modality mixes that hold welfare fixed and price the transition.

Phase 2 showed that asking a CLT alone to match a cash transfer in present
value is the wrong question once construction is rate-limited: while the cap
binds, programme size stops moving welfare, so no equivalent size exists.

The question Phase 3 asks instead is a constrained one. Each period the bloc
must deliver a target welfare level to its constrained households. CLT supplies
as much of that as the build rate allows; cash tops up the rest. The cash
top-up during the phase-in is the *transition cost* — the price of moving to a
non-fugitive modality — and it is an output rather than a free parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .heterogeneous import hetero_household_block, welfare_measure
from .params import Bloc, G


@dataclass(frozen=True)
class MixPeriod:
    """One period of a welfare-targeting mix."""

    year: int
    clt_units: float
    cash_size: float
    welfare: float
    target: float
    fiscal_clt: float
    fiscal_cash: float
    pH: float
    shortfall_covered: bool

    @property
    def fiscal_total(self) -> float:
        return self.fiscal_clt + self.fiscal_cash


def _welfare_at(b: Bloc, g: G, modality: str, size: float, year: float,
                mode: str) -> float:
    out = hetero_household_block(replace(b, modality=modality), g, b.gdp, 1.0,
                                 b.rate, True, size, years_since_start=year)
    return welfare_measure(out, mode)


def cash_topup(
    b: Bloc, g: G, target: float, clt_size: float, year: float,
    mode: str = "utilitarian", ceiling: float = 0.5,
) -> tuple[float, bool]:
    """Cash needed on top of the CLT stock in place to reach `target`.

    Returns ``(size, reached)``. ``reached`` is False when even the ceiling
    leaves welfare short, which is reported rather than silently clipped.
    """
    combined = _mixed_welfare(b, g, clt_size, 0.0, year, mode)
    if combined >= target:
        return 0.0, True

    lo, hi = 0.0, ceiling
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _mixed_welfare(b, g, clt_size, mid, year, mode) < target:
            lo = mid
        else:
            hi = mid
    size = 0.5 * (lo + hi)
    reached = _mixed_welfare(b, g, clt_size, size, year, mode) >= target - 1e-9
    return size, reached


def _mixed_welfare(b: Bloc, g: G, clt_size: float, cash_size: float,
                   year: float, mode: str) -> float:
    """Welfare when CLT housing and a cash transfer are delivered together.

    The two are combined inside one household problem rather than evaluated
    separately, so the cash a household receives is spent at the rent the CLT
    programme has already produced.
    """
    out = hetero_household_block(
        replace(b, modality="clt_cash", size=clt_size), g, b.gdp, 1.0, b.rate,
        True, clt_size, years_since_start=year, cash_size=cash_size)
    return welfare_measure(out, mode)


def transition_path(
    b: Bloc, g: G, target: float, clt_size: float, years: int = 30,
    mode: str = "utilitarian",
) -> list[MixPeriod]:
    """Year-by-year mix that holds welfare at `target` while CLT phases in."""
    path: list[MixPeriod] = []
    for year in range(years + 1):
        cash, reached = cash_topup(b, g, target, clt_size, float(year), mode)
        out = hetero_household_block(
            replace(b, modality="clt_cash", size=clt_size), g, b.gdp, 1.0, b.rate,
            True, clt_size, years_since_start=float(year), cash_size=cash)
        clt_only = hetero_household_block(
            replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, clt_size,
            years_since_start=float(year))
        path.append(MixPeriod(
            year=year, clt_units=clt_only.fiscal, cash_size=cash,
            welfare=welfare_measure(out, mode), target=target,
            fiscal_clt=100 * clt_only.fiscal / b.gdp,
            fiscal_cash=100 * cash * b.gdp / b.gdp,
            pH=out.pH, shortfall_covered=reached))
    return path


def transition_cost(path: list[MixPeriod], discount: float = 0.03) -> dict[str, Any]:
    """Discounted cash top-up over the phase-in — the cost of switching."""
    pv_cash = sum(p.fiscal_cash * (1 + discount) ** -p.year for p in path)
    pv_clt = sum(p.fiscal_clt * (1 + discount) ** -p.year for p in path)
    years_topped = sum(1 for p in path if p.cash_size > 1e-9)
    return {
        "pv_cash_topup": pv_cash,
        "pv_clt": pv_clt,
        "pv_total": pv_cash + pv_clt,
        "years_with_topup": years_topped,
        "all_targets_met": all(p.shortfall_covered for p in path),
        "final_cash_size": path[-1].cash_size,
    }


def max_attainable_welfare(
    b: Bloc, g: G, years: int = 30, discount: float = 0.03,
    mode: str = "utilitarian",
) -> dict[str, Any]:
    """Best present value a rate-limited CLT can reach, independent of `size`.

    Building flat out every year makes the stock path a property of the build
    rate alone, so this bound does not depend on the calibration target. If it
    falls short of what cash achieves, no CLT programme can match cash within
    the horizon — that is cause (a), a genuine constraint, as distinct from
    the calibration simply lacking a usable instrument.
    """
    discount_weights = [(1.0 + discount) ** -t for t in range(years)]

    # Build flat out: set the target beyond anything reachable so the cap,
    # not the target, is what limits the stock every year.
    unreachable = 1e3
    clt_pv = 0.0
    for t, w in enumerate(discount_weights):
        out = hetero_household_block(
            replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True,
            unreachable, years_since_start=float(t))
        clt_pv += w * welfare_measure(out, mode)

    # Cash benchmark: the equal-welfare cash transfer, available in full at once.
    from .heterogeneous import calibrate_hetero

    free = replace(g, clt_build_rate=None)
    cash_size = calibrate_hetero(b, free, 0.05, mode)["cash_t"]
    cash_pv = 0.0
    for t, w in enumerate(discount_weights):
        out = hetero_household_block(
            replace(b, modality="cash_t"), g, b.gdp, 1.0, b.rate, True, cash_size,
            years_since_start=float(t))
        cash_pv += w * welfare_measure(out, mode)

    return {
        "clt_max_pv": clt_pv,
        "cash_pv": cash_pv,
        "gap": clt_pv - cash_pv,
        "clt_can_match": clt_pv >= cash_pv,
        "cash_size": cash_size,
    }


def compare_with_cash(
    b: Bloc, g: G, target: float, clt_size: float, cash_size: float,
    years: int = 30, discount: float = 0.03, mode: str = "utilitarian",
) -> dict[str, Any]:
    """Transition to CLT against simply continuing with cash.

    Both arms hold welfare at `target`, so they differ only in how the money is
    spent. The break-even year is when the CLT arm's cumulative fiscal cost
    falls below the cash arm's — before that, switching is still being paid for.
    """
    path = transition_path(b, g, target, clt_size, years=years, mode=mode)

    cash_fiscal, cash_leak = [], []
    for year in range(years + 1):
        out = hetero_household_block(
            replace(b, modality="cash_t"), g, b.gdp, 1.0, b.rate, True, cash_size,
            years_since_start=float(year))
        base = hetero_household_block(
            replace(b, modality="none"), g, b.gdp, 1.0, b.rate, False, 0.0,
            years_since_start=float(year))
        cash_fiscal.append(100 * out.fiscal / b.gdp)
        cash_leak.append(100 * (out.imp_c - base.imp_c + out.imp_gov) / b.gdp)

    mix_leak = []
    for year, period in enumerate(path):
        out = hetero_household_block(
            replace(b, modality="clt_cash"), g, b.gdp, 1.0, b.rate, True, clt_size,
            years_since_start=float(year), cash_size=period.cash_size)
        base = hetero_household_block(
            replace(b, modality="none"), g, b.gdp, 1.0, b.rate, False, 0.0,
            years_since_start=float(year))
        mix_leak.append(100 * (out.imp_c - base.imp_c + out.imp_gov) / b.gdp)

    weights = [(1.0 + discount) ** -t for t in range(years + 1)]
    pv = lambda xs: sum(w * x for w, x in zip(weights, xs))
    mix_fiscal = [p.fiscal_total for p in path]

    cumulative_mix = cumulative_cash = 0.0
    break_even = None
    for year in range(years + 1):
        cumulative_mix += mix_fiscal[year] * weights[year]
        cumulative_cash += cash_fiscal[year] * weights[year]
        if break_even is None and cumulative_mix <= cumulative_cash:
            break_even = year

    return {
        "d_pv_fiscal": pv(mix_fiscal) - pv(cash_fiscal),
        "d_pv_leak": pv(mix_leak) - pv(cash_leak),
        "break_even_year": break_even,
        "pv_fiscal_mix": pv(mix_fiscal),
        "pv_fiscal_cash": pv(cash_fiscal),
        "pv_leak_mix": pv(mix_leak),
        "pv_leak_cash": pv(cash_leak),
    }
