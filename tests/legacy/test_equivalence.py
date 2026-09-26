"""Welfare equivalence of in-kind and cash transfers.

The editor's second objection was that cash/CLT welfare equivalence was never
established. These tests pin the mechanism: an *infra-marginal* in-kind grant
(one smaller than what the household would buy anyway) is exactly equivalent to
cash, while an *extramarginal* grant produces a corner solution and is strictly
worse per unit of fiscal cost.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from legacy.v7.household import calibrate, household_block, les_demand, utility
from legacy.v7.params import G, archetypes

TOL = 1e-6


@pytest.fixture
def setup():
    g = G()
    b = archetypes()["C"]
    return b, g


def test_inframarginal_voucher_equals_cash_in_utility(setup) -> None:
    """A food voucher below free-choice food demand gives cash-identical utility."""
    b, g = setup
    Y, e, r = b.gdp, 1.0, b.rate

    free = household_block(replace(b, modality="none"), g, Y, e, r, False, 0.0)
    food_qty = free["x"]["F"]
    small = 0.5 * food_qty / Y  # comfortably infra-marginal

    cash = household_block(replace(b, modality="cash_t"), g, Y, e, r, True, small)
    voucher = household_block(replace(b, modality="voucher"), g, Y, e, r, True, small)

    assert voucher["U"] == pytest.approx(cash["U"], abs=TOL)


def test_inframarginal_voucher_equals_cash_in_expenditure(setup) -> None:
    """Total consumption of every good matches, not just the utility index."""
    b, g = setup
    Y, e, r = b.gdp, 1.0, b.rate
    free = household_block(replace(b, modality="none"), g, Y, e, r, False, 0.0)
    small = 0.5 * free["x"]["F"] / Y

    cash = household_block(replace(b, modality="cash_t"), g, Y, e, r, True, small)
    voucher = household_block(replace(b, modality="voucher"), g, Y, e, r, True, small)

    for good in ("F", "G", "H"):
        assert voucher["x"][good] == pytest.approx(cash["x"][good], abs=TOL), good


def test_extramarginal_voucher_is_a_corner_solution(setup) -> None:
    """An oversized food grant pins food at the grant and is worse than cash."""
    b, g = setup
    Y, e, r = b.gdp, 1.0, b.rate
    free = household_block(replace(b, modality="none"), g, Y, e, r, False, 0.0)
    huge = 3.0 * free["x"]["F"] / Y  # far beyond free-choice food demand

    cash = household_block(replace(b, modality="cash_t"), g, Y, e, r, True, huge)
    voucher = household_block(replace(b, modality="voucher"), g, Y, e, r, True, huge)

    # Food is pinned to the grant: the household cannot resell the surplus.
    assert voucher["x"]["F"] == pytest.approx(huge * Y, rel=1e-9)
    # And the same nominal quantity is worth strictly less than cash.
    assert voucher["U"] < cash["U"] - TOL


def test_corner_solution_reallocates_remaining_budget() -> None:
    """When a good is pinned, the rest of the budget goes to the other goods."""
    p = {"F": 1.0, "G": 1.0, "H": 1.0}
    sub = {"F": 0.1, "G": 0.05, "H": 0.1}
    beta = {"F": 0.25, "G": 0.45, "H": 0.30}

    free_x, _ = les_demand(1.0, p, sub, beta)
    pinned_x, pinned_buy = les_demand(1.0, p, sub, beta, fixed={"F": free_x["F"] * 4})

    assert pinned_x["F"] == pytest.approx(free_x["F"] * 4)
    assert pinned_buy["F"] == pytest.approx(0.0)       # nothing bought on the market
    assert pinned_x["G"] > free_x["G"]                 # budget released to other goods


@pytest.mark.parametrize("key", ["A", "B", "C", "D"])
def test_welfare_calibration_equalizes_utility(key: str) -> None:
    """Calibrated sizes really do equalize constrained-household utility."""
    g = G()
    b = archetypes()[key]
    cal = calibrate(b, g, 0.05)

    for mod in ("cash_t", "voucher", "clt"):
        if cal.get("_infeasible_" + mod):
            continue
        hb = household_block(replace(b, modality=mod), g, b.gdp, 1.0, b.rate, True, cal[mod])
        assert hb["U"] == pytest.approx(cal["_U"], abs=1e-5), f"{key}:{mod}"


def test_utility_rejects_subsistence_violation() -> None:
    """Consumption at or below subsistence is not an interior optimum."""
    sub = {"F": 1.0, "G": 1.0, "H": 1.0}
    beta = {"F": 0.25, "G": 0.45, "H": 0.30}
    assert utility({"F": 1.0, "G": 2.0, "H": 2.0}, sub, beta) <= -1e8
