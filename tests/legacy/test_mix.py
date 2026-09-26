"""Welfare-targeting mixes and the cost of the transition (Phase 3 groundwork).

Asking a rate-limited CLT to match a cash transfer on its own has no solution
once the build cap binds. The mix formulation replaces that question with a
feasible one: hold welfare at a target every period, let CLT supply what it
can, and price the cash that covers the rest.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from legacy.v7.heterogeneous import calibrate_hetero, hetero_household_block, welfare_measure
from legacy.v7.mix import cash_topup, transition_cost, transition_path
from legacy.v7.params import G, archetypes


def _setup(build_rate: float | None = 0.005):
    b = archetypes()["C"]
    free = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    cal = calibrate_hetero(b, free, 0.05, "utilitarian")
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=build_rate)
    return b, g, cal["_U"], cal["clt"]


def test_clt_cash_with_zero_cash_equals_clt() -> None:
    """The combined modality must reduce to plain CLT when no cash is paid."""
    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    plain = hetero_household_block(replace(b, modality="clt"), g,
                                   b.gdp, 1.0, b.rate, True, 0.01)
    combined = hetero_household_block(replace(b, modality="clt_cash"), g,
                                      b.gdp, 1.0, b.rate, True, 0.01, cash_size=0.0)
    assert combined.fiscal == pytest.approx(plain.fiscal, rel=1e-12)
    assert combined.pH == pytest.approx(plain.pH, rel=1e-12)
    assert welfare_measure(combined, "utilitarian") == pytest.approx(
        welfare_measure(plain, "utilitarian"), rel=1e-12)


def test_cash_is_spent_at_the_rent_the_clt_produced() -> None:
    """Cash and housing interact, so they cannot be evaluated separately.

    Adding cash raises housing demand and therefore the clearing rent, which a
    sum of two independent runs would miss.
    """
    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    without = hetero_household_block(replace(b, modality="clt_cash"), g,
                                     b.gdp, 1.0, b.rate, True, 0.01, cash_size=0.0)
    with_cash = hetero_household_block(replace(b, modality="clt_cash"), g,
                                       b.gdp, 1.0, b.rate, True, 0.01, cash_size=0.02)
    assert with_cash.pH > without.pH


def test_topup_hits_the_target_every_period() -> None:
    """The mix holds welfare at the target throughout the phase-in."""
    b, g, target, clt_size = _setup()
    path = transition_path(b, g, target, clt_size, years=30)
    assert all(p.shortfall_covered for p in path)
    for period in path:
        assert period.welfare == pytest.approx(target, abs=1e-6), period.year


def test_topup_declines_as_the_stock_arrives() -> None:
    """Cash falls monotonically as CLT takes over.

    At 0.5%/yr the build is not quite complete after 30 years, so a small
    residual top-up remains; the test checks the decline, not that the
    transition happens to finish inside the window.
    """
    b, g, target, clt_size = _setup()
    path = transition_path(b, g, target, clt_size, years=30)
    cash = [p.cash_size for p in path]
    assert all(b_ <= a + 1e-12 for a, b_ in zip(cash, cash[1:]))
    assert cash[0] > 0
    assert cash[-1] < 0.1 * cash[0]      # nearly complete


def test_topup_reaches_zero_when_the_build_finishes() -> None:
    """A faster build does retire the cash top-up within the window."""
    b, _, target, clt_size = _setup()
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)
    path = transition_path(b, g, target, clt_size, years=30)
    assert path[-1].cash_size == pytest.approx(0.0, abs=1e-6)


def test_no_topup_needed_without_a_build_cap() -> None:
    """With the stock available immediately the transition costs nothing."""
    b, g, target, clt_size = _setup(build_rate=None)
    path = transition_path(b, g, target, clt_size, years=10)
    assert all(p.cash_size == pytest.approx(0.0, abs=1e-6) for p in path)
    assert transition_cost(path)["pv_cash_topup"] == pytest.approx(0.0, abs=1e-4)


def test_slower_building_costs_more_to_transition() -> None:
    """The transition cost is the price of the build constraint."""
    b, _, target, clt_size = _setup()
    costs = {}
    for rate in (0.02, 0.005):
        g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
              clt_build_rate=rate)
        costs[rate] = transition_cost(transition_path(b, g, target, clt_size, 30))
    assert costs[0.005]["pv_cash_topup"] > costs[0.02]["pv_cash_topup"]


def test_cash_topup_reports_when_the_target_is_unreachable() -> None:
    """An impossible target is flagged rather than silently clipped."""
    b, g, _, clt_size = _setup()
    _, reached = cash_topup(b, g, target=1e6, clt_size=clt_size, year=0.0)
    assert reached is False


# --- size-independent test of the genuine constraint ------------------------

def test_attainable_bound_does_not_depend_on_size() -> None:
    """The bound builds flat out, so no calibration target enters it.

    This is what makes it a clean test of cause (a): the search instrument's
    failure cannot contaminate the answer.
    """
    from legacy.v7.mix import max_attainable_welfare

    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.005)
    first = max_attainable_welfare(b, g, years=20, discount=0.03)
    second = max_attainable_welfare(b, g, years=20, discount=0.03)
    assert first["clt_max_pv"] == pytest.approx(second["clt_max_pv"], rel=1e-12)


def test_fast_building_can_match_cash_and_slow_cannot() -> None:
    """The bound separates a real constraint from a calibration artifact."""
    from legacy.v7.mix import max_attainable_welfare

    b = archetypes()["C"]
    fast = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
             clt_build_rate=0.05)
    slow = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
             clt_build_rate=0.005)
    assert max_attainable_welfare(b, fast, 30, 0.03)["clt_can_match"] is True
    assert max_attainable_welfare(b, slow, 30, 0.03)["clt_can_match"] is False


# --- comparison against continuing with cash --------------------------------

def test_comparison_reports_fiscal_and_leakage_differences() -> None:
    """Both arms hold welfare fixed, so only the spending differs."""
    from legacy.v7.mix import compare_with_cash

    b = archetypes()["C"]
    free = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    cal = calibrate_hetero(b, free, 0.05, "utilitarian")
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)

    result = compare_with_cash(b, g, cal["_U"], cal["clt"], cal["cash_t"],
                               years=30, discount=0.03)
    assert result["d_pv_fiscal"] == pytest.approx(
        result["pv_fiscal_mix"] - result["pv_fiscal_cash"], rel=1e-9)
    assert result["d_pv_leak"] == pytest.approx(
        result["pv_leak_mix"] - result["pv_leak_cash"], rel=1e-9)


def test_switching_is_fiscally_cheaper_than_staying_on_cash() -> None:
    """Holding welfare fixed, the CLT mix costs less in present value."""
    from legacy.v7.mix import compare_with_cash

    for key in ("A", "C", "D"):
        b = archetypes()[key]
        free = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
        cal = calibrate_hetero(b, free, 0.05, "utilitarian")
        g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
              clt_build_rate=0.02)
        result = compare_with_cash(b, g, cal["_U"], cal["clt"], cal["cash_t"],
                                   years=30, discount=0.03)
        assert result["d_pv_fiscal"] < 0, key


def test_transition_can_raise_leakage_even_while_saving_money() -> None:
    """Fiscal and external-leakage verdicts need not agree.

    In the developing bloc the build imports during the phase-in push leakage
    up even though the programme is cheaper, so Phase 3 cannot rank structures
    on cost alone.
    """
    from legacy.v7.mix import compare_with_cash

    b = archetypes()["D"]
    free = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    cal = calibrate_hetero(b, free, 0.05, "utilitarian")
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)
    result = compare_with_cash(b, g, cal["_U"], cal["clt"], cal["cash_t"],
                               years=30, discount=0.03)
    assert result["d_pv_fiscal"] < 0      # cheaper
    assert result["d_pv_leak"] > 0        # yet leakier


def test_identical_household_structures_give_identical_topups() -> None:
    """Blocs A and B need the same cash top-up because their households match.

    The top-up is set by the welfare gap of the constrained block, and A and B
    share c_income_share, c_pop_share and eps_supply. They differ in
    land_share, which shows up in the CLT cost rather than in the cash path.
    """
    from legacy.v7.mix import transition_cost

    paths = {}
    for key in ("A", "B"):
        b = archetypes()[key]
        free = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
        cal = calibrate_hetero(b, free, 0.05, "utilitarian")
        g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
              clt_build_rate=0.02)
        paths[key] = transition_cost(
            transition_path(b, g, cal["_U"], cal["clt"], years=30), 0.03)

    assert paths["A"]["pv_cash_topup"] == pytest.approx(
        paths["B"]["pv_cash_topup"], rel=1e-6)
    assert paths["A"]["pv_clt"] != pytest.approx(paths["B"]["pv_clt"], rel=1e-6)
