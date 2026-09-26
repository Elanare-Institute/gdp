"""Accounting and market-clearing invariants.

These hold for every modality and every bloc, and are what keep later phases
from silently breaking the model's internal consistency.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from legacy.v7.core import build, simulate
from legacy.v7.household import household_block
from legacy.v7.params import DT, G, archetypes

BLOCS = ["A", "B", "C", "D"]
MODS = ["none", "ubi", "cash_t", "voucher", "clt"]


@pytest.mark.parametrize("key", BLOCS)
@pytest.mark.parametrize("mod", MODS)
def test_budget_constraint_holds(key: str, mod: str) -> None:
    """Market spending equals cash income, every period.

    In-kind grants are excluded from the market side, since they are not bought.
    """
    g = G()
    b = archetypes()[key]
    size = 0.03
    hb = household_block(replace(b, modality=mod), g, b.gdp, 1.0, b.rate, True, size)

    cash_income = b.c_income_share * b.gdp
    if mod == "ubi":
        cash_income += size * b.gdp * b.c_pop_share
    elif mod == "cash_t":
        cash_income += size * b.gdp

    assert sum(hb["buy"].values()) == pytest.approx(cash_income, rel=1e-9)


@pytest.mark.parametrize("key", BLOCS)
@pytest.mark.parametrize("mod", MODS)
def test_housing_market_clears(key: str, mod: str) -> None:
    """Excess demand for housing is driven below 1e-8 by the bisection."""
    g = G()
    b = archetypes()[key]
    hb = household_block(replace(b, modality=mod), g, b.gdp, 1.0, b.rate, True, 0.03)

    demand = hb["buy"]["H"] / hb["pH"]
    supply = hb["S0"] * hb["pH"] ** (b.eps_supply * g.eps_mult)
    assert abs(demand - supply) < 1e-8, f"{key}:{mod}"


@pytest.mark.parametrize("key", BLOCS)
def test_zero_transfer_means_zero_leakage(key: str) -> None:
    """With no transfer, the treated and counterfactual blocks coincide."""
    g = G()
    b = archetypes()[key]
    on = household_block(replace(b, modality="none"), g, b.gdp, 1.0, b.rate, True, 0.0)
    off = household_block(replace(b, modality="none"), g, b.gdp, 1.0, b.rate, False, 0.0)

    assert on["imp_c"] == pytest.approx(off["imp_c"], abs=1e-12)
    assert on["imp_gov"] == pytest.approx(0.0, abs=1e-12)
    assert on["fiscal"] == pytest.approx(0.0, abs=1e-12)


def test_no_transfer_scenario_reports_no_leakage() -> None:
    """The 'none' scenario produces an empty leakage block."""
    g = G()
    assert simulate(build(g, "none"), g)["leakage"] == {}


@pytest.mark.parametrize("mod", MODS)
def test_state_variables_stay_in_domain(mod: str) -> None:
    """Prices, output and rates stay positive and bounded through the run."""
    g = G()
    result = simulate(build(g, mod), g)
    for name, fin in result["final"].items():
        assert fin["Y"] > 0, name
        assert fin["e"] > 0, name
        assert fin["debt"] >= 0, name
        assert fin["pi"] >= -0.02, name


@pytest.mark.parametrize("key", BLOCS)
def test_clt_lowers_rent_relative_to_cash(key: str) -> None:
    """CLT reduces the clearing rent; cash bids it up.

    This is the incidence channel, and it emerges from market clearing rather
    than from any modality-specific coefficient.
    """
    g = G()
    b = archetypes()[key]
    base = household_block(replace(b, modality="none"), g, b.gdp, 1.0, b.rate, False, 0.0)
    cash = household_block(replace(b, modality="cash_t"), g, b.gdp, 1.0, b.rate, True, 0.03)
    clt = household_block(replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, 0.03)

    assert cash["pH"] > base["pH"], key
    assert clt["pH"] < base["pH"], key


def test_simulation_is_deterministic() -> None:
    """Same inputs, same outputs — there is no hidden state or unseeded noise."""
    g = G()
    first = simulate(build(g, "clt"), g)
    second = simulate(build(g, "clt"), g)
    assert first["final"] == second["final"]
    assert first["crisis"] == second["crisis"]


# --- v7: domestic sourcing and the leakage metrics --------------------------

def test_domestic_sourcing_defaults_to_v6() -> None:
    """A sourcing rate of zero leaves the import content at m_H."""
    g = G()
    b = archetypes()["D"]
    hb = household_block(replace(b, modality="clt", clt_domestic_sourcing=0.0),
                         g, b.gdp, 1.0, b.rate, True, 0.03)
    struct = (1 - b.land_share) * 0.03 * b.gdp
    assert hb["imp_gov"] == pytest.approx(b.m_H * struct, rel=1e-9)


def test_domestic_sourcing_reduces_construction_imports() -> None:
    """Full domestic sourcing removes construction imports entirely."""
    g = G()
    b = archetypes()["D"]
    full = household_block(replace(b, modality="clt", clt_domestic_sourcing=1.0),
                           g, b.gdp, 1.0, b.rate, True, 0.03)
    assert full["imp_gov"] == pytest.approx(0.0, abs=1e-12)


def test_domestic_sourcing_does_not_touch_other_modalities() -> None:
    """The lever applies to CLT construction only, not to vouchers or cash."""
    g = G()
    b = archetypes()["D"]
    for mod in ("cash_t", "voucher", "ubi"):
        base = household_block(replace(b, modality=mod), g, b.gdp, 1.0, b.rate, True, 0.03)
        lever = household_block(replace(b, modality=mod, clt_domestic_sourcing=1.0),
                                g, b.gdp, 1.0, b.rate, True, 0.03)
        assert base["imp_gov"] == pytest.approx(lever["imp_gov"], abs=1e-12), mod


@pytest.mark.parametrize("mod", ["cash_t", "voucher", "clt"])
def test_three_leakage_metrics_are_reported(mod: str) -> None:
    """Every run reports leak_abs, leak_per_fiscal and leak_per_welfare."""
    g = G()
    leak = simulate(build(g, mod), g)["leakage"]
    for bloc, value in leak.items():
        assert value["leak_abs"] is not None, bloc
        assert value["leak_per_fiscal"] is not None, bloc
        assert value["welfare_gain"] > 0, bloc
        assert value["leak_per_welfare"] is not None, bloc


def test_crisis_variant_excludes_appreciation() -> None:
    """The no-appreciation severity never exceeds the full severity."""
    g = G()
    for mod in ("ubi", "cash_t", "clt"):
        crisis = simulate(build(g, mod), g)["crisis"]
        for bloc, value in crisis.items():
            if bloc == "_system":
                continue
            assert value["severity_no_appreciation"] <= value["severity"] + 1e-12, bloc
