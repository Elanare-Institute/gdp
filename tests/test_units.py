"""Nominal and real quantities must not be mixed.

The world output gap was once computed as nominal import spending divided by
real supply capacity. A price rise then registered as extra demand, which
raised the price again — a feedback loop made of nothing but the units, which
blew the model up as soon as the pass-through coefficient was raised.

These tests fix which quantities are real, which are nominal, and which one
each junction is entitled to use.

Specification: `specs/V8_UNITS.md`.
"""

from __future__ import annotations

import pytest

from model.core import build, simulate
from model.household import household_block
from model.params import G, archetypes
from model.trade import ADDING_UP_TOL

BLOC = archetypes()["D"]
WORLD_PRICES = (0.5, 1.0, 1.5, 2.0, 5.0)


def _block(world_price: float, **kw):
    g = kw.pop("g", G())
    return household_block(BLOC, g, BLOC.gdp, 1.0, BLOC.rate, False, 0.0,
                           world_price=world_price, **kw)


# --- the household block reports both, and they differ ---

def test_real_imports_match_the_quantities_bought() -> None:
    """`imp_q` must be the quantity, not spending rescaled by something."""
    for pw in WORLD_PRICES:
        out = _block(pw)
        expected = BLOC.m_F * out["x"]["F"] + BLOC.m_G * out["x"]["G"]
        assert out["imp_q"] == pytest.approx(expected, rel=1e-12)


def test_dearer_imports_reduce_the_quantity_bought() -> None:
    """Demand slopes down. If `imp_q` rose with the price it would be nominal."""
    quantities = [_block(pw)["imp_q"] for pw in WORLD_PRICES]
    assert quantities == sorted(quantities, reverse=True)


def test_nominal_import_spending_can_rise_while_the_quantity_falls() -> None:
    """Exactly the divergence that made the two impossible to interchange."""
    cheap, dear = _block(1.0), _block(5.0)
    assert dear["imp_c"] > cheap["imp_c"]
    assert dear["imp_q"] < cheap["imp_q"]


def test_deflating_spending_by_the_world_price_is_not_the_quantity() -> None:
    """Imported goods carry a domestic price component, so `P_w` is not their
    deflator. Dividing by it overstates the fall — this test records that the
    shortcut is wrong, so it cannot quietly come back."""
    out = _block(5.0)
    assert out["imp_c"] / 5.0 < out["imp_q"] * 0.6


def test_the_two_measures_agree_at_the_reference_price() -> None:
    out = _block(1.0)
    assert out["imp_c"] == pytest.approx(out["imp_q"], rel=1e-12)


# --- the world market clears on real quantities ---

def test_world_demand_is_the_real_quantity() -> None:
    """World capacity is real, so what is compared against it must be too."""
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    blocs = build(g, "none")
    result = simulate(blocs, g)
    opening = result["world"][0]
    # At t=0 the world price is 1, so real and nominal coincide and the gap is
    # exactly zero by the capacity normalisation.
    assert opening["demand"] == pytest.approx(opening["capacity"], rel=1e-9)


def test_a_uniform_price_rise_is_not_read_as_extra_demand() -> None:
    """The defect itself: raising the price level must not open the gap.

    Two households facing world prices an order of magnitude apart buy less in
    real terms at the higher one. If the model were comparing spending against
    capacity, the higher price would look like more demand.
    """
    cheap, dear = _block(1.0), _block(10.0)
    assert dear["imp_q"] < cheap["imp_q"]


def test_raising_the_pass_through_does_not_explode_the_model() -> None:
    """The symptom that exposed the unit error. Kept as a guard."""
    for kappa in (0.1, 0.35, 1.0, 2.0):
        g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02, kappa_w=kappa)
        result = simulate(build(g, "ubi", 0.01, (1, 1, 1, 1)), g)
        final = result["world_final"]["price"]
        assert 1e-3 < final < 1e3, f"kappa_w={kappa} gave P_w={final}"


# --- the balance of payments stays nominal ---

@pytest.mark.parametrize("modality", ["none", "ubi", "clt"])
def test_the_external_identities_hold_in_nominal_terms(modality: str) -> None:
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    u = 0.0 if modality == "none" else 0.01
    blocs = build(g, modality, u, (1, 1, 1, 1))
    result = simulate(blocs, g)
    total = sum(result["final"][b.name]["nfa"] for b in blocs)
    assert abs(total) < 1e-4, f"{modality}: net foreign assets sum to {total}"


# --- real allocations depend on relative prices only ---

def test_the_real_allocation_responds_to_relative_prices() -> None:
    """A rise in the world price is a rise in traded goods relative to
    non-traded ones, so the basket must tilt away from the traded goods."""
    cheap, dear = _block(1.0), _block(2.0)
    assert dear["x"]["G"] / dear["x"]["H"] < cheap["x"]["G"] / cheap["x"]["H"]


def test_a_closed_bloc_ignores_the_world_price_entirely() -> None:
    """With no import content there is no channel, nominal or real."""
    from dataclasses import replace
    closed = replace(BLOC, m_F=0.0, m_G=0.0, m_H=0.0)
    g = G()
    a = household_block(closed, g, closed.gdp, 1.0, closed.rate, False, 0.0,
                        world_price=1.0)
    b = household_block(closed, g, closed.gdp, 1.0, closed.rate, False, 0.0,
                        world_price=4.0)
    assert a["x"] == b["x"]
    assert a["imp_q"] == pytest.approx(b["imp_q"], abs=ADDING_UP_TOL)


# --- capacity does not answer to the demand a programme creates ---

def _capacity(modality: str, u: float) -> list[float]:
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    blocs = build(g, modality, u, (1, 1, 1, 1)) if u > 0 else build(g, "none")
    return [row["capacity"] for row in simulate(blocs, g)["world"]]


@pytest.mark.parametrize("modality,u", [("ubi", 0.01), ("ubi", 0.05),
                                        ("clt", 0.01), ("cash_t", 0.03)])
def test_capacity_is_the_same_whatever_is_handed_out(modality: str, u: float) -> None:
    """World capacity comes from the no-transfer run and is shared.

    If capacity followed the configuration's own demand, a transfer would call
    forth the supply to meet it, the output gap would close on its own, and the
    world-price pressure the programme generates — the quantity this project
    exists to measure — would be understated.
    """
    assert _capacity(modality, u) == pytest.approx(_capacity("none", 0.0), rel=1e-12)


def test_a_transfer_opens_a_gap_rather_than_growing_capacity() -> None:
    """The consequence: extra demand shows up as a gap, not as extra supply."""
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    quiet = simulate(build(g, "none"), g)["world"]
    paying = simulate(build(g, "ubi", 0.02, (1, 1, 1, 1)), g)["world"]
    assert paying[-1]["demand"] > quiet[-1]["demand"]
    assert paying[-1]["capacity"] == pytest.approx(quiet[-1]["capacity"], rel=1e-12)


def test_the_no_transfer_world_sits_at_a_zero_gap() -> None:
    """Capacity is solved jointly with baseline demand, so the reference path
    carries no trend of its own for later results to be read against."""
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    for row in simulate(build(g, "none"), g)["world"]:
        assert row["demand"] / row["capacity"] - 1.0 == pytest.approx(0.0, abs=1e-6)
