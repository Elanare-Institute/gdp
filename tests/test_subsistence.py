"""Whether the budget reaches the subsistence bundle.

The indicator exists because the allocation cannot show a shortfall: the
demand system clips supernumerary income at zero, so quantities never fall
below the floor however dear the bundle gets. These tests fix that the
indicator does register the squeeze, that it stays an observation rather than
becoming part of the dynamics, and that the two definitions of the floor —
behavioural and measured — do the different jobs they are meant to.

Specification: `specs/V8_U.md` §6.
"""

from __future__ import annotations

import pytest

from model.core import build, simulate
from model.params import G
from model.subsistence import (
    bundle_cost, positions, reference_bundles, required_transfer,
    subsistence_quantities, summarise,
)
from model.twoaxis import make_bloc

BLOC = make_bloc("probe", credit=0.5, self_sufficiency=0.0)
PRICES = {"F": 1.0, "G": 1.0, "H": 1.0}


def _g(**kw) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02, **kw)


def _ratio(prices: dict[str, float], **kw) -> float:
    g = _g(**{k: v for k, v in kw.items() if k in G.__dataclass_fields__})
    bundles = reference_bundles(BLOC, g, BLOC.gdp)
    ps = positions(BLOC, g, BLOC.gdp, prices, reference=bundles,
                   transfer=kw.get("transfer", 0.0),
                   income_factor=kw.get("income_factor", 1.0))
    return summarise(ps)["min"]


# --- the indicator responds to what it should ---

def test_dearer_goods_lower_the_ratio() -> None:
    """The squeeze the allocation cannot show."""
    assert _ratio({"F": 2.0, "G": 1.0, "H": 1.0}) < _ratio(PRICES)


def test_a_transfer_raises_the_ratio() -> None:
    assert _ratio(PRICES, transfer=0.05) > _ratio(PRICES)


def test_income_that_falls_behind_lowers_the_ratio() -> None:
    """The point of fixing the bundle as a quantity.

    A floor defined as a share of income cannot register this: halve the
    income and the floor halves with it. What a person needs to eat does not.
    """
    assert _ratio(PRICES, income_factor=0.6) < _ratio(PRICES)


def test_the_poorest_quantile_falls_short_first() -> None:
    g = _g()
    bundles = reference_bundles(BLOC, g, BLOC.gdp)
    ps = positions(BLOC, g, BLOC.gdp, {"F": 3.0, "G": 3.0, "H": 3.0},
                   reference=bundles)
    ratios = [p.budget_ratio for p in ps]
    assert ratios[0] == min(ratios), "q1 carries the largest subsistence scale"


def test_a_higher_floor_lowers_the_ratio() -> None:
    plain = _ratio(PRICES)
    raised = _ratio(PRICES, subsistence_scale=1.4)
    assert raised < plain


# --- the bundle is a quantity, fixed once ---

def test_the_bundle_is_repriced_not_redefined() -> None:
    """Later periods buy the same quantities at new prices."""
    g = _g()
    bundles = reference_bundles(BLOC, g, BLOC.gdp)
    cheap = bundle_cost({"F": 1.0, "G": 1.0, "H": 1.0}, bundles["q1"])
    dear = bundle_cost({"F": 2.0, "G": 2.0, "H": 2.0}, bundles["q1"])
    assert dear == pytest.approx(2 * cheap)


def test_the_measured_floor_does_not_shrink_with_income() -> None:
    g = _g()
    rich = reference_bundles(BLOC, g, BLOC.gdp)
    poor = reference_bundles(BLOC, g, BLOC.gdp * 0.5)
    assert poor["q1"]["F"] < rich["q1"]["F"], "the reference point still matters"
    # but once fixed, the bundle does not move with later income
    assert bundle_cost(PRICES, rich["q1"]) == bundle_cost(PRICES, rich["q1"])


def test_the_behavioural_floor_is_still_a_share_of_income() -> None:
    """The deliberate inconsistency, recorded so it cannot be lost.

    The household block keeps v6's income-proportional floor, because Phase 2's
    welfare calibration rests on it. Only the indicator uses quantities.
    """
    g = _g()
    half = subsistence_quantities(g, 0.5)
    full = subsistence_quantities(g, 1.0)
    assert half["F"] == pytest.approx(0.5 * full["F"])


# --- it is an observation, not a mechanism ---

def test_the_indicator_does_not_touch_the_dynamics() -> None:
    """Quantities secured must be identical whatever the floor is set to."""
    plain = simulate(build(_g(), "ubi", 0.02, (1, 1, 1, 1)), _g())
    g2 = _g(subsistence_scale=1.4)
    raised = simulate(build(g2, "ubi", 0.02, (1, 1, 1, 1)), g2)
    assert plain["quantities"] == raised["quantities"]
    assert plain["final"] == raised["final"]


# --- the linkage rules ---

def test_growth_linkage_is_the_default_and_the_most_generous() -> None:
    assert G().income_linkage == "growth_linked"
    ratios = {}
    for rule in ("growth_linked", "cpi_linked", "fixed_nominal"):
        g = _g(income_linkage=rule)
        ratios[rule] = simulate(build(g, "none"), g)["budget"][30]["min_3"]
    assert ratios["growth_linked"] > ratios["cpi_linked"]


def test_an_unknown_linkage_is_refused() -> None:
    from model.core import _income_factor
    with pytest.raises(ValueError):
        _income_factor(BLOC, _g(income_linkage="whatever"),
                       {"Y": 1.0, "Y0": 1.0, "P": 1.0})


# --- what the sweep is for ---

def test_no_transfer_is_needed_at_the_original_floor() -> None:
    """v6's 0.65 leaves a hand-to-mouth household 35% free, and nothing binds."""
    g = _g(income_linkage="cpi_linked")
    assert max(row["need_3"] for row in simulate(build(g, "none"), g)["budget"]) == 0.0


def test_a_higher_floor_makes_a_transfer_necessary() -> None:
    """Which is the finding the sweep reports: where the region appears."""
    g = _g(income_linkage="cpi_linked", subsistence_scale=1.4)
    assert max(row["need_3"] for row in simulate(build(g, "none"), g)["budget"]) > 0.0


def test_required_transfer_closes_the_gap_it_reports() -> None:
    g = _g(subsistence_scale=1.4)
    prices = {"F": 2.0, "G": 2.0, "H": 2.0}
    bundles = reference_bundles(BLOC, g, BLOC.gdp)
    need = required_transfer(BLOC, g, BLOC.gdp, prices, reference=bundles)
    assert need > 0
    after = summarise(positions(BLOC, g, BLOC.gdp, prices, transfer=need,
                                reference=bundles))["min"]
    assert after >= 1.0 - 1e-6


# --- the reference point is a choice, and the conclusion does not turn on it ---

def test_the_reference_defaults_to_the_opening_state() -> None:
    assert G().subsistence_reference == "start"


def test_both_references_give_the_same_answer() -> None:
    """Fixing the bundle somewhere is unavoidable and the choice is not
    neutral, so it is checked rather than assumed.

    "start" fixes it at the bloc's opening output; "baseline" fixes it at what
    a no-transfer world would have had, so a programme cannot move the
    yardstick it is judged by. Both are computed at t=0 and the programmes
    begin at year 1, so they coincide here — which is the result, not an
    assumption.
    """
    ratios = {}
    for reference in ("start", "baseline"):
        g = _g(income_linkage="cpi_linked", subsistence_scale=1.3,
               subsistence_reference=reference)
        budget = simulate(build(g, "ubi", 0.04, (1, 1, 1, 1)), g)["budget"]
        ratios[reference] = budget[30]["min_3"]
    assert ratios["start"] == pytest.approx(ratios["baseline"])


def test_an_unknown_reference_falls_back_to_the_opening_state() -> None:
    """Rather than failing mid-run on a typo in a sweep."""
    g = _g(subsistence_reference="whatever")
    assert simulate(build(g, "none"), g)["budget"]
