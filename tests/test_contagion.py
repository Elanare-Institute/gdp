"""When one borrower fails, lenders reassess everything that looks like it.

Every adjustment in phases A-C was continuous: pressure built, prices drifted,
nothing ever stopped. Lenders do not behave that way. A default is followed by
a withdrawal from blocs that resemble the one that failed, and the withdrawal
arrives as a step. Resemblance is measured on the same two axes that decide
whether a bloc can pay for its imports.

The reserve issuer is exempt, because a withdrawal of *foreign* lending does
not stop a bloc that settles in money it prints. That exemption is the phase D
claim, so it is tested rather than assumed.

Specification: `specs/V8_D.md`.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from model.core import simulate
from model.household import calibrate
from model.params import G
from model.settlement import axis_distance, contagion_weight, decay_contagion
from model.twoaxis import grid, make_bloc

U = 0.06  # large enough that defaults occur inside the horizon


def _g(strength: float = 0.0, **kw) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation="cpi_indexed", settlement=True, trade_elasticity=1.0,
             policy_response=1.0, contagion_strength=strength, **kw)


def _run(strength: float, u: float = U, **kw):
    g = _g(strength, **kw)
    blocs = [replace(b, modality="ubi", size=calibrate(b, g, u)["ubi"], start=1.0)
             for b in grid(5)]
    return simulate(blocs, g)


def _defaults(result) -> list[float]:
    return sorted(v["insolvent_year"] for k, v in result["crisis"].items()
                  if k != "_system" and v.get("insolvent_year") is not None)


# --- the primitives ---

def test_similar_blocs_are_close_on_the_plane() -> None:
    a = make_bloc("a", 0.2, 0.2)
    twin = make_bloc("twin", 0.22, 0.22)
    opposite = make_bloc("opp", 0.9, 0.9)
    assert axis_distance(a, twin) < axis_distance(a, opposite)


def test_the_withdrawal_falls_off_with_distance() -> None:
    near = contagion_weight(0.1, reach=0.5)
    far = contagion_weight(2.0, reach=0.5)
    assert 0 < far < near <= 1.0


def test_a_wider_reach_carries_the_withdrawal_further() -> None:
    assert contagion_weight(1.0, reach=1.0) > contagion_weight(1.0, reach=0.3)


def test_no_reach_means_no_contagion() -> None:
    assert contagion_weight(1.0, reach=0.0) == 0.0


def test_a_withdrawal_unwinds() -> None:
    level = 1.0
    for _ in range(20):
        level = decay_contagion(level, years=5.0, dt=0.25)
    assert 0.0 <= level < 0.5


# --- it changes the run ---

def test_contagion_is_off_by_default() -> None:
    assert G().contagion_strength == 0.0


def test_contagion_produces_more_defaults() -> None:
    """The step that continuous adjustment could not produce."""
    assert len(_defaults(_run(1.0))) > len(_defaults(_run(0.0)))


def test_contagion_brakes_the_world_price() -> None:
    """Several blocs stop importing at once, so world demand drops with them."""
    assert _run(1.0)["world_final"]["price"] < _run(0.0)["world_final"]["price"]


def test_the_first_default_is_unchanged() -> None:
    """Nothing has failed yet, so there is nothing to propagate from."""
    assert _defaults(_run(1.0))[0] == _defaults(_run(0.0))[0]


def test_contagion_events_are_recorded() -> None:
    result = _run(1.0)
    assert result["contagion"], "a default should have propagated"
    first = result["contagion"][0]
    assert first["defaulted"]
    assert first["year"] >= _defaults(result)[0]


def test_a_larger_transfer_makes_contagion_bite_harder() -> None:
    """Bigger programmes bring the first default forward, which leaves more of
    the horizon for the withdrawal to work through."""
    small = len(_run(1.0, u=0.02)["contagion"])
    large = len(_run(1.0, u=0.06)["contagion"])
    assert large >= small


# --- the reserve issuer is exempt ---

def _hits(result, blocs):
    issuer = next(i for i, b in enumerate(blocs) if b.reserve)
    worst = [max((e[f"hit_{i}"] for e in result["contagion"]), default=0.0)
             for i in range(len(blocs))]
    return issuer, worst


def test_distance_alone_spares_the_issuer_in_a_moderate_crisis() -> None:
    """The phase D claim, and the reason it is not simply asserted.

    Granting the reserve issuer an exemption would put the conclusion into the
    premise — v7's privileges under another name. On the main path it is
    marked down by the same rule as everyone else and comes off lighter
    because it sits far from the blocs that fail. That is a result about where
    it sits, not a rule about what it is.
    """
    result = _run(1.0, u=0.04, flight_to_reserve=0.0)
    issuer, worst = _hits(result, grid(5))
    others = [h for i, h in enumerate(worst) if i != issuer]
    assert worst[issuer] < sorted(others)[len(others) // 2]


def test_distance_stops_sparing_it_once_the_crisis_is_deep_enough() -> None:
    """Reported because it is what the model says.

    With enough defaults every bloc is marked down to the cap and the
    two-axis distance stops separating anybody: being far from the failures
    protects the issuer only while there are still blocs that have not failed.
    At that point the flight to its currency is the only thing left doing the
    work — which is a statement about how conditional the position is.
    """
    result = _run(1.0, u=0.06, flight_to_reserve=0.0)
    issuer, worst = _hits(result, grid(5))
    others = [h for i, h in enumerate(worst) if i != issuer]
    assert worst[issuer] == pytest.approx(max(others))


def test_the_issuer_is_not_exempted_by_rule_on_the_main_path() -> None:
    """It is hit — just less. A zero here would mean the rule was doing it."""
    assert G().contagion_exempts_reserve is False
    result = _run(1.0, u=0.04, flight_to_reserve=0.0)
    issuer, worst = _hits(result, grid(5))
    assert worst[issuer] > 0.0


def test_flight_to_the_reserve_currency_is_what_spares_it() -> None:
    """Money pulled out of a region buys the safest claim on offer. The
    issuer's relief is an inflow it receives *because* others are being
    withdrawn from, which is a mechanism rather than an exemption."""
    without = _hits(_run(1.0, u=0.04, flight_to_reserve=0.0), grid(5))
    with_flight = _hits(_run(1.0, u=0.04, flight_to_reserve=0.5), grid(5))
    assert with_flight[1][with_flight[0]] < without[1][without[0]]


def test_the_exemption_by_rule_remains_available_as_a_sensitivity() -> None:
    result = _run(1.0, u=0.04, contagion_exempts_reserve=True,
                  flight_to_reserve=0.0)
    issuer, worst = _hits(result, grid(5))
    assert worst[issuer] == 0.0


def test_someone_other_than_the_issuer_is_marked_down() -> None:
    """Otherwise the tests above would pass vacuously."""
    result = _run(1.0)
    blocs = grid(5)
    issuer, worst = _hits(result, blocs)
    assert max(h for i, h in enumerate(worst) if i != issuer) > 0.0


# --- sweeping the parameters ---

@pytest.mark.parametrize("strength", [0.5, 1.0, 2.0])
def test_stronger_contagion_is_never_milder(strength: float) -> None:
    baseline = _run(0.0)["world_final"]["price"]
    assert _run(strength)["world_final"]["price"] <= baseline


def test_a_longer_withdrawal_is_never_milder() -> None:
    brief = _run(1.0, contagion_years=2.0)["world_final"]["price"]
    lasting = _run(1.0, contagion_years=10.0)["world_final"]["price"]
    assert lasting <= brief


# --- where the issuer's protection gives way ---

def _issuer_hit(u: float, flight: float = 0.0) -> float:
    issuer, worst = _hits(_run(1.0, u=u, flight_to_reserve=flight), grid(5))
    return worst[issuer]


def test_the_issuers_protection_erodes_as_the_crisis_deepens() -> None:
    """Distance protects it only while there are still blocs that have not
    failed. As defaults reach positions closer to the issuer, the marks-down
    accumulate faster than they decay."""
    assert _issuer_hit(0.04) < _issuer_hit(0.05) <= _issuer_hit(0.06)


def test_what_reaches_the_issuer_is_proximity_not_the_count() -> None:
    """The mechanism is about where the failures are, not how many.

    Each individual mark-down on the issuer is small (under 0.21 even from the
    nearest failure). What overwhelms it is defaults arriving at positions
    close enough, and often enough, that the decay cannot keep up.
    """
    from model.settlement import axis_distance, contagion_weight
    blocs = grid(5)
    issuer = next(i for i, b in enumerate(blocs) if b.reserve)
    result = _run(1.0, u=0.06, flight_to_reserve=0.0)
    failed = [i for i, b in enumerate(blocs)
              if result["crisis"][b.name].get("insolvent_year") is not None]
    singles = [contagion_weight(axis_distance(blocs[issuer], blocs[i]), 0.5)
               for i in failed]
    assert max(singles) < 1.0, "no single default should saturate the issuer"
    assert sum(singles) > 1.0, "but together they must be able to"


def test_flight_to_the_currency_holds_even_where_distance_fails() -> None:
    """Once distance stops protecting it, the flight to its money is the only
    thing left — which is how conditional the position is."""
    assert _issuer_hit(0.06, flight=0.0) == pytest.approx(1.0)
    assert _issuer_hit(0.06, flight=0.5) < 0.5


# --- contagion also squeezes the blocs that do not fail ---

def _survivor_rationing(strength: float, u: float = 0.02) -> float:
    result = _run(strength, u=u, flight_to_reserve=0.5)
    blocs = grid(5)
    final = result["settlement"][-1]
    survivors = [i for i, b in enumerate(blocs)
                 if result["crisis"][b.name].get("insolvent_year") is None]
    return sum(final[f"share_{i}"] for i in survivors) / len(survivors)


def test_contagion_squeezes_blocs_that_never_default() -> None:
    """Why the world price falls under contagion even when the default count
    is unchanged. A narrowed borrowing window and a higher risk premium ration
    the imports of blocs that survive the whole run.
    """
    plain, spread = _run(0.0, u=0.02), _run(1.0, u=0.02)
    assert len(_defaults(plain)) == len(_defaults(spread)), "same defaults"
    assert spread["world_final"]["price"] < plain["world_final"]["price"]
    assert _survivor_rationing(1.0) > _survivor_rationing(0.0)


def test_the_world_price_must_be_read_with_the_default_count() -> None:
    """A lower world price under contagion is demand disappearing, not relief.

    Both channels push the same way: blocs that fail stop importing, and blocs
    that survive import less. Neither makes anyone better off.
    """
    plain, spread = _run(1.0, u=0.06), _run(0.0, u=0.06)
    assert plain["world_final"]["price"] < spread["world_final"]["price"]
    assert len(_defaults(plain)) > len(_defaults(spread))
