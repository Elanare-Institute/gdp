"""Who can pay for their imports.

Phase D's claim is that the reserve issuer's resilience comes from settling in
money it issues, not from the three parameters v7 used to assert it. Testing
that needs the constraint itself to behave: rationing has to reach what
households consume, it must not make a bloc look stronger for having run out
of money, and a depreciation has to earn the bloc something.

Specification: `specs/V8_D.md` §5.
"""

from __future__ import annotations

import pytest

from model.core import build, simulate
from model.params import G
from model.settlement import (
    borrowing_capacity, reserve_currency_demand, reserve_is_constrained,
    settle, update_reserves,
)
from model.trade import trade_shares

U = 0.02
BLOC_D = 3


def _g(settlement: bool = True, **kw) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation="cpi_indexed", settlement=settlement, **kw)


def _run(settlement: bool = True, modality: str = "ubi", **kw):
    g = _g(settlement, **kw)
    return simulate(build(g, modality, U, (1, 1, 1, 1)), g)


# --- the primitive ---

def test_imports_are_capped_at_what_can_be_paid_for() -> None:
    out = settle(desired=0.10, exports=0.06, reserves=0.02, borrowing=0.01)
    assert out.realised == pytest.approx(0.09)
    assert out.rationed == pytest.approx(0.01)
    assert out.bound


def test_a_bloc_with_enough_foreign_exchange_is_not_rationed() -> None:
    out = settle(desired=0.05, exports=0.06, reserves=0.02, borrowing=0.01)
    assert out.realised == pytest.approx(0.05)
    assert not out.bound


def test_the_reserve_issuer_settles_in_its_own_money() -> None:
    out = settle(desired=0.50, exports=0.01, reserves=0.0, borrowing=0.0,
                 exempt=True)
    assert out.realised == pytest.approx(0.50)
    assert not out.bound


def test_the_borrowing_window_closes_as_risk_rises() -> None:
    wide = borrowing_capacity(1.0, 0.05, risk_premium=0.0, sensitivity=4.0)
    narrow = borrowing_capacity(1.0, 0.05, risk_premium=0.1, sensitivity=4.0)
    shut = borrowing_capacity(1.0, 0.05, risk_premium=0.5, sensitivity=4.0)
    assert wide > narrow > 0
    assert shut == 0.0


def test_reserves_are_a_residual_and_cannot_go_negative() -> None:
    assert update_reserves(0.1, exports=0.05, realised_imports=0.04,
                           net_capital=0.0) == pytest.approx(0.11)
    assert update_reserves(0.01, exports=0.0, realised_imports=0.5,
                           net_capital=0.0) == 0.0


# --- the reserve currency's exemption is conditional ---

def test_demand_for_the_reserve_currency_falls_as_it_loses_value() -> None:
    strong = reserve_currency_demand(1.0, base_demand=1.0, elasticity=0.5)
    weak = reserve_currency_demand(3.0, base_demand=1.0, elasticity=0.5)
    assert weak < strong


def test_the_exemption_lapses_when_nobody_wants_the_currency() -> None:
    assert not reserve_is_constrained(0.9, base_demand=1.0, threshold=0.7)
    assert reserve_is_constrained(0.5, base_demand=1.0, threshold=0.7)


# --- the constraint reaches consumption ---

def test_rationing_lowers_what_households_secure() -> None:
    """The point of the constraint. If it only moved the balance of payments
    it would not be a constraint on anybody."""
    free = _run(settlement=False)["quantities"]
    bound = _run(settlement=True)["quantities"]
    for good in ("F", "H"):
        assert (bound[30][f"{good}_{BLOC_D}"] / bound[1][f"{good}_{BLOC_D}"]
                < free[30][f"{good}_{BLOC_D}"] / free[1][f"{good}_{BLOC_D}"])


def test_running_out_of_money_is_not_read_as_strength() -> None:
    """A rationed bloc buys less, so its current account improves. Taking that
    at face value would appreciate its currency exactly when it can no longer
    pay its bills — the model must depreciate it instead."""
    free = _run(settlement=False)["history"][30][f"e_{BLOC_D}"]
    bound = _run(settlement=True)["history"][30][f"e_{BLOC_D}"]
    assert bound > free


# --- depreciation has to earn something ---

def test_a_depreciated_bloc_wins_export_share() -> None:
    from model.params import archetypes
    blocs = list(archetypes().values())
    even = trade_shares(blocs, [1.0] * 4, elasticity=1.0)
    weak_d = trade_shares(blocs, [1.0, 1.0, 1.0, 2.0], elasticity=1.0)
    # From bloc A's perspective, D supplies more once D has depreciated.
    assert weak_d[0][3] > even[0][3]


def test_the_fixed_matrix_is_recovered_at_zero_elasticity() -> None:
    from model.params import archetypes
    blocs = list(archetypes().values())
    assert trade_shares(blocs) == trade_shares(blocs, [1.0, 2.0, 0.5, 3.0],
                                               elasticity=0.0)


def test_competitiveness_relieves_the_constraint() -> None:
    """Without this channel a bloc can never earn its way out, and the
    constraint tightens without limit — an artefact of the trade matrix rather
    than a property of the economy."""
    stuck = _run(settlement=True, trade_elasticity=0.0)["settlement"][-1]
    adjusting = _run(settlement=True, trade_elasticity=1.0)["settlement"][-1]
    assert adjusting[f"share_{BLOC_D}"] < stuck[f"share_{BLOC_D}"]


# --- identities survive rationing ---

@pytest.mark.parametrize("elasticity", [0.0, 1.0])
def test_the_external_identities_hold_under_rationing(elasticity: float) -> None:
    """Cutting one bloc's imports removes another's exports, so the world has
    to clear again on what was actually settled."""
    result = _run(settlement=True, trade_elasticity=elasticity)
    blocs = build(_g(trade_elasticity=elasticity), "ubi", U, (1, 1, 1, 1))
    total = sum(result["final"][b.name]["nfa"] for b in blocs)
    assert abs(total) < 1e-3, f"net foreign assets sum to {total}"


def test_world_demand_falls_when_imports_are_rationed() -> None:
    """Goods that could not be paid for were not bought: they leave world
    demand rather than being carried somewhere else."""
    free = _run(settlement=False)["world"]
    bound = _run(settlement=True)["world"]
    assert bound[15]["demand"] < free[15]["demand"]


# --- the flag is off by default ---

def test_settlement_is_off_by_default() -> None:
    assert G().settlement is False


def test_the_constraint_off_reproduces_phase_c() -> None:
    plain = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
              indexation="cpi_indexed")
    explicit = _g(settlement=False)
    a = simulate(build(plain, "ubi", U, (1, 1, 1, 1)), plain)
    b = simulate(build(explicit, "ubi", U, (1, 1, 1, 1)), explicit)
    assert a["quantities"] == b["quantities"]
