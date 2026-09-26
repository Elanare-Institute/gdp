"""The acceptance condition for phase A/B.

CLAUDE.md: "全ブロックが同一の現金給付を行ったときに世界物価が上がることを
確認してから C・D に進む". That is the headline test here, together with the
identities that have to hold inside the full simulation rather than only in
the pure functions of `test_trade.py`.

The main series runs at a transfer size small enough that no bloc becomes
insolvent (u <= 0.01). Beyond that, bloc D's debt runs away and it keeps
importing on credit that nothing supplies, which distorts everyone's exchange
rate. Rationing those imports is what phase D is for; until it exists, the
acceptance condition is checked where the question is well posed, and the
runaway is recorded as a known limit rather than papered over.

Specification: `specs/V8_A_B.md` §4.
"""

from __future__ import annotations

import pytest

from model.core import build, simulate
from model.params import G

#: Largest transfer at which no bloc goes insolvent within 30 years.
SAFE_U = 0.01
#: Adding-up tolerance inside the full simulation. Looser than the pure
#: functions' 1e-10 only because reported values are rounded to 5 decimals.
SIM_TOL = 1e-4


def _g() -> G:
    """Main series: heterogeneous households, phased build, no privileges."""
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)


def _world_price(result) -> float:
    return result["world_final"]["price"]


def _insolvent(result) -> list[str]:
    return [k for k, v in result["crisis"].items()
            if k != "_system" and v.get("insolvent_year") is not None]


# --- the acceptance condition ---

def test_a_universal_cash_transfer_raises_the_world_price() -> None:
    """The headline: cash escapes a country, not the world."""
    g = _g()
    quiet = simulate(build(g, "none"), g)
    paying = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)
    assert _world_price(paying) > _world_price(quiet)


def test_the_world_price_rises_monotonically_in_the_number_of_payers() -> None:
    """Leakage relocates the problem; it does not dispose of it."""
    g = _g()
    prices = []
    for which in ((), ("D",), ("C", "D"), ("B", "C", "D"), ("A", "B", "C", "D")):
        blocs = (build(g, "ubi", SAFE_U, (1, 1, 1, 1), "welfare", which)
                 if which else build(g, "none"))
        prices.append(_world_price(simulate(blocs, g)))
    assert prices == sorted(prices), prices
    assert prices[-1] > prices[0]


@pytest.mark.parametrize("u", [0.002, 0.005, 0.01])
def test_a_bigger_transfer_raises_the_world_price_by_more(u: float) -> None:
    g = _g()
    smaller = _world_price(simulate(build(g, "ubi", u / 2, (1, 1, 1, 1)), g))
    larger = _world_price(simulate(build(g, "ubi", u, (1, 1, 1, 1)), g))
    assert larger > smaller


def test_the_baseline_world_price_is_nearly_flat() -> None:
    """With no programme anywhere, world demand and capacity grow together.

    If the baseline drifted, every result above would be reading a trend
    rather than a transfer.
    """
    g = _g()
    assert _world_price(simulate(build(g, "none"), g)) == pytest.approx(1.0, abs=0.05)


def test_no_bloc_is_insolvent_in_the_main_series() -> None:
    """The range within which the acceptance condition is posed."""
    g = _g()
    assert _insolvent(simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)) == []


# --- spillover: the point of closing the world ---

def test_a_bloc_that_paid_nothing_still_faces_the_higher_world_price() -> None:
    g = _g()
    quiet = simulate(build(g, "none"), g)
    one_payer = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1), "welfare", ("D",)), g)
    assert _world_price(one_payer) > _world_price(quiet)


def test_one_blocs_transfer_moves_other_blocs_output() -> None:
    """In v7 this leakage left the model; here it arrives as somebody's export."""
    g = _g()
    quiet = simulate(build(g, "none"), g)
    one_payer = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1), "welfare", ("D",)), g)
    movers = [name for name in quiet["final"]
              if not name.startswith("D")
              and quiet["final"][name]["Y"] != one_payer["final"][name]["Y"]]
    assert movers, "no bloc outside D noticed D's programme"


# --- identities inside the running simulation ---

@pytest.mark.parametrize("modality", ["none", "ubi", "cash_t", "voucher", "clt"])
def test_net_foreign_assets_sum_to_zero_over_the_whole_run(modality: str) -> None:
    """The world cannot end up owing itself anything."""
    g = _g()
    u = 0.0 if modality == "none" else SAFE_U
    blocs = build(g, modality, u, (1, 1, 1, 1))
    result = simulate(blocs, g)
    total = sum(result["final"][b.name]["nfa"] for b in blocs)
    assert abs(total) < SIM_TOL, f"{modality}: net foreign assets sum to {total}"


def test_world_demand_is_positive_throughout() -> None:
    g = _g()
    result = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)
    assert all(row["demand"] > 0 for row in result["world"])


def test_the_run_is_reproducible() -> None:
    g = _g()
    a = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)
    b = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)
    assert a["world"] == b["world"]
    assert a["final"] == b["final"]


# --- reserve privileges are off ---

def test_reserve_privileges_are_off_by_default() -> None:
    assert G().reserve_privilege is False


def test_the_privilege_flag_changes_the_reserve_blocs_path() -> None:
    """Kept as a sensitivity, so it must still do something when switched on."""
    plain = simulate(build(_g(), "ubi", SAFE_U, (1, 1, 1, 1)), _g())
    privileged_g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
                     reserve_privilege=True)
    privileged = simulate(build(privileged_g, "ubi", SAFE_U, (1, 1, 1, 1)),
                          privileged_g)
    reserve = "A (reserve)"
    assert plain["final"][reserve] != privileged["final"][reserve]


# --- the known limit, recorded as a test so it cannot be forgotten ---

def test_large_transfers_drive_bloc_d_insolvent() -> None:
    """Not a defect to be silenced: the ceiling phase D has to explain.

    Above roughly u = 0.015 bloc D's debt passes the default threshold and it
    goes on importing on credit nobody is supplying. The settlement constraint
    is what will ration that; until then this test pins where the boundary is,
    so a later change that moves it is visible.
    """
    g = _g()
    assert _insolvent(simulate(build(g, "ubi", 0.03, (1, 1, 1, 1)), g)) != []


# --- where capital outflows go ---

def test_foreign_asset_purchases_reach_the_capital_account() -> None:
    """A bloc's purchase of foreign assets must be somebody's inflow.

    v7 logged this quantity as "leakage" and then dropped it, so the world was
    closed for goods and still open for capital. Turning a transfer on must
    move the net foreign asset positions of the blocs that did not run one.
    """
    g = _g()
    quiet = simulate(build(g, "none"), g)
    payer = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1), "welfare", ("D",)), g)
    others = [n for n in quiet["final"] if not n.startswith("D")]
    assert any(quiet["final"][n]["nfa"] != payer["final"][n]["nfa"] for n in others)


def test_leakage_is_reported_split_not_conflated() -> None:
    """Imports and foreign assets buy different things and are reported apart.

    Conflating them is what made Phase 1's and Phase 3's leakage numbers
    incomparable; the split must survive in v8's own output.
    """
    g = _g()
    cells = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)["leakage"]
    assert cells
    for cell in cells.values():
        for key in ("import_abs", "foreign_assets_abs",
                    "import_per_fiscal", "foreign_assets_per_fiscal"):
            assert key in cell, key
        assert cell["import_abs"] + cell["foreign_assets_abs"] == \
               pytest.approx(cell["leak_abs"], abs=1e-4)


def test_capital_inflow_to_demand_is_off_by_default() -> None:
    """The main series assumes an inflow buys claims, not goods."""
    assert G().capital_inflow_to_demand == 0.0


def test_charging_capital_inflows_to_demand_raises_the_world_price() -> None:
    """The sensitivity must do what it says, so the assumption can be judged."""
    def price(share: float) -> float:
        g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
              capital_inflow_to_demand=share)
        return simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)["world_final"]["price"]
    assert price(0.25) > price(0.0)


def test_the_modality_ordering_survives_the_capital_assumption() -> None:
    """UBI presses on the world market harder than CLT either way.

    Most of UBI's leakage is financial and most of CLT's is imports, so
    counting capital inflows as goods demand could in principle have reversed
    the ordering. It does not: it widens the gap, because it is UBI that has
    the financial leakage to re-count.
    """
    for share in (0.0, 0.25, 0.5):
        g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
              capital_inflow_to_demand=share)
        ubi = simulate(build(g, "ubi", SAFE_U, (1, 1, 1, 1)), g)["world_final"]["price"]
        clt = simulate(build(g, "clt", SAFE_U, (1, 1, 1, 1)), g)["world_final"]["price"]
        assert ubi > clt, f"ordering flipped at capital_inflow_to_demand={share}"
