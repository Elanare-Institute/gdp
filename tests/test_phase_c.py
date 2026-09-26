"""The self-defeating loop, wired into the macro dynamics.

Phases A/B size transfers as `size * Y` with `Y` real, so inflation never
reaches the transfer. Phase C carries the transfer as a nominal amount, which
opens the loop the paper's first claim is about:

    inflation -> the transfer is indexed up -> fiscal cost and demand rise
    -> world demand rises -> world prices rise -> inflation

These tests fix that the loop is closed, that the default reproduces phases
A/B exactly, and that an in-kind programme is not eroded the way a cash one is.

Specification: `specs/V8_C.md` §6.
"""

from __future__ import annotations

import pytest

from model.core import build, simulate
from model.params import G

U = 0.02


def _g(rule: str = "gdp_linked") -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02, indexation=rule)


def _run(rule: str, modality: str = "ubi", u: float = U):
    g = _g(rule)
    blocs = build(g, modality, u, (1, 1, 1, 1)) if u > 0 else build(g, "none")
    return simulate(blocs, g)


def _price(result) -> float:
    return result["world_final"]["price"]


# --- the default reproduces phases A/B ---

def test_gdp_linked_is_the_default() -> None:
    assert G().indexation == "gdp_linked"


@pytest.mark.parametrize("modality", ["none", "ubi", "cash_t", "voucher", "clt"])
def test_gdp_linked_reproduces_the_phase_ab_run(modality: str) -> None:
    """Phase C must be an addition, not a change to what came before."""
    g_default = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    g_explicit = _g("gdp_linked")
    u = 0.0 if modality == "none" else U
    a = simulate(build(g_default, modality, u, (1, 1, 1, 1)), g_default)
    b = simulate(build(g_explicit, modality, u, (1, 1, 1, 1)), g_explicit)
    assert a["world"] == b["world"]
    assert a["final"] == b["final"]


# --- the loop is closed under indexation ---

def test_indexation_generates_more_world_inflation_than_either_alternative() -> None:
    """The headline: indexing a cash transfer feeds the inflation it chases.

    The ordering is what the claim rests on, not the size of the gap. An
    indexed transfer costs more as prices rise, and that extra spending is
    extra demand; a fixed nominal one costs less in real terms every year and
    presses on the world market correspondingly less.
    """
    indexed = _price(_run("cpi_indexed"))
    linked = _price(_run("gdp_linked"))
    fixed = _price(_run("fixed_nominal"))
    assert indexed > linked > fixed


def test_the_loop_needs_indexation_to_run() -> None:
    """Without indexation the transfer cannot chase the price level."""
    assert _price(_run("fixed_nominal")) < _price(_run("cpi_indexed"))


def test_a_larger_indexed_transfer_drives_the_loop_harder() -> None:
    assert _price(_run("cpi_indexed", u=0.02)) > _price(_run("cpi_indexed", u=0.005))


def test_the_gap_between_the_rules_widens_with_the_transfer() -> None:
    """The loop compounds: a larger indexed transfer is disproportionately
    more inflationary than the same transfer left unindexed."""
    small = _price(_run("cpi_indexed", u=0.01)) / _price(_run("gdp_linked", u=0.01))
    large = _price(_run("cpi_indexed", u=0.03)) / _price(_run("gdp_linked", u=0.03))
    assert large > small


def test_the_price_path_rises_smoothly() -> None:
    """A jump in a single step would point at arithmetic, not at a mechanism."""
    path = [row["price"] for row in _run("cpi_indexed")["world"]]
    assert all(b >= a for a, b in zip(path, path[1:])), "no step reverses"
    steps = [b / a for a, b in zip(path, path[1:]) if a > 0]
    assert max(steps) < 1.2, "no single year may carry the whole increase"


def test_no_transfer_is_unaffected_by_the_indexation_rule() -> None:
    """With nothing handed out there is nothing to index."""
    assert _price(_run("cpi_indexed", u=0.0)) == pytest.approx(
        _price(_run("fixed_nominal", u=0.0)))


# --- an in-kind programme is not eroded the same way ---

def test_in_kind_housing_is_less_exposed_than_cash() -> None:
    """CLAUDE.md §E(i): what the trust built does not shrink when prices rise.

    Both are carried as nominal budgets under indexation, so both cost more as
    prices rise. The difference is what the money bought: housing already
    built goes on housing the same households, while cash handed out last year
    buys less this year.
    """
    cash = _price(_run("cpi_indexed", "ubi"))
    housing = _price(_run("cpi_indexed", "clt"))
    assert housing < cash


def test_every_modality_is_affected_by_the_rule() -> None:
    """A modality whose path ignores the rule would not be wired in."""
    for modality in ("ubi", "cash_t", "voucher", "clt"):
        indexed = _price(_run("cpi_indexed", modality))
        linked = _price(_run("gdp_linked", modality))
        assert indexed != linked, modality


# --- determinism ---

@pytest.mark.parametrize("rule", ["gdp_linked", "fixed_nominal", "cpi_indexed"])
def test_each_rule_is_reproducible(rule: str) -> None:
    assert _run(rule)["world"] == _run(rule)["world"]


# --- what the transfer ends up worth ---

def _real_value(rule: str, modality: str = "ubi", u: float = U) -> dict[str, float]:
    return _run(rule, modality, u)["real_value"]


def test_a_gdp_linked_transfer_cannot_lose_real_value() -> None:
    """`Y` is real, so `size * Y` is real. This is what phase C changes."""
    for value in _real_value("gdp_linked").values():
        assert value == pytest.approx(1.0, abs=1e-6)


def test_a_fixed_nominal_transfer_loses_most_of_its_value() -> None:
    for value in _real_value("fixed_nominal").values():
        assert value < 0.5


def test_indexation_holds_the_transfer_near_its_opening_value() -> None:
    """Near, not exactly: indexation lags by a period by construction."""
    for name, value in _real_value("cpi_indexed").items():
        assert 0.9 < value < 1.05, f"{name}: {value}"


def test_the_periphery_loses_the_most_when_nothing_is_indexed() -> None:
    """Higher inflation, so a fixed nominal transfer is eroded faster there."""
    values = _real_value("fixed_nominal")
    assert values["D (developing)"] < values["A (reserve)"]


def test_indexation_does_not_outrun_the_price_level_after_a_default() -> None:
    """An insolvent bloc's inflation is frozen along with the rest of its
    state. Indexing to a frozen rate would raise the transfer against a price
    level that has stopped moving, and the transfer would gain real value by
    virtue of the bloc having defaulted."""
    result = _run("cpi_indexed", u=0.02)
    defaulted = [name for name, cell in result["crisis"].items()
                 if name != "_system" and cell.get("insolvent_year") is not None]
    assert defaulted, "expected a default at this size"
    for name in defaulted:
        assert result["real_value"][name] < 1.05, name
