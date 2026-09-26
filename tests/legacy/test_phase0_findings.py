"""Findings recorded at Phase 0.

These are not desirable properties — they are observations about the v6
baseline that later phases must not erase silently. CLAUDE.md requires that
sign reversals be reported as primary findings rather than smoothed over, so
they are pinned here.
"""

from __future__ import annotations

import pytest

from legacy.v7.core import run_scenarios
from legacy.v7.params import G


@pytest.fixture(scope="module")
def baseline():
    return run_scenarios(G(), 0.05)


def _leak(res, basis: str, modality: str, bloc: str) -> float:
    return res["runs"][f"{basis}:{modality}"]["leakage"][bloc]["external_leak_per_fiscal"]


@pytest.mark.parametrize("bloc", ["B (non-reserve adv.)", "C (emerging)"])
def test_clt_leaks_less_than_cash_in_b_and_c(baseline, bloc: str) -> None:
    """The headline claim holds for the non-reserve advanced and emerging blocs."""
    assert _leak(baseline, "welfare", "clt", bloc) < _leak(baseline, "welfare", "cash_t", bloc)


def test_developing_bloc_reverses_under_both_bases(baseline) -> None:
    """In bloc D, CLT leaks MORE than cash — on both comparison bases.

    D has the highest import content of construction (m_H = 0.25) and of food
    (m_F = 0.30). CLT spends directly on construction, so a large share of the
    fiscal outlay leaves the country as imports, while cash is partly absorbed
    by domestic rents. This is the clearest sign reversal in the baseline.
    """
    d = "D (developing)"
    assert _leak(baseline, "welfare", "clt", d) > _leak(baseline, "welfare", "cash_t", d)
    assert _leak(baseline, "cost", "clt", d) > _leak(baseline, "cost", "cash_t", d)


def test_reserve_bloc_reverses_on_welfare_basis_only(baseline) -> None:
    """Bloc A reverses under welfare equivalence but not under equal cost.

    The direction of the comparison therefore depends on the basis, which is
    why both are reported rather than one.
    """
    a = "A (reserve)"
    assert _leak(baseline, "welfare", "clt", a) > _leak(baseline, "welfare", "cash_t", a)
    assert _leak(baseline, "cost", "clt", a) < _leak(baseline, "cost", "cash_t", a)


def test_cash_transfers_are_captured_as_rent(baseline) -> None:
    """Roughly 0.44-0.48 of each cash unit shows up as extra rent."""
    for bloc, leak in baseline["runs"]["welfare:cash_t"]["leakage"].items():
        assert 0.4 < leak["rent_capture_per_fiscal"] < 0.5, bloc


def test_clt_returns_rent_to_tenants(baseline) -> None:
    """CLT has strongly negative rent capture: it lowers the market rent."""
    for bloc, leak in baseline["runs"]["welfare:clt"]["leakage"].items():
        assert leak["rent_capture_per_fiscal"] < -1.0, bloc


def test_small_transfer_crises_are_entirely_reserve_flight() -> None:
    """Under small welfare-equivalent transfers, every crisis is an artifact.

    In `welfare:cash_t` and `welfare:voucher` the only bloc ever in crisis is
    A, and switching the safe-haven inflow off removes those crises completely.
    """
    v6 = run_scenarios(G(reserve_flight_mode="v6"), 0.05)
    off = run_scenarios(G(reserve_flight_mode="off"), 0.05)

    for scenario in ("welfare:cash_t", "welfare:voucher"):
        assert v6["runs"][scenario]["crisis"]["_system"]["bloc_quarters"] > 0, scenario
        assert off["runs"][scenario]["crisis"]["_system"]["bloc_quarters"] == 0, scenario


def test_large_transfer_crises_survive_without_reserve_flight() -> None:
    """Under large transfers, crises are not purely an artifact.

    Bloc A's severity is dominated by flight (roughly a ninefold reduction when
    it is switched off), but bloc B's is essentially untouched, because B's
    crises come from debt and FX dynamics rather than from safe-haven inflows.
    Reporting the v6 crisis count as a single number therefore mixes an
    artifact with a real effect.
    """
    v6 = run_scenarios(G(reserve_flight_mode="v6"), 0.05)
    off = run_scenarios(G(reserve_flight_mode="off"), 0.05)

    a_v6 = v6["runs"]["welfare:ubi"]["crisis"]["A (reserve)"]["severity"]
    a_off = off["runs"]["welfare:ubi"]["crisis"]["A (reserve)"]["severity"]
    assert a_off < 0.2 * a_v6  # most of A's severity is the artifact

    b_v6 = v6["runs"]["welfare:ubi"]["crisis"]["B (non-reserve adv.)"]["severity"]
    b_off = off["runs"]["welfare:ubi"]["crisis"]["B (non-reserve adv.)"]["severity"]
    assert b_off == pytest.approx(b_v6, rel=0.05)  # B is not driven by flight

    assert off["runs"]["welfare:ubi"]["crisis"]["_system"]["bloc_quarters"] > 0


def test_reserve_bloc_appreciates_implausibly_under_v6() -> None:
    """Bloc A's exchange rate falls far below any plausible range under v6."""
    v6 = run_scenarios(G(reserve_flight_mode="v6"), 0.05)
    e_final = v6["runs"]["welfare:cash_t"]["final"]["A (reserve)"]["e"]
    assert e_final < 0.3  # trips the appreciation threshold by construction


# --- metric dependence (added in Phase 1) -----------------------------------

def _leak_metric(res, basis: str, modality: str, bloc: str, metric: str) -> float:
    return res["runs"][f"{basis}:{modality}"]["leakage"][bloc][metric]


def test_sign_depends_on_the_leakage_metric(baseline) -> None:
    """In bloc D the CLT-vs-cash sign flips between the two normalisations.

    On leak_per_fiscal CLT looks worse (+0.087); on leak_abs it looks better
    (-0.409). Neither is wrong — they answer different questions — but a claim
    that omits the metric is not interpretable. This is why Phase 1 reports
    leak_abs, leak_per_fiscal and leak_per_welfare together.
    """
    d = "D (developing)"
    per_fiscal = (_leak_metric(baseline, "welfare", "clt", d, "leak_per_fiscal")
                  - _leak_metric(baseline, "welfare", "cash_t", d, "leak_per_fiscal"))
    absolute = (_leak_metric(baseline, "welfare", "clt", d, "leak_abs")
                - _leak_metric(baseline, "welfare", "cash_t", d, "leak_abs"))

    assert per_fiscal > 0  # CLT leaks more per unit of fiscal cost
    assert absolute < 0    # yet less as a share of GDP


def test_welfare_equivalent_clt_is_fiscally_smaller(baseline) -> None:
    """The mechanism behind the divergence: CLT buys the same utility for less.

    A smaller denominator makes per-fiscal ratios harsher, so the two metrics
    can rank the modalities differently.
    """
    for bloc in baseline["runs"]["welfare:cash_t"]["leakage"]:
        clt_cost = _leak_metric(baseline, "welfare", "clt", bloc, "fiscal_pct_gdp")
        cash_cost = _leak_metric(baseline, "welfare", "cash_t", bloc, "fiscal_pct_gdp")
        assert clt_cost < cash_cost, bloc
