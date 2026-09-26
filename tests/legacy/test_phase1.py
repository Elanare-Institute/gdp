"""Phase 1: parameter-space mapping and the leakage metrics.

These check the machinery of the sweep, not its conclusions: the sampler must
cover the cube, the mapping must respect declared bounds, and the three
leakage normalisations must be able to disagree (which is why all three are
reported).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from analysis.legacy.phase1_run import evaluate, lhs
from analysis.legacy.phase1_spec import (
    PARAM_NAMES, POST_HOC_CANDIDATE, PRIOR_EXPECTATION, apply_point, unit_to_value,
    PARAMS,
)
from legacy.v7.core import build, simulate
from legacy.v7.household import household_block
from legacy.v7.params import G, archetypes


def test_lhs_is_seeded_and_reproducible() -> None:
    """Same seed, same design."""
    assert lhs(50, 7) == lhs(50, 7)
    assert lhs(50, 7) != lhs(50, 8)


def test_lhs_stratifies_every_dimension() -> None:
    """Each parameter gets one draw per stratum — the defining LHS property."""
    n = 100
    design = lhs(n, 3)
    for name in PARAM_NAMES:
        strata = {int(point[name] * n) for point in design}
        assert len(strata) == n, name


def test_sampled_points_respect_declared_bounds() -> None:
    """Mapping the unit cube stays inside the documented caps."""
    for unit_value in (0.0, 0.5, 1.0):
        blocs, g, u = apply_point({name: unit_value for name in PARAM_NAMES})
        assert 0.02 <= u <= 0.15
        assert g.sub_F + g.sub_G + g.sub_H < 0.95
        for b in blocs:
            assert b.land_share <= 0.8
            assert 0.0 <= b.clt_domestic_sourcing <= 1.0
            for m in (b.m_F, b.m_G, b.m_H):
                assert 0.0 <= m <= 0.95


def test_phase1_uses_capped_reserve_flight() -> None:
    """The Phase 1 main series runs under the agreed reserve-flight mode."""
    _, g, _ = apply_point({name: 0.5 for name in PARAM_NAMES})
    assert g.reserve_flight_mode == "capped"


def test_log_scale_parameter_is_symmetric_in_logs() -> None:
    """eps_supply is swept log-uniformly, so the midpoint is geometric."""
    spec = {p.name: p for p in PARAMS}["eps_supply"]
    lo = unit_to_value(spec, 0.0, 1.0)
    mid = unit_to_value(spec, 0.5, 1.0)
    hi = unit_to_value(spec, 1.0, 1.0)
    assert mid == pytest.approx((lo * hi) ** 0.5, rel=1e-9)


def test_evaluate_returns_all_three_metrics() -> None:
    """A feasible point yields every metric for every bloc and basis."""
    row = evaluate(lhs(4, 11)[0], "staggered")
    assert row is not None
    for basis in ("welfare", "cost"):
        for bloc in ("A", "B", "C", "D"):
            for metric in ("leak_abs", "leak_per_fiscal", "leak_per_welfare"):
                assert f"{basis}.{bloc}.d_{metric}" in row


def test_metrics_can_disagree_in_sign() -> None:
    """The three normalisations are not redundant.

    Across a modest sample at least one point ranks CLT and cash differently
    under leak_abs than under leak_per_fiscal. This is the reason the report
    must always say which metric a claim refers to.
    """
    disagreements = 0
    for point in lhs(40, 5):
        row = evaluate(point, "staggered")
        if row is None:
            continue
        a = row.get("welfare.D.d_leak_abs")
        b = row.get("welfare.D.d_leak_per_fiscal")
        if a is not None and b is not None and (a < 0) != (b < 0):
            disagreements += 1
    assert disagreements > 0


def test_domestic_sourcing_reduces_leakage_monotonically() -> None:
    """Raising the sourcing lever cannot increase construction imports."""
    g = G()
    b = archetypes()["D"]
    previous = None
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        hb = household_block(replace(b, modality="clt", clt_domestic_sourcing=rate),
                             g, b.gdp, 1.0, b.rate, True, 0.03)
        if previous is not None:
            assert hb["imp_gov"] <= previous + 1e-12
        previous = hb["imp_gov"]


def test_prior_expectation_is_recorded_verbatim() -> None:
    """The pre-registered prior names eps_supply and land_share.

    It is kept unchanged so Phase 1 can score it honestly; the m_H candidate
    raised by Phase 0 is stored separately as post-hoc.
    """
    assert "eps_supply" in PRIOR_EXPECTATION and "land_share" in PRIOR_EXPECTATION
    assert "m_H" not in PRIOR_EXPECTATION
    assert "m_H" in POST_HOC_CANDIDATE
    assert "post" in POST_HOC_CANDIDATE.lower() or "after Phase 0" in POST_HOC_CANDIDATE


# --- Phase 1 follow-up: sourcing cost, scale-free targets ------------------

def test_sourcing_markup_defaults_to_free() -> None:
    """kappa = 0 reproduces the costless-sourcing behaviour."""
    g = G(sourcing_cost_kappa=0.0)
    b = archetypes()["D"]
    plain = household_block(replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, 0.03)
    sourced = household_block(replace(b, modality="clt", clt_domestic_sourcing=1.0),
                              g, b.gdp, 1.0, b.rate, True, 0.03)
    assert sourced["fiscal"] == pytest.approx(plain["fiscal"], rel=1e-12)


def test_sourcing_markup_raises_fiscal_cost() -> None:
    """With kappa > 0, sourcing domestically costs more per unit built."""
    g = G(sourcing_cost_kappa=0.5)
    b = archetypes()["D"]
    plain = household_block(replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, 0.03)
    sourced = household_block(replace(b, modality="clt", clt_domestic_sourcing=1.0),
                              g, b.gdp, 1.0, b.rate, True, 0.03)
    assert sourced["fiscal"] > plain["fiscal"]


def test_sourcing_markup_trades_imports_against_cost() -> None:
    """Sourcing still cuts imports, but no longer for free."""
    g = G(sourcing_cost_kappa=0.5)
    b = archetypes()["D"]
    plain = household_block(replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, 0.03)
    sourced = household_block(replace(b, modality="clt", clt_domestic_sourcing=1.0),
                              g, b.gdp, 1.0, b.rate, True, 0.03)
    assert sourced["imp_gov"] < plain["imp_gov"]
    assert sourced["dom_gov"] > plain["dom_gov"]


def test_evaluate_records_levels_and_scale_free_targets() -> None:
    """Levels, ratios and sign indicators are all stored for later analysis."""
    row = evaluate(lhs(4, 13)[0], "staggered")
    assert row is not None
    for field in ("cash_leak_abs", "clt_leak_abs", "ratio_leak_abs", "sign_leak_abs"):
        assert f"welfare.D.{field}" in row


def test_ratio_is_consistent_with_the_levels() -> None:
    """ratio = CLT / cash, and the sign indicator agrees with the difference."""
    row = evaluate(lhs(4, 17)[0], "staggered")
    assert row is not None
    cash = row["welfare.D.cash_leak_abs"]
    clt = row["welfare.D.clt_leak_abs"]
    assert row["welfare.D.ratio_leak_abs"] == pytest.approx(clt / cash, rel=1e-9)
    assert row["welfare.D.sign_leak_abs"] == float(clt - cash < 0)


def test_matched_ranges_equalize_the_two_levers() -> None:
    """The matched variant gives m_H and the sourcing lever comparable spans.

    In the headline sweep the policy lever spans the whole [0,1] of realised
    import content while m_H moves only +/-50%. Any 'policy beats structure'
    claim has to survive equalising that, so the variant exists to test it.
    """
    from analysis.legacy.phase1_spec import matched_params

    spec = {p.name: p for p in matched_params()}
    assert spec["m_H"].low == 0.0 and spec["m_H"].high == 2.0
    assert spec["clt_domestic_sourcing"].low == 0.0
    assert "matched" in spec["m_H"].note


# --- instrument choice for the sign question -------------------------------

def test_sign_sobol_indices_are_not_separable() -> None:
    """The Sobol decomposition cannot rank drivers of the SIGN.

    The sign is a binary with little variance, so at the sample sizes this
    project can afford the leading total-order indices have overlapping
    confidence intervals. Reporting their order would be reading noise. This
    is why the sign question is judged by LHS stratification instead.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase1" / "analysis.json"
    if not path.exists():
        pytest.skip("phase 1 results not present")
    sobol = json.loads(path.read_text()).get("sobol", {})
    idx = sobol.get("sign_leak_abs|cost|A")
    if not idx:
        pytest.skip("sign Sobol indices not present")

    ranked = sorted(idx["ST"].items(), key=lambda kv: -kv[1])
    (top_name, top_val), (second_name, second_val) = ranked[0], ranked[1]
    # The top two are within each other's confidence intervals.
    assert top_val - idx["ST_conf"][top_name] <= second_val + idx["ST_conf"][second_name]


def test_lhs_stratification_separates_sign_drivers() -> None:
    """The LHS instrument does separate them, by a wide margin.

    Land parameters clear t > 30 in the advanced blocs while the elasticity of
    supply sits near t = 1, which is what makes the prior judgeable at all.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase1" / "analysis.json"
    if not path.exists():
        pytest.skip("phase 1 results not present")
    drivers = json.loads(path.read_text()).get("sign_drivers", {}).get("cost|A")
    if not drivers:
        pytest.skip("sign drivers not present")

    by_name = {d["param"]: d["t"] for d in drivers["ranked"]}
    assert by_name["land_share"] > 20.0
    assert by_name["eps_supply"] < 5.0
    assert by_name["land_share"] > 10 * by_name["eps_supply"]


# --- corrected findings (Phase 1 revision) ---------------------------------

def test_m_H_and_sourcing_are_identical_on_the_import_channel() -> None:
    """Without the cost markup the two levers are the same parameter.

    ``imp_gov = m_H * (1 - s) * clt_build`` is the only place m_H appears, so
    moving realised imports via m_H or via s must give identical results when
    kappa = 0. This is what retracts the earlier 'policy beats structure'
    reading: the asymmetry came from kappa, not from the import channel.
    """
    g = G(sourcing_cost_kappa=0.0)
    b = archetypes()["D"]
    for frac in (1.0, 0.5, 0.25, 0.0):
        via_m = household_block(
            replace(b, modality="clt", m_H=0.25 * frac, clt_domestic_sourcing=0.0),
            g, b.gdp, 1.0, b.rate, True, 0.03)
        via_s = household_block(
            replace(b, modality="clt", m_H=0.25, clt_domestic_sourcing=1.0 - frac),
            g, b.gdp, 1.0, b.rate, True, 0.03)
        assert via_m["imp_gov"] == pytest.approx(via_s["imp_gov"], rel=1e-12)
        assert via_m["fiscal"] == pytest.approx(via_s["fiscal"], rel=1e-12)


def test_sourcing_differs_from_m_H_only_through_cost() -> None:
    """With kappa > 0 the levers diverge, and only in fiscal cost."""
    g = G(sourcing_cost_kappa=0.5)
    b = archetypes()["D"]
    via_m = household_block(
        replace(b, modality="clt", m_H=0.0, clt_domestic_sourcing=0.0),
        g, b.gdp, 1.0, b.rate, True, 0.03)
    via_s = household_block(
        replace(b, modality="clt", m_H=0.25, clt_domestic_sourcing=1.0),
        g, b.gdp, 1.0, b.rate, True, 0.03)

    assert via_m["imp_gov"] == pytest.approx(via_s["imp_gov"], abs=1e-12)  # same imports
    assert via_s["fiscal"] > via_m["fiscal"]                               # different cost


def test_m_H_appears_only_in_construction_imports() -> None:
    """m_H must not touch household consumption imports.

    Household imports run through m_F and m_G. A change to m_H may only move
    government construction imports.
    """
    g = G()
    b = archetypes()["D"]
    base = household_block(replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, 0.03)
    bumped = household_block(replace(b, modality="clt", m_H=b.m_H * 2),
                             g, b.gdp, 1.0, b.rate, True, 0.03)
    assert bumped["imp_c"] == pytest.approx(base["imp_c"], rel=1e-12)
    assert bumped["imp_gov"] > base["imp_gov"]


def test_composite_variables_lead_the_sign_regression() -> None:
    """The CLT's own inputs top the logistic regression in every fitted bloc.

    Land-intensive blocs are led by effective land rent; the import-intensive
    bloc by effective build imports. This is the unified principle.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase1" / "sign_logistic.json"
    if not path.exists():
        pytest.skip("sign regression results not present")
    fitted = json.loads(path.read_text())

    expected_leader = {"A": "eff_land_rent", "B": "eff_land_rent",
                       "D": "eff_build_imports"}
    for bloc, leader in expected_leader.items():
        if bloc not in fitted:
            continue
        coef = fitted[bloc]["coefficients"]
        top = max(coef, key=lambda k: abs(coef[k]))
        assert top == leader, f"bloc {bloc}: expected {leader}, got {top}"
        assert coef[leader] > 0, bloc  # more fugitive inputs -> more reversal


# --- classification quality of the sign models -----------------------------

def _eval_results():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase1" / "sign_eval.json"
    if not path.exists():
        pytest.skip("sign evaluation results not present")
    return json.loads(path.read_text())


@pytest.mark.parametrize("bloc", ["A", "B"])
def test_sign_model_beats_majority_in_land_blocs(bloc: str) -> None:
    """In A and B the classifier is clearly better than always saying 'no'."""
    r = _eval_results()[bloc]
    assert r["logistic"]["accuracy"] > r["majority_baseline"] + 0.05
    assert r["logistic"]["balanced_accuracy"] > 0.80
    assert r["logistic"]["auc"] > 0.90


def test_class_weighting_fixes_the_rare_class_in_bloc_D() -> None:
    """Bloc D's weak tree was an artifact of leaving the imbalance untreated.

    Unweighted, the tree mostly predicts the majority label (balanced accuracy
    0.625). With class_weight='balanced' it reaches A/B levels. The earlier
    reading of D as "cannot classify" was wrong for this reason.
    """
    r = _eval_results()["D"]
    assert r["n_reversals"] < 200
    assert r["tree"]["balanced_accuracy"] < 0.75             # untreated
    assert r["tree_balanced"]["balanced_accuracy"] > 0.85    # treated
    assert r["tree_balanced"]["balanced_accuracy"] > r["tree"]["balanced_accuracy"] + 0.2


def test_pr_auc_beats_the_event_rate_baseline() -> None:
    """PR-AUC is the honest metric under imbalance; every bloc clears it.

    Its baseline is the event rate, so bloc D's rarity makes its lift the
    largest rather than the smallest.
    """
    results = _eval_results()
    for bloc, r in results.items():
        assert r["logistic"]["pr_auc"] > r["event_rate"] * 2, bloc
    d = results["D"]
    assert d["logistic"]["pr_auc"] / d["event_rate"] > 10.0


def test_bloc_D_coefficients_are_stable_under_resampling() -> None:
    """D's leading coefficients do not cross zero, and ridge preserves them.

    With 144 events over 8 predictors (EPV 18) the point estimates need an
    interval before they can be reported; these are it.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase1" / "sign_ci.json"
    if not path.exists():
        pytest.skip("bootstrap results not present")
    d = json.loads(path.read_text())["D"]

    assert d["epv"] > 10.0
    for name in ("eff_build_imports", "omega0"):
        assert not d["ci"][name]["crosses_zero"], name
        # ridge shrinks but must not flip the sign
        assert d["ridge"][name] * d["point"][name] > 0, name
    assert d["point"]["eff_build_imports"] > 0
    assert d["point"]["omega0"] < 0


def test_cash_side_coefficients_are_negative_everywhere() -> None:
    """The cash-side channel enters with the sign the principle requires.

    More fugitive destinations for cash make cash the leakier option, so the
    CLT comparison reverses less often. Both cash-side regressors must
    therefore carry negative coefficients in every fitted bloc.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase1" / "sign_logistic.json"
    if not path.exists():
        pytest.skip("sign regression results not present")
    fitted = json.loads(path.read_text())

    for bloc, res in fitted.items():
        coef = res["coefficients"]
        assert coef["omega0"] < 0, bloc
        assert coef["mpc_u"] < 0, bloc


def test_clt_purchase_side_is_positive_where_it_leads() -> None:
    """The leading CLT-input regressor is positive in each bloc."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase1" / "sign_logistic.json"
    if not path.exists():
        pytest.skip("sign regression results not present")
    fitted = json.loads(path.read_text())

    leaders = {"A": "eff_land_rent", "B": "eff_land_rent", "D": "eff_build_imports"}
    for bloc, leader in leaders.items():
        if bloc in fitted:
            assert fitted[bloc]["coefficients"][leader] > 0, bloc
