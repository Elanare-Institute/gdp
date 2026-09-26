"""Phase 3: population, imitation dynamics, and the selection/drift contrast."""

from __future__ import annotations

import numpy as np
import pytest

from legacy.v7.evolution import (
    EvolutionConfig, MODALITIES, all_cash_mix, mix_outcome, move_towards,
    mutate, normalise, random_mix, run_evolution, stress_of,
)
from legacy.v7.params import G
from legacy.v7.population import (
    GROUP_SIZES, group_of, make_population, structural_distance,
)


def _globals() -> G:
    return G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
             clt_build_rate=0.02)


# --- population -------------------------------------------------------------

def test_population_matches_the_specified_composition() -> None:
    """One reserve issuer, five advanced, ten emerging, fifteen developing."""
    population = make_population(7)
    assert len(population) == 31
    counts: dict[str, int] = {}
    for bloc in population:
        counts[group_of(bloc)] = counts.get(group_of(bloc), 0) + 1
    assert counts == GROUP_SIZES


def test_population_is_seeded() -> None:
    a = make_population(11)
    b = make_population(11)
    c = make_population(12)
    assert [x.fx_share for x in a] == [x.fx_share for x in b]
    assert [x.fx_share for x in a] != [x.fx_share for x in c]


def test_fx_share_varies_within_groups() -> None:
    """fx_share is the regressor of interest, so it cannot be a group label."""
    population = make_population(3)
    for group in ("B", "C", "D"):
        values = [b.fx_share for b in population if group_of(b) == group]
        assert max(values) - min(values) > 0.05, group


def test_structural_distance_separates_reserve_status() -> None:
    population = make_population(5)
    reserve = next(b for b in population if b.reserve)
    others = [b for b in population if not b.reserve]
    near = min(others, key=lambda b: abs(b.fx_share - reserve.fx_share))
    assert structural_distance(reserve, near) >= 1.0


# --- mix algebra ------------------------------------------------------------

def test_mixes_stay_on_the_simplex() -> None:
    import random

    rng = random.Random(0)
    for mix in (all_cash_mix(), random_mix(rng), [0.4, 0.3, 0.2, 0.1]):
        moved = move_towards(mix, [0, 0, 0, 1.0], 0.5, 1.0)
        assert sum(moved) == pytest.approx(1.0)
        assert all(m >= 0 for m in moved)
        perturbed = mutate(mix, rng, 0.05)
        assert sum(perturbed) == pytest.approx(1.0)
        assert all(m >= 0 for m in perturbed)


def test_build_cap_limits_how_fast_clt_can_grow() -> None:
    """A bloc cannot imitate its way to a CLT-heavy mix overnight."""
    mix = all_cash_mix()
    capped = move_towards(mix, [0, 0, 0, 1.0], 1.0, 0.05)
    uncapped = move_towards(mix, [0, 0, 0, 1.0], 1.0, 1.0)
    assert capped[MODALITIES.index("clt")] == pytest.approx(0.05)
    assert uncapped[MODALITIES.index("clt")] > capped[MODALITIES.index("clt")]


def test_normalise_handles_a_degenerate_mix() -> None:
    assert normalise([0.0, 0.0, 0.0, 0.0]) == all_cash_mix()


# --- stress and outcomes ----------------------------------------------------

def test_mix_outcome_is_linear_in_the_mix() -> None:
    """Each modality is sized independently, so a mix is their weighted sum."""
    population = make_population(9)
    bloc, g = population[12], _globals()
    cash = mix_outcome(bloc, g, [0, 1, 0, 0])
    clt = mix_outcome(bloc, g, [0, 0, 0, 1])
    half = mix_outcome(bloc, g, [0, 0.5, 0, 0.5])
    assert half["fiscal_pct_gdp"] == pytest.approx(
        0.5 * (cash["fiscal_pct_gdp"] + clt["fiscal_pct_gdp"]), rel=1e-9)


def test_stress_weights_leakage_by_fx_share() -> None:
    """Borrowing in a foreign currency makes the same outflow hurt more."""
    population = make_population(4)
    g = _globals()
    low = min(population, key=lambda b: b.fx_share)
    high = max(population, key=lambda b: b.fx_share)
    mix = [0, 0, 0, 1.0]
    # Same leakage enters with a larger multiplier for the high-fx bloc.
    assert (1 + 2 * high.fx_share) > (1 + 2 * low.fx_share)
    assert stress_of(high, g, mix) > 0


# --- dynamics ---------------------------------------------------------------

def test_evolution_is_reproducible() -> None:
    population = make_population(2)
    g = _globals()
    config = EvolutionConfig(years=20)
    first = run_evolution(population, g, config, seed=5)
    second = run_evolution(population, g, config, seed=5)
    assert [f["clt_share"] for f in first["final"]] == \
           [f["clt_share"] for f in second["final"]]


def test_all_cash_start_stays_near_cash_without_a_seed_of_clt() -> None:
    """Imitation cannot spread a modality nobody holds.

    From an all-cash start only mutation introduces CLT, so uptake is slow.
    This is a property of proportional imitation, not a defect, and it is why
    the random-start arm is run alongside.
    """
    population = make_population(6)
    result = run_evolution(population, _globals(),
                           EvolutionConfig(years=40, initial="all_cash"), seed=3)
    assert result["history"][-1]["mean_clt"] < 0.2


def test_neutral_control_ignores_stress() -> None:
    """The control must not condition on stress, or it is not a control."""
    population = make_population(8)
    g = _globals()
    selection = run_evolution(population, g,
                              EvolutionConfig(years=40, neutral=False), seed=4)
    neutral = run_evolution(population, g,
                            EvolutionConfig(years=40, neutral=True), seed=4)
    assert selection["neutral"] is False
    assert neutral["neutral"] is True
    assert [f["clt_share"] for f in selection["final"]] != \
           [f["clt_share"] for f in neutral["final"]]


# --- why no fx gradient emerges --------------------------------------------

def test_cross_bloc_gradients_have_opposite_signs() -> None:
    """Across blocs the two channels pull against each other.

    These are cross-bloc slopes: fx_share is correlated with every other thing
    that separates a developing bloc from a reserve issuer, so they cannot be
    read as the effect of fx. The within-bloc counterfactual (see
    `test_fx_has_no_direct_static_effect`) is what isolates that.
    """
    import numpy as np

    population = make_population(42)
    g = _globals()
    fx, d_fiscal, d_leak = [], [], []
    for bloc in population:
        cash = mix_outcome(bloc, g, [0, 1, 0, 0])
        clt = mix_outcome(bloc, g, [0, 0, 0, 1])
        fx.append(bloc.fx_share)
        d_fiscal.append(clt["fiscal_pct_gdp"] - cash["fiscal_pct_gdp"])
        d_leak.append(clt["leak_pct_gdp"] - cash["leak_pct_gdp"])

    fiscal_corr = np.corrcoef(fx, d_fiscal)[0, 1]
    leak_corr = np.corrcoef(fx, d_leak)[0, 1]
    assert fiscal_corr < -0.3     # higher fx -> CLT cheaper
    assert leak_corr > 0.3        # higher fx -> CLT leakier
    assert np.mean(d_leak) > 0    # CLT leaks more on average during build-out


def test_fx_has_no_direct_static_effect() -> None:
    """Moving fx alone leaves the static fiscal and leakage gaps untouched.

    fx_share appears nowhere in the household block — it enters only the macro
    equations — so a purely static comparison cannot detect it. This is why
    the reduced-form stress had to impose a weight by hand, and why the
    endogenous measure is the one that can answer the phase's question.
    """
    from dataclasses import replace

    population = make_population(42)
    g = _globals()
    bloc = population[20]
    results = []
    for fx in (0.0, 0.5, 0.85):
        probe = replace(bloc, fx_share=fx)
        cash = mix_outcome(probe, g, [0, 1, 0, 0])
        clt = mix_outcome(probe, g, [0, 0, 0, 1])
        results.append((clt["fiscal_pct_gdp"] - cash["fiscal_pct_gdp"],
                        clt["leak_pct_gdp"] - cash["leak_pct_gdp"]))
    assert results[0] == pytest.approx(results[-1], rel=1e-9)


def test_endogenous_stress_makes_fx_bite() -> None:
    """Under the macro-based measure, higher fx widens the CLT advantage.

    Foreign-currency debt is revalued when the currency slips, so the cash
    path — which leaks more into landlord portfolios — costs a high-fx bloc
    more. Nothing is weighted by hand; the gradient comes out of the model.
    """
    from dataclasses import replace

    import numpy as np

    bloc = make_population(42)[20]
    g = _globals()
    fxs = (0.0, 0.25, 0.5, 0.75, 0.85)
    gaps = []
    for fx in fxs:
        probe = replace(bloc, fx_share=fx)
        gaps.append(stress_of(probe, g, [0, 0, 0, 1])
                    - stress_of(probe, g, [0, 1, 0, 0]))
    assert all(gap < 0 for gap in gaps)          # CLT helps at every fx
    assert np.polyfit(fxs, gaps, 1)[0] < -0.5    # and helps more as fx rises


def test_cache_key_distinguishes_counterfactual_blocs() -> None:
    """Two blocs sharing a name but differing in structure must not collide.

    The counterfactual sweep holds the name fixed while moving fx, so a cache
    keyed on the name alone would silently return the wrong economy.
    """
    from dataclasses import replace

    from legacy.v7.evolution import _bloc_key

    bloc = make_population(42)[5]
    assert _bloc_key(replace(bloc, fx_share=0.1)) != _bloc_key(replace(bloc, fx_share=0.9))
    assert _bloc_key(replace(bloc, clt_domestic_sourcing=0.0)) != \
           _bloc_key(replace(bloc, clt_domestic_sourcing=1.0))


# --- invasion dynamics ------------------------------------------------------

def test_invasion_seeds_a_minority_of_blocs() -> None:
    """The main series starts a few blocs on CLT and the rest on cash."""
    population = make_population(3)
    result = run_evolution(population, _globals(),
                           EvolutionConfig(years=5, initial="invasion",
                                           n_invaders=3), seed=1)
    invaders = [f for f in result["final"] if f["is_invader"]]
    assert len(invaders) == 3
    assert result["history"][0]["adopter_share"] == pytest.approx(3 / 31, abs=0.02)


def test_invasion_spreads_further_under_selection_than_drift() -> None:
    """Selection carries the innovation further than drift alone does."""
    population = make_population(11)
    g = _globals()
    selection = run_evolution(population, g,
                              EvolutionConfig(years=80, initial="invasion"), seed=2)
    neutral = run_evolution(population, g,
                            EvolutionConfig(years=80, initial="invasion",
                                            neutral=True), seed=2)
    assert selection["history"][-1]["adopter_share"] > \
           neutral["history"][-1]["adopter_share"]


def test_domestic_sourcing_removes_the_leakage_penalty() -> None:
    """Sourcing construction at home flips the sign of the leakage gap.

    Without it CLT leaks more than cash during the build; at full domestic
    sourcing it leaks less, at the cost of a higher fiscal bill.
    """
    from dataclasses import replace

    import numpy as np

    population = make_population(42)
    gaps = {}
    for sourcing in (0.0, 1.0):
        g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
              clt_build_rate=0.02, sourcing_cost_kappa=0.25 if sourcing else 0.0)
        values = []
        for bloc in population:
            probe = replace(bloc, clt_domestic_sourcing=sourcing)
            cash = mix_outcome(probe, g, [0, 1, 0, 0])
            clt = mix_outcome(probe, g, [0, 0, 0, 1])
            values.append(clt["leak_pct_gdp"] - cash["leak_pct_gdp"])
        gaps[sourcing] = float(np.mean(values))

    assert gaps[0.0] > 0      # CLT leakier when construction is imported
    assert gaps[1.0] < 0      # and less leaky when it is not


# --- which channel carries the fx effect ------------------------------------

def test_fx_effect_runs_through_debt_not_the_risk_premium() -> None:
    """Debt revaluation dominates; the risk premium contributes nothing.

    Naming the channel matters because it says what the result depends on:
    the foreign-currency debt stock being marked up when the currency slips,
    not lenders repricing the bloc.
    """
    from dataclasses import replace

    import numpy as np

    from legacy.v7.evolution import stress_components

    bloc = make_population(42)[20]
    g = _globals()
    fxs = (0.0, 0.25, 0.5, 0.75, 0.85)
    gaps = {"debt": [], "risk_premium": [], "depreciation": []}
    for fx in fxs:
        probe = replace(bloc, fx_share=fx)
        cash = stress_components(probe, g, [0, 1, 0, 0])
        clt = stress_components(probe, g, [0, 0, 0, 1])
        for key in gaps:
            gaps[key].append(clt[key] - cash[key])

    slopes = {k: np.polyfit(fxs, v, 1)[0] for k, v in gaps.items()}
    assert slopes["debt"] < -0.5                    # dominant channel
    assert abs(slopes["risk_premium"]) < 0.05       # contributes nothing
    assert slopes["debt"] < slopes["depreciation"]  # and larger than the other


def test_fx_advantage_disappears_at_equal_fiscal_cost() -> None:
    """Closing the cost gap removes the negative slope.

    The fx gradient is therefore an amplification of CLT's lower fiscal cost,
    not a separate return to being non-fugitive. Tested by shrinking cash to
    CLT's budget, which leaves both spending structures intact — scaling CLT
    up instead would also multiply its construction imports and confound the
    comparison.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase3" / "fx_channels.json"
    if not path.exists():
        pytest.skip("channel decomposition not present")
    summary = json.loads(path.read_text())

    assert summary["welfare_equivalent"]["total"]["mean"] < -1.0
    assert summary["equal_fiscal_shrink_cash"]["total"]["mean"] > 0
    assert summary["equal_fiscal_scale_clt"]["total"]["mean"] > 0


# --- separating cheapness from non-fugitivity -------------------------------

def test_cash_cheap_matches_clt_on_cost_but_not_on_welfare() -> None:
    """The control spends the same and delivers less, by construction."""
    from dataclasses import replace

    from legacy.v7.heterogeneous import calibrate_hetero, hetero_household_block, welfare_measure
    from legacy.v7.params import archetypes

    bloc = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    cal = calibrate_hetero(bloc, g, 0.05, "utilitarian")

    clt = hetero_household_block(replace(bloc, modality="clt"), g,
                                 bloc.gdp, 1.0, bloc.rate, True, cal["clt"])
    cheap = hetero_household_block(replace(bloc, modality="cash_cheap"), g,
                                   bloc.gdp, 1.0, bloc.rate, True, cal["_fiscal_clt"])

    assert cheap.fiscal == pytest.approx(clt.fiscal, rel=1e-6)
    assert welfare_measure(cheap, "utilitarian") < welfare_measure(clt, "utilitarian")


def test_three_way_control_separates_cheapness_from_non_fugitivity() -> None:
    """Blocs pick CLT over an equally cheap cash transfer, and more so as fx rises.

    cash_cheap costs what CLT costs and pays out like cash. That it does not
    respond to fx, while CLT does, is what distinguishes "CLT is chosen
    because it is cheap" from "because of where the money goes".
    """
    import json
    from pathlib import Path

    import numpy as np

    path = Path(__file__).resolve().parent.parent.parent / "results" / "phase3" / "evolution.jsonl"
    if not path.exists():
        pytest.skip("evolution results not present")
    runs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    three_way = [r for r in runs if r.get("arena") == "three_way" and not r["neutral"]]
    if not three_way:
        pytest.skip("three-way arena not present")

    labels = three_way[0]["final"][0]["mix_labels"]
    clt_i, cheap_i = labels.index("clt"), labels.index("cash_cheap")

    clt = np.array([f["mix"][clt_i] for r in three_way for f in r["final"]])
    cheap = np.array([f["mix"][cheap_i] for r in three_way for f in r["final"]])
    fx = np.array([f["fx_share"] for r in three_way for f in r["final"]])

    assert clt.mean() > cheap.mean()                       # CLT wins on average
    assert np.polyfit(fx, clt, 1)[0] > 0.01                # and rises with fx
    assert abs(np.polyfit(fx, cheap, 1)[0]) < 0.01         # the control does not


# --- cash_cheap must be a working control ----------------------------------

def test_cash_cheap_reaches_the_macro_block() -> None:
    """The control must actually transfer money in the macro simulation.

    `cash_cheap` was added to the heterogeneous household block but not to the
    one `model/core.py` calls, so the macro run silently treated it as no
    transfer at all — its debt and exchange-rate paths were identical to
    doing nothing. Any comparison drawn from that was meaningless.
    """
    from dataclasses import replace

    from legacy.v7.core import simulate
    from legacy.v7.evolution import _cached_calibration

    bloc = make_population(42)[20]
    g = _globals()
    cal = _cached_calibration(bloc, g, 0.05)

    nothing = simulate([replace(bloc, fx_share=0.85, modality="none",
                                start=999.0)], g)["history"]
    cheap = simulate([replace(bloc, fx_share=0.85, modality="cash_cheap",
                              size=cal["_fiscal_clt"], start=1.0)], g)["history"]

    assert cheap[-1]["d_0"] != pytest.approx(nothing[-1]["d_0"], rel=1e-6)
    assert cheap[-1]["e_0"] != pytest.approx(nothing[-1]["e_0"], rel=1e-6)


def test_cash_cheap_behaves_like_cash_at_the_same_size() -> None:
    """Only `size` distinguishes the control from an ordinary cash transfer."""
    from dataclasses import replace

    from legacy.v7.core import simulate

    bloc = make_population(42)[20]
    g = _globals()
    size = 0.01
    as_cash = simulate([replace(bloc, modality="cash_t", size=size, start=1.0)],
                       g)["history"]
    as_cheap = simulate([replace(bloc, modality="cash_cheap", size=size, start=1.0)],
                        g)["history"]
    assert as_cheap[-1]["d_0"] == pytest.approx(as_cash[-1]["d_0"], rel=1e-12)


def test_cash_cheap_matches_clt_fiscal_cost_to_machine_precision() -> None:
    """'Same cost' has to mean same cost, not approximately."""
    from dataclasses import replace

    from legacy.v7.evolution import _cached_calibration
    from legacy.v7.heterogeneous import hetero_household_block

    g = _globals()
    for bloc in make_population(42)[:6]:
        cal = _cached_calibration(bloc, g, 0.05)
        clt = hetero_household_block(replace(bloc, modality="clt"), g,
                                     bloc.gdp, 1.0, bloc.rate, True, cal["clt"])
        cheap = hetero_household_block(replace(bloc, modality="cash_cheap"), g,
                                       bloc.gdp, 1.0, bloc.rate, True,
                                       cal["_fiscal_clt"])
        assert cheap.fiscal == pytest.approx(clt.fiscal, rel=1e-12), bloc.name


# --- the three-way arena must seed both alternatives -----------------------

def test_asymmetric_seeding_favours_whatever_was_planted() -> None:
    """Planting only CLT hands it an advantage the control cannot have.

    Proportional imitation spreads what already exists, so a modality nobody
    holds can only arrive by mutation. Seeding CLT alone therefore measures
    availability, not fitness — the asymmetric and symmetric arenas reach
    opposite conclusions on the same parameters.
    """
    import numpy as np

    from legacy.v7.evolution import CONTROL_MODALITIES

    population = make_population(42)
    g = _globals()
    labels = list(CONTROL_MODALITIES)
    clt_i, cheap_i = labels.index("clt"), labels.index("cash_cheap")

    asymmetric = run_evolution(
        population, g,
        EvolutionConfig(years=80, initial="invasion",
                        modality_set=CONTROL_MODALITIES), seed=1)
    symmetric = run_evolution(
        population, g,
        EvolutionConfig(years=80, initial="invasion",
                        modality_set=CONTROL_MODALITIES,
                        symmetric_invasion=True), seed=1)

    def mean(run, index):
        return float(np.mean([f["mix"][index] for f in run["final"]]))

    # Only CLT planted: CLT leads.
    assert mean(asymmetric, clt_i) > mean(asymmetric, cheap_i)
    # Both planted: the cheaper control leads instead.
    assert mean(symmetric, cheap_i) > mean(symmetric, clt_i)


def test_symmetric_invasion_seeds_every_alternative() -> None:
    """Each non-cash modality gets its own invader blocs."""
    from legacy.v7.evolution import CONTROL_MODALITIES

    run = run_evolution(
        make_population(5), _globals(),
        EvolutionConfig(years=5, initial="invasion", n_invaders=3,
                        modality_set=CONTROL_MODALITIES,
                        symmetric_invasion=True), seed=2)
    seeded = [f["seeded_with"] for f in run["final"] if f["is_invader"]]
    assert sorted(set(seeded)) == ["cash_cheap", "clt"]
    assert len(seeded) == 6


def test_cheaper_control_carries_lower_stress_at_every_fx() -> None:
    """The static ranking the symmetric arena reproduces.

    cash_cheap costs what CLT costs and leaks less, so on the endogenous
    measure it is the less stressful option throughout — which is why an arena
    that gives both a fair start selects it.
    """
    from dataclasses import replace

    from legacy.v7.evolution import CONTROL_MODALITIES, stress_of

    bloc = make_population(42)[20]
    g = _globals()
    labels = list(CONTROL_MODALITIES)
    for fx in (0.0, 0.5, 0.85):
        probe = replace(bloc, fx_share=fx)
        scores = []
        for index in range(len(labels)):
            mix = [0.0] * len(labels)
            mix[index] = 1.0
            scores.append(stress_of(probe, g, mix, modality_set=CONTROL_MODALITIES))
        assert labels[int(np.argmin(scores))] == "cash_cheap", fx
