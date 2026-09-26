"""Phase 2: heterogeneous constrained households and CLT tenancy.

The central requirement is backward compatibility: with one household type and
unlimited tenancy the heterogeneous model must reproduce v6 exactly, so that
any later difference is attributable to heterogeneity rather than to a rewrite.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from legacy.v7.heterogeneous import (
    allocate_tenancies, calibrate_hetero, clt_units_available,
    hetero_household_block, rent_incidence, welfare_measure,
)
from legacy.v7.household import household_block
from legacy.v7.params import (
    SUB_SCALE_CAP, G, archetypes, income_quantiles, subsistence_headroom,
)

BLOCS = ["A", "B", "C", "D"]
MODS = ["none", "ubi", "cash_t", "voucher", "clt"]


# --- backward compatibility (acceptance condition) -------------------------

@pytest.mark.parametrize("key", BLOCS)
@pytest.mark.parametrize("mod", MODS)
def test_single_type_reproduces_v6(key: str, mod: str) -> None:
    """K=1 with tenancy for everyone must match the representative model."""
    g = G()
    b = replace(archetypes()[key], modality=mod)
    v6 = household_block(b, g, b.gdp, 1.0, b.rate, True, 0.03)
    het = hetero_household_block(b, g, b.gdp, 1.0, b.rate, True, 0.03)

    assert het.pH == pytest.approx(v6["pH"], rel=1e-12)
    assert het.U == pytest.approx(v6["U"], rel=1e-12)
    assert het.fiscal == pytest.approx(v6["fiscal"], rel=1e-12)
    assert het.rent == pytest.approx(v6["rent"], rel=1e-12)
    assert het.imp_c == pytest.approx(v6["imp_c"], rel=1e-12)
    assert het.imp_gov == pytest.approx(v6["imp_gov"], rel=1e-12)


@pytest.mark.parametrize("key", BLOCS)
def test_v6_defaults_are_the_heterogeneous_defaults(key: str) -> None:
    """The new options default to the v6 configuration."""
    g = G()
    assert g.n_types == 1
    assert g.clt_capacity is None      # everyone housed
    assert g.clt_build_rate is None    # whole programme available at once


def test_zero_spread_makes_types_identical() -> None:
    """With no inequality the quantiles collapse to the representative case."""
    g = G(n_types=5, income_spread=0.0)
    b = replace(archetypes()["C"], modality="clt")
    v6 = household_block(b, G(), b.gdp, 1.0, b.rate, True, 0.03)
    het = hetero_household_block(b, g, b.gdp, 1.0, b.rate, True, 0.03)
    assert het.pH == pytest.approx(v6["pH"], rel=1e-9)


# --- feasibility of the income distribution --------------------------------

@pytest.mark.parametrize("spread", [0.0, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("k", [3, 5, 7])
def test_every_quantile_can_afford_subsistence(spread: float, k: int) -> None:
    """No quantile may start below its subsistence bundle.

    Stone-Geary utility is undefined there, so an infeasible quantile would
    silently poison every welfare comparison rather than merely scoring low.
    """
    g = G()
    headroom = subsistence_headroom(income_quantiles(k, spread), g)
    assert min(headroom.values()) > 0.0, headroom


@pytest.mark.parametrize("key", BLOCS)
def test_all_types_are_feasible_without_transfers(key: str) -> None:
    """The untreated economy is well defined for every type in every bloc."""
    b = replace(archetypes()[key], modality="none")
    het = hetero_household_block(b, G(n_types=5), b.gdp, 1.0, b.rate, False, 0.0)
    for t in het.types:
        assert t.utility > -1e8, f"{key}:{t.name}"


def test_subsistence_cap_stays_below_the_theoretical_bound() -> None:
    """The cap must leave headroom against the budget-exhausting scale."""
    g = G()
    bound = 1.0 / (g.sub_F + g.sub_G + g.sub_H)
    assert SUB_SCALE_CAP < bound


# --- tenancy allocation ----------------------------------------------------

def test_priority_rule_fills_from_the_bottom() -> None:
    """Scarce tenancies go to the lowest-income quantiles first."""
    types = income_quantiles(5)
    alloc = allocate_tenancies(types, 0.4, "priority_low_income")
    assert alloc["q1"] == pytest.approx(0.2)
    assert alloc["q2"] == pytest.approx(0.2)
    assert alloc["q5"] == pytest.approx(0.0)


def test_lottery_rule_is_income_blind() -> None:
    """A lottery takes the same fraction of every quantile."""
    types = income_quantiles(5)
    alloc = allocate_tenancies(types, 0.4, "lottery")
    for name, share in alloc.items():
        assert share == pytest.approx(0.2 * 0.4), name


@pytest.mark.parametrize("rule", ["priority_low_income", "lottery"])
def test_allocation_never_exceeds_population(rule: str) -> None:
    """Allocated tenancies stay within each type and within capacity."""
    types = income_quantiles(5)
    for capacity in (0.0, 0.25, 0.5, 1.0, 1.5):
        alloc = allocate_tenancies(types, capacity, rule)
        assert sum(alloc.values()) <= min(1.0, max(capacity, 0.0)) + 1e-12
        for t in types:
            assert 0.0 <= alloc[t.name] <= t.pop_share + 1e-12


# --- build-rate cap --------------------------------------------------------

def test_build_rate_phases_the_programme_in() -> None:
    """A finite build rate delays full CLT stock instead of granting it at once."""
    g = G(clt_build_rate=0.01)
    target = 0.05
    first = clt_units_available(g, target, 1.0, years_since_start=1.0)
    later = clt_units_available(g, target, 1.0, years_since_start=3.0)
    full = clt_units_available(g, target, 1.0, years_since_start=100.0)

    assert first < later < target
    assert full == pytest.approx(target)


def test_no_build_rate_means_immediate_stock() -> None:
    """v6 behaviour: the whole programme exists from the first period."""
    g = G()
    assert clt_units_available(g, 0.05, 1.0, 0.0) == pytest.approx(0.05)


# --- utility must not depend on group size ---------------------------------

def test_utility_is_per_household_not_per_group() -> None:
    """Splitting a type in two must not change its members' welfare.

    Stone-Geary utility is logarithmic, so scaling a group's income and
    subsistence by its population share shifts utility by log(share). Without
    per-household normalisation a group would look worse off merely for being
    split, which would corrupt every tenant-vs-renter comparison.
    """
    b = replace(archetypes()["C"], modality="none")
    whole = hetero_household_block(b, G(n_types=5), b.gdp, 1.0, b.rate, False, 0.0)
    # A lottery splits every type without changing prices or incomes.
    split = hetero_household_block(
        replace(b, modality="clt"),
        G(n_types=5, clt_capacity=0.5, allocation_rule="lottery"),
        b.gdp, 1.0, b.rate, False, 0.0)

    by_name = {t.name: t.utility for t in whole.types}
    for t in split.types:
        assert t.utility == pytest.approx(by_name[t.name], rel=1e-9), t.name


def test_market_renters_gain_from_the_rent_fall() -> None:
    """The pecuniary externality reaches households with no tenancy.

    This is the Phase 2 question: CLT tenants gain directly, but renters also
    gain because the clearing rent falls. Both must be positive.
    """
    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=0.3)
    base = hetero_household_block(replace(b, modality="none"), g,
                                  b.gdp, 1.0, b.rate, False, 0.0)
    clt = hetero_household_block(replace(b, modality="clt"), g,
                                 b.gdp, 1.0, b.rate, True, 0.03)

    assert clt.pH < base.pH                      # rent falls
    incidence = rent_incidence(clt, base)
    tenants = [v["d_utility"] for k, v in incidence.items() if k.endswith("tenant")]
    renters = [v["d_utility"] for k, v in incidence.items() if k.endswith("renter")]

    assert all(v > 0 for v in tenants)           # tenants gain
    assert all(v > 0 for v in renters)           # renters gain too
    assert min(tenants) > max(renters)           # but tenants gain more


def test_cash_raises_rent_while_clt_lowers_it() -> None:
    """The incidence channel has opposite signs for the two modalities."""
    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=0.3)
    base = hetero_household_block(replace(b, modality="none"), g,
                                  b.gdp, 1.0, b.rate, False, 0.0)
    cash = hetero_household_block(replace(b, modality="cash_t"), g,
                                  b.gdp, 1.0, b.rate, True, 0.03)
    clt = hetero_household_block(replace(b, modality="clt"), g,
                                 b.gdp, 1.0, b.rate, True, 0.03)
    assert cash.pH > base.pH > clt.pH


# --- calibration modes -----------------------------------------------------

def test_calibration_modes_agree_when_everyone_is_a_tenant() -> None:
    """With universal tenancy the two welfare targets coincide."""
    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0)
    tenant = calibrate_hetero(b, g, 0.05, "tenant")
    util = calibrate_hetero(b, g, 0.05, "utilitarian")
    assert tenant["clt"] == pytest.approx(util["clt"], rel=1e-6)


def test_tenant_mode_costs_more_when_tenancy_is_scarce() -> None:
    """Targeting tenants alone ignores the benefit flowing to renters.

    Equalising only the tenants' utility therefore demands a larger — and more
    expensive — programme than the utilitarian target does.
    """
    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=0.3)
    tenant = calibrate_hetero(b, g, 0.05, "tenant")
    util = calibrate_hetero(b, g, 0.05, "utilitarian")
    assert tenant["_fiscal_clt"] > util["_fiscal_clt"]


def test_unknown_welfare_mode_is_rejected() -> None:
    b = archetypes()["C"]
    outcome = hetero_household_block(replace(b, modality="clt"), G(),
                                     b.gdp, 1.0, b.rate, True, 0.03)
    with pytest.raises(ValueError, match="welfare mode"):
        welfare_measure(outcome, "median")


# --- surplus CLT stock must reach the market -------------------------------

def test_surplus_clt_units_lower_the_market_rent() -> None:
    """Units beyond what tenants can absorb add to rental supply.

    With tenancy capped, a larger programme must still reduce the clearing
    rent — otherwise the pecuniary externality would vanish exactly where the
    Phase 2 question is sharpest, and calibration would demand absurd budgets
    to hit any welfare target.
    """
    b = replace(archetypes()["C"], modality="clt")
    g = G(n_types=5, clt_capacity=0.1)
    small = hetero_household_block(b, g, b.gdp, 1.0, b.rate, True, 0.01)
    large = hetero_household_block(b, g, b.gdp, 1.0, b.rate, True, 0.10)
    assert large.pH < small.pH - 1e-6


def test_calibrated_cost_stays_within_plausible_bounds() -> None:
    """No capacity setting should require an implausible share of GDP."""
    b = archetypes()["C"]
    for capacity in (1.0, 0.6, 0.3, 0.1):
        cal = calibrate_hetero(b, G(n_types=5, clt_capacity=capacity), 0.05,
                               "utilitarian")
        assert 0.0 < cal["_fiscal_clt"] < 0.05, capacity   # under 5% of GDP


def test_cost_is_non_monotone_in_capacity() -> None:
    """Mid-range tenancy is the cheapest way to hit a welfare target.

    Narrow tenancy forces the programme to work through market supply; full
    tenancy requires housing everyone in kind. The minimum sits in between.
    """
    b = archetypes()["C"]
    costs = {cap: calibrate_hetero(b, G(n_types=5, clt_capacity=cap), 0.05,
                                   "utilitarian")["_fiscal_clt"]
             for cap in (1.0, 0.3, 0.1)}
    assert costs[0.3] < costs[1.0]
    assert costs[0.3] < costs[0.1]


# --- calibration diagnostics ----------------------------------------------

def test_calibration_flags_a_search_ceiling() -> None:
    """A size pinned at the search bound is reported, not passed off as a solution.

    With a slow build rate the CLT cannot match UBI's discounted value within
    30 years at all, so the bisection runs to its ceiling. That has to be
    visible rather than appearing as a very large but valid answer.
    """
    from legacy.v7.heterogeneous import SEARCH_CEILING, discounted_calibrate

    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.005)
    result = discounted_calibrate(b, g, 0.05, "utilitarian", years=30, rate=0.03)

    assert result.get("_at_ceiling_clt") is True
    assert result["clt"] == pytest.approx(SEARCH_CEILING, rel=1e-3)


def test_present_value_matches_snapshot_without_a_build_cap() -> None:
    """With no phase-in the two calibrations must agree.

    The economy is stationary here, so discounting a constant path cannot
    change the equivalent size. Disagreement would mean the present-value
    routine is not solving the same problem.
    """
    from legacy.v7.heterogeneous import discounted_calibrate

    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0)
    snapshot = calibrate_hetero(b, g, 0.05, "utilitarian")
    pv = discounted_calibrate(b, g, 0.05, "utilitarian", years=30, rate=0.03)

    for modality in ("clt", "cash_t", "voucher"):
        assert pv[modality] == pytest.approx(snapshot[modality], rel=1e-4), modality


def test_snapshot_calibration_favours_the_slow_modality() -> None:
    """Calibrating at introduction credits CLT with benefits it has not delivered.

    Under a build cap the snapshot size is far smaller than the size needed to
    match on present value — the bias the discounted variant exists to expose.
    """
    from legacy.v7.heterogeneous import discounted_calibrate

    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    snapshot = calibrate_hetero(b, g, 0.05, "utilitarian")
    pv = discounted_calibrate(b, g, 0.05, "utilitarian", years=30, rate=0.01)

    assert pv["clt"] > snapshot["clt"]


# --- rent attribution ------------------------------------------------------

def test_clt_rent_attribution_changes_the_fiscal_cost() -> None:
    """Crediting CLT rent to the trust offsets the programme's cost."""
    b = replace(archetypes()["C"], modality="clt")
    to_trust = hetero_household_block(
        b, G(n_types=5, clt_capacity=0.3, clt_rent_to_landlords=False),
        b.gdp, 1.0, b.rate, True, 0.01)
    to_landlords = hetero_household_block(
        b, G(n_types=5, clt_capacity=0.3, clt_rent_to_landlords=True),
        b.gdp, 1.0, b.rate, True, 0.01)

    assert to_trust.fiscal < to_landlords.fiscal
    assert to_trust.rent < to_landlords.rent   # less rent reaches landlords


def test_v6_default_keeps_rent_with_landlords() -> None:
    """The default must stay v6-compatible so regression parity holds."""
    assert G().clt_rent_to_landlords is True


def test_cash_is_paid_per_capita_across_quantiles() -> None:
    """Targeted cash reaches every constrained household, not just some.

    CLT tenancies are rationed; cash is not. The comparison is only fair if
    this asymmetry is explicit.
    """
    b = replace(archetypes()["C"], modality="cash_t")
    out = hetero_household_block(b, G(n_types=5), b.gdp, 1.0, b.rate, True, 0.02)
    assert len(out.types) == 5
    for t in out.types:
        assert t.pop_share == pytest.approx(0.2)
        assert not t.is_tenant      # cash creates no tenancies


# --- annuity offset derived from the model, not invented --------------------

def test_annuity_factor_matches_the_formula() -> None:
    """The rent offset is r / (1 - (1+r)^-n), not a hand-picked constant."""
    from legacy.v7.heterogeneous import annuity_factor

    for rate, years in ((0.05, 40), (0.02, 20), (0.07, 60)):
        expected = rate / (1.0 - (1.0 + rate) ** -years)
        assert annuity_factor(rate, years) == pytest.approx(expected, rel=1e-12)


def test_annuity_factor_rises_with_the_interest_rate() -> None:
    """A costlier bloc amortises faster, so its offset share is larger."""
    from legacy.v7.heterogeneous import annuity_factor

    low = annuity_factor(archetypes()["A"].rate, 40)
    high = annuity_factor(archetypes()["D"].rate, 40)
    assert high > low


def test_housing_life_barely_moves_the_calibration() -> None:
    """Results are robust to the assumed building life (20 vs 60 years)."""
    b = archetypes()["C"]
    costs = {}
    for years in (20, 40, 60):
        g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
              housing_life_years=years)
        costs[years] = calibrate_hetero(b, g, 0.05, "utilitarian")["_fiscal_clt"]
    assert abs(costs[20] - costs[60]) < 0.0005


# --- decomposition of the 'no solution' result ------------------------------

def test_horizon_explains_the_moderate_build_cap() -> None:
    """At 2%/yr the failure is the 30-year cut-off, not the cap itself.

    Extending the horizon (or adding a terminal value) recovers a solution,
    which is what distinguishes cause (b) from cause (a).
    """
    from legacy.v7.heterogeneous import discounted_calibrate

    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)
    short = discounted_calibrate(b, g, 0.05, "utilitarian", years=30, rate=0.03,
                                 only=("clt",))
    long = discounted_calibrate(b, g, 0.05, "utilitarian", years=60, rate=0.03,
                                only=("clt",))

    assert short.get("_at_ceiling_clt") is True     # no solution at 30 years
    assert not long.get("_at_ceiling_clt")          # solved at 60


def test_slow_build_is_a_genuine_constraint() -> None:
    """At 0.5%/yr no horizon rescues it — cause (a), a real transition cost."""
    from legacy.v7.heterogeneous import discounted_calibrate

    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.005)
    for kwargs in (dict(years=30), dict(years=60),
                   dict(years=30, terminal_value=True, depreciation=0.02)):
        result = discounted_calibrate(b, g, 0.05, "utilitarian", rate=0.03,
                                      only=("clt",), **kwargs)
        assert result.get("_at_ceiling_clt") is True, kwargs


def test_size_works_backwards_under_a_binding_cap() -> None:
    """While the cap binds, a larger target makes households worse off.

    The stock standing is set by the build rate, so raising `size` does not
    build anything extra — it only lowers the programme's completion ratio,
    which narrows tenancy. Welfare therefore *falls* in `size`, the opposite
    of what a calibration search assumes, so the bisection cannot converge.
    That is the sense in which `size` is not a usable instrument here; it is
    not a claim that no equivalent programme exists. Phase 3 tops the
    shortfall up with cash instead of scaling size.
    """
    b = archetypes()["C"]
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.005)
    utilities = [
        welfare_measure(
            hetero_household_block(replace(b, modality="clt"), g, b.gdp, 1.0,
                                   b.rate, True, size, years_since_start=20.0),
            "utilitarian")
        for size in (0.02, 0.05, 0.20)
    ]
    assert all(b_ < a for a, b_ in zip(utilities, utilities[1:]))


def test_completed_build_matches_the_uncapped_run() -> None:
    """Once the stock reaches its target the cap must stop mattering.

    The capped and uncapped programmes are then the same programme, so any
    residual difference would be an artifact of the phase-in bookkeeping.
    """
    b = archetypes()["C"]
    free = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    size = calibrate_hetero(b, free, 0.05, "utilitarian")["clt"]
    capped = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
               clt_build_rate=0.05)

    uncapped_u = welfare_measure(
        hetero_household_block(replace(b, modality="clt"), free, b.gdp, 1.0,
                               b.rate, True, size), "utilitarian")
    completed_u = welfare_measure(
        hetero_household_block(replace(b, modality="clt"), capped, b.gdp, 1.0,
                               b.rate, True, size, years_since_start=30.0),
        "utilitarian")
    assert completed_u == pytest.approx(uncapped_u, rel=1e-9)

    # Without the cap the same sweep is well behaved.
    g_free = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False)
    free = [
        welfare_measure(
            hetero_household_block(replace(b, modality="clt"), g_free, b.gdp, 1.0,
                                   b.rate, True, size),
            "utilitarian")
        for size in (0.01, 0.05, 0.20)
    ]
    assert all(b_ > a for a, b_ in zip(free, free[1:]))
