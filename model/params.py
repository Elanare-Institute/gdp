"""Parameter containers for v8.

The bloc and household structure is carried over from v7 unchanged — the
archetypes, the Stone-Geary preferences and the import shares all mean what
they meant before. What changes is the world around them:

  * the reserve bloc's privileges (`reserve_flight`, `reserve_thr`,
    `reserve_damp`) are **off by default**. In v8 the reserve issuer's
    advantage has to come from being able to settle imports in its own money,
    not from parameters that hand it one. The v7 privileges are kept only so
    the sensitivity can be run.
  * world parameters (`kappa_w`, `lambda_w`, ...) are new: the world now has a
    price level and a supply capacity, and they are shared by every bloc.

See `specs/V8_A_B.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

YEARS: int = 30
DT: float = 0.25
STEPS: int = int(YEARS / DT)

# Crisis thresholds. These are reporting cut-offs, not structural boundaries:
# results are reported as signs, orderings, and shares of parameter space.
THR: dict[str, float] = dict(debt=1.5, infl=0.15, dep=2.0, app=0.3)

AllocationRule = Literal["priority_low_income", "lottery"]

#: Damping on the subsistence tilt across income quantiles. A full inverse
#: tilt (exponent 1.0) drives the poorest quantile below subsistence before any
#: transfer is made, which would make its utility undefined rather than merely
#: low.
SUB_SCALE_EXPONENT: float = 0.5

#: Hard ceiling on the subsistence multiplier, as a second guard.
SUB_SCALE_CAP: float = 1.35


@dataclass
class Bloc:
    """One economic bloc.

    Household and market structure are stylized: only orderings and orders of
    magnitude are meant to be meaningful.
    """

    name: str
    gdp: float
    debt: float
    rate: float
    base_growth: float
    decay: float
    openness: float
    fx_share: float
    reserve: bool = False
    capital: float = 1.0
    c_income_share: float = 0.2   # constrained households' income / GDP
    c_pop_share: float = 0.4      # constrained share of population
    m_F: float = 0.2              # import content of food
    m_G: float = 0.25             # import content of other goods
    m_H: float = 0.1              # import content of housing construction
    eps_supply: float = 0.4       # short-run rental supply elasticity
    land_share: float = 0.35      # land rent share of gross rent
    omega0: float = 0.2           # baseline foreign-asset share of saving
    start: float = 999.0
    modality: str = "none"
    """none | ubi | cash_t | voucher | clt | clt_cash | cash_cheap."""
    size: float = 0.0             # ubi: share of GDP; else calibrated quantity

    #: On the name "clt": the modality stands for **housing supplied as a
    #: non-fugitive real asset**, of which a community land trust is the
    #: example the paper draws on. It is not a faithful reconstruction of any
    #: particular institution — no ground-lease terms, no resale formula, no
    #: governance structure. What the model represents is the two features the
    #: argument turns on: the grant is a real quantity rather than a sum of
    #: money, and the resources it spends go disproportionately to domestic
    #: construction rather than to imports. Any programme with those two
    #: features would behave the same way here.
    #:
    #: Likewise "voucher" is a control, not a proposal: it exists to establish
    #: where an in-kind food grant is equivalent to cash (the infra-marginal
    #: case) and where it is not.
    credit_standing: float = 1.0
    """What lenders will advance this bloc, as a multiple of the global limit.

    The credit axis of `model.twoaxis`. Separating it from the initial debt
    level matters: giving a low-credit bloc a *low* opening debt makes its risk
    premium small and its borrowing window wide, which is the opposite of what
    "low credit" means. What lenders will advance has to be set directly.
    """

    reserve_standing: float = 1.0
    """Opening foreign exchange reserves, as a multiple of the global share."""

    clt_domestic_sourcing: float = 0.0
    """Policy effort to source CLT construction domestically, in [0, 1].

    Distinct from ``m_H``, which is a structural fact about how import-
    dependent construction is. The effective import share is
    ``m_H * (1 - clt_domestic_sourcing)``.

    In v8 this changes meaning: with the world closed and (from phase D) a
    settlement constraint, sourcing locally does not merely reduce leakage —
    it reduces the share of the programme that has to be paid for in foreign
    exchange. See CLAUDE.md §E.
    """


@dataclass(frozen=True)
class HouseholdType:
    """One income quantile within the constrained population."""

    name: str
    pop_share: float
    income_share: float
    sub_scale: float = 1.0


def income_quantiles(k: int, spread: float = 0.6) -> tuple[HouseholdType, ...]:
    """Build `k` equal-population income quantiles.

    ``spread`` controls inequality within the constrained block: 0 gives
    identical types, while larger values tilt income towards the upper
    quantiles. Subsistence scaling moves the opposite way, so lower quantiles
    are more housing-constrained.
    """
    if k == 1:
        return (HouseholdType("all", 1.0, 1.0, 1.0),)

    weights = [1.0 + spread * (2.0 * i / (k - 1) - 1.0) for i in range(k)]
    total = sum(weights)
    mean_w = total / k

    types = []
    for i, w in enumerate(weights):
        share = w / total
        sub_scale = min(SUB_SCALE_CAP, (mean_w / w) ** SUB_SCALE_EXPONENT)
        types.append(HouseholdType(f"q{i + 1}", 1.0 / k, share, sub_scale))
    return tuple(types)


def subsistence_headroom(types: Sequence[HouseholdType], g: "G") -> dict[str, float]:
    """Fraction of each type's income left after subsistence.

    Values at or below zero mean the type cannot reach its subsistence bundle
    even before any transfer, which makes its Stone-Geary utility undefined
    rather than merely low.
    """
    floor = g.sub_F + g.sub_G + g.sub_H
    return {t.name: 1.0 - floor * t.sub_scale for t in types}


@dataclass
class G:
    """Global parameters shared across blocs."""

    fiscal_mult: float = 1.2
    cap_sens: float = 0.8
    risk_coeff: float = 0.15
    compress: float = 0.3
    deficit_share: float = 0.6
    bop_coeff: float = 0.5        # external position -> FX pressure
    export_switch: float = 0.05   # expenditure switching on depreciation

    trade_elasticity: float = 0.0
    """How strongly a depreciation wins export market share.

    Zero keeps the fixed trade matrix of phases A-C. Above zero, a bloc whose
    currency has fallen supplies a larger share of world demand, which is the
    adjustment that lets a bloc under a settlement constraint earn its way
    out. Without it the constraint of phase D tightens without limit, which
    would be an artefact of the matrix rather than a property of the economy.
    """
    omega_sens: float = 2.0       # sensitivity of foreign-asset share to risk
    mpc_u: float = 0.3            # unconstrained households' MPC
    land_discount: float = 0.5    # CLT land acquisition discount
    default_thr: float = 3.0
    eps_mult: float = 1.0         # uniform multiplier on supply elasticity
    sourcing_cost_kappa: float = 0.0
    """Cost markup for domestic sourcing of CLT construction."""

    # Stone-Geary preferences for constrained households.
    sub_F: float = 0.35
    sub_G: float = 0.05
    sub_H: float = 0.25
    beta_F: float = 0.25
    beta_G: float = 0.45
    beta_H: float = 0.30

    # --- Household heterogeneity (ported from v7 Phase 2) ---
    n_types: int = 1
    income_spread: float = 0.6
    clt_capacity: float | None = None
    allocation_rule: AllocationRule = "priority_low_income"
    clt_build_rate: float | None = None
    housing_life_years: int = 40
    clt_rent_to_landlords: bool = False
    """Who collects rent on CLT-supplied housing.

    v8 defaults to ``False`` (the trust credits the rent against its own cost),
    which was v7's Phase 2 main series. v7 defaulted to ``True`` only to keep
    bit-for-bit parity with v6, a requirement v8 has dropped.
    """
    universal_split: float | None = None

    # --- v8: the world ---
    kappa_w: float = 0.35
    """Pass-through from the world output gap to world price inflation.

    World prices rise when world demand for tradables exceeds world capacity.
    This is the coefficient on that gap; it is a world constant, not a
    per-bloc or per-modality coefficient.
    """

    lambda_w: float = 0.05
    """Slow reversion of the world price level towards its trend.

    Without it a single demand shock shifts the price level permanently at
    full force. With it the level still moves, but transitory gaps decay.
    """

    world_supply_growth: float | None = None
    """Growth of world tradable capacity per year.

    ``None`` derives it from the size-weighted mean of the blocs' own growth
    rates, so world capacity and world income grow together by construction.
    A number overrides that, which is how a capacity shock is imposed.
    """

    # --- v8 phase D: settlement ---
    settlement: bool = False
    """Ration imports to what a bloc can pay for in foreign exchange.

    Off reproduces phase C. On, every non-reserve bloc's imports are capped at
    exports plus reserves plus new borrowing, and the shortfall is cut. The
    reserve issuer is exempt, because it settles in money it issues — the one
    structural privilege v8 grants it, in place of the three v7 asserted.
    """

    initial_reserves: float = 0.15
    """Opening foreign exchange reserves, as a share of GDP."""

    borrow_limit: float = 0.05
    """New foreign borrowing available per year, as a share of GDP, before
    any risk adjustment. Overridden per bloc by `Bloc.credit_standing`."""

    borrow_sensitivity: float = 4.0
    """How sharply the borrowing window closes as the risk premium rises.
    At a premium of 1/borrow_sensitivity the window is shut."""

    reserve_demand_elasticity: float = 0.5
    """How sharply demand for the reserve currency falls as it loses
    purchasing power. No empirical basis; swept rather than chosen."""

    reserve_demand_threshold: float = 0.7
    """Share of opening demand below which the reserve issuer loses its
    exemption and has to settle like everyone else. A reporting cut-off:
    the model is asked where the exemption goes, not told."""

    contagion_strength: float = 0.0
    """How much lenders withdraw from blocs resembling one that just defaulted.

    Zero leaves adjustment entirely continuous, which is how phases A-C
    behaved: pressure builds and prices drift, and nothing ever stops
    suddenly. Real lenders do not work that way. When a borrower fails they
    reassess everything that looks like it, and the withdrawal arrives as a
    step rather than a slope.

    At 1.0 a bloc identical to the defaulter loses its entire borrowing window
    for a time; blocs further away on the two axes lose proportionately less.
    Swept, not chosen.
    """

    contagion_reach: float = 0.5
    """How far across the two-axis plane a default's reassessment travels."""

    contagion_years: float = 5.0
    """How long a withdrawal takes to unwind."""

    contagion_exempts_reserve: bool = False
    """Exclude the reserve issuer from contagion by assertion.

    Off in the main series. Granting the exemption directly would reinstate
    v7's privileges in another form: the plan was to *derive* the reserve
    issuer's position from settlement, not to hand it one. Left on the main
    path, the issuer is marked down like anyone else at the same distance from
    the defaulter, and whether that leaves it better off is a result.

    Kept as a sensitivity so the two can be compared.
    """

    flight_to_reserve: float = 0.0
    """Capital that runs to the reserve currency when a bloc defaults.

    Lenders pulling out of a region do not hold the proceeds as cash; they buy
    the safest claim available, which is the reserve issuer's. This is the
    mechanism through which the issuer is spared — an inflow it receives
    *because* others are being withdrawn from, not an exemption written into
    the rule. Zero leaves the channel out.
    """

    contagion_risk_premium: float = 0.05
    """Extra risk premium a fully contagion-hit bloc pays, in annual points."""

    # --- v8 phase D: policy response ---
    policy_response: float = 0.0
    """How strongly authorities lean against deflation and stalled growth.

    Zero leaves the model as it was: governments hand out transfers and do
    nothing else, so a world whose imports are being rationed slides into
    deflation with nobody reacting. That is not a finding about the world, it
    is an omission — real authorities ease when prices fall and growth stops,
    because deflation stalls output and drives capital out.

    Above zero, both instruments respond **to the state**, not to a target:
    easing is proportional to how far inflation has fallen below
    `policy_inflation_floor` and growth below `policy_growth_floor`. No
    authority is told to hold inflation at a number, which would put the
    answer into the assumption.

    The response is identical for every bloc. The periphery's difficulty must
    come from the settlement constraint and foreign-currency debt, never from
    giving its authorities a weaker rule.
    """

    growth_differential_sensitivity: float = 0.4
    """How strongly capital chases the growth differential.

    Capital goes where growth is, not only where rates are. Without this term
    a bloc whose growth stalls keeps its capital as long as its policy rate
    holds up, and the route from "falling behind" to "reserves draining" — the
    second jaw of the periphery's vice — is missing.
    """

    policy_fiscal_share: float = 0.5
    """Share of the policy response carried by spending rather than by rates.

    General government spending, distinct from the transfer: easing is not
    only monetary.
    """

    growth_penalties: tuple[str, ...] = ("debt", "inflation", "fx", "capital")
    """Which reduced-form growth terms are active.

    v6 wrote the growth rate as the trend plus four adjustments: a fiscal
    stimulus, a capital-flow term, and penalties for high debt, high inflation
    and exchange-rate movement. The penalties are reduced-form — coefficients
    chosen to make growth fall when the macro position deteriorates, not
    mechanisms derived from anything — and they are exactly the kind of
    assumption the editor objected to when a result turns on them.

    Naming a subset switches the rest off, so a result that depends on them can
    be separated from one that does not. The default keeps all four, which is
    the v6/v7 behaviour.

      debt        -0.02 * max(0, d - 1.2)
      inflation   -0.03 * max(0, pi - 0.05)
      fx          -0.01 * |de|
      capital     +0.15 * flow
    """

    # --- v8 post_employment ---
    automation_max: float = 0.0
    """How far labour demand ultimately falls. Zero is the baseline series.

    Not an estimate — nobody knows — so it is swept rather than chosen.
    """

    automation_speed: float = 0.15
    """How fast automation approaches its ceiling, per year."""

    automation_capacity: float = 1.0
    """Potential output gained per unit of automation.

    Potential, not production: capacity nobody can buy from stands idle.
    """

    automation_cost: float = 0.3
    """Unit cost of domestic production saved per unit of automation."""

    demand_constrains_output: float = 0.0
    """How strongly a shortfall in demand holds production below potential.

    Zero is the v6 behaviour, carried through every phase so far: output
    follows a trend growth rate plus a fiscal impulse, and pays no attention
    to whether anybody can buy what is made. That is tolerable while labour
    income moves with output — demand and capacity grow together, and the
    question never arises.

    post_employment breaks that. Automation raises what could be made and
    removes the income that would buy it, so the two stop moving together and
    output has to be able to fall short of capacity. Above zero, production
    grows more slowly when demand lags potential, which is what closes the
    loop from lost income back to lost output.

    Not an estimate: swept.
    """

    automation_labour: float = 1.0
    """Constrained-block labour income lost per unit of automation.

    At 1.0, automation of `automation_max` removes that share of the block's
    earnings. What it stops earning accrues to the owners of the machines.
    """

    income_linkage: str = "growth_linked"
    """How the constrained block's income keeps up with the economy.

    ``growth_linked`` ties it to real GDP, which is what the model did before
    and what makes the budget ratio comfortable everywhere: output roughly
    doubles over thirty years and the constrained households' income doubles
    with it.

    That is the generous end. Van Mechelen & Marchal (2012, GINI DP 55) find
    that minimum income protection in the EU and US is mostly linked to prices
    rather than to wages, so it holds its real value while falling behind
    earnings — "when wages grow faster than prices … benefit levels subjected
    to such price-linking mechanisms may be more prone to welfare erosion".
    ``cpi_linked`` is the main series for that reason; ``fixed_nominal`` is the
    floor of the range.

    The same reasoning carries into post_employment: as labour income is
    replaced by transfers, household income stops tracking growth of its own
    accord.
    """

    subsistence_reference: str = "start"
    """Where the measured subsistence bundle is fixed.

    ``start`` uses the bloc's opening output, so the indicator asks whether a
    household can still afford what it needed on day one. ``baseline`` uses
    what the bundle would have been in a no-transfer run at the same date,
    which removes any effect the programme itself had on the reference.

    Fixing it somewhere is unavoidable and the choice is not neutral, so both
    are computed and the conclusion is checked against each.
    """

    subsistence_scale: float = 1.0
    """Multiplier on the Stone-Geary floor.

    The floor sums to 0.65 of income, which leaves a hand-to-mouth household
    35% to spend freely — generous for a block defined by being liquidity
    constrained. The share is not calibrated, so it is swept: this multiplier
    raises it towards the theoretical ceiling of 1.0, where the floor absorbs
    the whole budget.
    """

    indexation: str = "gdp_linked"
    """How a transfer's nominal amount moves over time.

    ``gdp_linked`` is the v7 and phase A/B behaviour: the transfer is a fixed
    share of real output, so inflation never touches it. ``fixed_nominal`` and
    ``cpi_indexed`` carry it as a nominal amount instead — see
    `model.indexation` and `specs/V8_C.md`. The default reproduces phases A/B.
    """

    capital_inflow_to_demand: float = 0.0
    """Share of a capital inflow that is spent on tradables on arrival.

    A bloc buying foreign assets is a capital outflow, and in a closed world it
    is somebody else's inflow. Whether that inflow bids for goods is an
    assumption, not a result, and it matters: if capital outflows became world
    goods demand one for one, a modality whose leakage is mostly financial
    (UBI) would press on the world price as hard as one whose leakage is mostly
    imports.

    ``0.0`` is the main series: an inflow buys claims, and reaches goods demand
    only later and indirectly, through the capital stock and growth. Set it
    above zero to charge some of the inflow straight to world demand and see
    how much the conclusion depends on the assumption.
    """

    # --- v8: reserve privileges (v7 behaviour, OFF by default) ---
    reserve_privilege: bool = False
    """Re-enable v7's exogenous reserve-issuer advantages.

    v7 gave the reserve bloc three things by assumption: safe-haven capital
    inflow during others' distress, a higher debt threshold before a risk
    premium appears, and damped exchange-rate movement. The editor's criticism
    was aimed squarely at advantages embedded as parameters, so v8 turns all
    three off and asks whether the reserve bloc's resilience survives on
    settlement alone. Set ``True`` for the sensitivity that reproduces v7's
    treatment.
    """

    reserve_flight: float = 0.3
    reserve_thr: float = 1.5
    reserve_damp: float = 0.5
    reserve_flight_floor: float = 0.3
    """Used only when ``reserve_privilege`` is True."""


def archetypes() -> dict[str, Bloc]:
    """The four archetype blocs, unchanged from v6/v7."""
    return {
        "A": Bloc("A (reserve)", gdp=1.2, debt=0.85, rate=0.020, base_growth=0.018, decay=0.0003,
                  openness=1.0, fx_share=0.0, reserve=True, capital=1.3,
                  c_income_share=0.15, c_pop_share=0.30, m_F=0.10, m_G=0.15, m_H=0.05,
                  eps_supply=0.3, land_share=0.45, omega0=0.05),
        "B": Bloc("B (non-reserve adv.)", gdp=0.9, debt=0.70, rate=0.025, base_growth=0.020,
                  decay=0.0003, openness=1.0, fx_share=0.15, capital=1.0,
                  c_income_share=0.15, c_pop_share=0.30, m_F=0.15, m_G=0.25, m_H=0.08,
                  eps_supply=0.3, land_share=0.40, omega0=0.15),
        "C": Bloc("C (emerging)", gdp=0.5, debt=0.40, rate=0.050, base_growth=0.035, decay=0.0005,
                  openness=0.75, fx_share=0.45, capital=0.45,
                  c_income_share=0.20, c_pop_share=0.40, m_F=0.20, m_G=0.30, m_H=0.15,
                  eps_supply=0.4, land_share=0.35, omega0=0.25),
        "D": Bloc("D (developing)", gdp=0.2, debt=0.35, rate=0.070, base_growth=0.038, decay=0.0008,
                  openness=0.50, fx_share=0.65, capital=0.20,
                  c_income_share=0.25, c_pop_share=0.50, m_F=0.30, m_G=0.35, m_H=0.25,
                  eps_supply=0.5, land_share=0.30, omega0=0.35),
    }


MODALITIES: tuple[str, ...] = ("none", "ubi", "cash_t", "voucher", "clt")
BASES: tuple[str, ...] = ("welfare", "cost")
