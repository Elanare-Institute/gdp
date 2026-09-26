"""Parameter containers for the provisioning model.

Split out of ``sim_v6.py`` without changing any numeric default. New v7
options are added as flags whose defaults reproduce v6 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

YEARS: int = 30
DT: float = 0.25
STEPS: int = int(YEARS / DT)

# Crisis thresholds. These are reporting cut-offs, not structural boundaries:
# results are reported as signs, orderings, and shares of parameter space.
THR: dict[str, float] = dict(debt=1.5, infl=0.15, dep=2.0, app=0.3)

ReserveFlightMode = Literal["v6", "off", "capped"]


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
    """none | ubi | cash_t | voucher | clt | clt_cash | cash_cheap.

    ``cash_cheap`` is a control, not a policy proposal: targeted cash paid the
    same way as ``cash_t`` but sized to the CLT's fiscal cost rather than to
    the welfare target. It therefore delivers less welfare. Its purpose is to
    separate "CLT is chosen because it is cheap" from "CLT is chosen because
    it is non-fugitive", which the welfare-equivalent comparison cannot do.
    """
    size: float = 0.0             # ubi: share of GDP; else calibrated quantity
    clt_domestic_sourcing: float = 0.0
    """Policy effort to source CLT construction domestically, in [0, 1].

    Distinct from ``m_H``. ``m_H`` is a structural fact about how import-
    dependent construction is; this is a policy lever (local content rules,
    domestic supplier development) that reduces the import share actually
    realised on CLT building. The effective import share is
    ``m_H * (1 - clt_domestic_sourcing)``, so 0 reproduces v6 exactly and 1
    means construction is fully domestically sourced.

    It applies only to CLT construction, not to household consumption imports
    or to voucher food procurement, which are not government build decisions.
    """


AllocationRule = Literal["priority_low_income", "lottery"]

#: Damping on the subsistence tilt across income quantiles. A full inverse
#: tilt (exponent 1.0) drives the poorest quantile below subsistence before any
#: transfer is made, which would make its utility undefined rather than merely
#: low. See `feasible_spread` for the binding constraint.
SUB_SCALE_EXPONENT: float = 0.5

#: Hard ceiling on the subsistence multiplier, as a second guard.
SUB_SCALE_CAP: float = 1.35


@dataclass(frozen=True)
class HouseholdType:
    """One income quantile within the constrained population.

    Shares are expressed relative to the constrained block as a whole, so
    ``pop_share`` and ``income_share`` each sum to 1 across the types.
    """

    name: str
    pop_share: float
    income_share: float
    sub_scale: float = 1.0
    """Multiplier on subsistence requirements.

    Poorer types spend a larger fraction of income on necessities, so their
    subsistence floor binds harder. A value of 1.0 reproduces the homogeneous
    v6 household.
    """


def income_quantiles(k: int, spread: float = 0.6) -> tuple[HouseholdType, ...]:
    """Build `k` equal-population income quantiles.

    ``spread`` controls inequality within the constrained block: 0 gives
    identical types (and therefore v6 behaviour), while larger values tilt
    income towards the upper quantiles. Subsistence scaling moves the opposite
    way, so lower quantiles are more housing-constrained.
    """
    if k == 1:
        return (HouseholdType("all", 1.0, 1.0, 1.0),)

    weights = [1.0 + spread * (2.0 * i / (k - 1) - 1.0) for i in range(k)]
    total = sum(weights)
    mean_w = total / k

    types = []
    for i, w in enumerate(weights):
        share = w / total
        # Poorer quantiles devote more of their income to necessities, but the
        # floor must stay strictly inside their budget or the type is
        # infeasible before any policy is applied. The exponent damps the tilt
        # and the cap enforces feasibility explicitly.
        sub_scale = min(SUB_SCALE_CAP, (mean_w / w) ** SUB_SCALE_EXPONENT)
        types.append(HouseholdType(f"q{i + 1}", 1.0 / k, share, sub_scale))
    return tuple(types)


def subsistence_headroom(types: Sequence["HouseholdType"], g: "G") -> dict[str, float]:
    """Fraction of each type's income left after subsistence.

    Values at or below zero mean the type cannot reach its subsistence bundle
    even before any transfer, which makes its Stone-Geary utility undefined
    rather than merely low. Every quantile must have positive headroom for the
    heterogeneous model to be well posed.
    """
    # Subsistence is scaled to each type's own income (see
    # model.heterogeneous.hetero_household_block), so the headroom is simply
    # one minus the scaled floor — no population reweighting.
    floor = g.sub_F + g.sub_G + g.sub_H
    return {t.name: 1.0 - floor * t.sub_scale for t in types}


@dataclass
class G:
    """Global parameters shared across blocs."""

    fiscal_mult: float = 1.2
    cap_sens: float = 0.8
    risk_coeff: float = 0.15
    reserve_flight: float = 0.3
    reserve_thr: float = 1.5
    reserve_damp: float = 0.5
    compress: float = 0.3
    deficit_share: float = 0.6
    bop_coeff: float = 0.5        # external leakage (share of GDP) -> FX pressure
    export_switch: float = 0.05   # expenditure switching on depreciation
    omega_sens: float = 2.0       # sensitivity of foreign-asset share to risk
    mpc_u: float = 0.3            # unconstrained households' MPC
    land_discount: float = 0.5    # CLT land acquisition discount
    default_thr: float = 3.0
    eps_mult: float = 1.0         # uniform multiplier on supply elasticity
    sourcing_cost_kappa: float = 0.0
    """Cost markup for domestic sourcing of CLT construction.

    Unit construction cost is scaled by ``1 + kappa * clt_domestic_sourcing``.
    Sourcing locally is not free: it means paying above the import price while
    domestic capacity is thin. ``0.0`` reproduces the earlier behaviour, in
    which sourcing was costless and therefore unrealistically attractive.
    """
    # Stone-Geary preferences for constrained households.
    sub_F: float = 0.35
    sub_G: float = 0.05
    sub_H: float = 0.25
    beta_F: float = 0.25
    beta_G: float = 0.45
    beta_H: float = 0.30

    # --- v7 flags (defaults reproduce v6 bit-for-bit) ---
    reserve_flight_mode: ReserveFlightMode = "v6"
    """How the reserve bloc's safe-haven inflow is handled.

    ``v6`` keeps the original behaviour. In v6 the reserve bloc's crises are
    driven by flight-induced appreciation (e < 0.3 trips the ``app``
    threshold), which conflates a modelling artifact with a policy result.
    ``off`` removes the inflow entirely; ``capped`` keeps it but floors the
    exchange rate at :attr:`reserve_flight_floor`, so appreciation cannot by
    itself manufacture a crisis. Used for robustness checks, not as the
    headline specification.
    """

    reserve_flight_floor: float = 0.3
    """Exchange-rate floor applied when ``reserve_flight_mode == 'capped'``."""

    # --- Phase 2: household heterogeneity (defaults reproduce v6) ---
    n_types: int = 1
    """Number of income quantiles the constrained block is split into.

    ``1`` collapses to the representative household of v6.
    """

    income_spread: float = 0.6
    """Inequality across the quantiles; 0 makes them identical."""

    clt_capacity: float | None = None
    """CLT tenancies available, as a share of the constrained population.

    ``None`` means every constrained household is housed by the CLT, which is
    the implicit v6 assumption. Below 1.0 the block splits into CLT tenants and
    market renters, and the rent relief reaches the latter only through the
    market — the pecuniary externality Phase 2 is meant to expose.
    """

    allocation_rule: AllocationRule = "priority_low_income"
    """Who gets a CLT tenancy when capacity is scarce."""

    clt_build_rate: float | None = None
    """Cap on new CLT stock per year, as a share of housing stock.

    ``None`` keeps v6's assumption that the whole programme exists from the
    first period. A finite rate phases it in.
    """

    housing_life_years: int = 40
    """Useful life of CLT housing, used to annualise the rent offset.

    The offset rate is the standard annuity factor ``r / (1 - (1+r)^-n)`` with
    ``r`` the bloc's own interest rate, so a high-rate bloc amortises faster.
    40 years is the main series; 20 and 60 are the sensitivities.
    """

    clt_rent_to_landlords: bool = True
    """Who collects rent on CLT-supplied housing.

    ``True`` is the v6 treatment and the default, so regression parity holds:
    CLT units are pooled with the private stock and their rent flows to
    landlords like any other rent, which puts it on the capital-flight path.

    ``False`` credits that rent to the trust instead, netting it off the
    programme's fiscal cost so it never becomes landlord income. Phase 2
    reports this as the main series — a trust that let its homes and handed the
    proceeds to landlords would not be a trust — and keeps ``True`` as the
    sensitivity.
    """

    universal_split: float | None = None
    """Share of a universal transfer going to constrained households.

    ``None`` uses ``bloc.c_pop_share`` (v6 behaviour). Setting it makes the
    split an explicit policy parameter rather than a demographic identity.
    """


def archetypes() -> dict[str, Bloc]:
    """The four v6 archetype blocs."""
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
