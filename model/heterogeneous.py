"""Heterogeneous constrained households, CLT tenancy, and rent incidence.

The v6 model treats the constrained block as one representative household that
receives the whole CLT programme. That hides the question Phase 2 exists to
answer: when CLT tenancies are scarce, who actually benefits — the tenants, or
also the market renters who face a lower clearing rent?

This module splits the block into income quantiles, allocates a limited number
of tenancies between them, and clears the housing market on the residual
market-renter demand. With ``n_types == 1`` and unlimited capacity it reduces
to :func:`model.household.household_block` exactly.

Ported to v8 with the same single change as `household.py`: traded goods are
priced at the world price converted at the bloc's exchange rate, so
``world_price=1.0`` reproduces v7 exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .household import les_demand, utility
from .params import Bloc, G, HouseholdType, income_quantiles

#: Upper bound of the calibration bisection, as a share of GDP. A solution at
#: this bound is reported as `_at_ceiling_*` rather than treated as converged.
SEARCH_CEILING: float = 0.5

def annuity_factor(rate: float, years: int) -> float:
    """Standard annuity factor ``r / (1 - (1+r)^-n)``.

    Rent is a flow and construction is a stock, so the rent credited against
    the programme's annual cost must be annualised. Deriving the factor from
    the bloc's own interest rate and the building's life means a high-rate
    bloc amortises faster, instead of every bloc sharing one invented constant.
    """
    if years <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return 1.0 / years
    return rate / (1.0 - (1.0 + rate) ** -years)


@dataclass(frozen=True)
class TypeOutcome:
    """One household type's allocation and welfare."""

    name: str
    pop_share: float
    is_tenant: bool
    income: float
    quantities: dict[str, float]
    market_spend: dict[str, float]
    utility: float
    rent_paid: float


@dataclass(frozen=True)
class HeteroOutcome:
    """Aggregate flows plus the per-type detail."""

    pH: float
    rent: float
    imp_c: float
    dom_c: float
    imp_gov: float
    dom_gov: float
    cash_u: float
    fiscal: float
    U: float
    types: tuple[TypeOutcome, ...]
    tenant_share: float

    @property
    def tenants(self) -> tuple[TypeOutcome, ...]:
        return tuple(t for t in self.types if t.is_tenant)

    @property
    def renters(self) -> tuple[TypeOutcome, ...]:
        return tuple(t for t in self.types if not t.is_tenant)


def allocate_tenancies(
    types: Sequence[HouseholdType], capacity: float, rule: str
) -> dict[str, float]:
    """Split each type's population between CLT tenancy and the market.

    Args:
        capacity: Tenancies available as a share of the constrained population.
        rule: ``priority_low_income`` fills from the bottom of the income
            distribution upwards; ``lottery`` takes the same fraction of every
            type.

    Returns:
        Tenant population share per type name.
    """
    capacity = max(0.0, min(1.0, capacity))
    if rule == "lottery":
        return {t.name: t.pop_share * capacity for t in types}

    # priority_low_income: poorest first, measured by income per person.
    order = sorted(types, key=lambda t: t.income_share / max(t.pop_share, 1e-12))
    remaining = capacity
    out = {t.name: 0.0 for t in types}
    for t in order:
        take = min(t.pop_share, remaining)
        out[t.name] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    return out


def _build_progress(g: G, size: float, Y: float, years_since_start: float,
                    housing_stock: float) -> float:
    """Share of the finished programme that is standing, in [0, 1]."""
    target = size * Y
    if g.clt_build_rate is None or target <= 0:
        return 1.0
    built = clt_units_available(g, size, Y, years_since_start, housing_stock)
    return min(1.0, built / target)


def clt_units_available(g: G, size: float, Y: float, years_since_start: float,
                        housing_stock: float | None = None) -> float:
    """CLT stock in place, respecting the build-rate cap.

    v6 assumes the entire programme exists immediately. A finite
    ``clt_build_rate`` phases it in linearly instead, expressed as a share of
    the existing *housing stock* per year (per TASKS.md), not of GDP: adding
    2% to the housing stock annually is an ambitious but meaningful pace,
    whereas 2% of GDP would complete a typical programme within a year.
    """
    target = size * Y
    if g.clt_build_rate is None or years_since_start < 0:
        return target
    # The cap is a construction *rate*: at most this share of the existing
    # housing stock can be added per year. It therefore sets how long the
    # programme takes to reach `target`, and a larger target simply takes
    # longer — it does not change what can be built in a given year.
    base = housing_stock if housing_stock is not None else Y
    return min(target, g.clt_build_rate * base * max(0.0, years_since_start))


def hetero_household_block(
    b: Bloc,
    g: G,
    Y: float,
    e: float,
    r: float,
    transfer_on: bool,
    size: float,
    years_since_start: float = 1e9,
    cash_size: float = 0.0,
    food_factor: float = 1.0,
    world_price: float = 1.0,
) -> HeteroOutcome:
    """One period of heterogeneous household demand and market clearing.

    Args:
        cash_size: Cash paid alongside in-kind housing, used by the
            ``clt_cash`` modality. Keeping both in one household problem
            matters: the cash is spent at the rent the CLT programme has
            already produced, which is not the same as adding two separate
            runs together.
        world_price: World-currency price of the traded good. ``1.0`` recovers
            v7's prices exactly.
    """
    types = income_quantiles(g.n_types, g.income_spread)
    Ic_total = b.c_income_share * Y
    beta = {"F": g.beta_F, "G": g.beta_G, "H": g.beta_H}
    traded = e * world_price
    pF = ((1 - b.m_F) + b.m_F * traded) * food_factor
    pG = (1 - b.m_G) + b.m_G * traded

    def subsistence(t: HouseholdType) -> dict[str, float]:
        """Per-type subsistence, scaled to that type's income."""
        income = Ic_total * t.income_share
        return {"F": g.sub_F * income * t.sub_scale,
                "G": g.sub_G * income * t.sub_scale,
                "H": g.sub_H * income * t.sub_scale}

    # Supply anchor: the no-transfer economy clears at pH = 1, as in v6.
    S0 = 0.0
    for t in types:
        income = Ic_total * t.income_share
        sub = subsistence(t)
        S0 += sub["H"] + beta["H"] * (income - sub["F"] - sub["G"] - sub["H"])

    cash_u = 0.0
    fiscal = struct_cost = land_cost = 0.0
    clt_units = 0.0
    cash_per_type: dict[str, float] = {t.name: 0.0 for t in types}
    voucher_per_type: dict[str, float] = {t.name: 0.0 for t in types}
    tenant_share_by_type: dict[str, float] = {t.name: 0.0 for t in types}

    if transfer_on:
        if b.modality == "ubi":
            T = size * Y
            split = b.c_pop_share if g.universal_split is None else g.universal_split
            cash_u = T * (1 - split)
            for t in types:  # universal: per capita within the constrained block
                cash_per_type[t.name] = T * split * t.pop_share
            fiscal = T
        elif b.modality in ("cash_t", "cash_cheap"):
            # Targeted cash is paid per capita across the whole constrained
            # block, so every quantile receives the same amount per household
            # regardless of income. This matters for the comparison with CLT:
            # cash reaches all constrained households, whereas CLT tenancies
            # are rationed by `clt_capacity`. Making the two comparable is why
            # the utilitarian calibration target is the main series.
            for t in types:
                cash_per_type[t.name] = size * Y * t.pop_share
            fiscal = size * Y
        elif b.modality == "voucher":
            for t in types:
                voucher_per_type[t.name] = size * Y * t.pop_share
            fiscal = pF * size * Y
        elif b.modality in ("clt", "clt_cash"):
            if b.modality == "clt_cash" and cash_size > 0:
                # Cash paid alongside the housing, per capita like cash_t so
                # the two instruments stay comparable.
                for t in types:
                    cash_per_type[t.name] = cash_size * Y * t.pop_share
            # S0 is the no-transfer housing stock, so the build cap is a share
            # of housing rather than of output.
            clt_units = clt_units_available(g, size, Y, years_since_start, S0)
            capacity = 1.0 if g.clt_capacity is None else g.clt_capacity
            if g.clt_build_rate is not None:
                # `clt_capacity` is the tenancy the finished programme offers;
                # part-built, it offers that share of it. Progress is measured
                # against the stock the same build rate would have delivered
                # by the time the programme completes, so it depends on the
                # construction path rather than on the welfare target `size`
                # — using `size` here would let a larger target shrink tenancy
                # even though the stock built is identical.
                capacity *= _build_progress(g, size, Y, years_since_start, S0)
            tenant_share_by_type = allocate_tenancies(types, capacity, g.allocation_rule)

            r0 = b.rate
            markup = 1.0 + g.sourcing_cost_kappa * b.clt_domestic_sourcing
            struct_cost = (1 - b.land_share) * clt_units * markup
            land_cost = (b.land_share * (1 - g.land_discount)
                         * (r + 0.02) / (r0 + 0.02) * clt_units)
            fiscal = struct_cost + land_cost
            if b.modality == "clt_cash":
                fiscal += cash_size * Y

    tenant_total = sum(tenant_share_by_type.values())
    # In-kind housing goes to tenants first. Anything the tenancy cannot absorb
    # is not wasted: it adds to the rental stock and so reaches market renters
    # through a lower clearing rent. Without this the market price would be
    # insensitive to the programme's size whenever tenancy is capped, which is
    # exactly the pecuniary externality Phase 2 is meant to measure.
    # How much of the programme the tenancy can absorb is decided by the
    # households' own demand, not by an imposed ceiling: a tenant consumes the
    # in-kind housing up to the point where it stops being infra-marginal, and
    # the corner solution in `les_demand` handles the rest. Allocating by
    # tenancy share leaves any genuine surplus for the market.
    units_per_tenant = (clt_units / tenant_total) if tenant_total > 1e-12 else 0.0

    def solve(pH: float):
        """Demand and welfare for every (type, tenancy status) at price `pH`."""
        p = {"F": pF, "G": pG, "H": pH}
        surplus_units = [0.0]
        outcomes: list[TypeOutcome] = []
        market_demand = 0.0
        for t in types:
            income = Ic_total * t.income_share
            sub = subsistence(t)
            tenant_pop = tenant_share_by_type[t.name]
            renter_pop = t.pop_share - tenant_pop
            for pop, is_tenant in ((tenant_pop, True), (renter_pop, False)):
                if pop <= 1e-12:
                    continue
                frac = pop / t.pop_share
                cash = cash_per_type[t.name] * frac
                sub_i = {k: v * frac for k, v in sub.items()}
                fixed: dict[str, float] = {}
                if voucher_per_type[t.name] > 0:
                    fixed["F"] = voucher_per_type[t.name] * frac
                if is_tenant and units_per_tenant > 0:
                    fixed["H"] = units_per_tenant * pop
                granted = fixed.get("H", 0.0)
                if granted > 0:
                    # A tenancy is capped at what the household would choose to
                    # occupy at these prices: a trust does not force surplus
                    # floorspace on its tenants, it lets the remainder instead.
                    # In equilibrium the rent adjusts until tenants want their
                    # whole allocation, so this surplus is normally zero; the
                    # cap matters because it stops an oversized grant from
                    # being scored as consumption the household did not want.
                    want, _ = les_demand(income * frac + cash, p, sub_i, beta)
                    occupied = min(granted, want["H"])
                    surplus_units[0] += granted - occupied
                    fixed = {**fixed, "H": occupied}

                x, buy = les_demand(income * frac + cash, p, sub_i, beta, fixed)
                market_demand += buy["H"] / pH

                # Utility is evaluated per household, not per group. Scaling a
                # group's income and subsistence by its population size shifts
                # log utility by log(frac) * sum(beta), which would otherwise
                # make a split group look worse off purely because it is
                # smaller. Dividing the bundle by `frac` removes that.
                x_pc = {k: v / frac for k, v in x.items()}
                sub_pc = {k: v / frac for k, v in sub_i.items()}
                outcomes.append(TypeOutcome(
                    name=t.name, pop_share=pop, is_tenant=is_tenant,
                    income=income * frac, quantities=x, market_spend=buy,
                    utility=utility(x_pc, sub_pc, beta), rent_paid=buy["H"]))
        supply = S0 * pH ** (b.eps_supply * g.eps_mult) + surplus_units[0]
        return market_demand - supply, outcomes, surplus_units[0]

    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if solve(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    pH = 0.5 * (lo + hi)
    _, outcomes, _ = solve(pH)

    rent = sum(o.rent_paid for o in outcomes)

    # Rent on CLT-supplied units. Under the main series the trust collects it
    # and it offsets the fiscal cost, so it never becomes landlord income and
    # never reaches the foreign-asset channel. The sensitivity variant leaves
    # it with landlords, as v6 implicitly did.
    clt_rent = 0.0
    if clt_units > 0 and not g.clt_rent_to_landlords:
        total_housing = S0 * pH ** (b.eps_supply * g.eps_mult) + clt_units
        if total_housing > 1e-12:
            clt_rent = rent * (clt_units / total_housing)
            rent -= clt_rent
            # `fiscal` is an annual flow while the build cost is a stock, so
            # the offset is annualised at the bloc's own cost of capital over
            # the building's life.
            clt_rent *= annuity_factor(r, g.housing_life_years)
    imp_c = sum(b.m_F * o.market_spend["F"] + b.m_G * o.market_spend["G"] for o in outcomes)
    dom_c = sum((1 - b.m_F) * o.market_spend["F"] + (1 - b.m_G) * o.market_spend["G"]
                for o in outcomes)

    clt_on = b.modality in ("clt", "clt_cash") and transfer_on
    clt_build = struct_cost if clt_on else 0.0
    land_pay = land_cost if clt_on else 0.0
    m_H_eff = b.m_H * (1.0 - b.clt_domestic_sourcing)
    imp_gov = m_H_eff * clt_build
    dom_gov = clt_build - imp_gov
    if b.modality == "voucher" and transfer_on:
        imp_gov += b.m_F * fiscal
        dom_gov += (1 - b.m_F) * fiscal

    # Aggregate welfare: population-weighted, matching v6 when K = 1.
    total_pop = sum(o.pop_share for o in outcomes)
    U = sum(o.utility * o.pop_share for o in outcomes) / total_pop if total_pop else -1e9

    return HeteroOutcome(
        pH=pH, rent=rent, imp_c=imp_c, dom_c=dom_c, imp_gov=imp_gov, dom_gov=dom_gov,
        cash_u=cash_u + land_pay, fiscal=fiscal - clt_rent, U=U,
        types=tuple(outcomes), tenant_share=tenant_total,
    )


def welfare_measure(outcome: HeteroOutcome, mode: str) -> float:
    """Aggregate a heterogeneous outcome into one welfare number.

    Args:
        mode: ``tenant`` weights only CLT tenants — the calibration implicit in
            v6, where the programme's recipients are the whole constrained
            block. ``utilitarian`` weights every constrained household,
            tenants and market renters alike, so the pecuniary externality on
            renters counts.

    The two coincide when every household is a tenant, which is why K=1 with
    unlimited capacity reproduces v6 under either mode.
    """
    if mode == "utilitarian":
        pop = sum(t.pop_share for t in outcome.types)
        return sum(t.utility * t.pop_share for t in outcome.types) / pop if pop else -1e9

    if mode != "tenant":
        raise ValueError(f"unknown welfare mode: {mode!r}")

    tenants = outcome.tenants
    pop = sum(t.pop_share for t in tenants)
    if pop <= 1e-12:  # no tenancies (e.g. cash transfers): fall back to everyone
        return welfare_measure(outcome, "utilitarian")
    return sum(t.utility * t.pop_share for t in tenants) / pop


def calibrate_hetero(b: Bloc, g: G, u_ubi: float, mode: str = "utilitarian") -> dict[str, Any]:
    """Welfare-equivalent sizes under a heterogeneous population.

    Mirrors :func:`model.household.calibrate`, but the target utility is the
    aggregate chosen by `mode`.
    """
    from dataclasses import replace as _replace

    Y, e, r = b.gdp, 1.0, b.rate
    target_out = hetero_household_block(_replace(b, modality="ubi"), g, Y, e, r, True, u_ubi)
    target = welfare_measure(target_out, mode)
    out: dict[str, Any] = {"ubi": u_ubi, "_U": target,
                           "_fiscal_ubi": target_out.fiscal / Y, "_mode": mode}

    for mod in ("cash_t", "voucher", "clt"):
        lo, hi = 0.0, SEARCH_CEILING
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            trial = hetero_household_block(_replace(b, modality=mod), g, Y, e, r, True, mid)
            if welfare_measure(trial, mode) < target:
                lo = mid
            else:
                hi = mid
        size = 0.5 * (lo + hi)
        final = hetero_household_block(_replace(b, modality=mod), g, Y, e, r, True, size)
        out[mod] = size
        out["_fiscal_" + mod] = final.fiscal / Y
        if welfare_measure(final, mode) < target - 1e-6:
            out["_infeasible_" + mod] = True
        # The bisection ran out of room rather than converging, so the reported
        # size is a bound, not a solution. Flagged separately from infeasible
        # because the equivalence may still exist above the search ceiling.
        if size > SEARCH_CEILING * (1 - 1e-6):
            out["_at_ceiling_" + mod] = True
    return out


def discounted_calibrate(
    b: Bloc, g: G, u_ubi: float, mode: str = "utilitarian",
    years: int = 30, rate: float = 0.03,
    terminal_value: bool = False, depreciation: float = 0.02,
    only: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Welfare-equivalent sizes matched on discounted present value.

    Calibrating at the moment of introduction flatters any modality that
    phases in slowly: a CLT under a build-rate cap delivers almost nothing in
    year one, yet a snapshot comparison treats it as if it did. Matching the
    present value of the utility gain over `years` removes that advantage.

    Truncating at `years` introduces the opposite bias. Housing is durable, so
    a horizon that ends while the stock is still standing discards benefits
    the programme genuinely delivers. ``terminal_value=True`` adds the present
    value of the flow beyond the horizon, decaying at `depreciation`, which
    separates "the build cap really is binding" from "the horizon was too
    short".

    Caveat on identification. While the build cap binds, the stock in place is
    set by the cap and not by `size`, so the welfare path is flat in `size`
    over exactly the years the cap is active. Raising `size` then only thins
    the per-tenant allocation once the cap lifts, which can make the objective
    non-monotone and defeat the bisection. Where that happens the routine
    reports `_at_ceiling_*`; read it as "size is not the right instrument
    here", not as "no equivalent programme exists". Phase 3 addresses this by
    treating the shortfall during the transition as a cash top-up instead.
    """
    from dataclasses import replace as _replace

    Y, e, r = b.gdp, 1.0, b.rate
    discount = [(1.0 + rate) ** -t for t in range(years)]

    def present_value(modality: str, size: float) -> float:
        """Discounted utility path, with the phase-in the modality actually has.

        Only CLT is subject to the build-rate cap; cash and vouchers arrive in
        full from the first year. Comparing present values is what stops the
        slow-starting modality from being credited with benefits it has not
        yet delivered.
        """
        total = 0.0
        last = 0.0
        for t, w in enumerate(discount):
            out = hetero_household_block(_replace(b, modality=modality), g, Y, e, r,
                                         True, size, years_since_start=float(t))
            last = welfare_measure(out, mode)
            total += w * last
        if terminal_value:
            # Perpetuity of the final-year flow, decaying at `depreciation`.
            tail = discount[-1] / (rate + depreciation) if (rate + depreciation) > 0 else 0.0
            total += tail * last
        return total

    target = present_value("ubi", u_ubi)
    out: dict[str, Any] = {"ubi": u_ubi, "_U": target, "_mode": mode,
                           "_discount_rate": rate, "_years": years,
                           "_terminal_value": terminal_value,
                           "_depreciation": depreciation if terminal_value else None}

    for mod in (only or ("cash_t", "voucher", "clt")):
        lo, hi = 0.0, SEARCH_CEILING
        # 30 halvings resolve the ceiling to ~5e-10, far finer than anything
        # reported; more would only cost time on the 60-year paths.
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if present_value(mod, mid) < target:
                lo = mid
            else:
                hi = mid
        size = 0.5 * (lo + hi)
        out[mod] = size
        final = hetero_household_block(_replace(b, modality=mod), g, Y, e, r, True, size)
        out["_fiscal_" + mod] = final.fiscal / Y
        if size > SEARCH_CEILING * (1 - 1e-6):
            out["_at_ceiling_" + mod] = True
    return out


def rent_incidence(
    treated: HeteroOutcome, baseline: HeteroOutcome
) -> dict[str, dict[str, float]]:
    """Change in rent paid and utility, split by type and tenancy status.

    This is the Phase 2 output that the representative model cannot produce:
    it separates the tenants' in-kind benefit from the market renters' gain
    through the lower clearing rent.
    """
    before = {(t.name, t.is_tenant): t for t in baseline.types}
    # Baseline has no tenants, so market renters are the comparison group.
    before_by_name = {t.name: t for t in baseline.types if not t.is_tenant}

    out: dict[str, dict[str, float]] = {}
    for t in treated.types:
        prior = before.get((t.name, t.is_tenant)) or before_by_name.get(t.name)
        if prior is None:
            continue
        key = f"{t.name}:{'tenant' if t.is_tenant else 'renter'}"
        scale = t.pop_share / prior.pop_share if prior.pop_share > 1e-12 else 1.0
        out[key] = {
            "pop_share": t.pop_share,
            "d_rent": t.rent_paid - prior.rent_paid * scale,
            "d_utility": t.utility - prior.utility,
            "rent_share_of_income": t.rent_paid / t.income if t.income > 0 else 0.0,
        }
    return out
