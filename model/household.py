"""Household demand and housing-market clearing.

Ported from v7 unchanged except for one thing: the prices of traded goods are
now built from the **world price** and the exchange rate, rather than from the
exchange rate alone.

v7 wrote ``pF = 1 + m_F * (e - 1)``, which is the same as assuming the world
price of the traded part is always 1 and only the exchange rate moves it. v8
carries an explicit ``P_w``, so the traded part costs ``e * P_w`` and the
domestic part still costs 1:

    pF = (1 - m_F) * 1 + m_F * (e * P_w)

At ``P_w = 1`` this is algebraically identical to v7, which is what
`tests/test_household_port.py` asserts. Once the world price moves, a bloc
pays more for imports even if its own exchange rate never budged — the channel
the paper's second claim runs through.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .params import Bloc, G


def les_demand(
    M: float,
    p: dict[str, float],
    sub: dict[str, float],
    beta: dict[str, float],
    fixed: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Stone-Geary / linear expenditure system demand.

    Args:
        M: Cash income available for market purchases.
        p: Prices by good.
        sub: Subsistence quantities by good.
        beta: Marginal budget shares by good.
        fixed: In-kind quantities received by good. When an in-kind grant
            exceeds what the household would freely choose, that good is
            *extramarginal*: the corner solution pins its quantity and the
            remaining budget is reallocated across the other goods. This is
            what makes an infra-marginal voucher equivalent to cash, and it is
            solved explicitly rather than assumed.

    Returns:
        ``(quantities, market_expenditure)`` by good.
    """
    goods = list(p)
    fixed = fixed or {}
    full = M + sum(p[k] * q for k, q in fixed.items())
    supern = full - sum(p[k] * sub[k] for k in goods)
    x = {k: sub[k] + beta[k] * max(supern, 0) / p[k] for k in goods}

    binding = [k for k, q in fixed.items() if x[k] < q]
    if binding:  # in-kind grant is extramarginal for these goods
        free = [k for k in goods if k not in binding]
        bsum = sum(beta[k] for k in free)
        rest = M + sum(p[k] * fixed[k] for k in fixed if k not in binding)
        sup2 = rest - sum(p[k] * sub[k] for k in free)
        for k in binding:
            x[k] = fixed[k]
        for k in free:
            x[k] = sub[k] + (beta[k] / bsum) * max(sup2, 0) / p[k]

    buy = {k: p[k] * max(0.0, x[k] - fixed.get(k, 0.0)) for k in goods}
    return x, buy


def utility(x: dict[str, float], sub: dict[str, float], beta: dict[str, float]) -> float:
    """Stone-Geary utility. Returns a large negative number below subsistence."""
    total = 0.0
    for k in x:
        d = x[k] - sub[k]
        if d <= 0:
            return -1e9
        total += beta[k] * math.log(d)
    return total


def household_block(
    b: Bloc, g: G, Y: float, e: float, r: float, transfer_on: bool, size: float,
    food_factor: float = 1.0, cash_size: float = 0.0, world_price: float = 1.0,
    nominal: float | None = None, scarcity: float = 0.0,
    income_factor: float = 1.0, cost_factor: float = 1.0,
) -> dict[str, Any]:
    """One period of household demand and housing-market clearing.

    Args:
        transfer_on: ``False`` gives the counterfactual without the transfer.
        world_price: World-currency price of the traded good. ``1.0`` recovers
            v7's prices exactly.
        nominal: The transfer's nominal amount, when an indexation rule is
            carrying it (phase C). ``None`` falls back to ``size * Y``, which
            is the real, GDP-linked transfer of v7 and phases A/B.
        income_factor: Multiplier on the constrained block's income, carrying
            the linkage rule (`G.income_linkage`) and, under post_employment,
            the labour income automation has removed.
        cost_factor: Multiplier on the unit cost of domestic production.
            Below 1.0 under post_employment: machines make domestic goods
            cheaper, which is the one way automation helps a household.
        scarcity: Share of last period's import demand that could not be
            settled (phase D), in [0, 1]. Goods a bloc cannot pay for do not
            become unavailable at the old price — they become dearer, and the
            household buys less of them. Entering the constraint as a price
            is what makes it reach consumption rather than only the balance
            of payments.

    Returns:
        Flows in units of GDP, plus the clearing rent ``pH`` and utility ``U``.
    """
    Ic = b.c_income_share * Y * income_factor
    # `g.subsistence_scale` deliberately does NOT enter here. It sweeps the
    # floor used by the budget indicator, and letting it into the household's
    # own problem would change the allocation, the welfare calibration and
    # everything Phase 2 rests on. The two floors are separate on purpose —
    # see `model.subsistence`.
    sub = {"F": g.sub_F * Ic, "G": g.sub_G * Ic, "H": g.sub_H * Ic}
    beta = {"F": g.beta_F, "G": g.beta_G, "H": g.beta_H}
    # Traded content is paid at the world price converted at this bloc's rate;
    # the rest is domestic and costs 1 by normalisation.
    domestic = cost_factor
    traded = e * world_price
    # Imports that cannot be settled are not simply absent: they are scarce,
    # and scarcity is a price. A bloc rationed to half its import demand faces
    # a traded-goods price that has risen enough to clear the quantity it can
    # actually pay for, which is how the settlement constraint reaches what a
    # household eats.
    traded *= 1.0 / max(1e-6, 1.0 - min(0.95, max(0.0, scarcity)))
    pF = ((1 - b.m_F) * domestic + b.m_F * traded) * food_factor
    pG = (1 - b.m_G) * domestic + b.m_G * traded
    # Housing supply is anchored so that the no-transfer economy clears at pH = 1.
    S0 = sub["H"] + beta["H"] * (Ic - sub["F"] - sub["G"] - sub["H"])

    cash_c = cash_u = 0.0
    fixed: dict[str, float] = {}
    clt_units = 0.0
    fiscal = 0.0
    struct_cost = land_cost = 0.0

    # The programme's own budget. Under an indexation rule this is a nominal
    # amount carried across periods; without one it is a fixed share of real
    # output, which is what makes the v7 transfer immune to inflation.
    budget = size * Y if nominal is None else nominal

    if transfer_on:
        if b.modality == "ubi":
            T = budget
            split = b.c_pop_share if g.universal_split is None else g.universal_split
            cash_c, cash_u = T * split, T * (1 - split)
            fiscal = T
        elif b.modality in ("cash_t", "cash_cheap"):
            # cash_cheap is the Phase 3 control: paid out exactly like cash_t,
            # but sized to CLT's fiscal cost instead of to the welfare target.
            # It must behave identically here — the two differ only in `size`.
            cash_c = budget
            fiscal = cash_c
        elif b.modality == "voucher":
            # A voucher hands over food, so its size is a quantity. Under an
            # indexation rule the programme is carried as a budget instead, and
            # what that budget buys falls as food gets dearer — which is how
            # inflation reaches an in-kind transfer. Without a rule the size is
            # the quantity directly, as in v7.
            quantity = size * Y if nominal is None else (budget / pF if pF > 0 else 0.0)
            fixed = {"F": quantity}
            fiscal = pF * quantity
        elif b.modality in ("clt", "clt_cash"):
            # clt_cash pays cash alongside the in-kind housing, per capita like
            # cash_t, so the two instruments stay comparable. It mirrors the
            # branch in `heterogeneous.py`; keeping the two in step is what the
            # macro-wiring test enforces.
            if b.modality == "clt_cash" and cash_size > 0:
                cash_c += cash_size * Y
                fiscal += cash_size * Y
            # In-kind housing is a real quantity: what the trust built does
            # not shrink because prices rose. An indexation rule therefore
            # applies to the construction budget, and the units follow from
            # what that budget builds — but once built they stay built, which
            # is the sense in which an in-kind programme holds its real value.
            clt_units = size * Y if nominal is None else budget
            fixed = {"H": clt_units}
            r0 = b.rate  # pre-transfer interest level
            # Domestic sourcing raises the unit construction cost: local supply
            # is thinner than the import market, so substituting away from
            # imports is paid for in price.
            sourcing_markup = 1.0 + g.sourcing_cost_kappa * b.clt_domestic_sourcing
            struct_cost = (1 - b.land_share) * clt_units * sourcing_markup
            land_cost = (
                b.land_share * (1 - g.land_discount) * (r + 0.02) / (r0 + 0.02) * clt_units
            )
            fiscal += struct_cost + land_cost

    def excess(pH: float):
        """Excess demand for rental housing at price `pH`."""
        p = {"F": pF, "G": pG, "H": pH}
        x, buy = les_demand(Ic + cash_c, p, sub, beta, fixed)
        return buy["H"] / pH - S0 * pH ** (b.eps_supply * g.eps_mult), x, buy, p

    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    pH = 0.5 * (lo + hi)
    _, x, buy, p = excess(pH)

    rent = buy["H"]  # landlord rental income
    imp_c = b.m_F * buy["F"] + b.m_G * buy["G"]
    # The same imports as a real quantity. `buy` is spending and `x` is the
    # quantity bought, so `imp_c` moves with prices while `imp_q` does not.
    # World capacity is a real quantity, so it is `imp_q` that has to be
    # compared against it; `imp_c` is what the balance of payments is
    # denominated in. Deflating `imp_c` by the world price would be wrong in
    # both directions — imported goods carry a domestic component too, so the
    # world price is not their deflator.
    imp_q = b.m_F * x["F"] + b.m_G * x["G"]
    dom_c = (1 - b.m_F) * buy["F"] + (1 - b.m_G) * buy["G"]

    clt_on = b.modality in ("clt", "clt_cash") and transfer_on
    clt_build = struct_cost if clt_on else 0.0   # real construction demand
    land_pay = land_cost if clt_on else 0.0      # transfer to landowners
    # Domestic-sourcing policy reduces the import content actually realised on
    # construction; it does not change the structural parameter m_H.
    m_H_eff = b.m_H * (1.0 - b.clt_domestic_sourcing)
    imp_gov = m_H_eff * clt_build
    dom_gov = clt_build - imp_gov
    # Construction is sized in real units, so its import content is already a
    # quantity. Voucher procurement is not: `fiscal` there is `pF * size * Y`,
    # a spending figure, so the real quantity procured is that divided by the
    # food price.
    imp_gov_q = imp_gov
    if b.modality == "voucher" and transfer_on:  # public food procurement
        imp_gov += b.m_F * fiscal
        dom_gov += (1 - b.m_F) * fiscal
        imp_gov_q += b.m_F * (fiscal / pF if pF > 0 else 0.0)

    return dict(
        pH=pH, rent=rent, imp_c=imp_c, imp_q=imp_q, dom_c=dom_c,
        imp_gov=imp_gov, imp_gov_q=imp_gov_q, dom_gov=dom_gov,
        cash_u=cash_u + land_pay, fiscal=fiscal, U=utility(x, sub, beta), x=x,
        buy=buy, S0=S0, prices=p,
    )


def calibrate(b: Bloc, g: G, u_ubi: float) -> dict[str, Any]:
    """Find each modality's size that matches UBI's constrained-household utility.

    Calibration uses the price system at introduction.
    """
    Y, e, r = b.gdp, 1.0, b.rate
    target = household_block(replace(b, modality="ubi"), g, Y, e, r, True, u_ubi)
    out: dict[str, Any] = {"ubi": u_ubi, "_U": target["U"], "_fiscal_ubi": target["fiscal"] / Y}

    for mod in ("cash_t", "voucher", "clt"):
        lo, hi = 0.0, 0.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if household_block(replace(b, modality=mod), g, Y, e, r, True, mid)["U"] < target["U"]:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)
        hb = household_block(replace(b, modality=mod), g, Y, e, r, True, s)
        out[mod] = s
        out["_fiscal_" + mod] = hb["fiscal"] / Y
        if hb["U"] < target["U"] - 1e-6:
            out["_infeasible_" + mod] = True
    return out


def equal_cost_size(b: Bloc, g: G, mod: str, fiscal_share: float) -> float:
    """Size of `mod` whose fiscal cost matches `fiscal_share` of GDP."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = household_block(
            replace(b, modality=mod), g, b.gdp, 1.0, b.rate, True, mid
        )["fiscal"] / b.gdp
        if f < fiscal_share:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
