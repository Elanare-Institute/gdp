"""Macro dynamics with a closed world.

The structure of a bloc's period is carried over from v7 — household demand,
housing clearing, debt accumulation, a risk premium, an exchange rate — but
the blocs are no longer solved in isolation. Each quarter runs in two passes:

  1. every bloc computes its import demand at the prices it currently faces;
  2. the world clears — those imports become somebody's exports, the desired
     capital flows are made mutually consistent, and the world price responds
     to world demand — and only then does each bloc update its state.

That ordering is what makes the world closed. In v7 a bloc's leakage was
subtracted from its own balance of payments and never appeared anywhere else;
here it is another bloc's export revenue, and if every bloc leaks at once the
world price rises rather than the leakage disappearing.

Three v7 mechanisms are **off by default** and gated behind
``G.reserve_privilege``: safe-haven inflow to the reserve bloc, its higher
debt threshold, and its damped exchange rate. See CLAUDE.md.

See `specs/V8_A_B.md`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from .household import calibrate, equal_cost_size, household_block
from .indexation import TransferState, advance
from .automation import capacity_overhang, state as automation_state
from .params import DT, STEPS, THR, Bloc, G, archetypes
from .policy import respond
from .subsistence import (
    positions, reference_bundles, required_transfer, summarise,
)
from .settlement import (
    axis_distance, borrowing_capacity, contagion_weight, decay_contagion,
    rationing_summary, reserve_currency_demand, reserve_is_constrained,
    settle, update_reserves,
)
from .trade import (
    balanced_capital_flows, current_accounts, export_capacity,
    exports_from_imports, net_foreign_asset_change, trade_shares,
)
from .world import WorldState, import_price, initial_capacity, step_world


def _bloc_demand(b: Bloc, g: G, s: dict[str, float], t: float,
                 world: WorldState,
                 transfer: TransferState | None = None) -> dict[str, Any]:
    """One bloc's demands at current prices, before the world clears.

    Returns the import demand the world has to satisfy, the domestic demand
    that stays at home, and the pieces the state update needs. Nothing here
    touches the bloc's state: this is the first of the two passes.
    """
    on = t >= b.start and b.modality != "none"
    Y, e, r = s["Y"], s["d_e"], s["r"]
    pw = world.price

    # Under an indexation rule the transfer is a nominal amount carried across
    # periods; the rule decides whether inflation erodes it, and at what
    # fiscal cost. Under the default it stays a share of real output, which is
    # the phase A/B behaviour.
    nominal = None
    # An insolvent bloc's state stops being updated, its inflation included.
    # Indexing a transfer to a frozen inflation rate would keep raising it
    # against a price level that no longer moves, so the transfer would gain
    # real value by virtue of the bloc having defaulted. Indexation stops when
    # the state it reads stops.
    if transfer is not None and g.indexation != "gdp_linked" and on and s["alive"]:
        # `pi` is an annual rate and this runs quarterly, so the transfer is
        # adjusted by the inflation of one quarter. Passing the annual rate
        # would index four times too fast and the transfer would outrun the
        # price level it is supposed to be tracking.
        nominal = advance(transfer, g.indexation, size=b.size, output=Y,
                          price=s["P"], inflation=s["pi_lag"] * DT,
                          rent=s["pH"])

    # Last period's unsettled share, which this period's prices carry.
    scarcity = s.get("scarcity", 0.0)
    # How far the constrained block's income has kept up. Under the default it
    # tracks real output exactly (factor 1); under the price-linked rule it
    # holds its real value but not its share of a growing economy, which is
    # what minimum income protection actually does in most of the OECD.
    factor = _income_factor(b, g, s)
    # Automation removes part of the constrained block's labour income and
    # makes domestic production cheaper. Both come from the same progress
    # variable, so neither can be tuned without the other.
    auto = automation_state(g, t)
    factor *= auto.labour_multiplier
    hb1 = household_block(b, g, Y, s["e"], r, on, b.size, world_price=pw,
                          nominal=nominal, scarcity=scarcity,
                          income_factor=factor, cost_factor=auto.cost_multiplier)
    hb0 = household_block(b, g, Y, s["e"], r, False, b.size, world_price=pw,
                          scarcity=scarcity, income_factor=factor,
                          cost_factor=auto.cost_multiplier)

    d_imp = (hb1["imp_c"] - hb0["imp_c"]) + hb1["imp_gov"]
    d_rent = hb1["rent"] - hb0["rent"]
    # What labour no longer earns is still produced and still paid for, so it
    # accrues to the owners of the machines — the unconstrained block, whose
    # saving goes abroad on the same path as any other income of theirs.
    displaced = b.c_income_share * Y * auto.displaced_labour_share
    d_inc_u = hb1["cash_u"] + d_rent + displaced
    d_imp += g.mpc_u * b.m_G * d_inc_u
    # The same increment as a real quantity, for the world market. The
    # unconstrained households' share is spending on the general good, so it
    # is deflated by that good's price rather than by the world price.
    pg = hb1["prices"]["G"]
    d_imp_q = ((hb1["imp_q"] - hb0["imp_q"]) + hb1["imp_gov_q"]
               + g.mpc_u * b.m_G * d_inc_u / pg if pg > 0 else 0.0)
    d_dom = ((hb1["dom_c"] - hb0["dom_c"]) + hb1["dom_gov"]
             + g.mpc_u * (1 - b.m_G) * d_inc_u)

    # Baseline imports: what the bloc buys from the world with no programme at
    # all. The world has to clear on the level, not on the policy increment,
    # or a world-wide programme would show up as demand against zero capacity.
    base_imp = hb0["imp_c"]

    return {
        "on": on, "hb1": hb1, "hb0": hb0,
        "d_imp": d_imp, "d_dom": d_dom, "d_rent": d_rent, "d_inc_u": d_inc_u,
        # Nominal, for the balance of payments; real, for the world market.
        "imports": base_imp + d_imp,
        "imports_q": hb0["imp_q"] + d_imp_q,
        "fiscal": hb1["fiscal"],
    }


def _foreign_asset_purchase(b: Bloc, g: G, s: dict[str, float],
                            d_inc_u: float, rp: float) -> float:
    """Households' purchase of foreign assets out of the transfer.

    Unconstrained households save part of what reaches them and place part of
    that saving abroad; the share rises with the risk they perceive at home.
    This is a capital outflow, not a purchase of goods: it is one bloc buying a
    claim on another, so it belongs in the capital account and not in world
    demand for tradables.
    """
    exp_dep = max(0.0, s["de"] / DT)
    omega = min(0.9, max(0.0, b.omega0 + g.omega_sens * (rp + exp_dep)))
    return (1 - g.mpc_u) * d_inc_u * omega * b.openness


def _income_factor(b: Bloc, g: G, s: dict[str, float]) -> float:
    """Constrained income relative to what real-output linkage would give.

    Income is computed as `c_income_share * Y * factor`, so a factor of 1 is
    full linkage to growth. Under `cpi_linked` the income holds its opening
    real value: it is unchanged in real terms, which against a growing `Y`
    means a falling share. Under `fixed_nominal` it loses even that, at the
    rate of the bloc's own inflation.
    """
    rule = g.income_linkage
    if rule == "growth_linked":
        return 1.0
    opening = s["Y0"]
    if s["Y"] <= 0 or opening <= 0:
        return 1.0
    if rule == "cpi_linked":
        # Constant in real terms, so the factor is the ratio of opening to
        # current output.
        return opening / s["Y"]
    if rule == "fixed_nominal":
        # Constant in nominal terms: deflate by the price level as well.
        return opening / (s["Y"] * max(1e-6, s["P"]))
    raise ValueError(f"unknown income linkage: {rule}")


def _budget_income_factor(b: Bloc, g: G, s: dict[str, float], t: float) -> float:
    """Income multiplier the budget indicator should use.

    The linkage rule *and* the labour income automation has removed. The
    household block already applies both; the indicator has to see the same
    income, or it reports a household that is richer than the one the model
    is actually simulating.
    """
    return _income_factor(b, g, s) * automation_state(g, t).labour_multiplier


def _risk_premium(b: Bloc, g: G, s: dict[str, float]) -> float:
    """Premium over the policy rate that lenders charge this bloc."""
    thr = g.reserve_thr if (b.reserve and g.reserve_privilege) else 0.9
    return (g.risk_coeff * max(0, s["d"] - thr)
            + g.risk_coeff * 0.5 * max(0, s["pi"] - 0.06)
            + g.risk_coeff * 0.5 * b.fx_share * max(0, s["d"] - 0.6))


def baseline_capacity_path(blocs: Sequence[Bloc], g: G) -> tuple[float, ...]:
    """World capacity over time, taken from the no-transfer run.

    Capacity must not answer to the demand a programme creates. If it did, a
    transfer would raise demand, capacity would follow, the gap would close on
    its own and the world-price pressure the programme generates would be
    understated — which is the quantity this project is trying to measure.

    So capacity is fixed once, from a run in which nobody hands anything out,
    and every configuration shares that path. Growth within the baseline run
    still comes from the blocs' own growth rates, so capacity expands with the
    world economy; it just does not expand because of a programme.

    Cached per (bloc structure, globals), since it is identical for every
    configuration that differs only in what is handed out.
    """
    key = (tuple((b.name, b.gdp, b.base_growth, b.decay, b.openness) for b in blocs),
           g.kappa_w, g.lambda_w, g.world_supply_growth)
    cached = _CAPACITY_CACHE.get(key)
    if cached is not None:
        return cached
    quiet = [replace(b, modality="none", size=0.0, start=999.0) for b in blocs]
    # Two passes. The first runs with capacity growing at the blocs' assumed
    # rates and records what real demand actually did; the second adopts that
    # demand path as the capacity path.
    #
    # The assumed rates are not what the blocs realise: fiscal flows, capital
    # movement and debt service all move output away from `base_growth`. Left
    # uncorrected, capacity outgrew real demand by 3% over thirty years and the
    # no-transfer world drifted into a 0.45%/year deflation — a trend with no
    # cause in the model, which every later result would then be read against.
    #
    # Taking the baseline's realised demand instead means the no-transfer world
    # opens and stays at a zero gap by construction. It is a normalisation of
    # the reference path, not a claim that supply meets demand: every
    # configuration that hands something out faces this same fixed capacity,
    # so the demand a programme creates still opens a gap.
    # Adopting the first pass's demand is not yet a fixed point: the new
    # capacity changes the price, the price changes real demand, and a residual
    # gap survives. Iterating closes it — each pass adopts the previous pass's
    # realised demand — and it converges in a handful of rounds because the
    # feedback from price back into quantity is weak.
    path = tuple(_run(quiet, g, capacity_path=None)["demand_quarterly"])
    for _ in range(CAPACITY_FIXED_POINT_ROUNDS):
        nxt = tuple(_run(quiet, g, capacity_path=path)["demand_quarterly"])
        if max(abs(a - b) for a, b in zip(path, nxt)) < CAPACITY_TOL:
            path = nxt
            break
        path = nxt
    _CAPACITY_CACHE[key] = path
    return path


#: Baseline capacity paths, keyed on the structure that determines them.
_CAPACITY_CACHE: dict[tuple, tuple[float, ...]] = {}

#: Iterations allowed when solving capacity and baseline demand together.
CAPACITY_FIXED_POINT_ROUNDS: int = 12

#: Convergence tolerance on the quarterly capacity path.
CAPACITY_TOL: float = 1e-9


def simulate(blocs: Sequence[Bloc], g: G) -> dict[str, Any]:
    """Run the 30-year quarterly simulation for a closed world of blocs.

    World capacity comes from the matching no-transfer run, so it is the same
    for every configuration and does not respond to the demand a programme
    creates. See `baseline_capacity_path`.
    """
    capacity = baseline_capacity_path(blocs, g)
    reference = None
    if g.subsistence_reference == "baseline":
        # The bundle is fixed at what a no-transfer world would have had, so a
        # programme cannot move the yardstick it is then judged against. The
        # reference run is given a reference of its own (None) so it takes the
        # opening level and stops.
        quiet = [replace(b, modality="none", size=0.0, start=999.0) for b in blocs]
        opening = _run(quiet, g, capacity_path=capacity)["history"][0]
        reference = [opening[f"Y_{i}"] for i in range(len(blocs))]
    return _run(blocs, g, capacity_path=capacity, reference_output=reference)


def _run(blocs: Sequence[Bloc], g: G,
         capacity_path: Sequence[float] | None,
         reference_output: Sequence[float] | None = None) -> dict[str, Any]:
    """The simulation proper.

    Args:
        capacity_path: World capacity, sampled annually, shared across
            configurations. ``None`` lets capacity grow at the assumed rate,
            which is how the baseline path itself is produced.
        reference_output: Per-bloc output the measured subsistence bundle is
            fixed at, when `G.subsistence_reference` is "baseline". Supplied by
            `simulate` from a no-transfer run, so the reference run itself does
            not try to compute one and recurse.
    """
    N = len(blocs)
    # `P` is the bloc's own price level, accumulated from its inflation, and
    # `pi_lag` is last period's inflation. Indexation reads both: a transfer is
    # adjusted to the inflation that has already happened, never to the
    # inflation it is about to cause.
    S = [dict(Y=b.gdp, d=b.debt, r=b.rate, pi=0.02, e=1.0, de=0.0, d_e=1.0,
              K=b.capital, nfa=0.0, P=1.0, pi_lag=0.02, pH=1.0,
              reserves=g.initial_reserves * b.gdp * b.reserve_standing,
              scarcity=0.0, unsettled=0.0, contagion=0.0,
              growth_rate=b.base_growth, Y0=b.gdp, potential=b.gdp,
              trend=b.gdp,
              alive=True, t_insolv=None) for b in blocs]
    transfers = [TransferState() for _ in blocs]

    shares = trade_shares(blocs)
    weights = export_capacity(blocs)
    world = WorldState()
    hist: list[dict[str, float]] = []
    world_hist: list[dict[str, float]] = []
    capacity_quarterly: list[float] = []
    demand_quarterly: list[float] = []
    quantities: list[dict[str, float]] = []
    budget: list[dict[str, float]] = []
    util: list[dict[str, float]] = []
    # The subsistence bundle is fixed once, as quantities, and re-priced each
    # period thereafter. Fixing it at the opening state means the indicator
    # asks "can they still afford what they needed on day one" rather than
    # redefining the need every time income moves. See `model.subsistence`
    # for why this differs from the floor the household block itself uses.
    # Under "baseline" the reference is the no-transfer path's output at the
    # same date, so the programme cannot move the yardstick it is judged by;
    # under "start" it is the opening level. Both are reported.
    if g.subsistence_reference == "baseline" and reference_output is not None:
        bundles = [reference_bundles(b, g, reference_output[i])
                   for i, b in enumerate(blocs)]
    else:
        bundles = [reference_bundles(b, g, b.gdp) for b in blocs]
    settlement_log: list[dict[str, Any]] = []
    contagion_log: list[dict[str, Any]] = []
    # Opening demand for the reserve currency, against which later demand is
    # judged. Size-weighted, so a larger world wants more of it.
    base_reserve_demand = sum(b.gdp * b.openness for b in blocs if not b.reserve)
    flags: list[list[bool]] = [[False] * N]
    sev = [0.0] * N
    onsets = [0] * N
    dur = [0] * N
    breadth: list[int] = []
    leak_log = [dict(n=0, imp=0.0, fa=0.0, fiscal=0.0, rentcap=0.0, dU=0.0, Y=0.0)
                for _ in blocs]

    for step in range(STEPS + 1):
        t = step * DT

        # --- pass 1: demands at current prices -------------------------------
        parts = [_bloc_demand(b, g, s, t, world, tr)
                 for b, s, tr in zip(blocs, S, transfers)]
        # An insolvent bloc keeps importing. Insolvency here means the debt
        # ratio passed `default_thr`, which stops the bloc's own state from
        # being updated further; it does not mean the country stopped buying
        # food. Zeroing its imports would quietly impose a total settlement
        # embargo — the mechanism phase D is supposed to introduce
        # deliberately and calibrate — and would make world demand collapse
        # the moment any bloc crossed the threshold.
        imports = [p["imports"] for p in parts]

        # Two measures of the same imports, and they are not interchangeable.
        #
        #   imports    spending, in the numeraire the balance of payments is
        #              written in. Trade shares, current accounts and the
        #              capital account all use this.
        #   real       the quantity bought. World capacity is a real quantity,
        #              so the world market clears on this one.
        #
        # Comparing spending against capacity would let a price rise register
        # as extra demand, which would raise the price again — a feedback loop
        # made of nothing but the units. Deflating spending by the world price
        # would be wrong in the other direction, since imported goods carry a
        # domestic price component; the quantity is carried through from the
        # household block instead of being reconstructed here.
        real_imports = [p["imports_q"] for p in parts]

        # World capacity is normalised on the first period's demand, so the
        # baseline opens with a zero output gap and any later gap is caused by
        # what the simulation does. When a shared path is supplied, capacity
        # comes from it instead: the same capacity for every configuration, so
        # a programme's demand cannot call forth the supply to meet it.
        if step == 0:
            opening = (capacity_path[0] if capacity_path
                       else initial_capacity(real_imports))
            world = WorldState(capacity=opening)
        elif capacity_path and step < len(capacity_path):
            world = WorldState(price=world.price, capacity=capacity_path[step],
                               anchor=world.anchor)
        capacity_quarterly.append(world.capacity)
        demand_quarterly.append(sum(real_imports))

        # --- pass 2: the world clears ---------------------------------------
        # Market shares move with competitiveness when the elasticity is on,
        # so a bloc that has depreciated earns more of the world's spending.
        if g.trade_elasticity > 0:
            shares = trade_shares(blocs, [st["e"] for st in S], g.trade_elasticity)
        exports = exports_from_imports(imports, shares)
        ca = current_accounts(imports, exports)

        avg_r = sum(s["r"] for s in S) / N
        # World growth and the world real rate, size-weighted: the yardsticks
        # every bloc measures itself against, both for capital flows and for
        # its own policy reaction.
        size = [b.gdp * b.openness for b in blocs]
        size_total = sum(size) or 1.0
        avg_growth = sum(w * st.get("growth_rate", 0.0)
                         for w, st in zip(size, S)) / size_total
        avg_real_rate = sum(w * (st["r"] - st["pi"])
                            for w, st in zip(size, S)) / size_total
        desired = []
        premia = []
        for i, (b, s) in enumerate(zip(blocs, S)):
            rp = _risk_premium(b, g, s)
            # Lenders who have just been burned charge more, and they charge
            # it to everyone that resembles the borrower who burned them.
            rp += g.contagion_risk_premium * s.get("contagion", 0.0)
            premia.append(rp)
            # Capital chases both the return offered and the growth behind
            # it. A bloc that falls behind on growth loses capital even if it
            # holds its rate, which is how "falling behind" turns into
            # "reserves draining" and, under settlement, into imports it can
            # no longer pay for.
            growth_gap = s.get("growth_rate", 0.0) - avg_growth
            want = b.openness * g.cap_sens * (s["r"] - rp - avg_r)
            want += b.openness * g.growth_differential_sensitivity * growth_gap
            # Households placing transfer income abroad is an outflow like any
            # other. In v7 this quantity was logged as "leakage" and then
            # dropped; leaving it out here would mean the world was closed for
            # goods but still open for capital, and the receiving blocs would
            # never see the claims being bought from them.
            want -= _foreign_asset_purchase(b, g, s, parts[i]["d_inc_u"], rp)
            if b.reserve and g.reserve_privilege:
                instab = sum(max(0, S[j]["d"] - 0.7) + max(0, S[j]["pi"] - 0.04)
                             for j in range(N) if j != i)
                want += g.reserve_flight * instab
            desired.append(want)
        # Closing this sum is what routes one bloc's asset purchase to the
        # blocs whose assets are bought: the excess is redistributed by size,
        # so an outflow here is an inflow elsewhere.
        net_k = balanced_capital_flows(desired, weights)

        # Sensitivity: charge part of a capital inflow straight to world goods
        # demand. Off in the main series, where an inflow buys claims and
        # reaches goods only later through the capital stock. The world is
        # re-cleared afterwards so the identities still hold on the quantities
        # actually used.
        if g.capital_inflow_to_demand > 0:
            extra = [g.capital_inflow_to_demand * max(0.0, k) for k in net_k]
            imports = [m + x for m, x in zip(imports, extra)]
            real_imports = [m / world.price for m in imports]
            exports = exports_from_imports(imports, shares)
            ca = current_accounts(imports, exports)

        # --- settlement: can each bloc pay for what it wants to buy? ---------
        if g.settlement:
            reserve_demand = reserve_currency_demand(
                world.price, base_reserve_demand, g.reserve_demand_elasticity)
            reserve_bound = reserve_is_constrained(
                reserve_demand, base_reserve_demand, g.reserve_demand_threshold)
            outcomes = []
            for i, (b, st) in enumerate(zip(blocs, S)):
                # A contagion-hit bloc finds the window narrowed regardless
                # of its own position: what closed it was somebody else's
                # default.
                standing = b.credit_standing * max(0.0, 1.0 - st.get("contagion", 0.0))
                borrowing = borrowing_capacity(
                    st["Y"], g.borrow_limit * DT * standing,
                    premia[i], g.borrow_sensitivity)
                # The reserve issuer settles in its own money, so it is exempt
                # — but only while other blocs still want to hold that money.
                exempt = b.reserve and not reserve_bound
                outcomes.append(settle(imports[i], exports[i], st["reserves"],
                                       borrowing, exempt=exempt))
            # Rationing one bloc's imports removes another bloc's exports, so
            # the world has to clear again on the quantities actually settled.
            imports = [o.realised for o in outcomes]
            scale = [o.realised / o.desired if o.desired > 0 else 1.0
                     for o in outcomes]
            # Real demand is cut in the same proportion: the constraint bites
            # on quantities, and it is the quantity of food that reaches a
            # household, not the spending on it.
            real_imports = [q * f for q, f in zip(real_imports, scale)]
            exports = exports_from_imports(imports, shares)
            ca = current_accounts(imports, exports)
            for i, st in enumerate(S):
                st["reserves"] = update_reserves(
                    st["reserves"], exports[i], imports[i], net_k[i] * DT)
                # Carried into next period's prices, not this one's: a bloc
                # discovers it cannot pay after it has tried to.
                st["scarcity"] = outcomes[i].rationing_share
                st["unsettled"] = outcomes[i].rationed
            settlement_log.append({
                "year": t,
                "reserve_demand": round(reserve_demand, 6),
                "reserve_constrained": reserve_bound,
                **rationing_summary(outcomes),
                **{f"share_{i}": round(outcomes[i].rationing_share, 6)
                   for i in range(N)},
                **{f"reserves_{i}": round(S[i]["reserves"], 6) for i in range(N)},
            })

        d_nfa = net_foreign_asset_change(ca, net_k)

        if step % 4 == 0:
            hist.append({"year": t, **{f"{k}_{i}": round(S[i][k], 5)
                                       for i in range(N) for k in ("Y", "d", "e", "pi", "r")}})
            # What the constrained household actually ends up with, in real
            # units. This is the quantity a recipient can eat and live in, and
            # it is the only measure that compares cash with an in-kind
            # programme on the same terms: a cash transfer's purchasing power
            # and a housing programme's tenancy are not otherwise commensurate.
            # Total real demand, domestic and imported. World demand alone
            # counts only the traded part, so it cannot answer what the
            # propensity gap does to demand overall — which is what
            # utilisation is supposed to be about.
            # Constrained households' quantities, plus what the unconstrained
            # block spends out of the income that reached it. Both halves are
            # needed: automation moves income from a block that spends all of
            # it to one that spends `mpc_u` of it, and the size of that shift
            # is the whole question.
            total_demand = sum(
                parts[i]["hb1"]["x"]["F"] + parts[i]["hb1"]["x"]["G"]
                + parts[i]["hb1"]["x"]["H"]
                + g.mpc_u * parts[i]["d_inc_u"] / max(1e-9,
                                                      parts[i]["hb1"]["prices"]["G"])
                for i in range(N))
            util.append({"year": t, "demand": round(total_demand, 6),
                         "automation": round(automation_state(g, t).progress, 6),
                         **{f"overhang_{i}": round(
                             capacity_overhang(S[i]["Y"], S[i]["potential"]), 6)
                            for i in range(N)},
                         "world": round(capacity_overhang(
                             sum(S[i]["Y"] for i in range(N)),
                             sum(S[i]["potential"] for i in range(N))), 6)})
            quantities.append({"year": t, **{
                f"{good}_{i}": round(parts[i]["hb1"]["x"][good], 6)
                for i in range(N) for good in ("F", "G", "H")}})
            # Whether the money reaches the subsistence bundle, which is a
            # different question from what was secured: the demand system
            # never allocates below the floor, so the quantities above cannot
            # show a shortfall even when the budget plainly has one. Purely an
            # observation — nothing here feeds back into the dynamics.
            budget.append({"year": t, **{
                f"{k}_{i}": round(v, 6)
                for i in range(N)
                for k, v in summarise(positions(
                    blocs[i], g, S[i]["Y"], parts[i]["hb1"]["prices"],
                    transfer=parts[i]["hb1"]["fiscal"],
                    reference=bundles[i],
                    income_factor=_budget_income_factor(blocs[i], g, S[i], t))).items()},
                **{f"need_{i}": round(required_transfer(
                    blocs[i], g, S[i]["Y"], parts[i]["hb1"]["prices"],
                    reference=bundles[i],
                    income_factor=_budget_income_factor(blocs[i], g, S[i], t),
                    erosion=transfers[i].real_value_ratio(S[i]["P"]),
                    ) / S[i]["Y"], 6)
                   for i in range(N)}})
            # World demand is recorded twice: over every bloc, and over the
            # blocs still solvent. A world whose inflation subsides only
            # because its members defaulted has not recovered, and the two
            # series are what lets the collapse indicators tell the two apart.
            live = [m for m, st in zip(real_imports, S) if st["alive"]]
            world_hist.append({"year": t, "price": round(world.price, 6),
                               "capacity": round(world.capacity, 6),
                               "demand": round(sum(real_imports), 6),
                               "demand_solvent": round(sum(live), 6),
                               "insolvent": sum(1 for st in S if not st["alive"])})
        if step == STEPS:
            break

        # Net foreign assets accrue for every bloc, insolvent ones included.
        # An insolvent bloc stops importing but does not stop existing: its
        # counterpart position is still part of the world's books, and
        # skipping it here would break the adding-up identity that the whole
        # phase rests on.
        for i in range(N):
            S[i]["nfa"] += d_nfa[i] * DT

        auto_now = automation_state(g, t)
        # How much capacity automation adds this quarter, as a multiplier on
        # the previous quarter's potential.
        auto_prev = automation_state(g, max(0.0, t - DT))
        defaults_this_step: list[int] = []
        new_flags: list[bool] = []
        for i, (b, s, p) in enumerate(zip(blocs, S, parts)):
            if not s["alive"]:
                new_flags.append(True)
                continue

            Y, r = s["Y"], s["r"]
            rp = premia[i]
            d_fa = _foreign_asset_purchase(b, g, s, p["d_inc_u"], rp)
            fiscal = p["fiscal"]

            if p["on"]:
                L = leak_log[i]
                L["n"] += 1
                L["imp"] += p["d_imp"]
                L["fa"] += d_fa
                L["fiscal"] += fiscal
                L["Y"] += Y
                L["rentcap"] += p["d_rent"]
                L["dU"] += p["hb1"]["U"] - p["hb0"]["U"]

            eff_g = max(0.005, b.base_growth - b.decay * t)
            debt_acc = g.deficit_share * fiscal / Y + r * s["d"] - eff_g * s["d"]
            debt_acc -= s["pi"] * s["d"] * g.compress * (1 - b.fx_share)

            # The exchange rate now answers to the bloc's own external
            # position, which the world has just made consistent, rather than
            # to a leakage term that balanced against nothing.
            # The exchange rate answers to the demand for foreign exchange,
            # not to the payments that were actually made. A bloc rationed
            # because its reserves ran out shows a *better* current account
            # afterwards — it bought less — and reading that as strength would
            # have its currency appreciate precisely when it can no longer pay
            # its bills. What presses on the rate is the shortfall, so the
            # unsettled demand is added back.
            pressure = ca[i] - S[i].get("unsettled", 0.0)
            flow = (net_k[i] + g.bop_coeff * pressure / Y) * DT
            # Proportional, not additive. An additive rule (v7's `de = -2*flow`)
            # moves the rate by the same number of points whether it stands at
            # 2.0 or at 0.2, so a bloc under sustained surplus walks the rate
            # into the floor and stops responding to anything. In a closed
            # world that is not an edge case: somebody must run the surplus
            # that matches everyone else's deficit. Expressing the move as a
            # rate of change keeps the response proportionate at every level.
            de = -2.0 * flow * s["e"]
            # Expenditure switching pulls the rate back towards parity: a
            # bloc that has appreciated buys more abroad and sells less, which
            # is what stops the appreciation.
            de += g.export_switch * (1.0 - s["e"]) * DT
            if b.reserve and g.reserve_privilege:
                de *= g.reserve_damp
                de = max(de, g.reserve_flight_floor - s["e"])

            # Inflation is the sum of three pushes on the *change* in the
            # rate, not a level the rate is set to.
            #
            #   world   what imports cost in this bloc's money, which moves
            #           with the world price and with the exchange rate. A
            #           bloc that ran no programme still imports the world's
            #           inflation through this term — the channel the phase
            #           exists to create.
            #   pull    domestic demand pressure, as in v7.
            #   drag    the central bank leaning against inflation it already
            #           has.
            #
            # The world term is a push on the change, weighted by import
            # content. Writing it as a level (`pi = theta*world + ...`) makes
            # an appreciating bloc's inflation track the exchange rate down to
            # the floor, which is not what an appreciation does to a price
            # index in which most items are domestic.
            pw_next = step_world(world, sum(real_imports), blocs, g, t,
                                 demand_weights=real_imports)
            p_imp_now = import_price(world, s["e"])
            p_imp_next = import_price(pw_next, max(0.05, s["e"] + de))
            world_push = (p_imp_next / p_imp_now - 1.0) / DT if p_imp_now > 0 else 0.0
            theta = min(0.9, b.openness * (b.m_F + b.m_G) / 2.0)
            pull = g.fiscal_mult * 0.15 * max(0.0, p["d_dom"] / Y)
            dpi = (0.3 * theta * world_push + pull
                   - 0.1 * max(0, s["pi"] - 0.03)) * DT

            # Growth: the trend, the fiscal stimulus, and up to four
            # reduced-form adjustments. The adjustments are switchable so that
            # a result which turns on them can be told apart from one that
            # does not — see `G.growth_penalties`.
            penalties = g.growth_penalties
            growth = eff_g + g.fiscal_mult * 0.08 * p["d_dom"] / Y
            # Output is held back when demand falls short of what could be
            # produced. Without this, production grows on trend however little
            # anybody can buy, and automation raises capacity against a
            # denominator that never responds.
            if g.demand_constrains_output > 0 and s["potential"] > 0:
                slack = max(0.0, 1.0 - Y / s["potential"])
                growth -= g.demand_constrains_output * slack
            if "capital" in penalties:
                growth += 0.15 * flow
            if "debt" in penalties:
                growth -= 0.02 * max(0, s["d"] - 1.2)
            if "inflation" in penalties:
                growth -= 0.03 * max(0, s["pi"] - 0.05)
            if "fx" in penalties:
                growth -= 0.01 * abs(de)
            growth *= DT
            # Authorities lean against deflation and stalled growth. The
            # rule is identical in every bloc: what differs is what easing
            # costs you when your debt is in someone else's money and your
            # imports have to be settled in it.
            impulse = respond(growth / DT, avg_growth,
                              s["r"] - s["pi"], avg_real_rate,
                              g.policy_response, g.policy_fiscal_share)
            if impulse.fiscal_boost > 0:
                # General government spending, distinct from the transfer. It
                # is domestic demand, so it lands in the same place the
                # transfer's domestic component does.
                growth += g.fiscal_mult * 0.08 * impulse.fiscal_boost * DT
                debt_acc += impulse.fiscal_boost
            r_tgt = max(0, 0.02 + 0.5 * (s["pi"] - 0.02) + 0.3 * rp
                        - impulse.rate_cut)

            s["d"] = max(0, s["d"] + debt_acc * DT)
            s["d"] = max(0, s["d"] + de * s["d"] * b.fx_share)
            s["K"] = max(0.05, s["K"] + flow)
            s["e"] = max(0.05, s["e"] + de)
            s["de"] = de
            s["d_e"] = s["e"]
            s["pH"] = p["hb1"]["pH"]
            s["pi_lag"] = s["pi"]
            s["pi"] = max(-0.02, s["pi"] + dpi)
            s["P"] = max(1e-6, s["P"] * (1.0 + s["pi"] * DT))
            # Potential output rises with automation; actual output is what
            # demand supports. The gap is idle capacity — goods that could be
            # made and that nobody has the income to buy.
            # Potential carries the bloc's own realised trend plus what
            # automation adds on top of it. Trend growth belongs to both
            # numerator and denominator — it is not what automation did — so
            # it is the *automation* multiplier that opens the gap, and
            # utilisation measures that gap alone.
            #
            # Deriving potential from `Y` directly made utilisation identically
            # 1/capacity_multiplier: arithmetic, not a finding.
            s["trend"] = max(0.01, s["trend"] * (1 + eff_g * DT))
            s["potential"] = s["trend"] * auto_now.capacity_multiplier
            s["growth_rate"] = growth / DT
            s["Y"] = max(0.01, Y * (1 + growth))
            s["r"] = max(0, min(0.30, r + 0.2 * (r_tgt - r) * DT))

            exc = [max(0, s["d"] - THR["debt"]) / THR["debt"],
                   max(0, s["pi"] - THR["infl"]) / THR["infl"],
                   max(0, s["e"] - THR["dep"]) / THR["dep"]]
            in_c = any(x > 0 for x in exc)
            if in_c:
                dur[i] += 1
                sev[i] += sum(exc) * DT
                if not flags[-1][i]:
                    onsets[i] += 1
            new_flags.append(in_c)

            if s["d"] > g.default_thr:
                s["alive"] = False
                s["t_insolv"] = t
                defaults_this_step.append(i)

        # The next period's capacity is read from the shared path when there
        # is one, so only the price actually evolves here.
        nxt = (capacity_path[step + 1]
               if capacity_path and step + 1 < len(capacity_path) else None)
        # A default is not a private matter between one borrower and its
        # lenders. Everything that looks like the failed bloc is marked down
        # at the same time, which is how an adjustment that had been gradual
        # becomes a stop.
        if g.contagion_strength > 0:
            for st in S:
                st["contagion"] = decay_contagion(
                    st["contagion"], g.contagion_years, DT)
            for failed in defaults_this_step:
                withdrawn = 0.0
                for i, b in enumerate(blocs):
                    if i == failed:
                        continue
                    # Exemption by assertion is a sensitivity, not the main
                    # path: handing the reserve issuer a pass would put the
                    # conclusion into the premise. On the main path it is
                    # marked down like anything else at the same distance.
                    if b.reserve and g.contagion_exempts_reserve:
                        continue
                    hit = g.contagion_strength * contagion_weight(
                        axis_distance(b, blocs[failed]), g.contagion_reach)
                    S[i]["contagion"] = min(1.0, S[i]["contagion"] + hit)
                    withdrawn += hit * S[i]["Y"]
                # Money pulled out of a region is not held as cash. It buys the
                # safest claim on offer, which is the reserve issuer's — so the
                # issuer receives an inflow *because* others are being
                # withdrawn from. That is the channel through which its
                # position is spared, and it is a consequence rather than a
                # rule.
                if g.flight_to_reserve > 0 and withdrawn > 0:
                    for i, b in enumerate(blocs):
                        if b.reserve:
                            relief = g.flight_to_reserve * withdrawn / max(1e-9, S[i]["Y"])
                            S[i]["contagion"] = max(0.0, S[i]["contagion"] - relief)
                            S[i]["reserves"] += g.flight_to_reserve * withdrawn
            if defaults_this_step:
                contagion_log.append({
                    "year": t,
                    "defaulted": [blocs[i].name for i in defaults_this_step],
                    **{f"hit_{i}": round(S[i]["contagion"], 6) for i in range(N)},
                })

        world = step_world(world, sum(real_imports), blocs, g, t,
                           demand_weights=real_imports, capacity=nxt)
        flags.append(new_flags)
        breadth.append(sum(new_flags))

    crisis: dict[str, Any] = {
        blocs[i].name: dict(onsets=onsets[i], duration_q=dur[i],
                            severity=round(sev[i], 3),
                            insolvent_year=S[i]["t_insolv"])
        for i in range(N)
    }
    crisis["_system"] = dict(max_breadth=max(breadth) if breadth else 0,
                             bloc_quarters=sum(breadth), total_onsets=sum(onsets))

    leak: dict[str, Any] = {}
    for i, L in enumerate(leak_log):
        if L["n"] and L["fiscal"] > 0:
            total = L["imp"] + L["fa"]
            leak[blocs[i].name] = dict(
                # Imports and foreign-asset purchases are reported apart.
                # Only the first buys goods on the world market; conflating
                # them is what made Phase 1 and Phase 3 incomparable.
                import_abs=round(100 * L["imp"] / L["Y"], 6),
                foreign_assets_abs=round(100 * L["fa"] / L["Y"], 6),
                import_per_fiscal=round(L["imp"] / L["fiscal"], 6),
                foreign_assets_per_fiscal=round(L["fa"] / L["fiscal"], 6),
                # Kept for continuity with the open-world phases; not the
                # right measure of pressure on the world goods market.
                leak_abs=round(100 * total / L["Y"], 6),
                leak_per_fiscal=round(total / L["fiscal"], 6),
                leak_per_welfare=(round(total / L["dU"], 6) if L["dU"] > 1e-12 else None),
                rent_capture_per_fiscal=round(L["rentcap"] / L["fiscal"], 3),
                fiscal_pct_gdp=round(100 * L["fiscal"] / L["Y"], 3),
                welfare_gain=round(L["dU"], 6),
            )

    # P3: what each transfer ends up worth, relative to its value on the day
    # it was introduced. 1.0 means the programme still buys what it bought;
    # below that, inflation has taken the difference. Under `gdp_linked` this
    # is 1.0 by construction, which is the point phase C exists to change.
    # Two numeraires, because they answer different questions. The CPI
    # reading says whether the transfer still buys the same basket; the
    # housing reading says whether it still buys the same shelter, and a cash
    # transfer bids up the rent it is then measured against.
    real_value = {blocs[i].name: round(transfers[i].real_value_ratio(S[i]["P"]), 6)
                  for i in range(N)}
    housing_value = {blocs[i].name: round(transfers[i].housing_value_ratio(S[i]["pH"]), 6)
                     for i in range(N)}

    final = {blocs[i].name: dict(debt=round(S[i]["d"], 3), e=round(S[i]["e"], 3),
                                 pi=round(S[i]["pi"], 3), Y=round(S[i]["Y"], 3),
                                 nfa=round(S[i]["nfa"], 5),
                                 real_value=real_value[blocs[i].name],
                                 housing_value=housing_value[blocs[i].name])
             for i in range(N)}
    return dict(crisis=crisis, leakage=leak, final=final, history=hist,
                world=world_hist, real_value=real_value,
                housing_value=housing_value, quantities=quantities,
                budget=budget, capacity=util,
                settlement=settlement_log, contagion=contagion_log,
                capacity_quarterly=capacity_quarterly,
                demand_quarterly=demand_quarterly,
                world_final=dict(price=round(world.price, 6),
                                 capacity=round(world.capacity, 6)))


def build(g: G, modality: str, u: float = 0.05,
          starts: Sequence[float] = (1, 5, 10, 15), basis: str = "welfare",
          which: Sequence[str] = ("A", "B", "C", "D")) -> list[Bloc]:
    """Assemble the bloc list for one scenario.

    Args:
        which: Blocs that actually run the programme. The others are present
            in the world — they trade and absorb world inflation — but hand
            out nothing. This is how "one bloc pays" is distinguished from
            "everybody pays".
    """
    arch = archetypes()
    out: list[Bloc] = []
    for key, st in zip(("A", "B", "C", "D"), starts):
        b = arch[key]
        if key in which and modality != "none":
            cal = calibrate(b, g, u)
            if basis == "welfare":
                size = cal[modality]
            else:
                size = u if modality == "ubi" else equal_cost_size(
                    b, g, modality, cal["_fiscal_ubi"])
            b = replace(b, modality=modality, size=size, start=st)
        out.append(b)
    return out
