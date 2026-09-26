"""Macro dynamics: debt, exchange rate, inflation, growth, and crisis metrics.

Extracted from ``sim_v6.py``. The update equations are unchanged; the only
additions are the ``reserve_flight_mode`` branch and the explicit
``universal_split``, both of which default to v6 behaviour.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from .household import calibrate, equal_cost_size, household_block
from .shocks import ShockPath
from .params import DT, STEPS, THR, Bloc, G, archetypes


def simulate(blocs: Sequence[Bloc], g: G,
             shocks: "ShockPath | None" = None) -> dict[str, Any]:
    """Run the 30-year quarterly simulation for a set of blocs.

    Args:
        shocks: Optional realised shock path. ``None`` leaves the model
            deterministic and reproduces v6 exactly.
    """
    S = [dict(Y=b.gdp, d=b.debt, r=b.rate, pi=0.02, e=1.0, K=b.capital, de=0.0,
              alive=True, t_insolv=None) for b in blocs]
    N = len(blocs)
    hist: list[dict[str, float]] = []
    flags: list[list[bool]] = [[False] * N]
    sev = [0.0] * N
    onsets = [0] * N
    dur = [0] * N
    sev_na = [0.0] * N   # severity excluding the appreciation trigger
    dur_na = [0] * N
    breadth: list[int] = []
    leak_log = [dict(n=0, imp=0.0, fa=0.0, fiscal=0.0, rentcap=0.0, dU=0.0) for _ in blocs]

    for step in range(STEPS + 1):
        t = step * DT
        if step % 4 == 0:
            hist.append({"year": t, **{f"{k}_{i}": round(S[i][k], 5)
                                       for i in range(N) for k in ("Y", "d", "e", "pi", "r")}})
        if step == STEPS:
            break

        avg_r = sum(s["r"] for s in S) / N
        if shocks is not None and step < len(shocks.rate_add):
            # Non-reserve blocs face the world rate, so a reserve-bloc shock
            # raises the outside option they are compared against.
            avg_r += shocks.rate_add[step]
        instab = [max(0, s["d"] - 0.7) + max(0, s["pi"] - 0.04) + max(0, s["e"] - 1.5) for s in S]
        new_flags: list[bool] = []

        for i, (b, s) in enumerate(zip(blocs, S)):
            if not s["alive"]:
                new_flags.append(True)
                continue

            on = t >= b.start and b.modality != "none"
            Y, e, r = s["Y"], s["e"], s["r"]
            food = (shocks.food_factor[step]
                    if shocks is not None and step < len(shocks.food_factor) else 1.0)
            hb1 = household_block(b, g, Y, e, r, on, b.size, food_factor=food)
            hb0 = household_block(b, g, Y, e, r, False, b.size, food_factor=food)

            d_imp = (hb1["imp_c"] - hb0["imp_c"]) + hb1["imp_gov"]
            d_rent = hb1["rent"] - hb0["rent"]
            d_inc_u = hb1["cash_u"] + d_rent
            d_imp += g.mpc_u * b.m_G * d_inc_u

            use_res = b.reserve
            thr = g.reserve_thr if use_res else 0.9
            rp = (g.risk_coeff * max(0, s["d"] - thr)
                  + g.risk_coeff * 0.5 * max(0, s["pi"] - 0.06)
                  + g.risk_coeff * 0.5 * b.fx_share * max(0, s["d"] - 0.6))
            exp_dep = max(0.0, s["de"] / DT)
            omega = min(0.9, max(0.0, b.omega0 + g.omega_sens * (rp + exp_dep)))
            d_fa = (1 - g.mpc_u) * d_inc_u * omega * b.openness
            d_dom = ((hb1["dom_c"] - hb0["dom_c"]) + hb1["dom_gov"]
                     + g.mpc_u * (1 - b.m_G) * d_inc_u)
            fiscal = hb1["fiscal"]

            if on:
                L = leak_log[i]
                L["n"] += 1
                L["imp"] += d_imp
                L["fa"] += d_fa
                L["fiscal"] += fiscal
                L["Y"] = L.get("Y", 0.0) + Y
                L["rentcap"] += d_rent
                L["dU"] += hb1["U"] - hb0["U"]
                L["leak"] = L.get("leak", 0.0) + (d_imp + d_fa)

            eff_g = max(0.005, b.base_growth - b.decay * t)
            debt_acc = g.deficit_share * fiscal / Y + r * s["d"] - eff_g * s["d"]
            debt_acc -= s["pi"] * s["d"] * g.compress * (1 - b.fx_share)

            flow = b.openness * g.cap_sens * (r - rp - avg_r) * DT
            if use_res and g.reserve_flight_mode != "off":
                flow += g.reserve_flight * sum(instab[j] for j in range(N) if j != i) * DT
            bop = -(d_imp + d_fa) / Y + g.export_switch * (e - 1)
            flow += g.bop_coeff * bop * DT
            if shocks is not None and step < len(shocks.sudden_stop) and not use_res:
                # A stop hits non-reserve blocs in proportion to how open they
                # are; the reserve issuer receives the counterpart inflow.
                flow -= shocks.sudden_stop[step] * b.openness * DT

            de = -2.0 * flow
            if use_res:
                de *= g.reserve_damp
            # 'capped' keeps the safe-haven inflow but prevents appreciation from
            # manufacturing a crisis on its own by flooring the exchange rate.
            if use_res and g.reserve_flight_mode == "capped":
                de = max(de, g.reserve_flight_floor - s["e"])

            pull = g.fiscal_mult * 0.15 * max(0.0, d_dom / Y)
            dpi = (pull + 0.3 * max(0, de) - 0.1 * max(0, s["pi"] - 0.03)) * DT
            growth = (eff_g + g.fiscal_mult * 0.08 * d_dom / Y + 0.15 * flow
                      - 0.02 * max(0, s["d"] - 1.2) - 0.03 * max(0, s["pi"] - 0.05)
                      - 0.01 * abs(de)) * DT
            r_tgt = max(0, 0.02 + 0.5 * (s["pi"] - 0.02) + 0.3 * rp)
            if shocks is not None and step < len(shocks.rate_add):
                # The reserve bloc sets the world rate; everyone else imports
                # it through the capital-flow term below.
                r_tgt += shocks.rate_add[step] if use_res else 0.0

            s["d"] = max(0, s["d"] + debt_acc * DT)
            s["d"] = max(0, s["d"] + de * s["d"] * b.fx_share)
            s["K"] = max(0.05, s["K"] + flow)
            s["e"] = max(0.05, s["e"] + de)
            s["de"] = de
            s["pi"] = max(-0.02, s["pi"] + dpi)
            s["Y"] = max(0.01, Y * (1 + growth))
            s["r"] = max(0, min(0.30, r + 0.2 * (r_tgt - r) * DT))

            exc = [max(0, s["d"] - THR["debt"]) / THR["debt"],
                   max(0, s["pi"] - THR["infl"]) / THR["infl"],
                   max(0, s["e"] - THR["dep"]) / THR["dep"],
                   max(0, THR["app"] - s["e"]) / THR["app"]]
            in_c = any(x > 0 for x in exc)
            if in_c:
                dur[i] += 1
                sev[i] += sum(exc) * DT
                if not flags[-1][i]:
                    onsets[i] += 1
            new_flags.append(in_c)
            # Variant that drops the appreciation trigger (exc[3]). In the
            # reserve bloc that trigger fires on safe-haven inflow rather than
            # on any policy effect, so both versions are reported.
            exc_na = exc[:3]
            if any(x > 0 for x in exc_na):
                dur_na[i] += 1
                sev_na[i] += sum(exc_na) * DT

            if s["d"] > g.default_thr:
                s["alive"] = False
                s["t_insolv"] = t

        flags.append(new_flags)
        breadth.append(sum(new_flags))

    crisis: dict[str, Any] = {
        blocs[i].name: dict(onsets=onsets[i], duration_q=dur[i], severity=round(sev[i], 3),
                            severity_no_appreciation=round(sev_na[i], 3),
                            duration_q_no_appreciation=dur_na[i],
                            insolvent_year=S[i]["t_insolv"])
        for i in range(N)
    }
    crisis["_system"] = dict(max_breadth=max(breadth), bloc_quarters=sum(breadth),
                             total_onsets=sum(onsets))

    leak: dict[str, Any] = {}
    for i, L in enumerate(leak_log):
        if L["n"] and L["fiscal"] > 0:
            total_leak = L["imp"] + L["fa"]
            # Three normalisations, reported together because they can rank
            # modalities differently:
            #   leak_abs         - share of GDP (primary on the welfare basis)
            #   leak_per_fiscal  - per unit of fiscal cost (primary on equal cost)
            #   leak_per_welfare - per unit of constrained-household utility gain
            leak[blocs[i].name] = dict(
                import_per_fiscal=round(L["imp"] / L["fiscal"], 3),
                foreign_assets_per_fiscal=round(L["fa"] / L["fiscal"], 3),
                external_leak_per_fiscal=round(total_leak / L["fiscal"], 3),
                rent_capture_per_fiscal=round(L["rentcap"] / L["fiscal"], 3),
                fiscal_pct_gdp=round(100 * L["fiscal"] / L["Y"], 3),
                external_leak_pct_gdp=round(100 * total_leak / L["Y"], 3),
                leak_abs=round(100 * total_leak / L["Y"], 6),
                leak_per_fiscal=round(total_leak / L["fiscal"], 6),
                leak_per_welfare=(round(total_leak / L["dU"], 6) if L["dU"] > 1e-12 else None),
                welfare_gain=round(L["dU"], 6),
            )

    final = {blocs[i].name: dict(debt=round(S[i]["d"], 3), e=round(S[i]["e"], 3),
                                 pi=round(S[i]["pi"], 3), Y=round(S[i]["Y"], 3))
             for i in range(N)}
    return dict(crisis=crisis, leakage=leak, final=final, history=hist)


def build(
    g: G,
    modality: str,
    u: float = 0.05,
    starts: Sequence[float] = (1, 5, 10, 15),
    basis: str = "welfare",
    which: Sequence[str] = ("A", "B", "C", "D"),
) -> list[Bloc]:
    """Assemble the bloc list for one scenario.

    Args:
        basis: ``welfare`` matches constrained-household utility across
            modalities; ``cost`` matches fiscal cost to universal UBI.
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
                size = u if modality == "ubi" else equal_cost_size(b, g, modality, cal["_fiscal_ubi"])
            b = replace(b, modality=modality, size=size, start=st)
        out.append(b)
    return out


def run_scenarios(g: G, u: float = 0.05, simultaneous: bool = False,
                  keep_history: bool = False) -> dict[str, Any]:
    """Run every (basis, modality) scenario, mirroring ``sim_v6.main``."""
    starts = (1, 1, 1, 1) if simultaneous else (1, 5, 10, 15)
    res: dict[str, Any] = {
        "calibration": {k: calibrate(b, g, u) for k, b in archetypes().items()},
        "runs": {},
    }
    for basis in ("welfare", "cost"):
        for mod in ("none", "ubi", "cash_t", "voucher", "clt"):
            if basis == "cost" and mod in ("none", "ubi"):
                continue
            r = simulate(build(g, mod, u, starts, basis), g)
            if not keep_history:
                r.pop("history")
            res["runs"][f"{basis}:{mod}"] = r
    return res
