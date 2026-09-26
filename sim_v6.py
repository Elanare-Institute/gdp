#!/usr/bin/env python3
"""
sim_v6.py — Transfer modality, endogenous external leakage, and FX-debt feedback
==============================================================================
v5 からの主要変更（エディター指摘への対応）
  1. CLT専用の減衰係数（clt_demand_dampening / clt_exch_dampening / cost×0.3）を全廃。
  2. 各ブロックに家計ブロックを導入:
       - 流動性制約世帯（hand-to-mouth, Stone–Geary/LES: 食料F・その他財G・住宅H）
       - 非制約世帯（家主を兼ねる。MPC・貯蓄・内外ポートフォリオ選択）
     輸入・海外資産購入・家賃帰着はここから内生的に決まる。
  3. 給付様式は「何を渡すか」だけで定義: cash / voucher(食料) / in-kind housing(CLT)。
     現物のインフラマージナル性（LESの端点解）を明示的に解く。
  4. 住宅市場を毎期均衡させ、家賃帰着（rent capture）→家主所得→海外資産の経路を内生化。
  5. CLTの財政コストは「土地地代を除いた住宅サービス費用 + 取得土地の年賦費用（金利連動）」から算出。
  6. 厚生等価（制約世帯の効用一致）で給付量を較正。等コスト比較も併記。
  7. 給付額は現在GDP比（T_t = u·Y_t）。v1–v5 は ubi_cost が GDP でスケールされず、
     小国ほど実効負担率が大きかった（例: D の「5%」は導入時 GDP 比 約15%）。
  8. 債務クランプ廃止。債務比率が default_threshold を超えたブロックは insolvent として凍結し、
     time-to-insolvency を報告。
  9. 危機指標を onset / duration / severity / breadth に分解。
 10. 為替に輸出の支出切替（expenditure switching）を導入し、漏出が無限ドリフトしないようにする。
Python 3 標準ライブラリのみ。
"""
import math, json, argparse, sys
from dataclasses import dataclass, field, replace
from copy import deepcopy

YEARS, DT = 30, 0.25
STEPS = int(YEARS / DT)

# ─────────────────────────────── parameters ───────────────────────────────
@dataclass
class Bloc:
    name: str
    gdp: float; debt: float; rate: float; base_growth: float; decay: float
    openness: float; fx_share: float; reserve: bool = False
    capital: float = 1.0
    # 家計・市場構造（stylized, 順序と桁のみ）
    c_income_share: float = 0.2   # 制約世帯の所得/GDP
    c_pop_share: float = 0.4      # 制約世帯の人口比（ユニバーサル給付の配分比）
    m_F: float = 0.2              # 食料の輸入比率
    m_G: float = 0.25             # その他財の輸入比率
    m_H: float = 0.1              # 住宅建設・維持の輸入比率
    eps_supply: float = 0.4       # 賃貸住宅の短期供給弾力性
    land_share: float = 0.35      # 家賃に占める土地地代比率
    omega0: float = 0.2           # 非制約世帯の貯蓄の海外資産比率（基準）
    # 政策
    start: float = 999.0
    modality: str = "none"        # none | ubi | cash_t | voucher | clt
    size: float = 0.0             # ubi: GDP比 / その他: 較正後の制約世帯あたり量（GDP比）

@dataclass
class G:
    fiscal_mult: float = 1.2
    cap_sens: float = 0.8
    risk_coeff: float = 0.15
    reserve_flight: float = 0.3
    reserve_thr: float = 1.5
    reserve_damp: float = 0.5
    compress: float = 0.3
    # v6 new
    deficit_share: float = 0.6
    bop_coeff: float = 0.5        # 対外漏出（GDP比）→為替圧力
    export_switch: float = 0.05   # 減価による輸出増（為替の平均回帰）
    omega_sens: float = 2.0       # 海外資産比率のリスク・期待減価感応度
    mpc_u: float = 0.3            # 非制約世帯の限界消費性向
    land_discount: float = 0.5    # CLT土地取得の市場価格比ディスカウント（寄付・公有地移管）
    default_thr: float = 3.0
    eps_mult: float = 1.0         # 住宅供給弾力性の一律倍率（感度分析用）
    # Stone–Geary (制約世帯): 基礎需要（所得比）と限界予算比率
    sub_F: float = 0.35; sub_G: float = 0.05; sub_H: float = 0.25
    beta_F: float = 0.25; beta_G: float = 0.45; beta_H: float = 0.30

def archetypes():
    return {
        "A": Bloc("A (reserve)", gdp=1.2, debt=0.85, rate=0.020, base_growth=0.018, decay=0.0003,
                  openness=1.0, fx_share=0.0, reserve=True, capital=1.3,
                  c_income_share=0.15, c_pop_share=0.30, m_F=0.10, m_G=0.15, m_H=0.05,
                  eps_supply=0.3, land_share=0.45, omega0=0.05),
        "B": Bloc("B (non-reserve adv.)", gdp=0.9, debt=0.70, rate=0.025, base_growth=0.020, decay=0.0003,
                  openness=1.0, fx_share=0.15, capital=1.0,
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

# ─────────────────────────────── household block ───────────────────────────────
def les_demand(M, p, sub, beta, fixed=None):
    """Stone–Geary/LES. fixed={'k': q} は現物給付量。端点解（extramarginal）を処理。
    返り値: 数量 dict と市場購入額 dict。"""
    goods = list(p)
    fixed = fixed or {}
    full = M + sum(p[k] * q for k, q in fixed.items())
    supern = full - sum(p[k] * sub[k] for k in goods)
    x = {k: sub[k] + beta[k] * max(supern, 0) / p[k] for k in goods}
    binding = [k for k, q in fixed.items() if x[k] < q]
    if binding:  # 現物が過剰 → その財は q に固定、残りの予算を他財へ
        free = [k for k in goods if k not in binding]
        bsum = sum(beta[k] for k in free)
        rest = M + sum(p[k] * fixed[k] for k in fixed if k not in binding)
        sup2 = rest - sum(p[k] * sub[k] for k in free)
        for k in binding: x[k] = fixed[k]
        for k in free: x[k] = sub[k] + (beta[k] / bsum) * max(sup2, 0) / p[k]
    buy = {k: p[k] * max(0.0, x[k] - fixed.get(k, 0.0)) for k in goods}
    return x, buy

def utility(x, sub, beta):
    s = 0.0
    for k in x:
        d = x[k] - sub[k]
        if d <= 0: return -1e9
        s += beta[k] * math.log(d)
    return s

def household_block(b, g, Y, e, r, transfer_on, size):
    """1期の家計・住宅市場。transfer_on=False で反実仮想。GDP単位の各フローを返す。"""
    Ic = b.c_income_share * Y
    sub = {"F": g.sub_F * Ic, "G": g.sub_G * Ic, "H": g.sub_H * Ic}
    beta = {"F": g.beta_F, "G": g.beta_G, "H": g.beta_H}
    pF = 1 + b.m_F * (e - 1); pG = 1 + b.m_G * (e - 1)
    S0 = sub["H"] + beta["H"] * (Ic - sub["F"] - sub["G"] - sub["H"])  # 無給付時にp_H=1で清算

    cash_c = cash_u = 0.0; fixed = {}; clt_units = 0.0; fiscal = 0.0
    struct_cost = land_cost = 0.0
    if transfer_on:
        if b.modality == "ubi":
            T = size * Y
            cash_c, cash_u = T * b.c_pop_share, T * (1 - b.c_pop_share); fiscal = T
        elif b.modality == "cash_t":
            cash_c = size * Y; fiscal = cash_c
        elif b.modality == "voucher":
            fixed = {"F": size * Y}; fiscal = pF * size * Y
        elif b.modality == "clt":
            clt_units = size * Y; fixed = {"H": clt_units}
            r0 = b.rate  # 導入前の金利水準（Bloc初期値）
            struct_cost = (1 - b.land_share) * clt_units
            land_cost = b.land_share * (1 - g.land_discount) * (r + 0.02) / (r0 + 0.02) * clt_units
            fiscal = struct_cost + land_cost

    # 住宅市場均衡（二分法）: 市場需要 = 制約世帯の市場購入量, 供給 = S0 (p)^eps
    def excess(pH):
        p = {"F": pF, "G": pG, "H": pH}
        x, buy = les_demand(Ic + cash_c, p, sub, beta, fixed)
        return buy["H"] / pH - S0 * pH ** (b.eps_supply * g.eps_mult), x, buy, p
    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if excess(mid)[0] > 0: lo = mid
        else: hi = mid
    pH = 0.5 * (lo + hi)
    _, x, buy, p = excess(pH)

    rent = buy["H"]                                   # 家主の家賃収入
    imp_c = b.m_F * buy["F"] + b.m_G * buy["G"]
    dom_c = (1 - b.m_F) * buy["F"] + (1 - b.m_G) * buy["G"]
    clt_on = b.modality == "clt" and transfer_on
    clt_build = struct_cost if clt_on else 0.0     # 建物部分 = 実需（輸入比率 m_H）
    land_pay = land_cost if clt_on else 0.0         # 土地代金 = 地主（非制約世帯）への所得移転
    imp_gov = b.m_H * clt_build
    dom_gov = clt_build - imp_gov
    if b.modality == "voucher" and transfer_on:       # 現物食料の調達（輸入比率は同じ）
        imp_gov += b.m_F * fiscal; dom_gov += (1 - b.m_F) * fiscal
    return dict(pH=pH, rent=rent, imp_c=imp_c, dom_c=dom_c, imp_gov=imp_gov, dom_gov=dom_gov,
                cash_u=cash_u + land_pay, fiscal=fiscal, U=utility(x, sub, beta), x=x)

# ─────────────────────────────── calibration ───────────────────────────────
def calibrate(b, g, u_ubi):
    """制約世帯の効用が ubi(u) と一致する各様式の給付量を求める（導入時の価格体系で）。"""
    Y, e, r = b.gdp, 1.0, b.rate
    target = household_block(replace(b, modality="ubi"), g, Y, e, r, True, u_ubi)
    out = {"ubi": u_ubi, "_U": target["U"], "_fiscal_ubi": target["fiscal"] / Y}
    for mod in ("cash_t", "voucher", "clt"):
        lo, hi = 0.0, 0.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if household_block(replace(b, modality=mod), g, Y, e, r, True, mid)["U"] < target["U"]: lo = mid
            else: hi = mid
        s = 0.5 * (lo + hi)
        hb = household_block(replace(b, modality=mod), g, Y, e, r, True, s)
        out[mod] = s; out["_fiscal_" + mod] = hb["fiscal"] / Y
        if hb["U"] < target["U"] - 1e-6: out["_infeasible_" + mod] = True
    return out

def equal_cost_size(b, g, mod, fiscal_share):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = household_block(replace(b, modality=mod), g, b.gdp, 1.0, b.rate, True, mid)["fiscal"] / b.gdp
        if f < fiscal_share: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

# ─────────────────────────────── macro dynamics ───────────────────────────────
THR = dict(debt=1.5, infl=0.15, dep=2.0, app=0.3)

def simulate(blocs, g):
    S = [dict(Y=b.gdp, d=b.debt, r=b.rate, pi=0.02, e=1.0, K=b.capital, de=0.0,
              alive=True, t_insolv=None) for b in blocs]
    N = len(blocs)
    hist, flags = [], [[False] * N]
    sev = [0.0] * N; onsets = [0] * N; dur = [0] * N; breadth = []
    leak_log = [dict(n=0, imp=0.0, fa=0.0, fiscal=0.0, rentcap=0.0, dU=0.0) for _ in blocs]
    ubi_ref = [None] * N

    for step in range(STEPS + 1):
        t = step * DT
        if step % 4 == 0:
            hist.append({"year": t, **{f"{k}_{i}": round(S[i][k], 5) for i in range(N) for k in ("Y", "d", "e", "pi", "r")}})
        if step == STEPS: break
        avg_r = sum(s["r"] for s in S) / N
        instab = [max(0, s["d"] - 0.7) + max(0, s["pi"] - 0.04) + max(0, s["e"] - 1.5) for s in S]
        new_flags = []
        for i, (b, s) in enumerate(zip(blocs, S)):
            if not s["alive"]:
                new_flags.append(True); continue
            on = t >= b.start and b.modality != "none"
            Y, e, r = s["Y"], s["e"], s["r"]
            hb1 = household_block(b, g, Y, e, r, on, b.size)
            hb0 = household_block(b, g, Y, e, r, False, b.size)
            d_imp = (hb1["imp_c"] - hb0["imp_c"]) + hb1["imp_gov"]
            d_rent = hb1["rent"] - hb0["rent"]
            d_inc_u = hb1["cash_u"] + d_rent
            d_imp += g.mpc_u * b.m_G * d_inc_u
            use_res = b.reserve
            thr = g.reserve_thr if use_res else 0.9
            rp = (g.risk_coeff * max(0, s["d"] - thr) + g.risk_coeff * 0.5 * max(0, s["pi"] - 0.06)
                  + g.risk_coeff * 0.5 * b.fx_share * max(0, s["d"] - 0.6))
            exp_dep = max(0.0, s["de"] / DT)
            omega = min(0.9, max(0.0, b.omega0 + g.omega_sens * (rp + exp_dep)))
            d_fa = (1 - g.mpc_u) * d_inc_u * omega * b.openness   # 追加の海外資産購入（負=回帰）
            d_dom = ((hb1["dom_c"] - hb0["dom_c"]) + hb1["dom_gov"]
                     + g.mpc_u * (1 - b.m_G) * d_inc_u)
            fiscal = hb1["fiscal"]
            if on:
                L = leak_log[i]; L["n"] += 1; L["imp"] += d_imp; L["fa"] += d_fa; L["fiscal"] += fiscal
                L["Y"] = L.get("Y", 0.0) + Y; L["rentcap"] += d_rent; L["dU"] += hb1["U"] - hb0["U"]

            eff_g = max(0.005, b.base_growth - b.decay * t)
            debt_acc = g.deficit_share * fiscal / Y + r * s["d"] - eff_g * s["d"]
            debt_acc -= s["pi"] * s["d"] * g.compress * (1 - b.fx_share)
            flow = b.openness * g.cap_sens * (r - rp - avg_r) * DT
            if use_res:
                flow += g.reserve_flight * sum(instab[j] for j in range(N) if j != i) * DT
            bop = -(d_imp + d_fa) / Y + g.export_switch * (e - 1)     # 経常+ポートフォリオ（GDP比）
            flow += g.bop_coeff * bop * DT
            de = -2.0 * flow
            if use_res: de *= g.reserve_damp
            pull = g.fiscal_mult * 0.15 * max(0.0, d_dom / Y)
            dpi = (pull + 0.3 * max(0, de) - 0.1 * max(0, s["pi"] - 0.03)) * DT
            growth = (eff_g + g.fiscal_mult * 0.08 * d_dom / Y + 0.15 * flow
                      - 0.02 * max(0, s["d"] - 1.2) - 0.03 * max(0, s["pi"] - 0.05) - 0.01 * abs(de)) * DT
            r_tgt = max(0, 0.02 + 0.5 * (s["pi"] - 0.02) + 0.3 * rp)
            s["d"] = max(0, s["d"] + debt_acc * DT)
            s["d"] = max(0, s["d"] + de * s["d"] * b.fx_share)
            s["K"] = max(0.05, s["K"] + flow)
            s["e"] = max(0.05, s["e"] + de); s["de"] = de
            s["pi"] = max(-0.02, s["pi"] + dpi)
            s["Y"] = max(0.01, Y * (1 + growth))
            s["r"] = max(0, min(0.30, r + 0.2 * (r_tgt - r) * DT))
            # crisis metrics
            exc = [max(0, s["d"] - THR["debt"]) / THR["debt"], max(0, s["pi"] - THR["infl"]) / THR["infl"],
                   max(0, s["e"] - THR["dep"]) / THR["dep"], max(0, THR["app"] - s["e"]) / THR["app"]]
            in_c = any(x > 0 for x in exc)
            if in_c:
                dur[i] += 1; sev[i] += sum(exc) * DT
                if not flags[-1][i]: onsets[i] += 1
            new_flags.append(in_c)
            if s["d"] > g.default_thr:
                s["alive"] = False; s["t_insolv"] = t
        flags.append(new_flags); breadth.append(sum(new_flags))

    crisis = {blocs[i].name: dict(onsets=onsets[i], duration_q=dur[i], severity=round(sev[i], 3),
                                  insolvent_year=S[i]["t_insolv"]) for i in range(N)}
    crisis["_system"] = dict(max_breadth=max(breadth), bloc_quarters=sum(breadth),
                             total_onsets=sum(onsets))
    leak = {}
    for i, L in enumerate(leak_log):
        if L["n"] and L["fiscal"] > 0:
            leak[blocs[i].name] = dict(import_per_fiscal=round(L["imp"] / L["fiscal"], 3),
                                       foreign_assets_per_fiscal=round(L["fa"] / L["fiscal"], 3),
                                       external_leak_per_fiscal=round((L["imp"] + L["fa"]) / L["fiscal"], 3),
                                       rent_capture_per_fiscal=round(L["rentcap"] / L["fiscal"], 3),
                                       fiscal_pct_gdp=round(100 * L["fiscal"] / L["Y"], 3),
                                       external_leak_pct_gdp=round(100 * (L["imp"] + L["fa"]) / L["Y"], 3))
    final = {blocs[i].name: dict(debt=round(S[i]["d"], 3), e=round(S[i]["e"], 3), pi=round(S[i]["pi"], 3),
                                 Y=round(S[i]["Y"], 3)) for i in range(N)}
    return dict(crisis=crisis, leakage=leak, final=final, history=hist)

# ─────────────────────────────── experiments ───────────────────────────────
def build(g, modality, u=0.05, starts=(1, 5, 10, 15), basis="welfare", which=("A", "B", "C", "D")):
    arch = archetypes(); out = []
    for key, st in zip(("A", "B", "C", "D"), starts):
        b = arch[key]
        if key in which and modality != "none":
            cal = calibrate(b, g, u)
            if basis == "welfare":
                size = cal[modality]
            else:  # equal fiscal cost to universal UBI
                size = u if modality == "ubi" else equal_cost_size(b, g, modality, cal["_fiscal_ubi"])
            b = replace(b, modality=modality, size=size, start=st)
        out.append(b)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--params", nargs="*", default=[])
    ap.add_argument("--simultaneous", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = G()
    for kv in a.params:
        k, v = kv.split("="); setattr(g, k, float(v))
    starts = (1, 1, 1, 1) if a.simultaneous else (1, 5, 10, 15)
    res = {"calibration": {k: calibrate(b, g, a.u) for k, b in archetypes().items()}, "runs": {}}
    for basis in ("welfare", "cost"):
        for mod in ("none", "ubi", "cash_t", "voucher", "clt"):
            if basis == "cost" and mod in ("none", "ubi"): continue
            r = simulate(build(g, mod, a.u, starts, basis), g)
            r.pop("history")
            res["runs"][f"{basis}:{mod}"] = r
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out: open(a.out, "w").write(txt)
    else: print(txt)

if __name__ == "__main__":
    main()
