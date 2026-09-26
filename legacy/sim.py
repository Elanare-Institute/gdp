#!/usr/bin/env python3
"""
UBI × Mundellian Trilemma: Dynamic Transition Simulator
========================================================
4-bloc differential equation model for hypothesis-driven analysis.
CLI tool - no GUI required. Outputs structured JSON or markdown tables.

Usage:
  python sim.py                     # Run all hypotheses, output summary
  python sim.py --hyp H1 H2        # Run specific hypotheses
  python sim.py --format json       # JSON output for programmatic use
  python sim.py --format md         # Markdown tables
  python sim.py --params fiscal_mult=1.5 cap_sens=1.2
  python sim.py --sweep ubiLevel 0.03 0.05 0.07 0.10 0.12 0.15
"""

import json, sys, math, argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from copy import deepcopy

# ═══════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════
YEARS = 30
DT = 0.25
STEPS = int(YEARS / DT)

@dataclass
class Country:
    name: str
    gdp: float = 1.0
    debt_ratio: float = 0.6
    interest_rate: float = 0.03
    inflation: float = 0.02
    exchange_rate: float = 1.0
    capital_stock: float = 1.0
    base_growth: float = 0.02
    tax_capacity: float = 0.35
    ubi_start_year: float = 999.0
    ubi_level: float = 0.05
    regime: str = "float"       # float | fixed | managed
    capital_openness: float = 1.0
    monetary_independence: float = 1.0
    clt_mode: bool = False
    country_type: str = "advanced"
    is_reserve_currency: bool = False   # 基軸通貨国フラグ（v2: 構造的非対称性）
    growth_decay_rate: float = 0.0      # v3: 年あたり成長率低下幅（趨勢的低下）
    foreign_debt_share: float = 0.0     # v4: 対外（ドル建て）債務の割合 (0-1)
    forex_debt_off: bool = False        # v4: このシナリオで forex 動態を無効化（v3再現用）

@dataclass
class GlobalParams:
    fiscal_mult: float = 1.2
    cap_sens: float = 0.8
    risk_coeff: float = 0.15
    # ── v2: 基軸通貨の構造的非対称性パラメータ ──
    reserve_flight_coeff: float = 0.3     # 逃避先流入（flight-to-safety）の感応度
    reserve_debt_threshold: float = 1.5   # リスクプレミアム発動閾値（対称は0.9）
    reserve_exch_damp: float = 0.5        # 為替変動の減衰係数（需要の構造的下支え）
    reserve_debt_compress: float = 0.3    # インフレによる実質債務圧縮係数
    reserve_asymmetry: float = 1.0        # マスタースイッチ（0で非対称OFF＝v1再現）
    # ── v3: 成長率の趨勢的低下（growth decay）パラメータ ──
    growth_decay_enabled: float = 1.0     # 1でdecay有効、0で無効（--no-growth-decay）
    growth_decay_rate_c: float = -1.0     # 新興国Cのdecay上書き（負=センチネル→国別既定を使用）
    growth_decay_rate_d: float = -1.0     # 途上国Dのdecay上書き（負=センチネル→国別既定を使用）
    # ── v4: 対外債務の通貨ミスマッチ パラメータ ──
    forex_debt_enabled: float = 1.0       # 1でforex動態有効、0で無効（--no-forex-debt→v3再現）
    foreign_debt_c: float = -1.0          # 新興国Cの対外債務割合上書き（負=センチネル）
    foreign_debt_d: float = -1.0          # 途上国Dの対外債務割合上書き（負=センチネル）
    # ── v5: CLT の通貨経路迂回 パラメータ（全て clt_mode の国にのみ作用）──
    clt_demand_dampening: float = 0.3     # CLTのインフレ圧力(demand_pull)減衰率
    clt_exch_dampening: float = 0.35      # CLTの為替圧力減衰率（exch_change *= 1-0.5*(1-これ)）
    clt_dynamic_cost: float = 0.0         # 1でCLTコストを動的（初年度高・以後急減衰）に

def archetypes(realistic=True):
    """v3: realistic=True で新興国C・途上国Dを現実的成長率に。
    返り値には realistic フラグに従う emerging/developing に加え、
    フラグ非依存の明示変種 *_real / *_opt を含める（新仮説がシナリオ内で
    成長レジームを固定するため）。"""
    def _mk(defaults):
        def factory(**kw):
            merged = {**defaults, **kw}
            return Country(**merged)
        return factory
    # 新興国C / 途上国D の楽観的（v1/v2互換）・現実的（v3）パラメータ
    emerging_opt = dict(
        name="新興国C", gdp=0.5, debt_ratio=0.40, interest_rate=0.05,
        base_growth=0.045, tax_capacity=0.22, capital_stock=0.55,
        capital_openness=0.75, country_type="emerging", growth_decay_rate=0.0005,
        foreign_debt_share=0.45)
    emerging_real = {**emerging_opt, "base_growth": 0.035, "capital_stock": 0.45}
    developing_opt = dict(
        name="途上国D", gdp=0.2, debt_ratio=0.35, interest_rate=0.07,
        base_growth=0.055, tax_capacity=0.15, capital_stock=0.25,
        capital_openness=0.5, country_type="developing", growth_decay_rate=0.0008,
        foreign_debt_share=0.65)
    developing_real = {**developing_opt, "base_growth": 0.038, "capital_stock": 0.20}
    return {
        "advancedA": _mk(dict(
            name="先進国A (USD圏)",
            gdp=1.2, debt_ratio=0.85, interest_rate=0.02, base_growth=0.018,
            tax_capacity=0.42, capital_stock=1.3, capital_openness=1.0,
            country_type="advanced", is_reserve_currency=True, growth_decay_rate=0.0003,
            foreign_debt_share=0.0)),
        "advancedB": _mk(dict(
            name="先進国B (EUR圏)",
            gdp=0.9, debt_ratio=0.70, interest_rate=0.025, base_growth=0.02,
            tax_capacity=0.40, capital_stock=1.0, capital_openness=1.0,
            country_type="advanced", growth_decay_rate=0.0003, foreign_debt_share=0.15)),
        # フラグに従う既定（既存 H1〜H13 用）
        "emerging": _mk(emerging_real if realistic else emerging_opt),
        "developing": _mk(developing_real if realistic else developing_opt),
        # フラグ非依存の明示変種（新仮説 H14〜H17 用）
        "emerging_real": _mk(emerging_real),
        "developing_real": _mk(developing_real),
        "emerging_opt": _mk(emerging_opt),
        "developing_opt": _mk(developing_opt),
    }

def simulate(countries: List[Country], params: GlobalParams):
    N = len(countries)
    states = [deepcopy(c) for c in countries]
    history = []  # annual snapshots only
    crisis_log = []
    absorption_breach_log = []   # v3: 成長率 < UBIコスト/GDP を記録（相転移検出）
    step_buf = {}

    for step in range(STEPS + 1):
        year = round(step * DT, 2)

        # record annually
        if step % 4 == 0:
            rec = {"year": year}
            for i, s in enumerate(states):
                rec[f"gdp_{i}"] = round(s.gdp, 6)
                rec[f"debt_{i}"] = round(s.debt_ratio, 6)
                rec[f"exch_{i}"] = round(s.exchange_rate, 6)
                rec[f"infl_{i}"] = round(s.inflation * 100, 4)
                rec[f"cap_{i}"] = round(s.capital_stock, 6)
                rec[f"rate_{i}"] = round(s.interest_rate * 100, 4)
            history.append(rec)

        if step == STEPS:
            break

        avg_rate = sum(s.interest_rate for s in states) / N

        # v2: 各国の「不安定性」寄与をステップ開始時スナップショットから算出（順序依存を回避）
        # v4: 数値安定のため寄与を 200 で上限クランプ（v3回帰セット最大 151.6 より十分上、
        #     forex 自己増幅ループの発散が逃避先流入を通じて基軸国GDPを overflow させるのを防ぐ）。
        instability = [min(200.0, max(0, s.debt_ratio - 0.7)
                           + max(0, s.inflation - 0.04)
                           + max(0, 0.5 - s.exchange_rate)) for s in states]

        for i, s in enumerate(states):
            use_reserve = params.reserve_asymmetry and s.is_reserve_currency
            # v4: forex 動態のゲートと実効的な対外債務割合（センチネル上書き）
            forex_on = params.forex_debt_enabled and not s.forex_debt_off
            fx_share = s.foreign_debt_share
            if s.country_type == "emerging" and params.foreign_debt_c >= 0:
                fx_share = params.foreign_debt_c
            elif s.country_type == "developing" and params.foreign_debt_d >= 0:
                fx_share = params.foreign_debt_d
            ubi_on = year >= s.ubi_start_year
            # v5-(1D): CLTコスト。既定は v4 と同一の静的0.3。動的コスト有効時は初年度高・以後急減衰。
            if not ubi_on:
                ubi_cost = 0.0
            elif s.clt_mode:
                if params.clt_dynamic_cost:
                    yrs = year - s.ubi_start_year
                    ubi_cost = s.ubi_level * (0.10 + 0.70 * math.exp(-0.3 * yrs))
                else:
                    ubi_cost = s.ubi_level * 0.3
            else:
                ubi_cost = s.ubi_level

            # v3: 実効成長率（趨勢的低下）。GlobalParams のセンチネル上書きを優先。
            decay = s.growth_decay_rate
            if s.country_type == "emerging" and params.growth_decay_rate_c >= 0:
                decay = params.growth_decay_rate_c
            elif s.country_type == "developing" and params.growth_decay_rate_d >= 0:
                decay = params.growth_decay_rate_d
            if not params.growth_decay_enabled:
                decay = 0.0
            effective_growth = max(0.005, s.base_growth - decay * year)

            # v3: 吸収閾値の相転移検出（成長率 < UBIコスト/GDP なら発散側）
            ubi_cost_gdp = ubi_cost / max(s.gdp, 0.1)
            if ubi_on and effective_growth < ubi_cost_gdp:
                absorption_breach_log.append({
                    "year": year, "country": s.name, "idx": i,
                    "growth": round(effective_growth, 5),
                    "ubi_cost_gdp": round(ubi_cost_gdp, 5),
                    "gap": round(ubi_cost_gdp - effective_growth, 5)})

            fiscal_gap = ubi_cost * 0.6
            debt_accum = (fiscal_gap / max(s.gdp, 0.1)
                          + s.interest_rate * s.debt_ratio
                          - effective_growth * s.debt_ratio)
            # v4-(1D): インフレによる債務圧縮を全ブロックに一般化（自国通貨建て部分のみ効く）。
            # forex 無効時は v2/v3 と同じ「基軸通貨国のみ圧縮」に退避（回帰保証）。
            if forex_on:
                debt_accum -= s.inflation * s.debt_ratio * params.reserve_debt_compress * (1.0 - fx_share)
            elif use_reserve:
                debt_accum -= s.inflation * s.debt_ratio * params.reserve_debt_compress

            # v2-(2): 基軸通貨国はリスクプレミアム発動閾値が高い（1.5 vs 0.9）
            threshold = params.reserve_debt_threshold if use_reserve else 0.9
            risk_premium = (params.risk_coeff * max(0, s.debt_ratio - threshold)
                            + params.risk_coeff * 0.5 * max(0, s.inflation - 0.06))
            # v4-(1C): 対外債務が高い国は同じ債務比率でもリスクプレミアムが上乗せ
            if forex_on:
                risk_premium += params.risk_coeff * 0.5 * fx_share * max(0, s.debt_ratio - 0.6)
            eff_return = s.interest_rate - risk_premium
            cap_flow = s.capital_openness * params.cap_sens * (eff_return - avg_rate) * DT

            # v2-(1): 逃避先流入（flight-to-safety）— 他ブロックの不安定性総和に比例
            if use_reserve:
                instability_others = sum(instability[j] for j in range(N) if j != i)
                cap_flow += params.reserve_flight_coeff * instability_others * DT

            exch_pressure = -cap_flow * 2.0
            if s.regime == "fixed":
                exch_change = 0.0
            elif s.regime == "managed":
                exch_change = exch_pressure * 0.3
            else:
                exch_change = exch_pressure

            # v2-(3): 基軸通貨国の為替変動は需要の構造的下支えで減衰
            if use_reserve:
                exch_change *= params.reserve_exch_damp

            # v5-(1B): CLTは現金が国境を越えないためUBI起因の為替圧力を減衰（通貨経路の迂回）
            if s.clt_mode and ubi_on:
                exch_change *= (1.0 - 0.5 * (1.0 - params.clt_exch_dampening))

            demand_pull = params.fiscal_mult * ubi_cost * 0.15 if ubi_on else 0.0
            # v5-(1A): CLTは現金を市場に流入させないためインフレ圧力(demand_pull)が弱い
            if s.clt_mode:
                demand_pull *= params.clt_demand_dampening
            import_infl = max(0, exch_change) * 0.3
            monetary_brake = s.monetary_independence * 0.1 * max(0, s.inflation - 0.03)
            infl_change = (demand_pull + import_infl - monetary_brake) * DT

            ubi_stim = params.fiscal_mult * ubi_cost * 0.08 if ubi_on else 0.0
            cap_contrib = cap_flow * 0.15
            debt_drag = -0.02 * max(0, s.debt_ratio - 1.2)
            infl_drag = -0.03 * max(0, s.inflation - 0.05)
            vol_drag = -0.01 * abs(exch_change)
            gdp_growth = (effective_growth + ubi_stim + cap_contrib + debt_drag + infl_drag + vol_drag) * DT

            taylor = 0.5 * (s.inflation - 0.02) + 0.3 * risk_premium
            rate_target = max(0, 0.02 + taylor)
            rate_change = s.monetary_independence * 0.2 * (rate_target - s.interest_rate) * DT
            fixed_force = 0.5 * exch_pressure * DT if s.regime == "fixed" else 0.0

            # v4: 債務比率の上限クランプ 2000（v3回帰セット最大 145.6 より十分上）。
            # forex 自己増幅ループの発散を「事実上のデフォルト」として有限値で頭打ちにし overflow を防ぐ。
            s.debt_ratio = min(2000.0, max(0, s.debt_ratio + debt_accum * DT))
            # v4-(1B): 通貨ミスマッチ動態。本モデルの為替慣行では exchange_rate 上昇=通貨安
            # （import_infl=max(0,exch_change) が減価時に発火）なので、減価=exch_change>0 で
            # ドル建て対外債務の自国通貨換算額が膨張、増価(<0)で圧縮（対称）。fx_share に比例。
            # ※指示書1-Bのコメントは exchange_rate 低下=減価 を前提とし符号が逆だが、本コードの
            #   慣行に合わせて符号を +exch_change に修正（自己増幅ループが正しく発動するため）。
            if forex_on and fx_share > 0:
                s.debt_ratio = min(2000.0, max(0, s.debt_ratio + exch_change * s.debt_ratio * fx_share))
            s.capital_stock = max(0.05, s.capital_stock + cap_flow)
            s.exchange_rate = max(0.05, s.exchange_rate + exch_change)
            s.inflation = max(-0.02, s.inflation + infl_change)
            s.gdp = max(0.05, s.gdp * (1 + gdp_growth))
            s.interest_rate = max(0, min(0.30, s.interest_rate + rate_change + fixed_force))

            if (s.debt_ratio > 1.5 or s.inflation > 0.15
                    or s.exchange_rate < 0.3 or s.capital_stock < 0.15):
                crisis_log.append({
                    "year": year, "country": s.name, "idx": i,
                    "debt": round(s.debt_ratio, 3),
                    "inflation": round(s.inflation, 4),
                    "exch": round(s.exchange_rate, 3),
                    "capital": round(s.capital_stock, 3),
                })

    return {
        "history": history,
        "crisis_log": crisis_log,
        "absorption_breach_log": absorption_breach_log,
        "final_states": [_country_summary(s) for s in states],
    }

def _first_breach_by_country(breach_log):
    """v3: 国別の初回 absorption breach（年・gap）を返す。"""
    first = {}
    for b in breach_log:
        key = str(b["idx"])
        if key not in first:
            first[key] = {"country": b["country"], "year": b["year"], "gap": b["gap"]}
    return first

def _country_summary(c: Country):
    return {
        "name": c.name,
        "gdp": round(c.gdp, 4),
        "debt_ratio": round(c.debt_ratio, 4),
        "exchange_rate": round(c.exchange_rate, 4),
        "inflation": round(c.inflation, 4),
        "capital_stock": round(c.capital_stock, 4),
        "interest_rate": round(c.interest_rate, 4),
    }


# ═══════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════
def compute_stats(history, n_countries):
    stats = {}
    for i in range(n_countries):
        gdp = [r[f"gdp_{i}"] for r in history]
        debt = [r[f"debt_{i}"] for r in history]
        exch = [r[f"exch_{i}"] for r in history]
        infl = [r[f"infl_{i}"] for r in history]
        cap = [r[f"cap_{i}"] for r in history]
        rate = [r[f"rate_{i}"] for r in history]

        exch_ret = [(exch[j] - exch[j-1]) / exch[j-1] for j in range(1, len(exch))]
        vol = _stddev(exch_ret) if exch_ret else 0

        stats[f"country_{i}"] = {
            "final_gdp": gdp[-1],
            "final_debt": debt[-1],
            "max_debt": max(debt),
            "final_capital": cap[-1],
            "max_inflation_pct": max(infl),
            "exch_volatility": round(vol, 6),
            "gdp_change_5y_pct": round((gdp[5] / gdp[0] - 1) * 100, 3) if len(gdp) > 5 else None,
            "gdp_change_10y_pct": round((gdp[10] / gdp[0] - 1) * 100, 3) if len(gdp) > 10 else None,
            "gdp_change_30y_pct": round((gdp[-1] / gdp[0] - 1) * 100, 3),
            "capital_change_5y": round(cap[5] - cap[0], 4) if len(cap) > 5 else None,
            "capital_change_10y": round(cap[10] - cap[0], 4) if len(cap) > 10 else None,
        }
    return stats

def _stddev(arr):
    n = len(arr)
    if n < 2: return 0.0
    m = sum(arr) / n
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (n - 1))


# ═══════════════════════════════════════════════════════
# HYPOTHESES
# ═══════════════════════════════════════════════════════
A = archetypes()

HYPOTHESES = {
    "H1": {
        "title": "先行導入国の短期コスト仮説",
        "statement": "最初にUBIを導入した国は移行期に資本流出と為替減価に直面し、追随国より短期的に不利になる。",
        "prediction": "先行国のYear1-5資本ストック減少率 > 後発国",
        "scenarios": [
            {"label": "A先行→B追随→C後発→D不参加",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10), A["developing"]()]},
            {"label": "C先行→A追随→B後発→D不参加",
             "countries": lambda: [A["advancedA"](ubi_start_year=5), A["advancedB"](ubi_start_year=10),
                                   A["emerging"](ubi_start_year=1), A["developing"]()]},
        ]
    },
    "H2": {
        "title": "最終拒否国の漁夫の利仮説",
        "statement": "最後まで拒否した国に資本が集中し短期的に「勝つ」が、他国の市場縮小で長期的には成長が止まる。",
        "prediction": "拒否国の資本Year1-10増加、Year10-30でGDP成長率低下",
        "scenarios": [
            {"label": "D以外全員導入、D拒否",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=1),
                                   A["emerging"](ubi_start_year=3), A["developing"]()]},
            {"label": "A以外全員導入、A拒否（基軸通貨国拒否）",
             "countries": lambda: [A["advancedA"](), A["advancedB"](ubi_start_year=1),
                                   A["emerging"](ubi_start_year=1), A["developing"](ubi_start_year=3)]},
        ]
    },
    "H3": {
        "title": "同時導入の安定性優位仮説",
        "statement": "全ブロック同時導入は段階的導入より移行期の為替・資本変動が小さい。",
        "prediction": "同時導入の為替ボラティリティ < 段階的導入",
        "scenarios": [
            {"label": "全員Year1同時",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=1),
                                   A["emerging"](ubi_start_year=1), A["developing"](ubi_start_year=1)]},
            {"label": "2年刻み段階的",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=3),
                                   A["emerging"](ubi_start_year=5), A["developing"](ubi_start_year=7)]},
            {"label": "5年刻み段階的",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=6),
                                   A["emerging"](ubi_start_year=11), A["developing"](ubi_start_year=16)]},
        ]
    },
    "H4": {
        "title": "資本規制による安定化仮説",
        "statement": "トリレンマの「資本移動の自由」を犠牲にすればUBI導入時の国際金融圧力を大幅に緩和できる。",
        "prediction": "資本規制あり(openness=0.2)の方が為替・債務の変動幅が小さい",
        "scenarios": [
            {"label": "段階的・資本自由",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10), A["developing"]()]},
            {"label": "段階的・資本規制",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, capital_openness=0.2),
                                   A["advancedB"](ubi_start_year=5, capital_openness=0.2),
                                   A["emerging"](ubi_start_year=10, capital_openness=0.2),
                                   A["developing"](capital_openness=0.2)]},
        ]
    },
    "H5": {
        "title": "CLT型の構造的優位仮説",
        "statement": "CLT/現物型UBIは財政コスト構造が根本的に異なるため、国際金融トリレンマの圧力経路自体が変わる。",
        "prediction": "CLT型は同じUBI水準でも債務蓄積・インフレ圧力が顕著に小さい",
        "scenarios": [
            {"label": "段階的・現金給付型",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10), A["developing"]()]},
            {"label": "段階的・CLT型",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, clt_mode=True),
                                   A["advancedB"](ubi_start_year=5, clt_mode=True),
                                   A["emerging"](ubi_start_year=10, clt_mode=True),
                                   A["developing"]()]},
        ]
    },
    "H6": {
        "title": "UBI水準の閾値仮説",
        "statement": "UBI水準にtipping pointが存在し、超えると非線形に破綻が連鎖する。閾値は経済構造で異なる。",
        "prediction": "GDP3%→7%→12%と上げた時にある水準で債務比率が急激に発散",
        "scenarios": [
            {"label": f"UBI = GDP {pct}%",
             "countries": lambda p=pct/100: [
                 A["advancedA"](ubi_start_year=1, ubi_level=p),
                 A["advancedB"](ubi_start_year=3, ubi_level=p),
                 A["emerging"](ubi_start_year=5, ubi_level=p),
                 A["developing"]()]}
            for pct in [3, 5, 7, 10, 12, 15]
        ]
    },
    "H7": {
        "title": "CLT/現金混合構造仮説",
        "statement": "CLT配置のパターン（全員・先進国のみ・新興国のみ）により国際金融圧力の経路が変わり、混合構成でも構造優位を部分継承できる。",
        "prediction": "全員CLT < 先進国CLT・新興国現金 < 先進国現金・新興国CLT < 全員現金、の順で債務・インフレが増大",
        "scenarios": [
            {"label": "全員現金（H1基準と同一）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10), A["developing"]()]},
            {"label": "全員CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, clt_mode=True),
                                   A["advancedB"](ubi_start_year=5, clt_mode=True),
                                   A["emerging"](ubi_start_year=10, clt_mode=True),
                                   A["developing"]()]},
            {"label": "先進国CLT・新興国現金",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, clt_mode=True),
                                   A["advancedB"](ubi_start_year=5, clt_mode=True),
                                   A["emerging"](ubi_start_year=10, clt_mode=False),
                                   A["developing"]()]},
            {"label": "先進国現金・新興国CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, clt_mode=False),
                                   A["advancedB"](ubi_start_year=5, clt_mode=False),
                                   A["emerging"](ubi_start_year=10, clt_mode=True),
                                   A["developing"]()]},
        ]
    },
    "H8": {
        "title": "固定為替制混在仮説",
        "statement": "トリレンマの3コーナーから「為替安定」を選ぶ国が混在する場合、その国は債務累積を緩和できるが、他国へ圧力を転嫁する。",
        "prediction": "固定制国は為替Volが極小・債務累積も抑制されるが、他国の為替Vol・債務が悪化",
        "scenarios": [
            {"label": "全員float（H1基準と同一）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10), A["developing"]()]},
            {"label": "新興国Cがfixed",
             "countries": lambda: [A["advancedA"](ubi_start_year=1), A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10, regime="fixed",
                                                  monetary_independence=0.3),
                                   A["developing"]()]},
            {"label": "先進国Bがfixed（EUR硬直性）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1),
                                   A["advancedB"](ubi_start_year=5, regime="fixed",
                                                  monetary_independence=0.3),
                                   A["emerging"](ubi_start_year=10),
                                   A["developing"]()]},
        ]
    },
    "H9": {
        "title": "国別UBI水準差仮説",
        "statement": "税収基盤に応じてUBI水準を調整すれば、一律水準より破綻リスクを抑えつつ移行できる。逆進的配分は途上国を破綻させる。",
        "prediction": "差別化(先進5%/新興3%/途上1%) < 一律5% < 逆進(先進3%/新興5%/途上7%) の順で危機件数増大",
        "scenarios": [
            {"label": "一律UBI=5%",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing"](ubi_start_year=15, ubi_level=0.05)]},
            {"label": "税収比例(先進5%/新興3%/途上1%)",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=10, ubi_level=0.03),
                                   A["developing"](ubi_start_year=15, ubi_level=0.01)]},
            {"label": "逆進的(先進3%/新興5%/途上7%)",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.03),
                                   A["emerging"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing"](ubi_start_year=15, ubi_level=0.07)]},
        ]
    },
    # ═══════════════ v2: 基軸通貨の構造的非対称性 仮説群 ═══════════════
    "H10": {
        "title": "基軸通貨国の構造的優位仮説",
        "statement": "基軸通貨国（米国）は消去法的ロックインにより、他ブロックより高水準のUBIを維持できる唯一の国である。",
        "prediction": "先進国AのUBI=GDP10%でも危機が発生しない一方、先進国Bは5%で危機に入る。",
        "scenarios": [
            {"label": "全員UBI=5%（A/B/C導入、D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=1, ubi_level=0.05),
                                   A["developing"]()]},
            {"label": "全員UBI=10%（A/B/C導入、D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.10),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.10),
                                   A["emerging"](ubi_start_year=1, ubi_level=0.10),
                                   A["developing"]()]},
            {"label": "非対称水準 A=10%/B=5%/C=3%（D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.10),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=1, ubi_level=0.03),
                                   A["developing"]()]},
        ]
    },
    "H11": {
        "title": "先進国B（非基軸先進国）最脆弱仮説",
        "statement": "基軸通貨の特権がなく途上国ほどの成長余力もない先進国B（EU型）が、UBI移行期において最も脆弱なブロックである。",
        "prediction": "同一UBI水準で先進国Bが最初に危機閾値を超える。",
        "scenarios": [
            {"label": "段階導入 A(Y1)→B(Y5)→C(Y10)、UBI=5%（D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing"]()]},
            {"label": "段階導入 A(Y1)→B(Y5)→C(Y10)、UBI=7%（D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.07),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.07),
                                   A["emerging"](ubi_start_year=10, ubi_level=0.07),
                                   A["developing"]()]},
            {"label": "段階導入 A(Y1)→B(Y5)→C(Y10)、UBI=10%（D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.10),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.10),
                                   A["emerging"](ubi_start_year=10, ubi_level=0.10),
                                   A["developing"]()]},
        ]
    },
    "H12": {
        "title": "米国UBI拒否の覇権強化仮説",
        "statement": "米国がUBIを拒否し他国が導入すると、逃避先流入が集中して米国の経済的覇権が強化される。世界的にはUBI導入が望ましくても、米国にとってはゲーム理論的に拒否が支配戦略になる。",
        "prediction": "米国拒否シナリオでは米国の資本ストック・GDPが全シナリオ中最大になる。",
        "scenarios": [
            {"label": "A拒否、B/C/D全員導入（UBI=5%）",
             "countries": lambda: [A["advancedA"](),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=3, ubi_level=0.05),
                                   A["developing"](ubi_start_year=5, ubi_level=0.05)]},
            {"label": "A拒否、B/C導入、D不参加（UBI=5%）",
             "countries": lambda: [A["advancedA"](),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=3, ubi_level=0.05),
                                   A["developing"]()]},
            {"label": "全員導入（比較用ベースライン、UBI=5%）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05),
                                   A["emerging"](ubi_start_year=3, ubi_level=0.05),
                                   A["developing"](ubi_start_year=5, ubi_level=0.05)]},
        ]
    },
    "H13": {
        "title": "基軸通貨国先行導入の波及効果仮説",
        "statement": "基軸通貨国が先行してUBIを導入すると、他国の資本流出が限定的になるため、全体の移行コストが下がる。H1（先行者不利）の結論が反転する。",
        "prediction": "A先行のH1-S1と同じ導入順序でも、基軸通貨非対称モデルでは先行者コストが消失または反転する。",
        "scenarios": [
            {"label": "A先行(Y1)→B(Y5)→C(Y10)→D不参加（非対称モデル）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1),
                                   A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10),
                                   A["developing"]()]},
            {"label": "H1-S1と同一設定（非対称モデル適用）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1),
                                   A["advancedB"](ubi_start_year=5),
                                   A["emerging"](ubi_start_year=10),
                                   A["developing"]()]},
        ]
    },
    # ═══════════════ v3: 成長率低下と吸収閾値 仮説群 ═══════════════
    # 新仮説はシナリオ内で成長レジームを固定するため、明示変種 *_real / *_opt を使用。
    "H14": {
        "title": "成長率現実化による結論反転仮説",
        "statement": "v1/v2の「途上国は高水準UBIでも持つ」という結論は成長率の過大推定に依存しており、現実的な成長率（3.5-3.8%）では途上国のtipping pointが大幅に低下する。",
        "prediction": "現実的成長率では同一UBI水準でも危機・吸収閾値超過が楽観的成長率より顕著に増える（D不参加のため直接のD閾値はH17で測定）。",
        "scenarios": [
            {"label": "現実的成長率・UBI=3%（A→B→C段階, D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.03),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.03),
                                   A["developing_real"]()]},
            {"label": "現実的成長率・UBI=5%（A→B→C段階, D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"]()]},
            {"label": "現実的成長率・UBI=7%（A→B→C段階, D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.07),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.07),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.07),
                                   A["developing_real"]()]},
            {"label": "楽観的成長率・UBI=7%（比較用ベースライン, v2相当）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.07),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.07),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.07),
                                   A["developing_opt"]()]},
        ]
    },
    "H15": {
        "title": "逆進的UBI再検証仮説",
        "statement": "v1のH9で『先進3%/新興5%/途上7%』が全体危機を圧縮したが、途上国の成長率を現実化すると途上国に高水準を割り当てる合理性が消える。",
        "prediction": "現実的成長率では『逆進的UBI』より『累進的UBI』（先進高/途上低）の方が全体危機が少ない。",
        "scenarios": [
            {"label": "逆進的 A3/B5/C5/D7・現実的成長率",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.07)]},
            {"label": "均一 全員5%・現実的成長率",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05)]},
            {"label": "累進的 A10/B5/C3/D1・現実的成長率",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.10),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.03),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.01)]},
            {"label": "逆進的 A3/B5/C5/D7・楽観的成長率（v1結論再現）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_opt"](ubi_start_year=15, ubi_level=0.07)]},
        ]
    },
    "H16": {
        "title": "閉まる窓仮説（closing window）",
        "statement": "成長率が趨勢的に低下する中でのUBI導入は、導入タイミングが遅れるほど持続可能性が下がる。今は通れる窓が10年後には閉まっている。",
        "prediction": "同一UBI水準でも、Year1導入よりYear10・Year20導入の方が危機件数・吸収閾値超過が多い。",
        "scenarios": [
            {"label": "全員Year1導入・UBI=5%・decay ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=1, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=1, ubi_level=0.05)]},
            {"label": "全員Year10導入・UBI=5%・decay ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=10, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=10, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=10, ubi_level=0.05)]},
            {"label": "全員Year20導入・UBI=5%・decay ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=20, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=20, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=20, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=20, ubi_level=0.05)]},
            {"label": "全員Year1導入・UBI=5%・decay OFF（比較用）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, growth_decay_rate=0.0),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05, growth_decay_rate=0.0),
                                   A["emerging_real"](ubi_start_year=1, ubi_level=0.05, growth_decay_rate=0.0),
                                   A["developing_real"](ubi_start_year=1, ubi_level=0.05, growth_decay_rate=0.0)]},
        ]
    },
    "H17": {
        "title": "「成長で吸収」の相転移仮説",
        "statement": "UBI財政コスト/GDPが成長率を超えた瞬間に動態が質的に変わり、連続的劣化ではなく発散的危機に移行する。この分岐は成長率=コスト/GDPを境界とする相転移である。",
        "prediction": "absorption breach発生後の債務増加率がbreach前の2倍以上に跳ね上がる。",
        "scenarios": [
            # A/B/Cは全シナリオ5%固定（資本フロー相互作用のため）、Dの水準/成長のみ変える
            {"label": "D UBI=3%・現実的成長率（breachしない想定）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.03)]},
            {"label": "D UBI=4%・現実的成長率（境界付近）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.04)]},
            {"label": "D UBI=5%・現実的成長率（breachする想定）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05)]},
            {"label": "D UBI=5%・楽観的成長率（breachしない想定, 比較用）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_opt"](ubi_start_year=15, ubi_level=0.05)]},
        ]
    },
    # ═══════════════ v4: 対外債務の通貨ミスマッチ 仮説群 ═══════════════
    # forex OFF シナリオは全4国に forex_debt_off=True を付与（v3再現）。realistic 成長既定。
    "H18": {
        "title": "通貨ミスマッチによるtipping point前倒し仮説",
        "statement": "対外債務の通貨ミスマッチを入れると、途上国・新興国のtipping pointがv3よりさらに下がる。自己増幅ループ（通貨安→債務膨張→通貨安）が相転移を前倒しする。",
        "prediction": "途上国Dのtipping pointがUBI=5%（v3）から3%以下に低下。新興国Cも低下。先進国Aは変化なし。",
        "scenarios": [
            {"label": "段階 A→B→C UBI=3%・forex ON（D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.03),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.03),
                                   A["developing_real"]()]},
            {"label": "段階 A→B→C UBI=5%・forex ON（D不参加）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"]()]},
            {"label": "段階 A→B→C UBI=5%・forex OFF（v3比較）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True),
                                   A["developing_real"](forex_debt_off=True)]},
            {"label": "D参加 UBI=3%・forex ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.03),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.03),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.03)]},
        ]
    },
    "H19": {
        "title": "対外債務自己増幅ループの定量化仮説",
        "statement": "通貨安→対外債務膨張→リスクプレミアム上昇→資本流出→さらに通貨安、というループが存在し、一度回り始めると自力では止まらない。",
        "prediction": "途上国Dで為替が一定以上下落した後、debt_ratioの増加率が非線形に加速する。forex OFFでは発生しない。",
        "scenarios": [
            # A/B/C=5%固定、Dの水準/forexのみ変える（4ブロック走行しD抽出）
            {"label": "D UBI=3%・forex ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.03)]},
            {"label": "D UBI=3%・forex OFF",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.03, forex_debt_off=True)]},
            {"label": "D UBI=5%・forex ON（ループ発動想定）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05)]},
            {"label": "D UBI=5%・forex OFF",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05, forex_debt_off=True)]},
        ]
    },
    "H20": {
        "title": "「全ての層が途上国に不利に効く」累積構造仮説",
        "statement": "基軸通貨非対称性（v2）、成長率低下（v3）、対外債務ミスマッチ（v4）の3層が全て途上国に不利に累積する。3層同時の効果が各層単独の増分の和より大きい（非線形累積）。",
        "prediction": "途上国Dの30年後債務比率について、3層同時 > Σ(各層単独の増分)。",
        "scenarios": [
            # 各層を country 構築で制御（reserve=is_reserve_currency, decay=growth_decay_rate=0, forex=forex_debt_off）
            {"label": "S1 v1相当（対称・楽観・forex OFF）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, is_reserve_currency=False, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.05, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["developing_opt"](growth_decay_rate=0.0, forex_debt_off=True)]},
            {"label": "S2 v2のみ（基軸非対称ON・楽観・forex OFF）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.05, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["developing_opt"](growth_decay_rate=0.0, forex_debt_off=True)]},
            {"label": "S3 v3のみ（対称・現実・forex OFF）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, is_reserve_currency=False, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True),
                                   A["developing_real"](forex_debt_off=True)]},
            {"label": "S4 v4のみ（対称・楽観・forex ON）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, is_reserve_currency=False, growth_decay_rate=0.0),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, growth_decay_rate=0.0),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.05, growth_decay_rate=0.0),
                                   A["developing_opt"](growth_decay_rate=0.0)]},
            {"label": "S5 全部ON（基軸非対称・現実・forex ON）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"]()]},
        ]
    },
    "H21": {
        "title": "「累進的UBI」再検証（対外債務込み）",
        "statement": "H15で逆進的UBI > 累進的UBIだったが、対外債務ミスマッチを入れると途上国の危機コストが跳ね上がるため結論が変わりうる。ただしH15の「律速はDではなくA」構造が支配的なら結論は変わらない。",
        "prediction": "律速構造が維持される限り、対外債務を入れても逆進的UBI優位は変わらない。",
        "scenarios": [
            {"label": "逆進的 A3/B5/C5/D7・forex ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.07)]},
            {"label": "均一 全員5%・forex ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05)]},
            {"label": "累進的 A10/B5/C3/D1・forex ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.10),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.03),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.01)]},
            {"label": "逆進的 A3/B5/C5/D7・forex OFF（v3=H15-S1比較）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.07, forex_debt_off=True)]},
        ]
    },
    # ═══════════════ v5: CLT × 対外債務の交差項（通貨経路の迂回）仮説群 ═══════════════
    # clt_mode=True の国は通貨経路を迂回。forex/decay/reserve は v4 と同じ per-scenario 制御。
    "H22": {
        "title": "CLTによる対外債務ループ遮断仮説",
        "statement": "CLTモードは通貨経路を迂回するため、対外債務の自己増幅ループを構造的に遮断する。v4 H19で確認された債務増加率5.1倍の加速が、CLTでは消失または大幅減衰する。",
        "prediction": "途上国D CLTモードで、forex ON/OFFの債務増加率の差が現金型の1/3以下に縮小。",
        "scenarios": [
            {"label": "D=現金UBI5%・forex ON（v4 H19相当）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=10, ubi_level=0.05)]},
            {"label": "D=CLT UBI5%・forex ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True)]},
            {"label": "D=現金UBI5%・forex OFF",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True),
                                   A["developing_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True)]},
            {"label": "D=CLT UBI5%・forex OFF",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, forex_debt_off=True),
                                   A["developing_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True, forex_debt_off=True)]},
        ]
    },
    "H23": {
        "title": "CLTによる三層累積構造の解体仮説",
        "statement": "v4 H20で三層累積が乗算的（D参加時3.45倍）だったのは、成長率低下→通貨安→対外債務膨張の連鎖が成立するから。CLTが通貨経路を遮断すれば、この乗算構造が崩れて加算的になる。",
        "prediction": "CLTモード下での累積項が、各層単独の和の1.5倍以下に低下（現金型3.45倍→CLT型1.5倍以下）。",
        # 注: 引用される乗算累積3.45倍はD参加時のみ顕在化するため、D=5%参加・全員CLTで構成（report明記）。
        "scenarios": [
            {"label": "S1 v1相当（対称・楽観・forexOFF）全CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, clt_mode=True, is_reserve_currency=False, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["developing_opt"](ubi_start_year=15, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0, forex_debt_off=True)]},
            {"label": "S2 v2のみ（基軸非対称ON）全CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0, forex_debt_off=True),
                                   A["developing_opt"](ubi_start_year=15, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0, forex_debt_off=True)]},
            {"label": "S3 v3のみ（現実成長）全CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, clt_mode=True, is_reserve_currency=False, forex_debt_off=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, clt_mode=True, forex_debt_off=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True, forex_debt_off=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05, clt_mode=True, forex_debt_off=True)]},
            {"label": "S4 v4のみ（forex）全CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, clt_mode=True, is_reserve_currency=False, growth_decay_rate=0.0),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0),
                                   A["emerging_opt"](ubi_start_year=10, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0),
                                   A["developing_opt"](ubi_start_year=15, ubi_level=0.05, clt_mode=True, growth_decay_rate=0.0)]},
            {"label": "S5 全部ON 全CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, clt_mode=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, clt_mode=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05, clt_mode=True)]},
        ]
    },
    "H24": {
        "title": "途上国限定CLTの最適性仮説",
        "statement": "CLTの効果は foreign_debt_share が高い国ほど大きい。通貨経路の遮断効果が対外債務ループの強度に比例するため。「途上国CLT・先進国現金」の混合が全員CLTと同程度に有効で先進国の政策柔軟性を保つ。",
        "prediction": "C+DのみCLT（A/B現金）が全員CLTの90%以上の危機削減効果を持つ。",
        "scenarios": [
            {"label": "S1 全員現金（ベースライン）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05)]},
            {"label": "S2 全員CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05, clt_mode=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, clt_mode=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05, clt_mode=True)]},
            {"label": "S3 C+D CLT・A+B現金",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05, clt_mode=True)]},
            {"label": "S4 D のみCLT・他現金",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.05, clt_mode=True)]},
        ]
    },
    "H25": {
        "title": "CLTの「窓を開く」効果仮説",
        "statement": "v3 H16のclosing window問題（成長率低下で導入が遅れるほど不利）がCLTで緩和される。CLTは通貨経路を迂回するため、成長率が低下しても持続可能性が保たれる期間が延びる。",
        "prediction": "現金型でYear10導入が危機となる水準のUBIが、CLT型ではYear20導入でも持続可能。",
        "scenarios": [
            {"label": "S1 全員Y1・現金5%・decay ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=1, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=1, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=1, ubi_level=0.05)]},
            {"label": "S2 全員Y10・現金5%・decay ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=10, ubi_level=0.05),
                                   A["advancedB"](ubi_start_year=10, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=10, ubi_level=0.05)]},
            {"label": "S3 全員Y10・CLT5%・decay ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=10, ubi_level=0.05, clt_mode=True),
                                   A["advancedB"](ubi_start_year=10, ubi_level=0.05, clt_mode=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True),
                                   A["developing_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True)]},
            {"label": "S4 全員Y20・CLT5%・decay ON",
             "countries": lambda: [A["advancedA"](ubi_start_year=20, ubi_level=0.05, clt_mode=True),
                                   A["advancedB"](ubi_start_year=20, ubi_level=0.05, clt_mode=True),
                                   A["emerging_real"](ubi_start_year=20, ubi_level=0.05, clt_mode=True),
                                   A["developing_real"](ubi_start_year=20, ubi_level=0.05, clt_mode=True)]},
        ]
    },
    "H26": {
        "title": "逆進的UBI + CLT混合戦略の最終検証",
        "statement": "v4 H21で「逆進的UBIはシステム最適だがDを犠牲にする」と出た。途上国にCLTを適用すれば、逆進的UBIのシステム最適を維持しつつDの犠牲を回避できるか。",
        "prediction": "逆進的UBI（C=5%CLT, D=7%CLT）でDの債務爆発が消失し、かつ総危機が累進的より少ない。",
        "scenarios": [
            {"label": "S1 逆進 全員現金（v4 H21-S1再現）",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.07)]},
            {"label": "S2 逆進 C+D CLT・A+B現金",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.03),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.05, clt_mode=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.07, clt_mode=True)]},
            {"label": "S3 累進 A10/B5/C3/D1 全現金",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.10),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.03),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.01)]},
            {"label": "S4 累進 全CLT",
             "countries": lambda: [A["advancedA"](ubi_start_year=1, ubi_level=0.10, clt_mode=True),
                                   A["advancedB"](ubi_start_year=5, ubi_level=0.05, clt_mode=True),
                                   A["emerging_real"](ubi_start_year=10, ubi_level=0.03, clt_mode=True),
                                   A["developing_real"](ubi_start_year=15, ubi_level=0.01, clt_mode=True)]},
        ]
    },
}


# ═══════════════════════════════════════════════════════
# OUTPUT FORMATTERS
# ═══════════════════════════════════════════════════════
def format_md(hyp_id, hyp, results):
    lines = []
    lines.append(f"# {hyp_id}: {hyp['title']}")
    lines.append(f"\n**仮説**: {hyp['statement']}")
    lines.append(f"\n**予測**: {hyp['prediction']}")

    for i, (sc, res) in enumerate(zip(hyp["scenarios"], results)):
        lines.append(f"\n## シナリオ{i+1}: {sc['label']}")
        countries = sc["countries"]()
        names = [c.name for c in countries]
        stats = res["stats"]

        header = "| 指標 | " + " | ".join(names) + " |"
        sep = "|---" + "|---" * len(names) + "|"
        lines.append(header)
        lines.append(sep)

        rows = [
            ("最終GDP", "final_gdp", ".4f"),
            ("最終債務比", "final_debt", ".4f"),
            ("最大債務比", "max_debt", ".4f"),
            ("最大インフレ%", "max_inflation_pct", ".2f"),
            ("最終資本", "final_capital", ".4f"),
            ("為替Vol", "exch_volatility", ".5f"),
            ("5年GDP変化%", "gdp_change_5y_pct", ".2f"),
            ("10年GDP変化%", "gdp_change_10y_pct", ".2f"),
            ("30年GDP変化%", "gdp_change_30y_pct", ".2f"),
            ("5年資本Δ", "capital_change_5y", ".4f"),
            ("10年資本Δ", "capital_change_10y", ".4f"),
        ]
        for label, key, fmt in rows:
            vals = []
            for ci in range(len(names)):
                v = stats[f"country_{ci}"].get(key)
                vals.append(f"{v:{fmt}}" if v is not None else "—")
            lines.append(f"| {label} | " + " | ".join(vals) + " |")

        crises = res["crisis_log"]
        if crises:
            unique = list(dict.fromkeys(f"{c['country']}(Y{c['year']})" for c in crises))[:8]
            lines.append(f"\n⚠ 危機検出: {', '.join(unique)}" +
                         (f" +{len(crises)-8}件" if len(crises) > 8 else ""))

        breaches = res.get("absorption_breach_log", [])
        if breaches:
            # 国別に初回 breach（年・gap）を抽出
            first = {}
            for b in breaches:
                if b["idx"] not in first:
                    first[b["idx"]] = b
            summary = ", ".join(
                f"{b['country']}(Y{b['year']}, gap={b['gap']:+.4f})"
                for b in sorted(first.values(), key=lambda x: x["idx"]))
            lines.append(f"\n🔺 吸収閾値超過(初回): {summary}  ／ 総breach {len(breaches)}件")

    return "\n".join(lines)


def run_hypothesis(hyp_id, hyp, params):
    results = []
    for sc in hyp["scenarios"]:
        countries = sc["countries"]()
        sim = simulate(countries, params)
        stats = compute_stats(sim["history"], len(countries))
        results.append({**sim, "stats": stats})
    return results


# ═══════════════════════════════════════════════════════
# SWEEP MODE
# ═══════════════════════════════════════════════════════
def run_sweep(param_name, values, base_params):
    """Sweep a single parameter across values, running all hypotheses."""
    all_results = {}
    for val in values:
        p = deepcopy(base_params)
        # Could be a global param or a country-level param
        if hasattr(p, param_name):
            setattr(p, param_name, val)
        label = f"{param_name}={val}"
        all_results[label] = {}
        for hid, hyp in HYPOTHESES.items():
            all_results[label][hid] = run_hypothesis(hid, hyp, p)
    return all_results


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="UBI × Trilemma Simulator")
    parser.add_argument("--hyp", nargs="*", default=None,
                        help="Hypothesis IDs to run (e.g. H1 H3). Default: all")
    parser.add_argument("--format", choices=["json", "md"], default="md",
                        help="Output format")
    parser.add_argument("--params", nargs="*", default=[],
                        help="Override params: fiscal_mult=1.5 cap_sens=1.2")
    parser.add_argument("--sweep", nargs="+", default=None,
                        help="Sweep mode: param_name val1 val2 val3 ...")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: stdout)")
    # v3: 成長率レジーム切替
    growth_grp = parser.add_mutually_exclusive_group()
    growth_grp.add_argument("--realistic-growth", dest="realistic", action="store_true",
                            default=True, help="現実的成長率（v3既定）")
    growth_grp.add_argument("--optimistic-growth", dest="realistic", action="store_false",
                            help="楽観的成長率（v1/v2互換）")
    parser.add_argument("--no-growth-decay", dest="growth_decay", action="store_false",
                        default=True, help="成長率の趨勢的低下を無効化")
    # v4: 対外債務の通貨ミスマッチ切替
    parser.add_argument("--no-forex-debt", dest="forex_debt", action="store_false",
                        default=True, help="対外債務の通貨ミスマッチ動態を無効化（v3再現）")
    args = parser.parse_args()

    # v3: 成長レジームに応じてアーキタイプを再構築（HYPOTHESES lambda は呼出時に global A を解決）
    global A
    A = archetypes(realistic=args.realistic)

    params = GlobalParams()
    if not args.growth_decay:
        params.growth_decay_enabled = 0.0
    if not args.forex_debt:
        params.forex_debt_enabled = 0.0
    for p in args.params:
        k, v = p.split("=")
        # v5: bool 文字列（true/false）を 1.0/0.0 に変換（例: clt_dynamic_cost=true）
        if v.lower() in ("true", "false"):
            setattr(params, k, 1.0 if v.lower() == "true" else 0.0)
        else:
            setattr(params, k, float(v))

    hyp_ids = args.hyp if args.hyp else list(HYPOTHESES.keys())

    if args.sweep:
        param_name = args.sweep[0]
        values = [float(v) for v in args.sweep[1:]]
        sweep_results = run_sweep(param_name, values, params)
        output = json.dumps(sweep_results, ensure_ascii=False, indent=2, default=str)
    else:
        output_parts = []
        for hid in hyp_ids:
            hyp = HYPOTHESES[hid]
            results = run_hypothesis(hid, hyp, params)
            if args.format == "json":
                output_parts.append({
                    "hypothesis": hid,
                    "title": hyp["title"],
                    "statement": hyp["statement"],
                    "prediction": hyp["prediction"],
                    "scenarios": [
                        {"label": sc["label"],
                         "countries": [c.name for c in sc["countries"]()],
                         "stats": res["stats"],
                         "crisis_count": len(res["crisis_log"]),
                         "breach_count": len(res["absorption_breach_log"]),
                         "first_breach": _first_breach_by_country(res["absorption_breach_log"]),
                         "final_states": res["final_states"]}
                        for sc, res in zip(hyp["scenarios"], results)
                    ]
                })
            else:
                output_parts.append(format_md(hid, hyp, results))

        if args.format == "json":
            output = json.dumps(output_parts, ensure_ascii=False, indent=2)
        else:
            output = "\n\n---\n\n".join(output_parts)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()