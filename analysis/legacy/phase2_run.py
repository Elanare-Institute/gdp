"""Phase 2 experiments: who benefits from a CLT when tenancy is scarce."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.heterogeneous import (  # noqa: E402
    calibrate_hetero, hetero_household_block, rent_incidence,
)
from legacy.v7.household import calibrate  # noqa: E402
from legacy.v7.params import G, archetypes  # noqa: E402

RESULTS = ROOT / "results" / "phase2"
U_LEVELS = (0.03, 0.05, 0.10)          # fixed per the Phase 1 handoff
MAIN_CAPACITY = 1.0
"""Main series. Chosen for comparability with v6 rather than for cost:
picking the cheapest capacity would mean selecting the result via a
parameter."""

CAPACITIES = (1.0, 0.6, 0.3, 0.1)      # 0.6/0.3 sensitivity, 0.1 see report
MAIN_MODE = "utilitarian"
"""Main calibration target: cash reaches every constrained household, so the
comparison is only like-for-like against the whole block's welfare."""

MAIN_RULE = "priority_low_income"      # fixed; lottery is one sensitivity run
BLOCS = ("A", "B", "C", "D")


def main_globals(**overrides) -> G:
    """Globals for the main series, with CLT rent credited to the trust."""
    base = dict(n_types=5, clt_capacity=MAIN_CAPACITY,
                allocation_rule=MAIN_RULE, clt_rent_to_landlords=False)
    base.update(overrides)
    return G(**base)


def incidence_experiment(k: int = 5) -> list[dict[str, Any]]:
    """Type-level incidence across blocs, capacities, rules and u levels."""
    rows: list[dict[str, Any]] = []
    for key in BLOCS:
        b = archetypes()[key]
        for u in U_LEVELS:
            for capacity in CAPACITIES:
                for rule in ("priority_low_income", "lottery"):
                    g = main_globals(n_types=k, clt_capacity=capacity,
                                     allocation_rule=rule)
                    cal = calibrate_hetero(b, g, u, MAIN_MODE)
                    base = hetero_household_block(
                        replace(b, modality="none"), g, b.gdp, 1.0, b.rate, False, 0.0)
                    clt = hetero_household_block(
                        replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, cal["clt"])
                    cash = hetero_household_block(
                        replace(b, modality="cash_t"), g, b.gdp, 1.0, b.rate,
                        True, cal["cash_t"])

                    for label, out in (("clt", clt), ("cash_t", cash)):
                        for name, detail in rent_incidence(out, base).items():
                            type_name, status = name.split(":")
                            rows.append({
                                "bloc": key, "u": u, "capacity": capacity,
                                "rule": rule, "modality": label,
                                "at_ceiling": bool(cal.get("_at_ceiling_clt")),
                                "type": type_name, "status": status,
                                "pop_share": detail["pop_share"],
                                "d_rent": detail["d_rent"],
                                "d_utility": detail["d_utility"],
                                "rent_share_of_income": detail["rent_share_of_income"],
                                "pH": out.pH, "pH_base": base.pH,
                                "fiscal_pct_gdp": 100 * out.fiscal / b.gdp,
                            })
    return rows


def calibration_modes(k: int = 5) -> list[dict[str, Any]]:
    """Fiscal cost of matching welfare under each calibration target."""
    rows: list[dict[str, Any]] = []
    for key in BLOCS:
        b = archetypes()[key]
        for u in U_LEVELS:
            for capacity in CAPACITIES:
                g = main_globals(n_types=k, clt_capacity=capacity)
                for mode in ("tenant", "utilitarian"):
                    cal = calibrate_hetero(b, g, u, mode)
                    rows.append({
                        "bloc": key, "u": u, "capacity": capacity, "mode": mode,
                        "at_ceiling": bool(cal.get("_at_ceiling_clt")),
                        "clt_size": cal["clt"],
                        "fiscal_clt": 100 * cal["_fiscal_clt"],
                        "fiscal_cash_t": 100 * cal["_fiscal_cash_t"],
                        "fiscal_ubi": 100 * cal["_fiscal_ubi"],
                    })
    return rows


def build_rate_path(k: int = 5, years: int = 30) -> list[dict[str, Any]]:
    """Time path of CLT coverage, rent and leakage under a build-rate cap."""
    rows: list[dict[str, Any]] = []
    for key in BLOCS:
        b = archetypes()[key]
        for rate in (None, 0.02, 0.005):
            g_cal = main_globals(n_types=k)
            cal = calibrate_hetero(b, g_cal, 0.05, MAIN_MODE)
            size = cal["clt"]
            for year in range(years + 1):
                g = main_globals(n_types=k, clt_build_rate=rate)
                base = hetero_household_block(
                    replace(b, modality="none"), g, b.gdp, 1.0, b.rate, False, 0.0)
                out = hetero_household_block(
                    replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, size,
                    years_since_start=float(year))
                rows.append({
                    "bloc": key, "build_rate": rate, "year": year,
                    "coverage": out.tenant_share,
                    "clt_units_frac": (out.fiscal / b.gdp) / max(cal["_fiscal_clt"], 1e-12),
                    "pH": out.pH, "pH_base": base.pH,
                    "leak_pct_gdp": 100 * (out.imp_c - base.imp_c + out.imp_gov) / b.gdp,
                    "fiscal_pct_gdp": 100 * out.fiscal / b.gdp,
                })
    return rows


def v6_comparison(k: int = 5) -> list[dict[str, Any]]:
    """Representative (v6) versus heterogeneous outcomes, side by side."""
    rows: list[dict[str, Any]] = []
    for key in BLOCS:
        b = archetypes()[key]
        for u in U_LEVELS:
            v6_cal = calibrate(b, G(), u)
            for capacity in CAPACITIES:
                g = main_globals(n_types=k, clt_capacity=capacity)
                het_cal = calibrate_hetero(b, g, u, MAIN_MODE)
                base = hetero_household_block(
                    replace(b, modality="none"), g, b.gdp, 1.0, b.rate, False, 0.0)
                clt = hetero_household_block(
                    replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True, het_cal["clt"])
                rows.append({
                    "bloc": key, "u": u, "capacity": capacity,
                    "at_ceiling": bool(het_cal.get("_at_ceiling_clt")),
                    "v6_clt_size": v6_cal["clt"],
                    "het_clt_size": het_cal["clt"],
                    "v6_fiscal_clt": 100 * v6_cal["_fiscal_clt"],
                    "het_fiscal_clt": 100 * het_cal["_fiscal_clt"],
                    "pH": clt.pH, "pH_base": base.pH,
                })
    return rows


def present_value_calibration(k: int = 5) -> list[dict[str, Any]]:
    """Snapshot versus discounted-present-value calibration under a build cap."""
    from legacy.v7.heterogeneous import discounted_calibrate

    rows: list[dict[str, Any]] = []
    for key in BLOCS:
        b = archetypes()[key]
        for build_rate in (None, 0.02, 0.005):
            g = main_globals(n_types=k, clt_build_rate=build_rate)
            snapshot = calibrate_hetero(b, g, 0.05, MAIN_MODE)
            for discount in (0.01, 0.03, 0.05):
                pv = discounted_calibrate(b, g, 0.05, MAIN_MODE,
                                          years=30, rate=discount)
                rows.append({
                    "bloc": key, "build_rate": build_rate, "discount": discount,
                    "snapshot_clt": snapshot["clt"], "pv_clt": pv["clt"],
                    "ratio": pv["clt"] / snapshot["clt"],
                    "at_ceiling": bool(pv.get("_at_ceiling_clt")),
                    "snapshot_fiscal": 100 * snapshot["_fiscal_clt"],
                    "pv_fiscal": 100 * pv["_fiscal_clt"],
                })
    return rows


def rent_attribution_sensitivity(k: int = 5) -> list[dict[str, Any]]:
    """Main series (rent to the trust) against the v6 treatment.

    Also sweeps the annualisation share, since that constant is a modelling
    choice rather than an estimate.
    """
    import model.heterogeneous as het

    rows: list[dict[str, Any]] = []
    original = het.CLT_RENT_OFFSET_SHARE
    for share in (0.0, 0.05, 0.10):
        het.CLT_RENT_OFFSET_SHARE = share
        for key in BLOCS:
            b = archetypes()[key]
            g = main_globals(n_types=k)
            cal = calibrate_hetero(b, g, 0.05, MAIN_MODE)
            rows.append({
                "bloc": key, "sweep": "offset_share", "offset_share": share,
                "rent_to_landlords": False,
                "clt_size": cal["clt"], "fiscal_clt": 100 * cal["_fiscal_clt"],
            })
    het.CLT_RENT_OFFSET_SHARE = original

    for key in BLOCS:
        b = archetypes()[key]
        for to_landlords in (False, True):
            g = main_globals(n_types=k, clt_rent_to_landlords=to_landlords)
            cal = calibrate_hetero(b, g, 0.05, MAIN_MODE)
            base = hetero_household_block(replace(b, modality="none"), g,
                                          b.gdp, 1.0, b.rate, False, 0.0)
            clt = hetero_household_block(replace(b, modality="clt"), g,
                                         b.gdp, 1.0, b.rate, True, cal["clt"])
            rows.append({
                "bloc": key, "sweep": "attribution",
                "rent_to_landlords": to_landlords,
                "clt_size": cal["clt"], "fiscal_clt": 100 * cal["_fiscal_clt"],
                "landlord_rent": clt.rent, "landlord_rent_base": base.rent,
                "pH": clt.pH,
            })
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, fn in (("incidence", incidence_experiment),
                     ("calibration_modes", calibration_modes),
                     ("build_rate_path", build_rate_path),
                     ("v6_comparison", v6_comparison),
                     ("present_value_calibration", present_value_calibration),
                     ("rent_attribution", rent_attribution_sensitivity),
                     ("transition_costs", transition_costs)):
        rows = fn()
        path = RESULTS / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"  wrote {path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()


def transition_costs(k: int = 5) -> list[dict[str, Any]]:
    """Cost of moving to CLT while holding welfare at the cash-equivalent level.

    This is the Phase 3 framing: CLT supplies what the build rate allows and
    cash covers the rest, so the question becomes which structures find the
    transition worth paying for.
    """
    from legacy.v7.mix import (
        compare_with_cash, max_attainable_welfare, transition_cost, transition_path,
    )

    rows: list[dict[str, Any]] = []
    for key in BLOCS:
        b = archetypes()[key]
        free = main_globals(n_types=k)
        cal = calibrate_hetero(b, free, 0.05, MAIN_MODE)
        target, clt_size = cal["_U"], cal["clt"]

        for build_rate in (0.05, 0.02, 0.01, 0.005):
            g = main_globals(n_types=k, clt_build_rate=build_rate)
            path = transition_path(b, g, target, clt_size, years=30, mode=MAIN_MODE)
            for discount in (0.01, 0.03, 0.05):
                cost = transition_cost(path, discount)
                rows.append({
                    "bloc": key, "build_rate": build_rate, "discount": discount,
                    "clt_size": clt_size, **cost,
                    "cash_year0": path[0].cash_size,
                    **{f"vs_cash_{k}": v for k, v in
                       compare_with_cash(b, g, target, clt_size, cal["cash_t"],
                                         years=30, discount=discount,
                                         mode=MAIN_MODE).items()},
                    **{f"bound_{k}": v for k, v in
                       max_attainable_welfare(b, g, years=30, discount=discount,
                                              mode=MAIN_MODE).items()
                       if k in ("clt_max_pv", "cash_pv", "gap", "clt_can_match")},
                    "years_to_complete": next(
                        (p.year for p in path if p.cash_size <= 1e-9), None),
                })
    return rows
