"""Does cash_cheap differ from CLT in the debt channel, or only in depreciation?

cash_cheap spends exactly what CLT spends but pays it out as cash. If the two
move the debt path identically, whatever separates them must come from where
the money goes rather than from how much is spent — which is the claim the
three-way control was built to test.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.evolution import _cached_calibration  # noqa: E402
from legacy.v7.heterogeneous import hetero_household_block, welfare_measure  # noqa: E402
from legacy.v7.params import G  # noqa: E402
from legacy.v7.population import make_population  # noqa: E402
from analysis.legacy.phase3_channels import pv_components  # noqa: E402

RESULTS = ROOT / "results" / "phase3"
FX_GRID = (0.0, 0.25, 0.5, 0.75, 0.85)
COMPONENTS = ("debt", "risk_premium", "depreciation")


def globals_for() -> G:
    return G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
             clt_build_rate=0.02)


def levels(n_blocs: int = 12, seed: int = 42) -> list[dict[str, Any]]:
    """Fiscal cost, leakage and welfare of each modality, side by side."""
    g = globals_for()
    rows: list[dict[str, Any]] = []
    for bloc in make_population(seed)[:n_blocs]:
        cal = _cached_calibration(bloc, g, 0.05)
        base = hetero_household_block(replace(bloc, modality="none"), g,
                                      bloc.gdp, 1.0, bloc.rate, False, 0.0)
        sizes = {"cash_t": cal["cash_t"], "clt": cal["clt"],
                 "cash_cheap": cal["_fiscal_clt"]}
        row: dict[str, Any] = {"bloc": bloc.name, "group": bloc.name[0],
                               "fx_share": bloc.fx_share}
        for name, size in sizes.items():
            out = hetero_household_block(replace(bloc, modality=name), g,
                                         bloc.gdp, 1.0, bloc.rate, True, size)
            row[f"{name}_fiscal"] = 100 * out.fiscal / bloc.gdp
            row[f"{name}_leak"] = 100 * (out.imp_c - base.imp_c + out.imp_gov) / bloc.gdp
            row[f"{name}_welfare"] = welfare_measure(out, "utilitarian")
            row[f"{name}_size"] = size
        row["fiscal_gap_abs"] = abs(row["clt_fiscal"] - row["cash_cheap_fiscal"])
        row["fiscal_gap_rel"] = (row["fiscal_gap_abs"] / row["clt_fiscal"]
                                 if row["clt_fiscal"] else 0.0)
        rows.append(row)
    return rows


def channel_slopes(n_blocs: int = 12, seed: int = 42) -> list[dict[str, Any]]:
    """Per-component fx slope for CLT and for cash_cheap, each against cash_t."""
    g = globals_for()
    rows: list[dict[str, Any]] = []
    for bloc in make_population(seed)[:n_blocs]:
        cal = _cached_calibration(bloc, g, 0.05)
        sizes = {"clt": cal["clt"], "cash_cheap": cal["_fiscal_clt"]}
        for name, size in sizes.items():
            per_fx: dict[str, list[float]] = {k: [] for k in COMPONENTS}
            for fx in FX_GRID:
                probe = replace(bloc, fx_share=fx)
                cash = pv_components(probe, g, "cash_t", cal["cash_t"])
                other = pv_components(probe, g, name, size)
                for key in COMPONENTS:
                    per_fx[key].append(other[key] - cash[key])
            row = {"bloc": bloc.name, "group": bloc.name[0], "modality": name}
            for key in COMPONENTS:
                row[f"slope_{key}"] = float(np.polyfit(FX_GRID, per_fx[key], 1)[0])
                row[f"level_{key}"] = float(np.mean(per_fx[key]))
            row["slope_total"] = sum(row[f"slope_{k}"] for k in COMPONENTS)
            rows.append(row)
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    level_rows = levels()
    (RESULTS / "cheap_levels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in level_rows) + "\n", encoding="utf-8")
    print(f"  wrote cheap_levels.jsonl ({len(level_rows)} rows)")

    gaps = np.array([r["fiscal_gap_rel"] for r in level_rows])
    print(f"\nFiscal cost match, CLT vs cash_cheap: "
          f"max relative gap {gaps.max():.2e} over {len(gaps)} blocs")

    slope_rows = channel_slopes()
    (RESULTS / "cheap_channels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in slope_rows) + "\n", encoding="utf-8")
    print(f"  wrote cheap_channels.jsonl ({len(slope_rows)} rows)")

    print("\nfx slope of each component, relative to cash_t:")
    print(f"  {'component':14s} {'CLT':>10s} {'cash_cheap':>12s} {'difference':>12s}")
    summary: dict[str, Any] = {}
    for key in (*COMPONENTS, "total"):
        clt = np.array([r[f"slope_{key}"] for r in slope_rows if r["modality"] == "clt"])
        cheap = np.array([r[f"slope_{key}"] for r in slope_rows
                          if r["modality"] == "cash_cheap"])
        diff = clt - cheap
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        summary[key] = {"clt": float(clt.mean()), "cash_cheap": float(cheap.mean()),
                        "difference": float(diff.mean()),
                        "t": float(diff.mean() / se) if se else float("nan")}
        print(f"  {key:14s} {clt.mean():+10.4f} {cheap.mean():+12.4f} "
              f"{diff.mean():+12.4f}")

    (RESULTS / "cheap_channels.json").write_text(json.dumps(summary, indent=1),
                                                 encoding="utf-8")
    print("\nPaired t on the difference (per bloc):")
    for key in (*COMPONENTS, "total"):
        print(f"  {key:14s} t = {summary[key]['t']:+.2f}")


if __name__ == "__main__":
    main()
