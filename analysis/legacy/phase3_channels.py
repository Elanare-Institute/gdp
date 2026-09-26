"""Which channel carries the fx effect, and does it survive equal fiscal cost?

The endogenous stress measure responds to fx_share while the static ones do
not. This script names the component responsible, then removes the fiscal-cost
gap between CLT and cash to test whether the effect is anything more than an
amplification of that gap.
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

from legacy.v7.core import simulate  # noqa: E402
from legacy.v7.evolution import _cached_calibration  # noqa: E402
from legacy.v7.params import G  # noqa: E402
from legacy.v7.population import make_population  # noqa: E402

RESULTS = ROOT / "results" / "phase3"
FX_GRID = (0.0, 0.25, 0.5, 0.75, 0.85)
COMPONENTS = ("debt", "risk_premium", "depreciation")


def pv_components(bloc, g: G, modality: str, size: float,
                  years: int = 30, discount: float = 0.03) -> dict[str, float]:
    """Present value of each macro deviation from the no-transfer path."""
    base = simulate([replace(bloc, modality="none", start=999.0)], g)["history"][:years]
    run = simulate([replace(bloc, modality=modality, size=size, start=1.0)],
                   g)["history"][:years]
    weights = [(1.0 + discount) ** -t for t in range(len(base))]
    pv = lambda xs: sum(w * x for w, x in zip(weights, xs))
    return {
        "debt": pv([r["d_0"] - q["d_0"] for r, q in zip(run, base)]),
        "risk_premium": pv([max(0.0, r["r_0"] - bloc.rate) - max(0.0, q["r_0"] - bloc.rate)
                            for r, q in zip(run, base)]),
        "depreciation": pv([max(0.0, r["e_0"] - 1.0) - max(0.0, q["e_0"] - 1.0)
                            for r, q in zip(run, base)]),
    }


def sweep(n_blocs: int = 12, seed: int = 42) -> list[dict[str, Any]]:
    """Per-bloc fx counterfactual, with and without the fiscal gap closed."""
    g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
          clt_build_rate=0.02)
    population = make_population(seed)[:n_blocs]
    rows: list[dict[str, Any]] = []

    # Three ways to compare. Equalising the fiscal cost can be done from
    # either side, and the two are not interchangeable: scaling CLT up also
    # scales its construction imports, whereas shrinking cash leaves the
    # spending structure untouched. Both are reported.
    for mode in ("welfare_equivalent", "equal_fiscal_scale_clt",
                 "equal_fiscal_shrink_cash"):
        for bloc in population:
            cal = _cached_calibration(bloc, g, 0.05)
            clt_size, cash_size = cal["clt"], cal["cash_t"]
            if mode == "equal_fiscal_scale_clt" and cal["_fiscal_clt"] > 0:
                clt_size *= cal["_fiscal_cash_t"] / cal["_fiscal_clt"]
            elif mode == "equal_fiscal_shrink_cash" and cal["_fiscal_cash_t"] > 0:
                cash_size *= cal["_fiscal_clt"] / cal["_fiscal_cash_t"]
            for fx in FX_GRID:
                probe = replace(bloc, fx_share=fx)
                cash = pv_components(probe, g, "cash_t", cash_size)
                clt = pv_components(probe, g, "clt", clt_size)
                row = {"bloc": bloc.name, "group": bloc.name[0], "cf_fx": fx,
                       "mode": mode}
                total = 0.0
                for key in COMPONENTS:
                    row[f"d_{key}"] = clt[key] - cash[key]
                    total += row[f"d_{key}"]
                row["d_total"] = total
                rows.append(row)
    return rows


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in ("welfare_equivalent", "equal_fiscal_scale_clt",
                  "equal_fiscal_shrink_cash"):
        subset = [r for r in rows if r["mode"] == label]
        slopes: dict[str, list[float]] = {k: [] for k in (*COMPONENTS, "total")}
        for name in sorted({r["bloc"] for r in subset}):
            per_bloc = sorted([r for r in subset if r["bloc"] == name],
                              key=lambda r: r["cf_fx"])
            fx = np.array([r["cf_fx"] for r in per_bloc])
            for key in slopes:
                ys = np.array([r[f"d_{key}"] for r in per_bloc])
                slopes[key].append(float(np.polyfit(fx, ys, 1)[0]))
        out[label] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                      for k, v in slopes.items()}
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = sweep()
    path = RESULTS / "fx_channels.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  wrote {path.name} ({len(rows)} rows)")

    summary = report(rows)
    (RESULTS / "fx_channels.json").write_text(json.dumps(summary, indent=1),
                                              encoding="utf-8")

    print("\nSlope of each stress component against counterfactual fx:")
    print(f"  {'component':14s} {'welfare-equiv':>14s} {'scale CLT up':>14s} "
          f"{'shrink cash':>14s}")
    for key in (*COMPONENTS, "total"):
        cells = "".join(
            f"{summary[m][key]['mean']:+14.4f}"
            for m in ("welfare_equivalent", "equal_fiscal_scale_clt",
                      "equal_fiscal_shrink_cash"))
        print(f"  {key:14s} {cells}")

    print("\nReading: a negative slope means CLT's advantage widens with fx.")
    print("  The two equal-cost variants are not equivalent. Scaling CLT up to")
    print("  cash's budget multiplies its construction imports, so any change")
    print("  in slope there confounds the cost gap with a larger build.")
    print("  Shrinking cash to CLT's budget leaves both spending structures")
    print("  intact and is the cleaner test.")


if __name__ == "__main__":
    main()
