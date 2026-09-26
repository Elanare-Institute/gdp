"""Counterfactual in fx_share: is the slope in figure 2 fx, or group structure?

In the population run, fx_share is correlated with everything else that
distinguishes a developing bloc from a reserve issuer. A regression across
blocs therefore cannot say whether fx itself matters. Holding every other
parameter fixed and moving fx alone answers that directly.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.evolution import mix_outcome, stress_of  # noqa: E402
from legacy.v7.params import G  # noqa: E402
from legacy.v7.population import group_of, make_population  # noqa: E402

RESULTS = ROOT / "results" / "phase3"
FX_GRID = (0.0, 0.25, 0.5, 0.75, 0.85)
SOURCING_GRID = (0.0, 0.5, 1.0)
CASH_MIX = [0.0, 1.0, 0.0, 0.0]
CLT_MIX = [0.0, 0.0, 0.0, 1.0]


def base_globals(sourcing: float = 0.0) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
             clt_build_rate=0.02, sourcing_cost_kappa=0.25 if sourcing > 0 else 0.0)


def sweep(seed: int = 42) -> list[dict[str, Any]]:
    """For each bloc, move fx alone across the grid and record the deltas."""
    population = make_population(seed)
    rows: list[dict[str, Any]] = []

    for sourcing in SOURCING_GRID:
        g = base_globals(sourcing)
        for bloc in population:
            for fx in FX_GRID:
                # Only fx_share moves; every structural parameter is held at
                # this bloc's own value, so the comparison is within-bloc.
                probe = replace(bloc, fx_share=fx, clt_domestic_sourcing=sourcing)
                cash = mix_outcome(probe, g, CASH_MIX)
                clt = mix_outcome(probe, g, CLT_MIX)
                d_fiscal = clt["fiscal_pct_gdp"] - cash["fiscal_pct_gdp"]
                d_leak = clt["leak_pct_gdp"] - cash["leak_pct_gdp"]
                rows.append({
                    "bloc": bloc.name, "group": group_of(bloc),
                    "own_fx": bloc.fx_share, "cf_fx": fx,
                    "sourcing": sourcing,
                    "d_fiscal": d_fiscal, "d_leak": d_leak,
                    # Endogenous stress: fx acts through the model's own debt
                    # revaluation rather than an imposed weight.
                    "d_net_endogenous": (stress_of(probe, g, CLT_MIX)
                                         - stress_of(probe, g, CASH_MIX)),
                    # The legacy weighting, kept so the two stress definitions
                    # can be compared like for like.
                    "d_net_weighted": d_fiscal + (1.0 + 2.0 * fx) * d_leak,
                })
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = sweep()
    path = RESULTS / "fx_counterfactual.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  wrote {path.name} ({len(rows)} rows)")

    import numpy as np

    print("\nWithin-bloc effect of fx alone (sourcing = 0):")
    print(f"  {'metric':12s} {'slope vs cf_fx':>15s}")
    subset = [r for r in rows if r["sourcing"] == 0.0]
    fx = np.array([r["cf_fx"] for r in subset])
    for metric in ("d_fiscal", "d_leak", "d_net_endogenous", "d_net_weighted"):
        ys = np.array([r[metric] for r in subset])
        slope = np.polyfit(fx, ys, 1)[0]
        print(f"  {metric:12s} {slope:15.4f}")

    print("\nCompare: cross-bloc slope at each bloc's own fx (the figure-2 view):")
    own = [r for r in rows if r["sourcing"] == 0.0 and abs(r["cf_fx"] - r["own_fx"]) < 0.13]
    if own:
        fx_own = np.array([r["own_fx"] for r in own])
        for metric in ("d_fiscal", "d_leak"):
            ys = np.array([r[metric] for r in own])
            print(f"  {metric:12s} {np.polyfit(fx_own, ys, 1)[0]:15.4f}")

    print("\nEffect of domestic sourcing on the fx slope of leakage:")
    for sourcing in SOURCING_GRID:
        subset = [r for r in rows if r["sourcing"] == sourcing]
        fx = np.array([r["cf_fx"] for r in subset])
        ys = np.array([r["d_leak"] for r in subset])
        print(f"  sourcing={sourcing:.1f}: slope={np.polyfit(fx, ys, 1)[0]:+.4f} "
              f"mean={ys.mean():+.4f}")


if __name__ == "__main__":
    main()
