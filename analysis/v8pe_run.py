"""post_employment: does the trap appear once labour income goes?

Phase U found a trapped region only under three conditions together — no
rationing tolerated, a subsistence floor of 75% or more, and a transfer that
is not indexed to prices. The question here is what happens to those
conditions when labour demand falls worldwide.

The decisive one is indexation. `income_linkage` moves to the price-linked
side on its own as labour income is replaced by transfers, but `indexation` is
a design choice and does not. So the test that matters is whether the trap
appears **with `cpi_indexed`**, which is what most OECD minimum income schemes
actually do (Van Mechelen & Marchal 2012).

Reported as a response surface in two dimensions — automation against the
subsistence floor — because neither is an estimate and neither gets a single
headline value.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core import simulate  # noqa: E402
from model.household import calibrate  # noqa: E402
from model.params import G  # noqa: E402
from model.twoaxis import grid, locate_archetypes  # noqa: E402

RESULTS = ROOT / "results" / "v8pe"
STEPS = 5
#: Neither is an estimate; both are swept. 0.0 is the baseline series.
AUTOMATION = (0.0, 0.2, 0.4, 0.6, 0.8)
#: Phase U's range, minus 0.65: a household that has lost its labour income
#: spends nearly all of what is left on necessities, so the value that assumed
#: 35% free income no longer describes it. Kept for regression only.
FLOOR_SHARES = (0.65, 0.75, 0.85, 0.95)
#: The decisive comparison. cpi_indexed is the main series; fixed_nominal is
#: what happens if indexation is neglected.
INDEXATIONS = ("cpi_indexed", "fixed_nominal")
TOLERANCES = (0.0, 0.05, 0.10)
U_CANDIDATES = (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12)
BASE_FLOOR = 0.65


def globals_for(automation: float, floor_share: float, indexation: str) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation=indexation, settlement=True, trade_elasticity=1.0,
             policy_response=1.0, contagion_strength=1.0, flight_to_reserve=0.5,
             income_linkage="cpi_linked",
             subsistence_scale=floor_share / BASE_FLOOR,
             automation_max=automation)


def _with_ubi(blocs, g: G, u: float):
    if u <= 0:
        return list(blocs)
    return [replace(b, modality="ubi", size=calibrate(b, g, u)["ubi"], start=1.0)
            for b in blocs]


def scan(automation: float, floor_share: float, indexation: str) -> dict[str, Any]:
    """Need, affordability and the three categories, at one point of the sweep."""
    g = globals_for(automation, floor_share, indexation)
    blocs = grid(STEPS)

    needs: dict[float, dict[str, float]] = {}
    rationing: dict[float, dict[str, float]] = {}
    for u in U_CANDIDATES:
        result = simulate(_with_ubi(blocs, g, u), g)
        budget, settlement = result["budget"], result["settlement"]
        n = max(1, len(settlement))
        needs[u] = {b.name: max(row[f"need_{i}"] for row in budget)
                    for i, b in enumerate(blocs)}
        rationing[u] = {b.name: sum(row[f"share_{i}"] for row in settlement) / n
                        for i, b in enumerate(blocs)}

    cells: dict[str, Any] = {}
    for b in blocs:
        credit, selfsuf = (float(x) for x in
                           b.name.replace("c", "").replace("s", "").split("_"))
        # Need, solved against the transfer that creates it.
        need = needs[0.0][b.name]
        fixed_point = None
        for _ in range(6):
            candidate = next((u for u in U_CANDIDATES if u >= need), None)
            if candidate is None:
                break
            revised = needs[candidate][b.name]
            if revised <= candidate:
                fixed_point = candidate
                break
            need = revised

        affordable = {}
        trapped = {}
        for tol in TOLERANCES:
            allowed = [u for u in U_CANDIDATES
                       if rationing[u][b.name] <= tol + 1e-12]
            limit = max(allowed) if allowed else None
            affordable[f"{tol:g}"] = limit
            trapped[f"{tol:g}"] = (
                needs[0.0][b.name] > 0
                and rationing[0.0][b.name] <= 1e-12
                and (fixed_point is None or limit is None or fixed_point > limit))

        cells[b.name] = {
            "credit": credit, "self_sufficiency": selfsuf, "reserve": b.reserve,
            "baseline_rationing": rationing[0.0][b.name],
            "already_constrained": rationing[0.0][b.name] > 1e-12,
            "u_needed_static": needs[0.0][b.name],
            "u_needed_fixed_point": fixed_point,
            "no_fixed_point": fixed_point is None and needs[0.0][b.name] > 0,
            "u_affordable": affordable,
            "trapped_by_tolerance": trapped,
        }

    return {
        "automation": automation, "floor_share": floor_share,
        "indexation": indexation, "cells": cells,
        "trapped_by_tolerance": {
            f"{tol:g}": sum(1 for c in cells.values()
                            if c["trapped_by_tolerance"][f"{tol:g}"])
            for tol in TOLERANCES},
        "already_constrained_count": sum(1 for c in cells.values()
                                         if c["already_constrained"]),
        "no_fixed_point_count": sum(1 for c in cells.values()
                                    if c["no_fixed_point"]),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "archetype_positions": locate_archetypes(),
        "automation_levels": list(AUTOMATION),
        "floor_shares": list(FLOOR_SHARES),
        "indexations": list(INDEXATIONS),
        "tolerances": list(TOLERANCES),
        "scans": {f"{a:g}|{f:g}|{ix}": scan(a, f, ix)
                  for a in AUTOMATION for f in FLOOR_SHARES
                  for ix in INDEXATIONS},
    }
    path = RESULTS / "postemp.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    for ix in INDEXATIONS:
        print(f"=== Trapped positions, {ix} (of 25) ===")
        print("  automation x subsistence floor, no rationing tolerated\n")
        print(f"  {'auto':>6s} " + "".join(f"{f:>8.2f}" for f in FLOOR_SHARES))
        for a in AUTOMATION:
            row = "".join(
                f"{report['scans'][f'{a:g}|{f:g}|{ix}']['trapped_by_tolerance']['0']:8d}"
                for f in FLOOR_SHARES)
            print(f"  {a:6.1f} {row}")
        print()

    print("=== Baseline against post_employment (floor 0.85, cpi_indexed) ===")
    print(f"  {'auto':>6s} {'trapped':>9s} {'already':>9s} {'no-fp':>7s} "
          f"{'max need':>10s}")
    for a in AUTOMATION:
        sc = report["scans"][f"{a:g}|0.85|cpi_indexed"]
        need = max(c["u_needed_static"] for c in sc["cells"].values())
        print(f"  {a:6.1f} {sc['trapped_by_tolerance']['0']:9d} "
              f"{sc['already_constrained_count']:9d} "
              f"{sc['no_fixed_point_count']:7d} {need:10.4f}")

    print("\n=== Does tolerance still dissolve the trap? (floor 0.85) ===")
    print(f"  {'auto':>6s} {'indexation':>14s} " +
          "".join(f"{'tol ' + t:>10s}" for t in ("0%", "5%", "10%")))
    for a in AUTOMATION:
        for ix in INDEXATIONS:
            sc = report["scans"][f"{a:g}|0.85|{ix}"]
            row = "".join(f"{sc['trapped_by_tolerance'][t]:10d}"
                          for t in ("0", "0.05", "0.1"))
            print(f"  {a:6.1f} {ix:>14s} {row}")


if __name__ == "__main__":
    main()
