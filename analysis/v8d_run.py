"""Phase D: the settlement constraint on the credit x self-sufficiency plane.

Three questions, each answered on the plane rather than on the four
archetypes, where credit and self-sufficiency move together and cannot be told
apart:

  where      which blocs are rationed, and when
  how        whether they run out of reserves or run up debt
  contagion  what changes when a default takes its neighbours with it

Reported as pressure and as paths, never as a verdict: the model covers the
run-up to a bloc becoming unable to sustain itself, and stops there.
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

RESULTS = ROOT / "results" / "v8d"
STEPS = 5
MODALITIES = ("none", "ubi", "cash_t", "clt")
#: Centred where defaults are still few, so contagion has room to matter.
#: At u = 0.06 more than two thirds of the plane has already gone without it,
#: and the effect saturates; that level is kept as an upper reference only.
U_GRID = (0.02, 0.025, 0.03, 0.035, 0.04, 0.06)
CONTAGION = (0.0, 1.0)
MAIN_U = 0.03


def globals_for(contagion: float = 0.0, **kw) -> G:
    """Main series. The reserve issuer gets no contagion exemption: whether
    it is spared has to come out of the two-axis distance and the flight to
    its currency, not out of a rule that says so."""
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation="cpi_indexed", settlement=True, trade_elasticity=1.0,
             policy_response=1.0, contagion_strength=contagion,
             flight_to_reserve=0.5, **kw)


def _programme(blocs, g: G, modality: str, u: float):
    if modality == "none":
        return list(blocs)
    return [replace(b, modality=modality, size=calibrate(b, g, u)[modality],
                    start=1.0) for b in blocs]


def _first(series, predicate) -> float | None:
    for row in series:
        if predicate(row):
            return float(row["year"])
    return None


def cell_report(result, blocs) -> dict[str, Any]:
    settlement = result["settlement"]
    quantities = result["quantities"]
    out: dict[str, Any] = {}
    for i, b in enumerate(blocs):
        credit, selfsuf = (float(x) for x in
                           b.name.replace("c", "").replace("s", "").split("_"))
        shares = [row[f"share_{i}"] for row in settlement]
        out[b.name] = {
            "credit": credit,
            "self_sufficiency": selfsuf,
            "reserve": b.reserve,
            "first_rationed": _first(settlement, lambda r, i=i: r[f"share_{i}"] > 0.01),
            "rationing_final": shares[-1],
            "rationing_max": max(shares),
            "reserves_out": _first(settlement, lambda r, i=i: r[f"reserves_{i}"] <= 1e-9),
            "debt_final": result["final"][b.name]["debt"],
            "insolvent_year": result["crisis"][b.name]["insolvent_year"],
            "housing": 100.0 * quantities[30][f"H_{i}"] / quantities[1][f"H_{i}"],
            "food": 100.0 * quantities[30][f"F_{i}"] / quantities[1][f"F_{i}"],
        }
    return out


def run(modality: str, u: float, contagion: float) -> dict[str, Any]:
    g = globals_for(contagion)
    blocs = _programme(grid(STEPS), g, modality, u)
    result = simulate(blocs, g)
    return {
        "modality": modality, "u": u, "contagion": contagion,
        "world_price": result["world_final"]["price"],
        "world_path": [(r["year"], r["price"]) for r in result["world"]],
        "defaults": sorted(v["insolvent_year"] for k, v in result["crisis"].items()
                           if k != "_system" and v.get("insolvent_year") is not None),
        "contagion_events": len(result["contagion"]),
        "cells": cell_report(result, blocs),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "archetype_positions": locate_archetypes(),
        "steps": STEPS,
        "main_u": MAIN_U,
        "plane": {f"{m}|{c:g}": run(m, MAIN_U, c)
                  for m in MODALITIES for c in CONTAGION},
        "by_u": {f"{u:g}|{c:g}": run("ubi", u, c)
                 for u in U_GRID for c in CONTAGION},
    }
    path = RESULTS / "plane.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    print("=== Where the archetypes sit ===")
    for key, (c, s) in report["archetype_positions"].items():
        print(f"  {key}: credit={c:+.2f}  self-sufficiency={s:+.2f}")
    print("  -> on the diagonal; the axes cannot be separated there\n")

    cells = report["plane"][f"ubi|0"]["cells"]
    axes = sorted({c["credit"] for c in cells.values()})

    def table(source, field, title, fmt="7.3f"):
        print(f"=== {title} ===")
        print("  credit \\ self " + "".join(f"{s:>9.2f}" for s in axes))
        for credit in axes:
            row = []
            for selfsuf in axes:
                cell = next(c for c in source.values()
                            if abs(c["credit"] - credit) < 1e-9
                            and abs(c["self_sufficiency"] - selfsuf) < 1e-9)
                v = cell[field]
                row.append("        —" if v is None else f"{v:>9{fmt[1:]}}")
            print(f"  {credit:13.2f}" + "".join(row))
        print()

    table(cells, "first_rationed", f"First year rationed (UBI, u={MAIN_U:g})", "7.0f")
    table(cells, "insolvent_year", f"Year of default (UBI, u={MAIN_U:g})", "7.1f")
    table(cells, "housing", f"Housing secured at year 30 (UBI, u={MAIN_U:g})", "7.1f")

    print("=== What contagion changes ===")
    print("  Defaults and world price are reported together throughout: the "
          "price falls\n  because blocs stopped importing, so reading it "
          "alone inverts the finding.\n")
    print(f"  {'modality':9s} {'contagion':>10s} {'defaults':>9s} "
          f"{'world price':>12s} {'events':>7s}")
    for m in MODALITIES:
        for c in CONTAGION:
            r = report["plane"][f"{m}|{c:g}"]
            print(f"  {m:9s} {c:10.1f} {len(r['defaults']):9d} "
                  f"{r['world_price']:12.4f} {r['contagion_events']:7d}")

    print("\n=== Transfer level against contagion (UBI) ===")
    print("  World price and defaults together: a lower world price under "
          "contagion is\n  demand disappearing with the blocs that stopped "
          "importing, not relief.\n")
    print(f"  {'u':>6s} {'defaults':>9s} {'price':>8s} | "
          f"{'defaults':>9s} {'price':>8s} | {'extra':>6s}")
    print(f"  {'':>6s} {'(no contagion)':>18s} | {'(contagion)':>18s} |")
    for u in U_GRID:
        a = report["by_u"][f"{u:g}|0"]
        b = report["by_u"][f"{u:g}|1"]
        print(f"  {u:6.3f} {len(a['defaults']):9d} {a['world_price']:8.4f} | "
              f"{len(b['defaults']):9d} {b['world_price']:8.4f} | "
              f"{len(b['defaults']) - len(a['defaults']):+6d}")


if __name__ == "__main__":
    main()
