"""What happens on the credit x self-sufficiency plane.

The four archetypes sit on the diagonal of this plane — credit and
self-sufficiency rise together — so nothing computed on them can say which
axis does the work. Here the two are varied independently.

The mechanism expected, and to be checked rather than assumed:

  low credit           reserves run out and imports stop
  low self-sufficiency debt accumulates and the borrowing window shuts later
  both low             the earliest exit

If that is not what happens, the reason is reported instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core import simulate  # noqa: E402
from model.household import calibrate  # noqa: E402
from model.params import G  # noqa: E402
from model.twoaxis import grid, locate_archetypes, make_bloc  # noqa: E402

RESULTS = ROOT / "results" / "v8d"
U = 0.02
STEPS = 5
MODALITIES = ("none", "ubi", "cash_t", "clt")


def globals_for() -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation="cpi_indexed", settlement=True,
             trade_elasticity=1.0, policy_response=1.0)


def _with_programme(blocs, g: G, modality: str, u: float):
    """Give every bloc the same welfare-equivalent programme."""
    from dataclasses import replace
    if modality == "none":
        return list(blocs)
    out = []
    for b in blocs:
        size = calibrate(b, g, u)[modality]
        out.append(replace(b, modality=modality, size=size, start=1.0))
    return out


def _first_year(series, predicate) -> float | None:
    for row in series:
        if predicate(row):
            return float(row["year"])
    return None


def run(modality: str = "ubi") -> dict[str, Any]:
    g = globals_for()
    blocs = _with_programme(grid(STEPS), g, modality, U)
    result = simulate(blocs, g)
    settlement = result["settlement"]
    quantities = result["quantities"]

    cells: dict[str, Any] = {}
    for i, b in enumerate(blocs):
        credit, selfsuf = (float(x) for x in
                           b.name.replace("c", "").replace("s", "").split("_"))
        shares = [row[f"share_{i}"] for row in settlement]
        crisis = result["crisis"][b.name]
        cells[b.name] = {
            "credit": credit,
            "self_sufficiency": selfsuf,
            "reserve": b.reserve,
            "first_rationed_year": _first_year(
                settlement, lambda r, i=i: r[f"share_{i}"] > 0.01),
            "rationing_share_final": shares[-1],
            "rationing_share_max": max(shares),
            "reserves_final": settlement[-1][f"reserves_{i}"],
            "reserves_exhausted_year": _first_year(
                settlement, lambda r, i=i: r[f"reserves_{i}"] <= 1e-9),
            "debt_final": result["final"][b.name]["debt"],
            "insolvent_year": crisis["insolvent_year"],
            "housing_index": (100.0 * quantities[30][f"H_{i}"]
                              / quantities[1][f"H_{i}"]),
            "food_index": (100.0 * quantities[30][f"F_{i}"]
                           / quantities[1][f"F_{i}"]),
        }
    return {"modality": modality, "u": U, "cells": cells}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"archetype_positions": locate_archetypes(),
                              "runs": {m: run(m) for m in MODALITIES}}
    path = RESULTS / "twoaxis.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    print("=== Where the archetypes sit (credit, self-sufficiency) ===")
    for key, (c, s) in report["archetype_positions"].items():
        print(f"  {key}: credit={c:+.2f}  self-sufficiency={s:+.2f}")
    print("  -> all on the diagonal; the two effects cannot be told apart there\n")

    cells = report["runs"]["ubi"]["cells"]
    axes = sorted({c["credit"] for c in cells.values()})

    def table(field: str, title: str, fmt: str = "7.3f") -> None:
        print(f"=== {title} (UBI, u={U:g}) ===")
        print("  credit \\ self-suff " + "".join(f"{s:>9.2f}" for s in axes))
        for credit in axes:
            row = []
            for selfsuf in axes:
                cell = next(c for c in cells.values()
                            if abs(c["credit"] - credit) < 1e-9
                            and abs(c["self_sufficiency"] - selfsuf) < 1e-9)
                value = cell[field]
                row.append("        —" if value is None else f"{value:>9{fmt[1:]}}")
            print(f"  {credit:18.2f}" + "".join(row))
        print()

    table("first_rationed_year", "First year imports are rationed", "7.0f")
    table("rationing_share_final", "Rationing share at year 30")
    table("housing_index", "Housing secured at year 30 (start = 100)", "7.1f")
    table("debt_final", "Debt at year 30")

    print("=== Mode of failure ===")
    print("  Expected: low credit -> reserves exhausted; "
          "low self-sufficiency -> debt accumulates then borrowing stops.\n")
    print(f"  {'bloc':16s} {'credit':>7s} {'self':>6s} {'resv out':>9s} "
          f"{'debt':>7s} {'rationed':>9s}")
    for name, c in sorted(cells.items(),
                          key=lambda kv: (kv[1]["credit"], kv[1]["self_sufficiency"])):
        out_year = c["reserves_exhausted_year"]
        print(f"  {name:16s} {c['credit']:7.2f} {c['self_sufficiency']:6.2f} "
              f"{('—' if out_year is None else f'{out_year:.0f}'):>9s} "
              f"{c['debt_final']:7.3f} {c['rationing_share_final']:9.3f}")


if __name__ == "__main__":
    main()
