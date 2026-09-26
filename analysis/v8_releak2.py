"""Where the reversal comes from, and where capital outflows go.

Two questions the first re-read left open.

**Which correction flips the ranking?** Going from "leakage per unit of fiscal
cost" to "the demand a programme actually places on the world market" is two
corrections at once, and they are not the same correction:

  (a) rate,  imports + foreign assets   the quantity Phase 1 reported
  (b) level, imports + foreign assets   scale correction only
  (c) level, imports only               composition correction as well

Reporting only (a) against (c) hides which of the two is doing the work.

**Where does a capital outflow go?** In a closed world a bloc's purchase of
foreign assets is another bloc's capital inflow. Whether that inflow becomes
demand for goods decides whether (c) is the right measure at all, so the model
is asked directly rather than argued about.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.core import build as v7_build, simulate as v7_simulate  # noqa: E402
from legacy.v7.params import G as V7G  # noqa: E402
from model.core import build, simulate  # noqa: E402
from model.params import G  # noqa: E402

RESULTS = ROOT / "results" / "v8a"
MODALITIES = ("ubi", "cash_t", "voucher", "clt")
BLOCS = ("A (reserve)", "B (non-reserve adv.)", "C (emerging)", "D (developing)")


def _v7_globals() -> V7G:
    return V7G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
               clt_build_rate=0.02)


def staged_rankings(u: float = 0.05) -> dict[str, Any]:
    """The three stages, so the reversal can be attributed to one of them.

    Read off v7, because these are v7's reported quantities: the point is to
    re-read what Phase 1-3 said, not to recompute it under new dynamics.
    """
    g = _v7_globals()
    cells: dict[str, dict[str, dict[str, float]]] = {}
    for modality in MODALITIES:
        result = v7_simulate(v7_build(g, modality, u), g)
        for name, c in result["leakage"].items():
            imp_rate = c["import_per_fiscal"]
            fa_rate = c["foreign_assets_per_fiscal"]
            total_rate = c["external_leak_per_fiscal"]
            # leak_abs is (imports + foreign assets) as a share of GDP, so the
            # import-only level is that scaled by the import share of the total.
            level_total = c["leak_abs"]
            level_imports = level_total * (imp_rate / total_rate) if total_rate else 0.0
            cells.setdefault(name, {})[modality] = {
                "a_rate_total": total_rate,
                "b_level_total": level_total,
                "c_level_imports": level_imports,
                "import_rate": imp_rate,
                "fa_rate": fa_rate,
                "fiscal_pct_gdp": c["fiscal_pct_gdp"],
            }

    out: dict[str, Any] = {"u": u, "cells": cells, "rankings": {}}
    for name, per_mod in cells.items():
        ranks = {}
        for stage in ("a_rate_total", "b_level_total", "c_level_imports"):
            ranks[stage] = sorted(per_mod, key=lambda m: per_mod[m][stage], reverse=True)
        out["rankings"][name] = {
            **ranks,
            "a_vs_b": "same" if ranks["a_rate_total"] == ranks["b_level_total"] else "DIFFERENT",
            "b_vs_c": "same" if ranks["b_level_total"] == ranks["c_level_imports"] else "DIFFERENT",
        }
    return out


def capital_inflow_destination() -> dict[str, Any]:
    """Does a capital inflow become demand for goods in the receiving bloc?

    Asked of the model rather than of the code comments: run the world with an
    exogenous capital inflow to one bloc and see whether world demand for
    tradables moves in the same period.
    """
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)
    base = simulate(build(g, "none"), g)
    probe = simulate(build(g, "ubi", 0.01, (1, 1, 1, 1), "welfare", ("D",)), g)

    # Bloc D hands out cash; part of it leaves as foreign assets. Track whether
    # the blocs receiving that capital buy more from the world market.
    rows = []
    for i, (b0, b1) in enumerate(zip(base["world"], probe["world"])):
        rows.append({"year": b0["year"],
                     "demand_base": b0["demand"], "demand_probe": b1["demand"],
                     "d_demand": b1["demand"] - b0["demand"]})
    return {
        "note": "D alone runs UBI at u=0.01; its households place part abroad",
        "world_demand_path": rows,
        "final": {name: {"nfa_base": base["final"][name]["nfa"],
                         "nfa_probe": probe["final"][name]["nfa"],
                         "Y_base": base["final"][name]["Y"],
                         "Y_probe": probe["final"][name]["Y"]}
                  for name in BLOCS},
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    staged = staged_rankings()
    dest = capital_inflow_destination()
    (RESULTS / "releak_stages.json").write_text(
        json.dumps({"staged": staged, "capital": dest}, indent=1), encoding="utf-8")
    print(f"wrote {RESULTS / 'releak_stages.json'}\n")

    print(f"=== Stage-by-stage rankings (v7, u={staged['u']}, most leakage first) ===\n")
    for name in BLOCS:
        r = staged["rankings"][name]
        print(f"  {name}")
        print(f"    (a) rate,  imp+fa : {' > '.join(r['a_rate_total'])}")
        print(f"    (b) level, imp+fa : {' > '.join(r['b_level_total'])}   [{r['a_vs_b']}]")
        print(f"    (c) level, imp only: {' > '.join(r['c_level_imports'])}   [{r['b_vs_c']}]")
    print()
    print("  Where the reversal happens:")
    ab = sum(1 for n in BLOCS if staged["rankings"][n]["a_vs_b"] == "DIFFERENT")
    bc = sum(1 for n in BLOCS if staged["rankings"][n]["b_vs_c"] == "DIFFERENT")
    print(f"    rate -> level (scale)      : {ab}/4 blocs change order")
    print(f"    +fa -> imports (composition): {bc}/4 blocs change order")

    print("\n=== Bloc D, the three stages side by side ===")
    print(f"  {'modality':9s} {'(a) rate':>9s} {'(b) level':>10s} {'(c) imports':>12s} "
          f"{'fiscal%':>8s} {'fa rate':>8s}")
    for m in MODALITIES:
        c = staged["cells"]["D (developing)"][m]
        print(f"  {m:9s} {c['a_rate_total']:9.3f} {c['b_level_total']:10.4f} "
              f"{c['c_level_imports']:12.4f} {c['fiscal_pct_gdp']:8.3f} {c['fa_rate']:8.3f}")


if __name__ == "__main__":
    main()
