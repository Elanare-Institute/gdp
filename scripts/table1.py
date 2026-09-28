"""Table 1: four modalities at the same welfare gain, one bloc, year 30.

The four programmes are sized so that a constrained household in the bloc is
indifferent between them at the moment of introduction (`household.calibrate`).
Everything the table reports is then a *consequence* of that equivalence rather
than a difference in generosity: the same welfare at year zero, and whatever
the macroeconomy does with it over thirty years.

Configuration follows `analysis/v8c_run.py`, which is where the manuscript's
surrounding numbers come from: the four archetypes, all introducing at year 1,
`cpi_indexed`, u=0.02, no automation. The bloc reported is D (developing) --
the highest import content and foreign-currency share of the four, which is
the manuscript's "low-creditworthiness, low-self-sufficiency" position.

Provisioning columns are indexed to the bloc's own year-1 level, as in
`reports/PHASE_V8_C.md` and `analysis/v8c_decompose.py`: the quantities are
model units whose absolute size means nothing, and what the comparison is about
is the path away from introduction. Year 1 rather than year 0 because that is
when the programme starts -- at year 0 no transfer has been paid yet, so all
five runs are identical there and indexing to it would fold the first year's
divergence into the number. The no-transfer row is the reference the text
compares against.

Columns:
  fiscal        outlay as a share of GDP, at introduction
  imports       cumulative real imports over the run, indexed to no transfer
  depreciation  exchange rate at year 30 against its year-1 value, in %
  food, housing physical quantities secured by the constrained household,
                indexed to the same run's year 1 = 100
  budget_ratio  poorest quantile's (income + transfer) / cost of the bundle

Writes output/table1.json and prints the LaTeX body.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core import build, simulate  # noqa: E402
from model.household import calibrate  # noqa: E402
from model.params import G, archetypes  # noqa: E402

OUTPUT = ROOT / "output"
#: Paper names on the left, the model's own names on the right.
MODALITIES = {"ubi": "ubi", "targeted": "cash_t",
              "voucher": "voucher", "clt": "clt"}
#: Every bloc introduces at year 1: the world-price channel the phase exists
#: to create only appears when nobody is left outside the programme.
SIMULTANEOUS = (1.0, 1.0, 1.0, 1.0)
#: Bloc D, the developing archetype.
TARGET = 3
#: UBI size the other three are matched to, as in Phase C.
U_UBI = 0.02
YEAR = 30


def globals_for() -> G:
    """Phase C's main series: cost-of-living adjustment, no automation.

    The settlement constraint is off here, which is deliberate and is the one
    thing about this table that needs stating. With it on, bloc D is rationed
    about 30% of its desired imports *with no transfer at all* — the phase
    diagram classifies that corner of the plane as `already_constrained` — and
    the rationing then dominates every difference between the four forms,
    compressing the housing column to 103.3-103.5 across all of them. That is a
    real result and it is what the phase diagram reports; it is not what this
    table is for. This table isolates the demand-side consequence of holding
    welfare equal, which requires a bloc whose imports are not already being
    cut for reasons that have nothing to do with what it hands out.
    """
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation="cpi_indexed", kappa_w=0.35)


def _at_year(rows: list[dict[str, Any]], year: float, key: str) -> float:
    """Value of `key` in the last row at or before `year`."""
    eligible = [r for r in rows if r["year"] <= year + 1e-9]
    return (eligible[-1] if eligible else rows[-1])[key]


def run(modality: str | None) -> dict[str, Any]:
    """Simulate the world once and pull bloc D's numbers out.

    `modality is None` is the no-transfer run, which the import index is
    measured against and which the manuscript compares the four forms to.
    """
    g = globals_for()
    if modality is None:
        blocs, fiscal = build(g, "none"), 0.0
    else:
        blocs = build(g, modality, U_UBI, SIMULTANEOUS)
        fiscal = calibrate(archetypes()["D"], g, U_UBI)["_fiscal_" + modality]

    r = simulate(blocs, g)
    i = TARGET
    # Year 1: when the programme starts. `quantities` is annual, so the
    # row index is the year.
    base = r["quantities"][1]
    return {
        "fiscal": fiscal,
        # Cumulative, not terminal: what presses on the world market is the
        # whole path of demand, and a run that front-loads imports and is then
        # rationed is not the same as one that never asked.
        "imports": sum(row[f"imports_{i}"] for row in r["world"]),
        "e_start": r["history"][1][f"e_{i}"],
        "e": _at_year(r["history"], YEAR, f"e_{i}"),
        "food": 100.0 * _at_year(r["quantities"], YEAR, f"F_{i}") / base[f"F_{i}"],
        "housing": 100.0 * _at_year(r["quantities"], YEAR, f"H_{i}") / base[f"H_{i}"],
        # The poorest quantile: `subsistence.summarise` reports it as the
        # minimum over the quantiles.
        "budget": _at_year(r["budget"], YEAR, f"min_{i}"),
        "size": None if modality is None else blocs[i].size,
    }


LABELS = {"ubi": "UBI", "targeted": "Targeted cash",
          "voucher": "Food voucher", "clt": "In-kind housing",
          "none": "No transfer"}
ORDER = ("ubi", "targeted", "voucher", "clt", "none")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rows = {"none": run(None)}
    for paper_name, model_name in MODALITIES.items():
        rows[paper_name] = run(model_name)
        print(f"  {paper_name:10s} done", flush=True)

    base_imports = rows["none"]["imports"]
    for row in rows.values():
        row["import_index"] = round(100.0 * row["imports"] / base_imports, 1)
        row["depreciation"] = round(
            100.0 * (row["e"] / row["e_start"] - 1.0), 1)

    (OUTPUT / "table1.json").write_text(
        json.dumps({"u_ubi": U_UBI, "bloc": "D (developing)", "year": YEAR,
                    "indexation": "cpi_indexed", "automation": 0.0,
                    "rows": rows}, indent=1))

    print(f"\n=== Table 1: bloc D, cpi_indexed, u={U_UBI}, year {YEAR} ===")
    print(f"{'modality':16s} {'fiscal%':>8s} {'imports':>8s} {'deprec%':>8s} "
          f"{'food':>8s} {'housing':>8s} {'budget':>8s}")
    for key in ORDER:
        r = rows[key]
        print(f"{LABELS[key]:16s} {100 * r['fiscal']:8.1f} "
              f"{r['import_index']:8.1f} {r['depreciation']:8.1f} "
              f"{r['food']:8.1f} {r['housing']:8.1f} {r['budget']:8.3f}")

    print("\n--- LaTeX rows ---")
    for key in ORDER:
        r = rows[key]
        label = (LABELS[key] if key != "none"
                 else f"\\textit{{{LABELS[key]}}}")
        print(f"{label:24s} & {100 * r['fiscal']:.1f} & "
              f"{r['import_index']:.1f} & {r['depreciation']:+.1f} & "
              f"{r['food']:.1f} & {r['housing']:.1f} & {r['budget']:.3f} \\\\")


if __name__ == "__main__":
    main()
