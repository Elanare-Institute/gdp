"""Where does "a cash transfer leaves recipients worse off" come from?

Phase C found that in bloc D, thirty years of CPI-indexed universal cash ends
with constrained households holding less housing than if nothing had been
handed out. Before that is reported as a finding, it has to be established
that it comes from the household block and the world market rather than from
the reduced-form growth penalties inherited from v6.

Those penalties — growth falls when debt is high, when inflation is high, when
the exchange rate moves — are coefficients chosen to make the macro position
matter, not mechanisms derived from anything. A result that rests on them is a
result about the coefficients.

Each is switched off in turn, and then all together, and the gap to the
no-transfer baseline is measured each time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core import build, simulate  # noqa: E402
from model.params import G  # noqa: E402

RESULTS = ROOT / "results" / "v8c"
ALL_PENALTIES = ("debt", "inflation", "fx", "capital")
MODALITIES = ("none", "ubi", "cash_t", "voucher", "clt")
U = 0.02
BLOC = 3  # bloc D
RULE = "cpi_indexed"


def _globals(penalties: tuple[str, ...], rule: str = RULE) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation=rule, growth_penalties=penalties)


def _quantity_index(g: G, modality: str, good: str = "H") -> float:
    """Quantity secured at year 30, indexed to the programme's start."""
    blocs = build(g, modality, U, (1, 1, 1, 1)) if modality != "none" else build(g, "none")
    rows = simulate(blocs, g)["quantities"]
    return 100.0 * rows[30][f"{good}_{BLOC}"] / rows[1][f"{good}_{BLOC}"]


def scenario(penalties: tuple[str, ...], good: str = "H") -> dict[str, Any]:
    g = _globals(penalties)
    values = {m: _quantity_index(g, m, good) for m in MODALITIES}
    baseline = values["none"]
    return {
        "penalties": list(penalties),
        "levels": values,
        "gap_to_no_transfer": {m: values[m] - baseline for m in MODALITIES},
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"u": U, "rule": RULE, "bloc": BLOC, "good": "H",
                              "scenarios": {}}

    report["scenarios"]["all"] = scenario(ALL_PENALTIES)
    for dropped in ALL_PENALTIES:
        kept = tuple(p for p in ALL_PENALTIES if p != dropped)
        report["scenarios"][f"without_{dropped}"] = scenario(kept)
    report["scenarios"]["none_of_them"] = scenario(())

    # The same decomposition on food, to check the story is not housing-specific.
    report["food"] = {"all": scenario(ALL_PENALTIES, "F"),
                      "none_of_them": scenario((), "F")}

    path = RESULTS / "decomposition.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    print(f"=== Housing secured at year 30, bloc D, {RULE}, u={U:g} "
          f"(start = 100) ===\n")
    head = f"  {'growth penalties':22s} " + " ".join(f"{m:>9s}" for m in MODALITIES)
    print(head); print("  " + "-" * (len(head) - 2))
    for name, cell in report["scenarios"].items():
        row = " ".join(f"{cell['levels'][m]:9.1f}" for m in MODALITIES)
        print(f"  {name:22s} {row}")

    print(f"\n=== Gap to the no-transfer baseline (negative = worse off) ===\n")
    print(head); print("  " + "-" * (len(head) - 2))
    for name, cell in report["scenarios"].items():
        row = " ".join(f"{cell['gap_to_no_transfer'][m]:+9.1f}" for m in MODALITIES)
        print(f"  {name:22s} {row}")

    print("\n=== How much of the gap survives with every penalty off? ===\n")
    full = report["scenarios"]["all"]["gap_to_no_transfer"]
    bare = report["scenarios"]["none_of_them"]["gap_to_no_transfer"]
    print(f"  {'modality':10s} {'with penalties':>15s} {'without':>10s} {'share kept':>11s}")
    for m in MODALITIES:
        if m == "none":
            continue
        share = bare[m] / full[m] if abs(full[m]) > 1e-9 else float("nan")
        print(f"  {m:10s} {full[m]:+15.1f} {bare[m]:+10.1f} {share:11.1%}")


if __name__ == "__main__":
    main()
