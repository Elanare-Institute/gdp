"""Which provisioning forms are usable where.

The final output. Not "which modality lasts longest" — that would be a ranking
of the kind CLAUDE.md rules out — but which ones a bloc at a given position
can actually run, and which are closed to it.

A modality is **unusable** at a position when running it at the size needed to
reach subsistence puts the bloc outside what it can settle, or when it cannot
be built at all. Three ways that happens:

  settlement   the imports the programme requires cannot be paid for
  need         no transfer covers the need it creates (no fixed point)
  capacity     the construction the programme requires exceeds what the bloc
               can build within the horizon (phase 2's build cap)

The plane is credit x self-sufficiency throughout, and each figure holds two
of the four structural dimensions fixed while sweeping the others: land rent
share, housing supply elasticity, construction import share, build rate.

Two automation levels, side by side: the baseline the redistribution
literature assumes, and a world where labour demand has fallen.
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

RESULTS = ROOT / "results" / "v8final"
STEPS = 5
FLOOR = 0.85
AUTOMATION = (0.0, 0.4)
MODALITIES = ("ubi", "cash_t", "voucher", "clt")
U_CANDIDATES = (0.01, 0.02, 0.04, 0.06, 0.08)
#: Average rationing a bloc may carry before a modality counts as unusable.
TOLERANCE = 0.05

#: The four structural dimensions, swept two at a time. Values span what the
#: archetypes cover.
DIMENSIONS = {
    "land_share": (0.30, 0.45),
    "eps_supply": (0.30, 0.50),
    "m_H": (0.05, 0.25),
    "clt_build_rate": (0.01, 0.04),
}


def globals_for(automation: float, **overrides) -> G:
    base = dict(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
                indexation="cpi_indexed", settlement=True, trade_elasticity=1.0,
                policy_response=1.0, contagion_strength=1.0,
                flight_to_reserve=0.5, income_linkage="cpi_linked",
                subsistence_scale=FLOOR / 0.65, automation_max=automation)
    base.update({k: v for k, v in overrides.items() if k in G.__dataclass_fields__})
    return G(**base)


def _blocs(**overrides):
    out = grid(STEPS)
    structural = {k: v for k, v in overrides.items()
                  if k in ("land_share", "eps_supply", "m_H")}
    return [replace(b, **structural) for b in out] if structural else out


def usable(automation: float, **overrides) -> dict[str, Any]:
    """For each position and modality: can it be run at all?"""
    g = globals_for(automation, **overrides)
    blocs = _blocs(**overrides)

    need_by_bloc: dict[str, float] = {}
    quiet = simulate(blocs, g)
    for i, b in enumerate(blocs):
        need_by_bloc[b.name] = max(row[f"need_{i}"] for row in quiet["budget"])

    cells: dict[str, Any] = {}
    for i, b in enumerate(blocs):
        credit, selfsuf = (float(x) for x in
                           b.name.replace("c", "").replace("s", "").split("_"))
        cells[b.name] = {"credit": credit, "self_sufficiency": selfsuf,
                         "need": need_by_bloc[b.name], "modalities": {}}

    for modality in MODALITIES:
        # The smallest transfer of this kind that covers the need, if any, and
        # whether the bloc can settle the imports it requires.
        for u in U_CANDIDATES:
            sized = [replace(b, modality=modality,
                             size=calibrate(b, g, u)[modality], start=1.0)
                     for b in blocs]
            result = simulate(sized, g)
            settlement = result["settlement"]
            n = max(1, len(settlement))
            for i, b in enumerate(blocs):
                cell = cells[b.name]
                if modality in cell["modalities"]:
                    continue
                if u < cell["need"]:
                    continue
                rationing = sum(row[f"share_{i}"] for row in settlement) / n
                if rationing <= TOLERANCE:
                    cell["modalities"][modality] = {"u": u, "rationing": rationing}
        for name, cell in cells.items():
            cell["modalities"].setdefault(modality, None)

    return {"automation": automation, "overrides": overrides, "cells": cells}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "archetype_positions": locate_archetypes(),
        "floor": FLOOR, "tolerance": TOLERANCE,
        "automation_levels": list(AUTOMATION),
        "modalities": list(MODALITIES),
        "base": {f"{a:g}": usable(a) for a in AUTOMATION},
        "sections": {},
    }
    # Two-dimensional sections: each structural dimension at both ends.
    for name, (lo, hi) in DIMENSIONS.items():
        for label, value in (("low", lo), ("high", hi)):
            key = f"{name}_{label}"
            report["sections"][key] = {
                f"{a:g}": usable(a, **{name: value}) for a in AUTOMATION}

    path = RESULTS / "phase_diagram.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    for a in AUTOMATION:
        cells = report["base"][f"{a:g}"]["cells"]
        label = "baseline" if a == 0 else f"automation {a:.0%}"
        print(f"=== Usable modalities, {label} (floor {FLOOR:.0%}, "
              f"tolerance {TOLERANCE:.0%}) ===")
        axes = sorted({c["credit"] for c in cells.values()})
        print(f"  {'credit':>7s} " + "".join(f"{s:>22.2f}" for s in axes))
        for credit in axes:
            row = []
            for selfsuf in axes:
                cell = next(c for c in cells.values()
                            if abs(c["credit"] - credit) < 1e-9
                            and abs(c["self_sufficiency"] - selfsuf) < 1e-9)
                codes = {"ubi": "U", "cash_t": "C", "voucher": "V", "clt": "H"}
                usable_names = [codes[m] for m in MODALITIES
                                if cell["modalities"].get(m)]
                row.append("".join(usable_names) or "—")
            print(f"  {credit:7.2f} " + "".join(f"{r:>10s}" for r in row))
        print("  (U=ubi, C=targeted cash, V=voucher, H=in-kind housing; "
              "— none usable)\n")


if __name__ == "__main__":
    main()
