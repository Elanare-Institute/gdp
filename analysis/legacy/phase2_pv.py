"""Decompose the 'no solution' result of the present-value calibration.

Two candidate explanations were proposed: (a) the build cap genuinely starves
the early years, and (b) a 30-year horizon discards a durable asset's later
benefits. Both are tested — 60-year and terminal-value variants — but the
diagnostic below shows a third cause dominates: while the cap binds, the stock
in place is set by the cap rather than by `size`, so the calibration has no
working instrument. That is reported rather than the (a)/(b) split alone.
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

from legacy.v7.heterogeneous import (  # noqa: E402
    discounted_calibrate, hetero_household_block, welfare_measure,
)
from legacy.v7.params import G, archetypes  # noqa: E402

RESULTS = ROOT / "results" / "phase2"
BUILD_RATES = (None, 0.05, 0.02, 0.01, 0.005)
DISCOUNTS = (0.01, 0.03, 0.05)
DEPRECIATIONS = (0.015, 0.020, 0.025)


def variants(bloc: str = "C") -> list[dict[str, Any]]:
    """PV calibration under each horizon / terminal-value variant."""
    b = archetypes()[bloc]
    rows: list[dict[str, Any]] = []
    for build in BUILD_RATES:
        g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
              clt_build_rate=build)
        for disc in DISCOUNTS:
            specs = [("30y", dict(years=30)),
                     ("60y", dict(years=60))]
            for dep in DEPRECIATIONS:
                specs.append((f"30y+term{dep:.3f}",
                              dict(years=30, terminal_value=True, depreciation=dep)))
            for label, kw in specs:
                res = discounted_calibrate(b, g, 0.05, "utilitarian",
                                           rate=disc, only=("clt",), **kw)
                rows.append({
                    "bloc": bloc, "build_rate": build, "discount": disc,
                    "variant": label, "clt": res["clt"],
                    "solved": not res.get("_at_ceiling_clt", False),
                })
    return rows


def instrument_diagnostic(bloc: str = "C") -> list[dict[str, Any]]:
    """How responsive welfare is to `size` while the build cap binds.

    If the derivative is zero (or negative) the calibration has no instrument,
    which is a different failure from the programme being genuinely too slow.
    """
    b = archetypes()[bloc]
    rows: list[dict[str, Any]] = []
    sizes = (0.01, 0.02, 0.05, 0.10, 0.20)
    for build in BUILD_RATES:
        g = G(n_types=5, clt_capacity=1.0, clt_rent_to_landlords=False,
              clt_build_rate=build)
        for year in (1, 5, 10, 20, 30, 45, 59):
            utils = []
            for size in sizes:
                out = hetero_household_block(
                    replace(b, modality="clt"), g, b.gdp, 1.0, b.rate, True,
                    size, years_since_start=float(year))
                utils.append(welfare_measure(out, "utilitarian"))
            arr = np.asarray(utils)
            rows.append({
                "bloc": bloc, "build_rate": build, "year": year,
                "spread": float(arr.max() - arr.min()),
                "monotone_increasing": bool(np.all(np.diff(arr) > -1e-12)),
                "best_size": float(sizes[int(np.argmax(arr))]),
            })
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, fn in (("pv_variants", variants),
                     ("pv_instrument", instrument_diagnostic)):
        rows = fn()
        path = RESULTS / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        print(f"  wrote {path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
