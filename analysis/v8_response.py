"""Response surfaces: how much pressure does a configuration put on the world?

Not "does the world survive" — the model contains no trade-bloc formation, no
migration, no conflict, no policy reversal, so it cannot answer that. What it
can show is how the pressure on the world market responds to the choices a
configuration makes, and where that response stops being proportional to the
choice.

Two surfaces here, both from phases A/B:

  u x modality        transfer size against what is handed out
  u x coalition       transfer size against how many blocs hand it out

The in-kind share and the indexation rule need phase C and are added there.

Curvature is reported as the ratio of pressure to transfer size: flat means
pressure grows in step with the programme, rising means it grows faster.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.collapse import measure  # noqa: E402
from model.core import build, simulate  # noqa: E402
from model.params import G  # noqa: E402

RESULTS = ROOT / "results" / "v8a"
U_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)
MODALITIES = ("ubi", "cash_t", "voucher", "clt")
COALITIONS = {"D": ("D",), "C+D": ("C", "D"), "B+C+D": ("B", "C", "D"),
              "all": ("A", "B", "C", "D")}
SIMULTANEOUS = (1.0, 1.0, 1.0, 1.0)


def globals_for() -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)


def _cell(g: G, modality: str, u: float, which) -> dict[str, Any]:
    blocs = (build(g, modality, u, SIMULTANEOUS, "welfare", which)
             if u > 0 else build(g, "none"))
    m = measure(simulate(blocs, g))
    d = m.as_dict()
    # Pressure per unit of transfer. Flat in u means the world absorbs the
    # programme in proportion; rising means it does not.
    d["inflation_peak_per_u"] = m.inflation_peak / u if u > 0 else None
    d["gap_max_per_u"] = m.gap_max / u if u > 0 else None
    return d


def surface_by_modality(g: G) -> dict[str, Any]:
    return {modality: {f"{u:g}": _cell(g, modality, u, ("A", "B", "C", "D"))
                       for u in U_GRID}
            for modality in MODALITIES}


def surface_by_coalition(g: G) -> dict[str, Any]:
    return {name: {f"{u:g}": _cell(g, "ubi", u, which) for u in U_GRID}
            for name, which in COALITIONS.items()}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    g = globals_for()
    report = {
        "u_grid": list(U_GRID),
        "by_modality": surface_by_modality(g),
        "by_coalition": surface_by_coalition(g),
    }
    path = RESULTS / "response.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    print("=== Pressure by transfer size and modality (all blocs pay) ===")
    print(f"  {'u':>7s} " + " ".join(f"{m:>22s}" for m in MODALITIES))
    print(f"  {'':>7s} " + " ".join(f"{'peak pi / gap / dflt':>22s}" for _ in MODALITIES))
    for u in U_GRID:
        cells = [report["by_modality"][m][f"{u:g}"] for m in MODALITIES]
        row = " ".join(
            f"{100 * c['inflation_peak']:7.2f}% {100 * c['gap_max']:6.1f}% {c['insolvent_blocs']:3d}"
            for c in cells)
        print(f"  {u:7.4f} {row}")

    print("\n=== Curvature: peak inflation per unit of u ===")
    print(f"  {'u':>7s} " + " ".join(f"{m:>10s}" for m in MODALITIES))
    for u in U_GRID:
        if u == 0:
            continue
        vals = [report["by_modality"][m][f"{u:g}"]["inflation_peak_per_u"]
                for m in MODALITIES]
        print(f"  {u:7.4f} " + " ".join(f"{v:10.3f}" for v in vals))

    print("\n=== Pressure by how many blocs pay (UBI) ===")
    print(f"  {'u':>7s} " + " ".join(f"{n:>18s}" for n in COALITIONS))
    for u in U_GRID:
        cells = [report["by_coalition"][n][f"{u:g}"] for n in COALITIONS]
        row = " ".join(
            f"{100 * c['inflation_peak']:6.2f}% {100 * c['gap_max']:5.1f}% {c['insolvent_blocs']:2d}"
            for c in cells)
        print(f"  {u:7.4f} {row}")


if __name__ == "__main__":
    main()
