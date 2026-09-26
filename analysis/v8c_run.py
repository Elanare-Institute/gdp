"""Phase C: how the indexation rule changes the pressure a transfer generates.

Three rules, and the difference between them is the whole point:

  gdp_linked      the transfer is a share of real output, so inflation never
                  touches it — phases A/B, kept as the reference
  fixed_nominal   the transfer is never adjusted, so inflation takes it
  cpi_indexed     the transfer is adjusted to last period's inflation, so it
                  keeps its value and feeds the inflation it is chasing

Reported as pressure, not as a verdict: levels and their response to the
configuration. Everything is stated against the no-transfer baseline, which by
construction sits at a zero output gap and a world price of exactly 1.
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
from model.indexation import RULES  # noqa: E402
from model.params import G  # noqa: E402

RESULTS = ROOT / "results" / "v8c"
U_GRID = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
MODALITIES = ("ubi", "cash_t", "voucher", "clt")
KAPPAS = (0.1, 0.35, 1.0)
SIMULTANEOUS = (1.0, 1.0, 1.0, 1.0)
#: Beyond this the indexation loop has taken the price level out of any range
#: worth reporting a number for; the run is recorded as "ran away" instead.
RUNAWAY = 1e4


def globals_for(rule: str, kappa: float = 0.35) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation=rule, kappa_w=kappa)


def _cell(g: G, modality: str, u: float) -> dict[str, Any]:
    blocs = build(g, modality, u, SIMULTANEOUS) if u > 0 else build(g, "none")
    result = simulate(blocs, g)
    m = measure(result, real_value=None)
    price = result["world_final"]["price"]
    return {
        "world_price": price,
        "runaway": price > RUNAWAY,
        "inflation_peak": m.inflation_peak,
        "inflation_final": m.inflation_final,
        "gap_max": m.gap_max,
        "insolvent_blocs": m.insolvent_blocs,
        "first_default_year": m.first_default_year,
        "demand_share_solvent": m.demand_share_solvent,
        "real_value": result["real_value"],
        # Pressure per unit of transfer: flat means the world absorbs the
        # programme in proportion, rising means it does not.
        "gap_per_u": m.gap_max / u if u > 0 else None,
    }


def surface(kappa: float = 0.35) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rule in RULES:
        g = globals_for(rule, kappa)
        out[rule] = {modality: {f"{u:g}": _cell(g, modality, u) for u in U_GRID}
                     for modality in MODALITIES}
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"u_grid": list(U_GRID), "kappas": list(KAPPAS),
              "by_kappa": {f"{k:g}": surface(k) for k in KAPPAS}}
    path = RESULTS / "indexation.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    main_surface = report["by_kappa"]["0.35"]
    print("=== World price by indexation rule (all blocs, UBI, kappa_w=0.35) ===")
    print(f"  {'u':>7s} " + " ".join(f"{r:>16s}" for r in RULES))
    for u in U_GRID:
        cells = [main_surface[r]["ubi"][f"{u:g}"] for r in RULES]
        row = " ".join(
            ("        runaway" if c["runaway"] else f"{c['world_price']:16.4f}")
            for c in cells)
        print(f"  {u:7.4f} {row}")

    print("\n=== Real value of the transfer at year 30 (bloc D, UBI) ===")
    print(f"  {'u':>7s} " + " ".join(f"{r:>16s}" for r in RULES))
    for u in U_GRID:
        if u == 0:
            continue
        vals = []
        for r in RULES:
            rv = main_surface[r]["ubi"][f"{u:g}"]["real_value"]
            v = rv.get("D (developing)") if rv else None
            vals.append("             n/a" if v is None else f"{v:16.4f}")
        print(f"  {u:7.4f} " + " ".join(vals))

    print("\n=== Does the ranking of modalities survive indexation? ===")
    for rule in RULES:
        ranked = sorted(MODALITIES,
                        key=lambda m: main_surface[rule][m]["0.02"]["world_price"],
                        reverse=True)
        prices = [main_surface[rule][m]["0.02"]["world_price"] for m in ranked]
        print(f"  {rule:14s} " + " > ".join(
            f"{m}({p:.3g})" for m, p in zip(ranked, prices)))

    print("\n=== Sensitivity to the pass-through coefficient (UBI, u=0.02) ===")
    print(f"  {'kappa_w':>8s} " + " ".join(f"{r:>16s}" for r in RULES))
    for k in KAPPAS:
        cells = [report["by_kappa"][f"{k:g}"][r]["ubi"]["0.02"] for r in RULES]
        row = " ".join(
            ("        runaway" if c["runaway"] else f"{c['world_price']:16.4f}")
            for c in cells)
        print(f"  {k:8.2f} {row}")


if __name__ == "__main__":
    main()
