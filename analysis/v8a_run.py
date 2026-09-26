"""Phase A/B experiments: does closing the world do what it was meant to?

Three questions, one run each:

  spread     how does the world price respond to how many blocs pay?
  size       how does it respond to how much they pay?
  modality   does the answer depend on what is handed out?

Everything is deterministic — phase A/B has no stochastic component — so the
output is a single JSON file rather than a sampled set.
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

RESULTS = ROOT / "results" / "v8a"
#: Main series: no bloc goes insolvent at or below this size, so the world
#: price is reading the transfer rather than a debt runaway (see
#: `tests/test_closed_world.py`).
SAFE_U = 0.01
COALITIONS = ((), ("D",), ("C", "D"), ("B", "C", "D"), ("A", "B", "C", "D"))
SIZES = (0.0, 0.002, 0.005, 0.0075, 0.01)
MODALITIES = ("ubi", "cash_t", "voucher", "clt")
SIMULTANEOUS = (1.0, 1.0, 1.0, 1.0)


def globals_for_v8a() -> G:
    """Heterogeneous households, phased build, no reserve privileges."""
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02)


def _summary(result: dict[str, Any], blocs) -> dict[str, Any]:
    return {
        "world_price": result["world_final"]["price"],
        "world_peak": max(row["price"] for row in result["world"]),
        "world_path": [(row["year"], row["price"]) for row in result["world"]],
        "gap_path": [(row["year"], row["demand"] / row["capacity"] - 1.0)
                     for row in result["world"]],
        "final": {b.name: result["final"][b.name] for b in blocs},
        "insolvent": [k for k, v in result["crisis"].items()
                      if k != "_system" and v.get("insolvent_year") is not None],
        "nfa_sum": sum(result["final"][b.name]["nfa"] for b in blocs),
    }


def run_coalitions(g: G, u: float) -> dict[str, Any]:
    """World price against the number of blocs running the same programme."""
    out = {}
    for which in COALITIONS:
        blocs = (build(g, "ubi", u, SIMULTANEOUS, "welfare", which)
                 if which else build(g, "none"))
        out["+".join(which) or "none"] = _summary(simulate(blocs, g), blocs)
    return out


def run_sizes(g: G) -> dict[str, Any]:
    """World price against how much everyone hands out."""
    out = {}
    for u in SIZES:
        blocs = build(g, "ubi", u, SIMULTANEOUS) if u > 0 else build(g, "none")
        out[f"{u:g}"] = _summary(simulate(blocs, g), blocs)
    return out


def run_modalities(g: G, u: float) -> dict[str, Any]:
    """World price against what is handed out, at a matched welfare target."""
    out = {}
    blocs = build(g, "none")
    out["none"] = _summary(simulate(blocs, g), blocs)
    for modality in MODALITIES:
        blocs = build(g, modality, u, SIMULTANEOUS)
        out[modality] = _summary(simulate(blocs, g), blocs)
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    g = globals_for_v8a()
    report = {
        "u": SAFE_U,
        "coalitions": run_coalitions(g, SAFE_U),
        "sizes": run_sizes(g),
        "modalities": run_modalities(g, SAFE_U),
    }
    path = RESULTS / "world.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}")

    print("\ncoalition            P_w final   peak    insolvent")
    for key, v in report["coalitions"].items():
        print(f"  {key:18s} {v['world_price']:.4f}   {v['world_peak']:.4f}  {v['insolvent']}")
    print("\nsize (u)             P_w final   peak    insolvent")
    for key, v in report["sizes"].items():
        print(f"  {key:18s} {v['world_price']:.4f}   {v['world_peak']:.4f}  {v['insolvent']}")
    print("\nmodality             P_w final   peak    insolvent")
    for key, v in report["modalities"].items():
        print(f"  {key:18s} {v['world_price']:.4f}   {v['world_peak']:.4f}  {v['insolvent']}")


if __name__ == "__main__":
    main()
