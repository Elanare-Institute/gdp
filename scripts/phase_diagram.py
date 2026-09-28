"""Figure 2: which provisioning forms a bloc can actually run.

Each position on the credit x self-sufficiency plane is classified for each
modality into one of four states:

  already_constrained  rationed even with no transfer at all. Its imports are
                       unaffordable regardless of what is handed out, so this
                       is not a statement about the transfer
  no_fixed_point       handing out more raises the need by more than it
                       covers, so no transfer ever reaches the bundle
  trapped              a transfer that reaches the bundle exists, but running
                       it puts the bloc outside what it can settle
  feasible             a transfer reaches the bundle and can be settled

The need is solved as a fixed point rather than read off the no-transfer run:
a transfer raises demand, demand raises prices, and prices raise the cost of
the bundle it was meant to cover. `subsistence.required_transfer` reports the
need at given prices and does not iterate, so the iteration is done here.

Writes output/phase_diagram.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core import simulate  # noqa: E402
from model.household import calibrate  # noqa: E402
from model.params import G  # noqa: E402
from model.twoaxis import grid  # noqa: E402

OUTPUT = ROOT / "output"
STEPS = 5
AUTOMATION_LEVELS = (0.0, 0.4)
#: Paper names; the model's own names are on the right.
MODALITIES = {"ubi": "ubi", "targeted": "cash_t",
              "voucher": "voucher", "clt": "clt"}
#: Average rationing a bloc may carry before a transfer counts as unsettleable.
TOLERANCE = 0.10
#: Transfers considered, as a share of GDP. The fixed-point search walks this
#: grid; a need above the largest is treated as having no fixed point.
U_CANDIDATES = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12)
FIXED_POINT_ROUNDS = 6
#: Subsistence floor as a share of income. 0.65 is v6's, which leaves a
#: hand-to-mouth household 35% to spend freely; 0.85 is within the range the
#: reports sweep.
FLOOR = 0.85
BASE_FLOOR = 0.65


def globals_for(automation: float) -> G:
    """Main series of the reports, with settlement on."""
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation="cpi_indexed", settlement=True, trade_elasticity=1.0,
             policy_response=1.0, contagion_strength=1.0, flight_to_reserve=0.5,
             income_linkage="cpi_linked",
             subsistence_scale=FLOOR / BASE_FLOOR,
             automation_max=automation)


def _sized(blocs, g: G, modality: str, u: float):
    """Give every bloc the same programme at a welfare-equivalent size."""
    if u <= 0:
        return list(blocs)
    return [replace(b, modality=modality, size=calibrate(b, g, u)[modality],
                    start=1.0)
            for b in blocs]


def _run(blocs, g: G, modality: str, u: float) -> dict[str, Any]:
    """Need and average rationing per bloc, for one transfer size."""
    result = simulate(_sized(blocs, g, modality, u), g)
    settlement = result["settlement"]
    periods = max(1, len(settlement))
    return {
        b.name: {
            "need": max(row[f"need_{i}"] for row in result["budget"]),
            "rationing": sum(row[f"share_{i}"] for row in settlement) / periods,
        }
        for i, b in enumerate(blocs)
    }


def classify(blocs, runs: dict[float, dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort each position into one of the four states."""
    out: list[dict[str, Any]] = []
    for b in blocs:
        credit, self_sufficiency = (
            float(x) for x in b.name.replace("c", "").replace("s", "").split("_"))
        quiet = runs[0.0][b.name]

        # The need, solved against the transfer that creates it.
        need = quiet["need"]
        fixed_point = None
        for _ in range(FIXED_POINT_ROUNDS):
            candidate = next((u for u in U_CANDIDATES if u >= need), None)
            if candidate is None:
                break
            revised = runs[candidate][b.name]["need"]
            if revised <= candidate:
                fixed_point = candidate
                break
            need = revised

        if quiet["rationing"] > 1e-12:
            status = "already_constrained"
        elif fixed_point is None:
            status = "no_fixed_point"
        elif runs[fixed_point][b.name]["rationing"] > TOLERANCE + 1e-12:
            status = "trapped"
        else:
            status = "feasible"

        out.append({
            "credit": credit,
            "self_sufficiency": self_sufficiency,
            "reserve": bool(b.reserve),
            "status": status,
            "need_static": quiet["need"],
            "need_fixed_point": fixed_point,
            "baseline_rationing": quiet["rationing"],
            "rationing_at_fixed_point": (
                runs[fixed_point][b.name]["rationing"]
                if fixed_point is not None else None),
        })
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    errors: list[dict[str, str]] = []

    total = len(AUTOMATION_LEVELS) * len(MODALITIES) * len(U_CANDIDATES)
    done = 0
    started = time.time()

    for automation in AUTOMATION_LEVELS:
        g = globals_for(automation)
        blocs = grid(STEPS)
        key = f"automation_{automation}"
        report[key] = {}

        for label, modality in MODALITIES.items():
            runs: dict[float, dict[str, Any]] = {}
            failed = False
            for u in U_CANDIDATES:
                try:
                    runs[u] = _run(blocs, g, modality, u)
                except Exception as exc:  # recorded, not swallowed
                    errors.append({"automation": automation, "modality": label,
                                   "u": u, "error": f"{type(exc).__name__}: {exc}"})
                    failed = True
                done += 1
                elapsed = time.time() - started
                print(f"\r  {done}/{total} runs  "
                      f"automation={automation:g} {label:9s} u={u:<5g} "
                      f"{elapsed:5.0f}s", end="", flush=True)
            if failed:
                report[key][label] = []
                continue
            report[key][label] = classify(blocs, runs)

    print()
    path = OUTPUT / "phase_diagram.json"
    payload = {
        "grid_steps": STEPS,
        "tolerance": TOLERANCE,
        "floor": FLOOR,
        "u_candidates": list(U_CANDIDATES),
        "errors": errors,
        **report,
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")
    if errors:
        print(f"{len(errors)} run(s) failed; see the 'errors' key")

    for key, by_modality in report.items():
        print(f"\n=== {key} ===")
        for label, cells in by_modality.items():
            if not cells:
                print(f"  {label:9s} (no data)")
                continue
            counts: dict[str, int] = {}
            for c in cells:
                counts[c["status"]] = counts.get(c["status"], 0) + 1
            summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  {label:9s} {summary}")


if __name__ == "__main__":
    main()
