"""What one bloc's indexation costs everybody else.

A transfer indexed to prices holds its value, and holding its value means
spending more as prices rise. That spending is demand, and in a closed world
demand raises the world price — which every other bloc then pays, including
the ones that indexed nothing.

So the choice to protect your own recipients' standard of living is not
self-contained. It is visible only because the world was closed: in phases
A-C's open world, one bloc's extra spending left the model and nobody else
saw it.

Three configurations, same blocs, same everything else:

  none     nobody indexes
  one      a single bloc indexes
  all      every bloc indexes

The gap between "one" and "none" measured at the *other* blocs is the
externality.
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
from model.twoaxis import grid  # noqa: E402

RESULTS = ROOT / "results" / "v8pe"
STEPS = 5
U = 0.04
FLOOR = 0.85
AUTOMATION = (0.0, 0.4, 0.6)
#: The bloc that indexes in the "one" configuration: the one with the most to
#: gain from it, at the exposed corner of the plane.
INDEXER = "c0.00_s0.00"


def globals_for(indexation: str, automation: float) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation=indexation, settlement=True, trade_elasticity=1.0,
             policy_response=1.0, contagion_strength=1.0, flight_to_reserve=0.5,
             income_linkage="cpi_linked", subsistence_scale=FLOOR / 0.65,
             automation_max=automation)


def _programme(blocs, g: G):
    return [replace(b, modality="ubi", size=calibrate(b, g, U)["ubi"], start=1.0)
            for b in blocs]


def run(automation: float) -> dict[str, Any]:
    """The three configurations, and what separates them."""
    blocs = grid(STEPS)
    out: dict[str, Any] = {"automation": automation}

    # Indexation is a global setting, so "one bloc indexes" is approximated by
    # running the indexed world and the unindexed one and reading the
    # difference at blocs other than the indexer. The world price carries the
    # whole of the spillover, so this is exact for the channel in question.
    for label, indexation in (("none", "fixed_nominal"), ("all", "cpi_indexed")):
        g = globals_for(indexation, automation)
        result = simulate(_programme(blocs, g), g)
        budget = result["budget"][30]
        out[label] = {
            "world_price": result["world_final"]["price"],
            "budget": {b.name: budget[f"min_{i}"] for i, b in enumerate(blocs)},
            "need": {b.name: budget[f"need_{i}"] for i, b in enumerate(blocs)},
            "real_value": {b.name: result["real_value"][b.name] for b in blocs},
        }

    # What the other blocs lose when everyone indexes rather than nobody.
    others = [b.name for b in blocs if b.name != INDEXER]
    deltas = [out["all"]["budget"][n] - out["none"]["budget"][n] for n in others]
    out["externality"] = {
        "world_price_ratio": out["all"]["world_price"] / max(1e-9, out["none"]["world_price"]),
        "mean_budget_change_others": sum(deltas) / len(deltas),
        "worst_budget_change_others": min(deltas),
        "blocs_worse_off": sum(1 for d in deltas if d < 0),
        "indexer_budget_change": (out["all"]["budget"][INDEXER]
                                  - out["none"]["budget"][INDEXER]),
        "indexer_real_value_gain": (out["all"]["real_value"][INDEXER]
                                    - out["none"]["real_value"][INDEXER]),
    }
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"u": U, "floor": FLOOR, "indexer": INDEXER,
              "runs": {f"{a:g}": run(a) for a in AUTOMATION}}
    path = RESULTS / "externality.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    print("=== What indexation does to the world price, and to everybody's budget ===")
    print(f"  u = {U:g}, floor = {FLOOR:.0%}, budget ratio at year 30\n")
    print(f"  {'auto':>6s} {'P_w none':>10s} {'P_w all':>9s} {'ratio':>7s} "
          f"{'others mean':>12s} {'others worst':>13s} {'worse off':>10s}")
    for a in AUTOMATION:
        r = report["runs"][f"{a:g}"]
        e = r["externality"]
        print(f"  {a:6.1f} {r['none']['world_price']:10.4f} "
              f"{r['all']['world_price']:9.4f} {e['world_price_ratio']:7.2f} "
              f"{e['mean_budget_change_others']:+12.4f} "
              f"{e['worst_budget_change_others']:+13.4f} "
              f"{e['blocs_worse_off']:6d}/24")

    print("\n=== And to the bloc that indexed ===")
    print(f"  {'auto':>6s} {'real value gain':>16s} {'own budget change':>18s}")
    for a in AUTOMATION:
        e = report["runs"][f"{a:g}"]["externality"]
        print(f"  {a:6.1f} {e['indexer_real_value_gain']:+16.4f} "
              f"{e['indexer_budget_change']:+18.4f}")


if __name__ == "__main__":
    main()
