"""Does the transfer a bloc needs exceed the transfer it can pay for?

The question the phase exists to answer. Two quantities on the same plane:

  u_needed      the transfer that holds the poorest quantile's budget at the
                subsistence bundle, given what that bundle costs
  u_affordable  the largest transfer the bloc can hand out while its imports
                stay within what it can settle

Both are harder to define than they look, and the first version of this scan
got both wrong.

**A bloc can be rationed with no programme at all.** Then `u_affordable` is
zero, and reporting it as "the transfer it needs exceeds what it can afford"
misdescribes the situation: what it cannot afford is its imports, transfer or
no transfer. Those positions are counted separately as *already constrained*.

**The transfer moves the need.** Handing out money raises demand, which raises
prices, which raises the cost of the subsistence bundle — so the transfer
required at u=0 understates what is required once the transfer is running.
The need is therefore solved as a fixed point, and a bloc where no fixed point
exists — where each increase in the transfer raises the need by more — is a
trap of its own kind.

**Rationing is a matter of degree.** Disqualifying a bloc the moment a single
quarter is rationed is a harsh test, so the affordable transfer is also
reported against tolerances on the average rationing share.
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

RESULTS = ROOT / "results" / "v8u"
STEPS = 5
#: Floor as a share of income. 0.65 is v6's, unexamined; 0.95 leaves a
#: hand-to-mouth household almost nothing free, which is closer to what the
#: term means. The theoretical ceiling is 1.0.
FLOOR_SHARES = (0.65, 0.75, 0.85, 0.95)
LINKAGES = ("growth_linked", "cpi_linked", "fixed_nominal")
U_CANDIDATES = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
BASE_FLOOR = 0.65


def globals_for(floor_share: float, linkage: str, **kw) -> G:
    return G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02,
             indexation="cpi_indexed", settlement=True, trade_elasticity=1.0,
             policy_response=1.0, contagion_strength=1.0,
             flight_to_reserve=0.5,
             subsistence_scale=floor_share / BASE_FLOOR,
             income_linkage=linkage, **kw)


def _with_ubi(blocs, g: G, u: float):
    if u <= 0:
        return list(blocs)
    return [replace(b, modality="ubi", size=calibrate(b, g, u)["ubi"], start=1.0)
            for b in blocs]


#: Average rationing share a bloc is allowed before a transfer counts as
#: unaffordable. Zero is the strict reading — a single rationed quarter
#: disqualifies — and the others say how much the answer depends on that.
TOLERANCES = (0.0, 0.05, 0.10)

#: Iterations allowed when solving the transfer against the need it creates.
FIXED_POINT_ROUNDS = 6


def _need_at(blocs, g: G, u: float) -> dict[str, float]:
    """Peak transfer each bloc needs, when a transfer of `u` is running.

    Peak rather than final: a transfer that covers the bundle in year 1 and
    not in year 30 has not covered it.
    """
    result = simulate(_with_ubi(blocs, g, u), g)
    return {b.name: max(row[f"need_{i}"] for row in result["budget"])
            for i, b in enumerate(blocs)}


def _rationing_at(blocs, g: G, u: float) -> dict[str, float]:
    """Average rationing share over the run, per bloc."""
    settlement = simulate(_with_ubi(blocs, g, u), g)["settlement"]
    n = max(1, len(settlement))
    return {b.name: sum(row[f"share_{i}"] for row in settlement) / n
            for i, b in enumerate(blocs)}


def scan(floor_share: float, linkage: str) -> dict[str, Any]:
    """For each bloc: what it needs, and the most it can settle."""
    g = globals_for(floor_share, linkage)
    blocs = grid(STEPS)

    needs = {u: _need_at(blocs, g, u) for u in U_CANDIDATES}
    rationing = {u: _rationing_at(blocs, g, u) for u in U_CANDIDATES}
    baseline_rationing = rationing[0.0]

    cells: dict[str, Any] = {}
    for b in blocs:
        credit, selfsuf = (float(x) for x in
                           b.name.replace("c", "").replace("s", "").split("_"))

        # The need, solved against the transfer that creates it: start from
        # the need at u=0 and walk up to the smallest candidate that covers
        # the need *it* produces. No such candidate means each increase is
        # outrun by the need it causes.
        need = needs[0.0][b.name]
        fixed_point = None
        for _ in range(FIXED_POINT_ROUNDS):
            candidate = next((u for u in U_CANDIDATES if u >= need), None)
            if candidate is None:
                break
            revised = needs[candidate][b.name]
            if revised <= candidate:
                fixed_point = candidate
                break
            need = revised

        affordable = {}
        for tol in TOLERANCES:
            allowed = [u for u in U_CANDIDATES
                       if rationing[u][b.name] <= tol + 1e-12]
            affordable[f"{tol:g}"] = max(allowed) if allowed else None

        already = baseline_rationing[b.name] > 1e-12
        # Trapped under each tolerance: it needs a transfer, it can pay for
        # its imports without one, and no affordable transfer covers the need
        # it would create. Reported per tolerance because "rationed at all" is
        # a harsher test than the question warrants, and the answer moves.
        trapped_by_tol = {}
        for tol in TOLERANCES:
            limit = affordable[f"{tol:g}"]
            trapped_by_tol[f"{tol:g}"] = (
                needs[0.0][b.name] > 0 and not already
                and (fixed_point is None or limit is None or fixed_point > limit))
        strict = affordable["0"]
        cells[b.name] = {
            "trapped_by_tolerance": trapped_by_tol,
            "credit": credit,
            "self_sufficiency": selfsuf,
            "reserve": b.reserve,
            "baseline_rationing": baseline_rationing[b.name],
            "already_constrained": already,
            "u_needed_static": needs[0.0][b.name],
            "u_needed_fixed_point": fixed_point,
            "no_fixed_point": fixed_point is None and needs[0.0][b.name] > 0,
            "u_affordable": affordable,
            # Trapped: it needs a transfer, it is not already unable to pay
            # for its imports without one, and no affordable transfer covers
            # the need it would create.
            "trapped": (needs[0.0][b.name] > 0 and not already
                        and (fixed_point is None
                             or strict is None
                             or fixed_point > strict)),
        }

    return {
        "floor_share": floor_share, "linkage": linkage, "cells": cells,
        "trapped_count": sum(1 for c in cells.values() if c["trapped"]),
        "trapped_by_tolerance": {
            f"{tol:g}": sum(1 for c in cells.values()
                            if c["trapped_by_tolerance"][f"{tol:g}"])
            for tol in TOLERANCES},
        "already_constrained_count": sum(1 for c in cells.values()
                                         if c["already_constrained"]),
        "no_fixed_point_count": sum(1 for c in cells.values()
                                    if c["no_fixed_point"]),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "archetype_positions": locate_archetypes(),
        "floor_shares": list(FLOOR_SHARES),
        "linkages": list(LINKAGES),
        "scans": {f"{f:g}|{l}": scan(f, l)
                  for f in FLOOR_SHARES for l in LINKAGES},
    }
    path = RESULTS / "trap.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}\n")

    print("=== Positions with no way out (of 25 on the plane) ===")
    print("  trapped: needs a transfer, can pay for its imports without one,")
    print("           and no affordable transfer covers the need it creates\n")
    print(f"  {'floor':>7s} " + "".join(f"{l:>28s}" for l in LINKAGES))
    print(f"  {'':>7s} " + "".join(f"{'trap / already / no-fp':>28s}"
                                   for _ in LINKAGES))
    for floor in FLOOR_SHARES:
        parts = []
        for linkage in LINKAGES:
            sc = report["scans"][f"{floor:g}|{linkage}"]
            parts.append(f"{sc['trapped_count']:10d} /{sc['already_constrained_count']:7d} /"
                         f"{sc['no_fixed_point_count']:7d}")
        print(f"  {floor:7.2f} " + "".join(f"{p:>28s}" for p in parts))

    print("\n  already = rationed even with no transfer at all: it cannot pay")
    print("            for its imports, which is a different situation")
    print("  no-fp   = each increase in the transfer is outrun by the need it")
    print("            creates, so no transfer ever covers the bundle\n")

    print("=== Does the strict rationing test drive the answer? ===")
    print("  Largest affordable transfer under three tolerances on the average")
    print("  rationing share (cpi_linked, floor = 0.85)\n")
    cells = report["scans"]["0.85|cpi_linked"]["cells"]
    print(f"  {'credit':>7s} {'self':>6s} {'base ration':>12s} "
          + "".join(f"{'tol ' + t:>10s}" for t in ("0%", "5%", "10%")))
    for name, c in sorted(cells.items(),
                          key=lambda kv: (kv[1]["credit"], kv[1]["self_sufficiency"])):
        if not (c["already_constrained"] or c["trapped"]):
            continue
        def _fmt(tol: str) -> str:
            value = c["u_affordable"][tol]
            return "—" if value is None else f"{value:.3f}"

        row = "".join(f"{_fmt(t):>10s}" for t in ("0", "0.05", "0.1"))
        print(f"  {c['credit']:7.2f} {c['self_sufficiency']:6.2f} "
              f"{c['baseline_rationing']:12.4f} {row}")

    print("\n=== The need moves with the transfer ===")
    print("  static = need measured with no transfer running")
    print("  fixed  = smallest transfer covering the need it itself creates\n")
    print(f"  {'credit':>7s} {'self':>6s} {'static':>9s} {'fixed point':>12s}")
    for name, c in sorted(cells.items(),
                          key=lambda kv: (kv[1]["credit"], kv[1]["self_sufficiency"])):
        if c["u_needed_static"] <= 0:
            continue
        fp = c["u_needed_fixed_point"]
        print(f"  {c['credit']:7.2f} {c['self_sufficiency']:6.2f} "
              f"{c['u_needed_static']:9.4f} "
              f"{('none' if fp is None else f'{fp:.4f}'):>12s}")


if __name__ == "__main__":
    main()
