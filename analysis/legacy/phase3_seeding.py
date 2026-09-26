"""Is table 1 / table 2 / section 6 an artefact of asymmetric seeding?

The four-way invasion arena planted CLT and nothing else. Proportional
imitation can only spread what someone already holds, so a modality that is
the sole invader has a source to be copied from that the alternatives lack.
This script re-reads the same three results under both seedings:

  four_way            only CLT planted (the original main series)
  four_way_symmetric  every non-cash modality planted in its own blocs

and prints, for each: the invasion fx gradient (table 1), the group-by-group
selection effect (table 2), and the spread statistics (section 6).

Every number is a difference from the matched neutral-drift control run on
the same seeds; nothing is compared against zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase3_analyze import (  # noqa: E402
    _select, _slope, fx_gradient, spread_stats, welch,
)

RESULTS = ROOT / "results" / "phase3"
ARENAS = ("four_way", "four_way_symmetric")
LABEL = {"four_way": "asymmetric (CLT only)",
         "four_way_symmetric": "symmetric (each seeded)"}


def load() -> list[dict[str, Any]]:
    """Read the evolution results, whether written whole or in shards.

    The runner writes one file per process (`evolution.<shard>.jsonl`) because
    a shared multiprocessing pool wedged on this workload. A single
    `evolution.jsonl` is still read if present.
    """
    paths = sorted(RESULTS.glob("evolution.[0-9]*.jsonl"))
    if not paths:
        paths = [RESULTS / "evolution.jsonl"]
    out: list[dict[str, Any]] = []
    for p in paths:
        out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def _run_means(runs: list[dict]) -> np.ndarray:
    """Mean final CLT share, one value per run."""
    return np.array([np.mean([f["clt_share"] for f in r["final"]]) for r in runs])


def _run_slopes(runs: list[dict]) -> np.ndarray:
    """Within-run fx slope of the final CLT share, one value per run."""
    return np.array([_slope(np.array([f["fx_share"] for f in r["final"]]),
                            np.array([f["clt_share"] for f in r["final"]]))
                     for r in runs])


def group_effect(runs: list[dict], arena: str) -> dict[str, dict[str, float]]:
    """Selection minus drift, group by group, paired on the seed.

    Runs are paired by population seed so the difference is within-seed: the
    drift control sees the same 31 blocs and the same imitation draws.
    """
    sel = {r["population_seed"]: r for r in _select(runs, "invasion", False, True, arena)}
    neu = {r["population_seed"]: r for r in _select(runs, "invasion", True, True, arena)}
    shared = sorted(set(sel) & set(neu))
    out: dict[str, dict[str, float]] = {}
    groups = sorted({f["group"] for r in sel.values() for f in r["final"]})
    for g in groups:
        diffs = []
        for seed in shared:
            a = np.mean([f["clt_share"] for f in sel[seed]["final"] if f["group"] == g])
            b = np.mean([f["clt_share"] for f in neu[seed]["final"] if f["group"] == g])
            diffs.append(a - b)
        d = np.array(diffs)
        sd = d.std(ddof=1)
        out[g] = {
            "selection": float(np.mean([np.mean([f["clt_share"] for f in sel[s]["final"]
                                                 if f["group"] == g]) for s in shared])),
            "neutral": float(np.mean([np.mean([f["clt_share"] for f in neu[s]["final"]
                                               if f["group"] == g]) for s in shared])),
            "diff": float(d.mean()),
            "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float("nan"),
            "n": len(d),
        }
    return out


def main() -> None:
    runs = load()
    report: dict[str, Any] = {}

    print("=" * 74)
    print("TABLE 1 — invasion fx gradient, by seeding")
    print("=" * 74)
    head = f"{'arena':26s} {'arm':10s} {'mean CLT':>9s} {'fx slope':>9s} {'lvl t':>7s} {'slp t':>7s}"
    print(head); print("-" * len(head))
    for arena in ARENAS:
        sel = _select(runs, "invasion", False, True, arena)
        neu = _select(runs, "invasion", True, True, arena)
        if not sel or not neu:
            print(f"{LABEL[arena]:26s} (no runs)")
            continue
        lvl = welch(_run_means(sel), _run_means(neu))
        slp = welch(_run_slopes(sel), _run_slopes(neu))
        for neutral, label in ((False, "selection"), (True, "neutral")):
            st = fx_gradient(runs, "invasion", neutral, True, arena)
            t_lvl = f"{lvl['t']:+7.2f}" if not neutral else " " * 7
            t_slp = f"{slp['t']:+7.2f}" if not neutral else " " * 7
            print(f"{LABEL[arena] if not neutral else '':26s} {label:10s} "
                  f"{st['mean_clt']:9.4f} {st['slope_mean']:+9.4f} {t_lvl} {t_slp}")
        report[arena] = {"welch_level": lvl, "welch_slope": slp,
                         "selection": fx_gradient(runs, "invasion", False, True, arena),
                         "neutral": fx_gradient(runs, "invasion", True, True, arena)}

    print()
    print("=" * 74)
    print("TABLE 2 — selection effect by group (paired on seed), by seeding")
    print("=" * 74)
    for arena in ARENAS:
        eff = group_effect(runs, arena)
        if not eff:
            continue
        report.setdefault(arena, {})["groups"] = eff
        print(f"\n{LABEL[arena]}  (n={next(iter(eff.values()))['n']} seeds)")
        print(f"  {'group':6s} {'selection':>10s} {'neutral':>9s} {'diff':>9s} {'t':>8s}")
        for g, v in eff.items():
            print(f"  {g:6s} {v['selection']:10.4f} {v['neutral']:9.4f} "
                  f"{v['diff']:+9.4f} {v['t']:+8.1f}")

    print()
    print("=" * 74)
    print("SECTION 6 — spread from the invaders, by seeding")
    print("=" * 74)
    for arena in ARENAS:
        print(f"\n{LABEL[arena]}")
        for neutral, label in ((False, "selection"), (True, "neutral")):
            st = spread_stats(runs, "invasion", neutral, True, arena)
            if not st:
                continue
            report.setdefault(arena, {})[f"spread_{label}"] = st
            print(f"  {label:10s} spread in {st['share_spread']:.0%} of seeds, "
                  f"final adopters {st['final_adopter_share']:.1%}, "
                  f"reached 25% in {st['share_reaching_quarter']:.0%}")

    out = RESULTS / "seeding.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
