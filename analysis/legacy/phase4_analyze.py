"""Phase 4: read the Monte Carlo.

Three kinds of quantity are reported, kept separate on purpose:

  threshold-free   severity, final debt, final e — distributions, no cutoff
  survival         Kaplan-Meier for time to insolvency
  threshold-crossing  crisis onsets, insolvency rate

The thresholds are model constants (`THR`, `default_thr`). They are reported
because TASKS.md asks for a crisis probability, but they are not treated as
structural boundaries: the modality comparison is carried by the
threshold-free distributions and by the paired per-path differences.

Modality differences are paired: every modality sees the same shock path for
a given seed (common random numbers), so the difference is taken path by path
and its distribution is reported, not the difference of two means.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "phase4"
BLOCS = ("A", "B", "C", "D")
MODALITIES = ("none", "ubi", "cash_t", "voucher", "clt")
HORIZON = 30.0


def load() -> list[dict[str, Any]]:
    """Read the Monte Carlo output, whether written whole or in shards."""
    paths = sorted(RESULTS.glob("montecarlo.[0-9]*.jsonl"))
    if not paths:
        paths = [RESULTS / "montecarlo.jsonl"]
    out: list[dict[str, Any]] = []
    for path in paths:
        out += [json.loads(l) for l in path.read_text().splitlines() if l]
    return out


def kaplan_meier(times: list[float | None], horizon: float = HORIZON
                 ) -> tuple[list[float], list[float]]:
    """Survival curve for time-to-insolvency.

    A path that never becomes insolvent is censored at the horizon. With a
    fixed horizon and no other loss to follow-up this reduces to one minus the
    empirical CDF, but it is written as Kaplan-Meier so that a future variable
    horizon does not silently change the estimand.
    """
    n = len(times)
    events = sorted(t for t in times if t is not None and t <= horizon)
    xs, ys = [0.0], [1.0]
    surv, at_risk, i = 1.0, n, 0
    while i < len(events):
        t = events[i]
        d = sum(1 for e in events[i:] if e == t)
        surv *= (1 - d / at_risk)
        at_risk -= d
        xs.append(t); ys.append(surv)
        i += d
    xs.append(horizon); ys.append(surv)
    return xs, ys


def _key(rec: dict) -> tuple:
    return (rec["regime"], rec["modality"], rec["u"], rec["pattern"])


def summarise_cell(recs: list[dict], bloc: str) -> dict[str, Any]:
    """Distributional summary for one (regime, modality, u, pattern, bloc)."""
    sev = np.array([r[bloc]["severity"] for r in recs])
    sev_na = np.array([r[bloc]["severity_na"] for r in recs])
    debt = np.array([r[bloc]["debt"] for r in recs])
    fx = np.array([r[bloc]["e"] for r in recs])
    onsets = np.array([r[bloc]["onsets"] for r in recs])
    ins = [r[bloc]["insolvent_year"] for r in recs]
    return {
        "n": len(recs),
        "severity_p50": float(np.percentile(sev, 50)),
        "severity_p90": float(np.percentile(sev, 90)),
        "severity_p99": float(np.percentile(sev, 99)),
        "severity_mean": float(sev.mean()),
        "severity_na_p90": float(np.percentile(sev_na, 90)),
        "debt_p50": float(np.percentile(debt, 50)),
        "debt_p90": float(np.percentile(debt, 90)),
        "e_p90": float(np.percentile(fx, 90)),
        "crisis_prob": float((onsets > 0).mean()),
        "insolvency_prob": float(sum(1 for t in ins if t is not None) / len(ins)),
    }


def paired_diff(a: list[dict], b: list[dict], bloc: str, field: str) -> dict[str, float]:
    """Per-path difference a - b on the same seeds (common random numbers)."""
    ma = {r["seed"]: r[bloc][field] for r in a}
    mb = {r["seed"]: r[bloc][field] for r in b}
    shared = sorted(set(ma) & set(mb))
    d = np.array([ma[s] - mb[s] for s in shared], dtype=float)
    if d.size == 0:
        return {"n": 0}
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return {
        "n": int(d.size),
        "mean": float(d.mean()),
        "p50": float(np.percentile(d, 50)),
        "share_negative": float((d < 0).mean()),
        "t": float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan"),
    }


def main() -> None:
    recs = load()
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        cells[_key(r)].append(r)

    report: dict[str, Any] = {"n_records": len(recs)}

    # 1. Distributions per cell.
    summary: dict[str, Any] = {}
    for key, rs in sorted(cells.items()):
        regime, modality, u, pattern = key
        for bloc in BLOCS:
            summary[f"{regime}|{modality}|{u}|{pattern}|{bloc}"] = summarise_cell(rs, bloc)
    report["cells"] = summary

    # 2. Survival curves (main series: all shocks, staggered, u=0.05).
    curves: dict[str, Any] = {}
    for regime in sorted({r["regime"] for r in recs}):
        for modality in MODALITIES:
            rs = cells.get((regime, modality, 0.05, "staggered")) or \
                 cells.get((regime, modality, 0.03, "staggered"))
            if not rs:
                continue
            for bloc in BLOCS:
                xs, ys = kaplan_meier([r[bloc]["insolvent_year"] for r in rs])
                curves[f"{regime}|{modality}|{bloc}"] = {"t": xs, "s": ys}
    report["survival"] = curves

    # 3. CLT against each alternative, paired on the shock path.
    diffs: dict[str, Any] = {}
    for regime in sorted({r["regime"] for r in recs}):
        for u in sorted({r["u"] for r in recs}):
            for pattern in sorted({r["pattern"] for r in recs}):
                clt = cells.get((regime, "clt", u, pattern))
                if not clt:
                    continue
                for other in ("cash_t", "voucher", "ubi"):
                    rs = cells.get((regime, other, u, pattern))
                    if not rs:
                        continue
                    for bloc in BLOCS:
                        for field in ("severity", "debt", "e"):
                            k = f"{regime}|u{u}|{pattern}|clt-{other}|{bloc}|{field}"
                            diffs[k] = paired_diff(clt, rs, bloc, field)
    report["paired"] = diffs

    out = RESULTS / "analysis.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(recs)} records, {len(cells)} cells)")

    # Console view of the main series.
    print("\nMain series (all shocks, staggered, u=0.05) — bloc D")
    head = f"{'modality':9s} {'sev p50':>8s} {'sev p90':>8s} {'sev p99':>8s} {'crisis':>7s} {'insolv':>7s}"
    print(head); print("-" * len(head))
    for m in MODALITIES:
        k = f"all|{m}|{0.05 if m != 'none' else 0.03}|staggered|D"
        c = summary.get(k)
        if c:
            print(f"{m:9s} {c['severity_p50']:8.3f} {c['severity_p90']:8.3f} "
                  f"{c['severity_p99']:8.3f} {c['crisis_prob']:7.1%} {c['insolvency_prob']:7.1%}")


if __name__ == "__main__":
    main()
