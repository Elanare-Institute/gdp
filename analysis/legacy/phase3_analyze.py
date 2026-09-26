"""Phase 3 analysis: separate selection from drift.

The claim under test is that a bloc's external-finance position predicts the
provisioning form it converges to. Any such pattern must be shown against the
neutral-drift control, not against zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "phase3"


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


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """OLS slope of ys on xs."""
    if len(xs) < 3 or xs.std() == 0:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def _select(runs: list[dict], initial: str, neutral: bool,
            endogenous: bool = True, arena: str = "four_way") -> list[dict]:
    """Runs from one arm.

    `arena` must be given explicitly: the file holds both the asymmetric
    four-way arena (only CLT planted) and the symmetric one (each non-cash
    modality planted in its own blocs), and pooling them would mix a seeding
    artefact into the selection estimate.
    """
    return [r for r in runs if r["initial"] == initial
            and r["neutral"] == neutral
            and r.get("endogenous", True) == endogenous
            and r.get("arena", "four_way") == arena]


def spread_stats(runs: list[dict], initial: str, neutral: bool,
                 endogenous: bool = True, arena: str = "four_way") -> dict[str, Any]:
    """Whether and how fast CLT spreads from the invaders."""
    subset = _select(runs, initial, neutral, endogenous, arena)
    if not subset:
        return {}
    finals = np.array([r["history"][-1]["adopter_share"] for r in subset])
    starts = np.array([r["history"][0]["adopter_share"] for r in subset])
    spread = finals > starts + 1e-9
    # Years until adoption first exceeds a quarter of the population.
    times = []
    for run in subset:
        hit = next((h["year"] for h in run["history"] if h["adopter_share"] >= 0.25), None)
        if hit is not None:
            times.append(hit)
    return {
        "n_runs": len(subset),
        "share_spread": float(spread.mean()),
        "final_adopter_share": float(finals.mean()),
        "final_adopter_sd": float(finals.std(ddof=1)) if len(finals) > 1 else 0.0,
        "share_reaching_quarter": len(times) / len(subset),
        "median_years_to_quarter": float(np.median(times)) if times else None,
    }


def fx_gradient(runs: list[dict], initial: str, neutral: bool,
                endogenous: bool = True, arena: str = "four_way") -> dict[str, Any]:
    """Slope of final CLT share on fx_share, pooled within each run.

    Reported per run so the spread across seeds is visible: a mean slope that
    is large but inconsistent across seeds is path dependence, not selection.
    """
    slopes, finals = [], []
    for run in _select(runs, initial, neutral, endogenous, arena):
        fx = np.array([f["fx_share"] for f in run["final"]])
        clt = np.array([f["clt_share"] for f in run["final"]])
        slopes.append(_slope(fx, clt))
        finals.append(clt.mean())
    slopes = np.array([s for s in slopes if not np.isnan(s)])
    finals = np.array(finals)
    return {
        "n_runs": int(len(slopes)),
        "slope_mean": float(slopes.mean()) if len(slopes) else float("nan"),
        "slope_sd": float(slopes.std(ddof=1)) if len(slopes) > 1 else float("nan"),
        "slope_share_positive": float((slopes > 0).mean()) if len(slopes) else float("nan"),
        "mean_clt": float(finals.mean()) if len(finals) else float("nan"),
        "mean_clt_sd": float(finals.std(ddof=1)) if len(finals) > 1 else float("nan"),
    }


def welch(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Welch's t-test; unequal variances are expected between the two arms."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan")}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": float("nan"), "df": float("nan")}
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return {"t": float(t), "df": float(df)}


def group_means(runs: list[dict], initial: str, neutral: bool,
                arena: str = "four_way") -> dict[str, float]:
    """Mean final CLT share by archetype group."""
    buckets: dict[str, list[float]] = {}
    for run in _select(runs, initial, neutral, True, arena):
        for f in run["final"]:
            buckets.setdefault(f["group"], []).append(f["clt_share"])
    return {g: float(np.mean(v)) for g, v in sorted(buckets.items())}


def _arena_for(initial: str) -> str:
    """Which arena a given initial condition is read from.

    The invasion series is read from the symmetric arena: planting only CLT
    hands it an imitation source the alternatives lack, so the asymmetric
    arena measures the seeding as much as the selection. `all_cash` and
    `random` do not seed, so they are read from the single arena they ran in.
    """
    return "four_way_symmetric" if initial == "invasion" else "four_way"


def main() -> None:
    runs = load()
    report: dict[str, Any] = {"n_runs": len(runs)}

    print("Final CLT share, selection vs neutral drift (endogenous stress)\n")
    header = (f"{'initial':10s} {'arm':9s} {'mean CLT':>9s} {'sd':>7s} "
              f"{'fx slope':>9s} {'sd':>7s} {'>0':>6s}")
    print(header)
    print("-" * len(header))

    for initial in ("invasion", "all_cash", "random"):
        arms = {}
        for neutral in (False, True):
            stats = fx_gradient(runs, initial, neutral, True, _arena_for(initial))
            if not stats["n_runs"]:
                continue
            arms["neutral" if neutral else "selection"] = stats
            label = "neutral" if neutral else "selection"
            print(f"{initial:10s} {label:9s} {stats['mean_clt']:9.4f} "
                  f"{stats['mean_clt_sd']:7.4f} {stats['slope_mean']:9.4f} "
                  f"{stats['slope_sd']:7.4f} {stats['slope_share_positive']:6.0%}")
        if not arms:
            continue
        report[initial] = arms

        arena = _arena_for(initial)
        sel = _select(runs, initial, False, True, arena)
        neu = _select(runs, initial, True, True, arena)
        level = welch(np.array([np.mean([f["clt_share"] for f in r["final"]]) for r in sel]),
                      np.array([np.mean([f["clt_share"] for f in r["final"]]) for r in neu]))
        slope = welch(
            np.array([_slope(np.array([f["fx_share"] for f in r["final"]]),
                             np.array([f["clt_share"] for f in r["final"]])) for r in sel]),
            np.array([_slope(np.array([f["fx_share"] for f in r["final"]]),
                             np.array([f["clt_share"] for f in r["final"]])) for r in neu]))
        # Effect size on the level, so a significant-but-tiny gap is visible.
        a = np.array([np.mean([f["clt_share"] for f in r["final"]]) for r in sel])
        b = np.array([np.mean([f["clt_share"] for f in r["final"]]) for r in neu])
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        cohen_d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")
        report[initial]["welch_level"] = level
        report[initial]["welch_slope"] = slope
        report[initial]["cohen_d_level"] = cohen_d
        report[initial]["level_gap"] = float(a.mean() - b.mean())
        print(f"{'':10s} {'vs ctrl':9s} level t={level['t']:+.2f} (d={cohen_d:+.2f}, "
              f"gap={a.mean() - b.mean():+.4f})  slope t={slope['t']:+.2f}")
        print()

    print("Spread from the invaders (main series):")
    for neutral in (False, True):
        stats = spread_stats(runs, "invasion", neutral, True, _arena_for("invasion"))
        if not stats:
            continue
        report[f"spread_{'neutral' if neutral else 'selection'}"] = stats
        label = "neutral" if neutral else "selection"
        print(f"  {label:9s} spread in {stats['share_spread']:.0%} of seeds, "
              f"final adopters {stats['final_adopter_share']:.1%} "
              f"(sd {stats['final_adopter_sd']:.3f}), "
              f"reached 25% in {stats['share_reaching_quarter']:.0%}")

    print("\nSensitivity: reduced-form stress (1 + 2*fx) x leakage, invasion start")
    arena_rf = _arena_for("invasion")
    for neutral in (False, True):
        stats = fx_gradient(runs, "invasion", neutral, False, arena_rf)
        if not stats["n_runs"]:
            continue
        report[f"reduced_form_{'neutral' if neutral else 'selection'}"] = stats
        label = "neutral" if neutral else "selection"
        print(f"  {label:9s} mean CLT {stats['mean_clt']:.4f}  "
              f"fx slope {stats['slope_mean']:+.4f} "
              f"(>0 in {stats['slope_share_positive']:.0%})")

    print("\nMean final CLT share by group (selection arm):")
    for initial in ("invasion", "all_cash", "random"):
        means = group_means(runs, initial, False, _arena_for(initial))
        ctrl = group_means(runs, initial, True, _arena_for(initial))
        if not means:
            continue
        report[f"groups_{initial}"] = {"selection": means, "neutral": ctrl}
        cells = "  ".join(f"{g}:{means[g]:.3f}({ctrl[g]:.3f})" for g in sorted(means))
        print(f"  {initial:10s} {cells}")
    print("  (control in parentheses)")

    (RESULTS / "analysis.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'analysis.json'}")


if __name__ == "__main__":
    main()
