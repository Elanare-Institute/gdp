"""Read the orthogonal-population sensitivity against the archetype baseline.

Both populations are read the same way: selection is compared with the
neutral-drift control run on the same seeds, and the fx gradient is taken on
the per-bloc difference (selection minus drift). What differs is only whether
a group fixed effect is needed.

  archetype    fx is confounded with structure, so the pooled slope is
               reported next to the within-group slope; they disagree.
  orthogonal   fx is independent of structure by construction, so the pooled
               slope is the estimate. There are no groups to fix.
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
ARENA = "four_way_symmetric"


def _read(pattern: str) -> list[dict[str, Any]]:
    paths = sorted(RESULTS.glob(pattern))
    return [json.loads(l) for p in paths for l in p.read_text().splitlines() if l.strip()]


def _paired(runs: list[dict]) -> tuple[dict[int, dict], dict[int, dict]]:
    """Selection and drift runs keyed by seed."""
    sel = {r["population_seed"]: r for r in runs if not r["neutral"]}
    neu = {r["population_seed"]: r for r in runs if r["neutral"]}
    return sel, neu


def _ols(x: np.ndarray, y: np.ndarray, dummies: np.ndarray | None = None
         ) -> tuple[float, float]:
    """Slope of y on x and its t statistic, optionally with fixed effects."""
    cols = [np.ones(len(x))] if dummies is None else [dummies]
    X = np.column_stack(cols + [x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(x) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return float(beta[-1]), float(beta[-1] / se[-1])


def _welch(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"t": float(t), "gap": float(a.mean() - b.mean()),
            "d": float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")}


def analyse(runs: list[dict], label: str, grouped: bool) -> dict[str, Any]:
    sel, neu = _paired(runs)
    seeds = sorted(set(sel) & set(neu))
    if not seeds:
        print(f"{label}: no runs"); return {}

    # 1. Level effect, one observation per seed.
    a = np.array([np.mean([f["clt_share"] for f in sel[s]["final"]]) for s in seeds])
    b = np.array([np.mean([f["clt_share"] for f in neu[s]["final"]]) for s in seeds])
    level = _welch(a, b)

    # 2. fx gradient on the per-bloc paired difference.
    fx, d, grp = [], [], []
    for s in seeds:
        for f1, f0 in zip(sel[s]["final"], neu[s]["final"]):
            fx.append(f1["fx_share"])
            d.append(f1["clt_share"] - f0["clt_share"])
            grp.append(f1["group"])
    fx, d, grp = np.array(fx), np.array(d), np.array(grp)
    slope, t = _ols(fx, d)
    out = {"n_seeds": len(seeds), "n_blocs": len(fx), "level": level,
           "mean_clt_selection": float(a.mean()), "mean_clt_neutral": float(b.mean()),
           "fx_slope_pooled": slope, "fx_t_pooled": t,
           "fx_sd": float(fx.std()), "fx_min": float(fx.min()), "fx_max": float(fx.max()),
           "n_low_fx": int((fx < 0.1).sum())}

    print(f"\n--- {label} ---")
    print(f"  seeds {len(seeds)}, blocs {len(fx)}, fx in [{fx.min():.2f}, {fx.max():.2f}], "
          f"{(fx < 0.1).sum()} blocs with fx<0.1")
    print(f"  level:  selection {a.mean():.4f} vs drift {b.mean():.4f}  "
          f"gap {level['gap']:+.4f}  t={level['t']:+.2f}  d={level['d']:+.2f}")
    print(f"  fx slope (pooled, no fixed effect): {slope:+.4f}  t={t:+.2f}")

    if grouped:
        groups = sorted(set(grp))
        D = np.column_stack([(grp == g).astype(float) for g in groups])
        slope_fe, t_fe = _ols(fx, d, D)
        out["fx_slope_fe"] = slope_fe
        out["fx_t_fe"] = t_fe
        print(f"  fx slope (group fixed effects):     {slope_fe:+.4f}  t={t_fe:+.2f}")
        print(f"    -> the two disagree: fx is confounded with structure here")
    else:
        print("    -> no groups; fx is independent of structure by construction")

    # 3. Per-seed slope, so the estimate does not rest on pooling blocs.
    per_seed = []
    for s in seeds:
        x = np.array([f["fx_share"] for f in sel[s]["final"]])
        y = (np.array([f["clt_share"] for f in sel[s]["final"]])
             - np.array([f["clt_share"] for f in neu[s]["final"]]))
        per_seed.append(np.polyfit(x, y, 1)[0])
    ps = np.array(per_seed)
    t_ps = ps.mean() / (ps.std(ddof=1) / np.sqrt(len(ps)))
    out["fx_slope_per_seed"] = float(ps.mean())
    out["fx_t_per_seed"] = float(t_ps)
    out["fx_share_positive"] = float((ps > 0).mean())
    print(f"  fx slope (per seed, n={len(ps)}):  {ps.mean():+.4f}  t={t_ps:+.2f}  "
          f">0 in {(ps > 0).mean():.0%} of seeds")
    return out


def main() -> None:
    arch = [r for r in _read("evolution.[0-9]*.jsonl")
            if r.get("arena") == ARENA and r["initial"] == "invasion"
            and r.get("endogenous", True)]
    ortho = _read("orthogonal*.jsonl")

    print("=" * 74)
    print("POPULATION SENSITIVITY — archetype groups vs orthogonal fx")
    print("=" * 74)
    report = {
        "archetype": analyse(arch, "archetype population (A1/B5/C10/D15)", True),
        "orthogonal": analyse(ortho, "orthogonal population (fx ~ U[0,0.85])", False),
    }
    out = RESULTS / "orthogonal_analysis.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
