"""Bootstrap confidence intervals for the sign-regression coefficients.

Bloc D has 144 events against 10 predictors (EPV 14.4), which is close enough
to the conventional floor of 10 that point estimates alone would overstate the
precision. Intervals are resampled, and a ridge-penalised fit is reported
alongside so the effect of shrinkage is visible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase1_sign import design, logistic  # noqa: E402

RESULTS = ROOT / "results" / "phase1"
N_BOOT = 1000
SEED = 20260922


def ridge_logistic(X: np.ndarray, y: np.ndarray, lam: float,
                   iters: int = 200) -> np.ndarray:
    """L2-penalised logistic regression (the intercept is not penalised)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xb.shape[1])
    penalty = lam * np.eye(Xb.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        W = p * (1 - p) + 1e-9
        grad = Xb.T @ (y - p) - penalty @ beta
        H = Xb.T @ (Xb * W[:, None]) + penalty
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def bootstrap(X: np.ndarray, y: np.ndarray, names: list[str],
              n_boot: int = N_BOOT) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals, resampling within each class.

    Stratified resampling keeps the event count fixed across replicates, so the
    intervals reflect coefficient uncertainty rather than variation in how many
    reversals happen to be drawn.
    """
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    draws = np.empty((n_boot, X.shape[1]))
    failures = 0

    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        try:
            draws[b] = logistic(X[idx], y[idx])[1:]
        except np.linalg.LinAlgError:
            draws[b] = np.nan
            failures += 1

    out = {}
    for j, name in enumerate(names):
        col = draws[:, j]
        col = col[~np.isnan(col)]
        out[name] = {
            "lo": float(np.percentile(col, 2.5)),
            "hi": float(np.percentile(col, 97.5)),
            "median": float(np.median(col)),
            "crosses_zero": bool(np.percentile(col, 2.5) < 0 < np.percentile(col, 97.5)),
        }
    if failures:
        print(f"    ({failures} bootstrap fits failed and were dropped)")
    return out


def main() -> None:
    rows = [json.loads(l) for l in (RESULTS / "lhs.jsonl").read_text().splitlines() if l.strip()]
    report: dict[str, dict] = {}

    for bloc in ("A", "B", "D"):
        key = f"cost.{bloc}.d_leak_abs"
        vals = [r for r in rows if r.get(key) is not None]
        y = np.asarray([1.0 if r[key] > 0 else 0.0 for r in vals])
        if y.sum() < 50:
            continue
        X, names = design(vals, bloc)
        epv = float(y.sum()) / len(names)

        plain = logistic(X, y)[1:]
        ridge = ridge_logistic(X, y, lam=len(y) * 0.01)[1:]
        ci = bootstrap(X, y, names)

        print(f"\nbloc {bloc}: {int(y.sum())} events / {len(names)} predictors "
              f"(EPV {epv:.1f}), {N_BOOT} bootstrap replicates")
        print(f"  {'parameter':22s} {'point':>7s} {'ridge':>7s} "
              f"{'95% CI':>20s}  stable?")
        order = np.argsort(-np.abs(plain))
        for j in order:
            c = ci[names[j]]
            flag = "crosses 0" if c["crosses_zero"] else "yes"
            print(f"  {names[j]:22s} {plain[j]:+7.2f} {ridge[j]:+7.2f} "
                  f"[{c['lo']:+7.2f},{c['hi']:+7.2f}]  {flag}")

        report[bloc] = {
            "n_events": int(y.sum()), "n_predictors": len(names), "epv": epv,
            "point": {n: float(v) for n, v in zip(names, plain)},
            "ridge": {n: float(v) for n, v in zip(names, ridge)},
            "ci": ci,
        }

    (RESULTS / "sign_ci.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'sign_ci.json'}")


if __name__ == "__main__":
    main()
