"""Diagnose why m_H and clt_domestic_sourcing differ in influence.

Both enter the model through a single expression,
``imp_gov = m_H * (1 - s) * clt_build``, so on the import channel alone they
should be near-symmetric. They are not. The candidate explanation is the
second channel the sourcing lever has and m_H does not: the cost markup
``kappa``, which raises ``clt_build`` itself.

The test conditions on kappa. If the asymmetry is a kappa artifact it should
shrink sharply in the sub-sample where kappa is near zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "phase1"
TARGET = "welfare.D.d_leak_abs"


def load(name: str) -> list[dict]:
    path = RESULTS / name
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def variance_explained(rows: list[dict], param: str, target: str,
                       bins: int = 8) -> float:
    """First-order effect: variance of the conditional mean, normalised.

    A model-free stand-in for a first-order Sobol index that works on any
    sub-sample, including ones that break the Saltelli design.
    """
    xs = np.asarray([r[param] for r in rows], dtype=float)
    ys = np.asarray([r[target] for r in rows], dtype=float)
    ok = ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]
    if ys.size < 50 or ys.var() == 0:
        return float("nan")

    edges = np.quantile(xs, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    means, weights = [], []
    for i in range(bins):
        m = (xs >= edges[i]) & (xs < edges[i + 1])
        if m.sum() >= 10:
            means.append(ys[m].mean())
            weights.append(m.sum())
    if len(means) < 3:
        return float("nan")
    means = np.asarray(means)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    grand = float((means * weights).sum())
    between = float((weights * (means - grand) ** 2).sum())
    return between / float(ys.var())


def report(rows: list[dict], label: str) -> dict[str, float]:
    out = {}
    for param in ("m_H", "clt_domestic_sourcing", "sourcing_cost_kappa"):
        if param not in rows[0]:
            continue
        out[param] = variance_explained(rows, param, TARGET)
    print(f"  {label:38s} " + "  ".join(
        f"{k}={v:.4f}" for k, v in out.items()))
    return out


def main() -> None:
    print("First-order variance explained for", TARGET)
    print("(model-free; comparable across sub-samples)\n")

    for name, tag in (("sobol.jsonl", "headline ranges"),
                      ("sobol_matched.jsonl", "matched ranges")):
        rows = load(name)
        print(f"{tag} (n={len(rows)}):")
        report(rows, "full sample")

        # Condition on kappa: keep only draws where the cost markup is small.
        low = [r for r in rows if r.get("sourcing_cost_kappa", 1.0) < 0.1]
        if low:
            report(low, f"kappa < 0.1 sub-sample (n={len(low)})")
        high = [r for r in rows if r.get("sourcing_cost_kappa", 0.0) > 0.4]
        if high:
            report(high, f"kappa > 0.4 sub-sample (n={len(high)})")
        print()

    print("Interpretation:")
    print("  If the m_H / sourcing gap narrows sharply at low kappa, the")
    print("  asymmetry runs through the fiscal-cost channel, not the import")
    print("  channel. If it persists, the import channel is genuinely")
    print("  asymmetric and the 'policy beats structure' reading stands.")


if __name__ == "__main__":
    main()
