"""Classification quality for the sign models.

Accuracy alone is misleading when reversals are rare: in bloc D a model that
always predicts "no reversal" is already 96.4% accurate. Balanced accuracy and
AUC, both cross-validated, say whether the model is doing better than that.
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
FOLDS = 5
SEED = 20260922


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]

    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision: area under the precision-recall curve.

    More informative than ROC-AUC when the positive class is rare, because its
    baseline is the event rate rather than 0.5.
    """
    order = np.argsort(-score)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    return float((precision * y).sum() / n_pos)


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean of sensitivity and specificity."""
    pos, neg = y == 1, y == 0
    sens = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    spec = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    return 0.5 * (sens + spec)


def cross_validate(X: np.ndarray, y: np.ndarray, kind: str,
                   weight: str | None = None) -> dict[str, float]:
    """Stratified k-fold evaluation of one classifier."""
    rng = np.random.default_rng(SEED)
    folds = np.empty(len(y), dtype=int)
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % FOLDS

    acc, bal, auc, pr = [], [], [], []
    for k in range(FOLDS):
        train, test = folds != k, folds == k
        if y[test].sum() == 0 or (1 - y[test]).sum() == 0:
            continue
        if kind == "logistic":
            beta = logistic(X[train], y[train])
            score = np.column_stack([np.ones(test.sum()), X[test]]) @ beta
            pred = (score > 0).astype(float)
        else:
            from sklearn.tree import DecisionTreeClassifier
            # class_weight='balanced' so a rare positive class is not simply
            # ignored in favour of the majority label.
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40,
                                          class_weight=weight,
                                          random_state=0).fit(X[train], y[train])
            score = tree.predict_proba(X[test])[:, 1]
            pred = tree.predict(X[test]).astype(float)
        acc.append(float((pred == y[test]).mean()))
        bal.append(_balanced_accuracy(y[test], pred))
        auc.append(_auc(y[test], score))
        pr.append(_pr_auc(y[test], score))

    return {"accuracy": float(np.mean(acc)), "balanced_accuracy": float(np.mean(bal)),
            "auc": float(np.mean(auc)), "pr_auc": float(np.mean(pr))}


def main() -> None:
    rows = [json.loads(l) for l in (RESULTS / "lhs.jsonl").read_text().splitlines() if l.strip()]
    out: dict[str, dict] = {}

    print(f"{FOLDS}-fold cross-validated, equal-cost basis, LHS {len(rows)} points\n")
    header = (f"{'bloc':5s} {'reversals':>12s} {'majority':>9s} | "
              f"{'logit acc':>9s} {'bal acc':>8s} {'AUC':>6s} {'PR-AUC':>6s} | "
              f"{'tree-bal':>9s} {'bal acc':>8s} {'AUC':>6s}")
    print(header)
    print("-" * len(header))

    for bloc in ("A", "B", "D"):
        key = f"cost.{bloc}.d_leak_abs"
        vals = [r for r in rows if r.get(key) is not None]
        y = np.asarray([1.0 if r[key] > 0 else 0.0 for r in vals])
        if y.sum() < 50:
            continue
        X, _ = design(vals, bloc)
        majority = float(max(y.mean(), 1 - y.mean()))

        lg = cross_validate(X, y, "logistic")
        tr = cross_validate(X, y, "tree")
        trb = cross_validate(X, y, "tree", weight="balanced")
        out[bloc] = {"n": len(y), "n_reversals": int(y.sum()),
                     "event_rate": float(y.mean()),
                     "majority_baseline": majority, "logistic": lg,
                     "tree": tr, "tree_balanced": trb}
        print(f"{bloc:5s} {int(y.sum()):6d}/{len(y):<5d} {majority:9.3f} | "
              f"{lg['accuracy']:9.3f} {lg['balanced_accuracy']:8.3f} {lg['auc']:6.3f} "
              f"{lg['pr_auc']:6.3f} | "
              f"{trb['accuracy']:9.3f} {trb['balanced_accuracy']:8.3f} {trb['auc']:6.3f}")

    print("\nTree: unweighted vs class_weight='balanced' (balanced accuracy)")
    for bloc, r in out.items():
        print(f"  {bloc}: {r['tree']['balanced_accuracy']:.3f} -> "
              f"{r['tree_balanced']['balanced_accuracy']:.3f}")

    print("\nPR-AUC vs its baseline (the event rate):")
    for bloc, r in out.items():
        print(f"  {bloc}: logit {r['logistic']['pr_auc']:.3f}  "
              f"(baseline {r['event_rate']:.3f}, "
              f"lift x{r['logistic']['pr_auc'] / r['event_rate']:.1f})")

    (RESULTS / "sign_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'sign_eval.json'}")


if __name__ == "__main__":
    main()
