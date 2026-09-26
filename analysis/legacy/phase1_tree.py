"""Decision boundary for the sign reversal, as a depth-3 tree."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase1_sign import design  # noqa: E402

RESULTS = ROOT / "results" / "phase1"
FIG = ROOT / "figures" / "phase1"
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}


def main() -> None:
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    rows = [json.loads(l) for l in (RESULTS / "lhs.jsonl").read_text().splitlines() if l.strip()]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "font.size": 9})

    blocs = ["A", "B", "D"]
    fig, axes = plt.subplots(1, len(blocs), figsize=(6.4 * len(blocs), 5.4))

    for ax, bloc in zip(axes, blocs):
        key = f"cost.{bloc}.d_leak_abs"
        vals = [r for r in rows if r.get(key) is not None]
        y = np.asarray([1 if r[key] > 0 else 0 for r in vals])
        X, names = design(vals, bloc)

        # class_weight='balanced' so rare reversals are not ignored in favour
        # of the majority label (this lifted bloc D's balanced accuracy from
        # 0.625 to 0.885).
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40,
                                      class_weight="balanced", random_state=0)
        tree.fit(X, y)
        acc = tree.score(X, y)
        # max_depth=2 in the drawing keeps the boxes legible; the reported
        # accuracy is still the depth-3 model's.
        plot_tree(tree, feature_names=names, class_names=["CLT better", "reversal"],
                  filled=True, impurity=False, proportion=True, rounded=True,
                  fontsize=8, ax=ax, max_depth=2, label="root")
        ax.set_title(f"bloc {bloc} — {int(y.sum())}/{len(y)} reversals "
                     f"({y.mean():.1%}), class-weighted depth-3",
                     pad=10, loc="left", fontsize=11)

    fig.suptitle("Where the sign flips: the CLT's own inputs lead in every bloc "
                 "(standardised, class-weighted)",
                 fontsize=12.5, x=0.008, ha="left", y=1.02)
    fig.savefig(FIG / "fig7_sign_tree.png")
    plt.close(fig)
    print("  wrote fig7_sign_tree.png")

    # Report the root split of each tree: the single most informative cut.
    for bloc in blocs:
        key = f"cost.{bloc}.d_leak_abs"
        vals = [r for r in rows if r.get(key) is not None]
        y = np.asarray([1 if r[key] > 0 else 0 for r in vals])
        X, names = design(vals, bloc)
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=40,
                                      class_weight="balanced",
                                      random_state=0).fit(X, y)
        root = tree.tree_.feature[0]
        print(f"  bloc {bloc}: root split on {names[root]} "
              f"(standardised threshold {tree.tree_.threshold[0]:+.3f}), "
              f"accuracy {tree.score(X, y):.3f}")


if __name__ == "__main__":
    main()
