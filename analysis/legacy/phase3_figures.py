"""Phase 3 figures: convergence patterns against the drift control."""

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

RESULTS = ROOT / "results" / "phase3"
FIG = ROOT / "figures" / "phase3"

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10.5, "legend.fontsize": 9,
        "axes.edgecolor": INK["axis"], "xtick.color": INK["muted"],
        "ytick.color": INK["muted"], "grid.color": INK["grid"],
        "grid.linewidth": 0.6, "grid.linestyle": "-",
        "savefig.bbox": "tight", "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def load() -> list[dict]:
    """Read the evolution results, whether written whole or in shards."""
    paths = sorted(RESULTS.glob("evolution.[0-9]*.jsonl"))
    if not paths:
        paths = [RESULTS / "evolution.jsonl"]
    out: list[dict] = []
    for path in paths:
        out += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return out


def _finish(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.5, points: int = 40):
    """Simple locally-weighted mean, adequate for a visual trend line."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    grid = np.linspace(x.min(), x.max(), points)
    width = frac * (x.max() - x.min())
    out = []
    for g in grid:
        w = np.clip(1 - (np.abs(x - g) / width) ** 3, 0, None) ** 3
        out.append(np.sum(w * y) / np.sum(w) if np.sum(w) > 0 else np.nan)
    return grid, np.asarray(out)


def _arena_for(initial: str) -> str:
    """Which arena a given initial condition is read from.

    The invasion series is read from the symmetric arena, where every non-cash
    modality is planted in its own blocs. Planting only CLT gives it a source
    to be imitated from that the alternatives lack, which is a property of the
    seeding rather than of the blocs. `all_cash` and `random` do not seed at
    all, so the distinction does not arise and they are read from the single
    arena they were run in.
    """
    return "four_way_symmetric" if initial == "invasion" else "four_way"


def fig1_scatter(runs: list[dict], out: Path) -> None:
    """Final CLT share against fx_share, selection versus control."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    for ax, initial in zip(axes, ("invasion", "random")):
        for neutral, color, label in ((False, PALETTE["blue"], "Selection"),
                                      (True, PALETTE["orange"], "Neutral drift")):
            xs, ys = [], []
            for run in runs:
                if (run["initial"] != initial or run["neutral"] != neutral
                        or not run.get("endogenous", True)
                        or run.get("arena", "four_way") != _arena_for(initial)):
                    continue
                for f in run["final"]:
                    xs.append(f["fx_share"])
                    ys.append(f["clt_share"])
            xs, ys = np.asarray(xs), np.asarray(ys)
            ax.scatter(xs, ys, s=3, alpha=0.12, color=color, linewidths=0)
            gx, gy = _lowess(xs, ys)
            ax.plot(gx, gy, color=color, linewidth=2.4, label=label)

        ax.set_xlabel("Foreign-currency share of debt")
        ax.set_title(f"initial mix: {initial}", pad=10, loc="left", fontsize=11)

    axes[0].set_ylabel("Final CLT share of the mix")
    axes[0].legend(loc="upper left")
    fig.suptitle("Under the endogenous stress measure the selection arm tilts "
                 "upward in fx_share; the control stays flat",
                 fontsize=12.5, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig1_fx_gradient.png")


def fig2_counterfactual(out: Path) -> None:
    """Move fx alone, holding every other parameter at its own value."""
    rows = [json.loads(l) for l in
            (RESULTS / "fx_counterfactual.jsonl").read_text().splitlines() if l.strip()]
    base = [r for r in rows if r["sourcing"] == 0.0]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    panels = (("d_fiscal", PALETTE["blue"], "Fiscal cost", "CLT − cash, % of GDP"),
              ("d_leak", PALETTE["orange"], "External leakage", "CLT − cash, % of GDP"),
              ("d_net_endogenous", PALETTE["aqua"], "Endogenous stress",
               "CLT − cash, PV of debt + premium + depreciation"))

    for ax, (metric, color, title, ylab) in zip(axes, panels):
        xs = np.array([r["cf_fx"] for r in base])
        ys = np.array([r[metric] for r in base])
        ax.scatter(xs, ys, s=16, color=color, alpha=0.45, linewidths=0)
        slope, intercept = np.polyfit(xs, ys, 1)
        line = np.linspace(xs.min(), xs.max(), 20)
        ax.plot(line, slope * line + intercept, color=INK["secondary"], linewidth=1.8)
        ax.axhline(0, color=INK["axis"], linewidth=1.1, zorder=0)
        ax.set_xlabel("Counterfactual fx_share (all else held fixed)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}  (slope {slope:+.3f})", pad=10, loc="left", fontsize=11)

    axes[0].annotate("flat: fx does not enter\nthe household block at all",
                     xy=(0.05, 0.12), xycoords="axes fraction", fontsize=8.5,
                     color=INK["secondary"])
    axes[2].annotate("steep: fx acts through\nforeign-currency debt revaluation",
                     xy=(0.05, 0.12), xycoords="axes fraction", fontsize=8.5,
                     color=INK["secondary"])

    fig.suptitle("Within-bloc counterfactual: the static channels are flat in fx; "
                 "only the macro measure responds",
                 fontsize=12.5, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig2_counterfactual.png")


def fig4_sourcing(out: Path) -> None:
    """Does domestic sourcing remove the leakage penalty?"""
    rows = [json.loads(l) for l in
            (RESULTS / "fx_counterfactual.jsonl").read_text().splitlines() if l.strip()]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    colors = {0.0: PALETTE["orange"], 0.5: PALETTE["yellow"], 1.0: PALETTE["aqua"]}
    for sourcing, color in colors.items():
        subset = [r for r in rows if r["sourcing"] == sourcing]
        xs = np.array([r["cf_fx"] for r in subset])
        ys = np.array([r["d_leak"] for r in subset])
        grid = sorted(set(xs))
        means = [ys[xs == x].mean() for x in grid]
        ax.plot(grid, means, color=color, linewidth=2.4, marker="o", markersize=5,
                label=f"domestic sourcing {sourcing:.0%}")

    ax.axhline(0, color=INK["secondary"], linewidth=1.2, zorder=0)
    ax.annotate("above zero: CLT leaks more than cash", xy=(0.02, 0.9),
                xycoords="axes fraction", fontsize=8.5, color=INK["secondary"])
    ax.set_xlabel("Counterfactual fx_share")
    ax.set_ylabel("Leakage gap, CLT − cash (% of GDP)")
    ax.set_title("Sourcing construction at home flips the leakage penalty",
                 pad=12, loc="left")
    ax.legend(loc="lower left")
    _finish(fig, out / "fig4_sourcing.png")


def fig3_paths(runs: list[dict], out: Path) -> None:
    """Mean CLT share over time, by arm, with the spread across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)

    for ax, initial in zip(axes, ("invasion", "random")):
        for neutral, color, label in ((False, PALETTE["blue"], "Selection"),
                                      (True, PALETTE["orange"], "Neutral drift")):
            series = [r["history"] for r in runs
                      if r["initial"] == initial and r["neutral"] == neutral
                      and r.get("endogenous", True)
                      and r.get("arena", "four_way") == _arena_for(initial)]
            if not series:
                continue
            years = [h["year"] for h in series[0]]
            matrix = np.array([[h["mean_clt"] for h in run] for run in series])
            mean = matrix.mean(axis=0)
            lo = np.percentile(matrix, 10, axis=0)
            hi = np.percentile(matrix, 90, axis=0)
            ax.fill_between(years, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(years, mean, color=color, linewidth=2.4, label=label)

        ax.set_xlabel("Year")
        ax.set_title(f"initial mix: {initial}", pad=10, loc="left", fontsize=11)

    axes[0].set_ylabel("Mean CLT share")
    axes[0].legend(loc="upper left")
    fig.suptitle("Selection raises the CLT share above drift; bands are the "
                 "10-90% range across 50 seeds",
                 fontsize=12.5, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig3_paths.png")




def fig5_channels(out: Path) -> None:
    """Which stress component carries the fx effect, and does it survive equal cost?"""
    path = RESULTS / "fx_channels.json"
    if not path.exists():
        print("  [skip] channel decomposition not available")
        return
    summary = json.loads(path.read_text())

    components = ["debt", "risk_premium", "depreciation", "total"]
    modes = [("welfare_equivalent", "Welfare-equivalent", PALETTE["blue"]),
             ("equal_fiscal_shrink_cash", "Equal cost (cash shrunk)", PALETTE["aqua"]),
             ("equal_fiscal_scale_clt", "Equal cost (CLT scaled up)", PALETTE["orange"])]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    width = 0.26
    for offset, (key, label, color) in enumerate(modes):
        values = [summary[key][c]["mean"] for c in components]
        xs = [i + (offset - 1) * width for i in range(len(components))]
        bars = ax.bar(xs, values, width * 0.9, color=color, label=label, linewidth=0)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:+.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3 if v >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=8, color=INK["secondary"])

    ax.axhline(0, color=INK["secondary"], linewidth=1.2, zorder=3)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(["debt", "risk premium", "depreciation", "total"])
    ax.set_ylabel("Slope against counterfactual fx_share")
    ax.set_title("Debt revaluation carries the fx effect — and it vanishes once "
                 "the cost gap is closed", pad=12, loc="left")
    ax.annotate("negative = CLT's advantage widens with fx", xy=(0.015, 0.06),
                xycoords="axes fraction", fontsize=8.5, color=INK["secondary"])
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _finish(fig, out / "fig5_channels.png")


def fig6_three_way(runs: list[dict], out: Path) -> None:
    """CLT against a cash transfer of identical fiscal cost."""
    # The symmetric arena is the main series: seeding only CLT would hand it a
    # source to be imitated from that the control lacks.
    three_way = [r for r in runs if r.get("arena") == "three_way_symmetric"
                 and not r["neutral"]]
    asymmetric = [r for r in runs if r.get("arena") == "three_way"
                  and not r["neutral"]]
    if not three_way:
        print("  [skip] symmetric three-way arena not available")
        return

    labels = three_way[0]["final"][0]["mix_labels"]
    clt_i, cheap_i = labels.index("clt"), labels.index("cash_cheap")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    groups = ["A", "B", "C", "D"]
    width = 0.36
    for offset, (index, color, label) in enumerate((
        (clt_i, PALETTE["blue"], "CLT"),
        (cheap_i, PALETTE["orange"], "cash_cheap (same cost, less welfare)"),
    )):
        values = [float(np.mean([f["mix"][index] for r in three_way
                                 for f in r["final"] if f["group"] == g]))
                  for g in groups]
        xs = [i + (offset - 0.5) * width for i in range(len(groups))]
        axes[0].bar(xs, values, width * 0.9, color=color, label=label, linewidth=0)

    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups)
    axes[0].set_xlabel("Bloc group (fx rises left to right)")
    axes[0].set_ylabel("Mean final share of the mix")
    axes[0].set_title("Symmetric seeding: CLT and the control run level",
                      pad=10, loc="left", fontsize=11)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="x", visible=False)

    fx = np.array([f["fx_share"] for r in three_way for f in r["final"]])
    for index, color, label in ((clt_i, PALETTE["blue"], "CLT"),
                                (cheap_i, PALETTE["orange"], "cash_cheap")):
        ys = np.array([f["mix"][index] for r in three_way for f in r["final"]])
        axes[1].scatter(fx, ys, s=3, alpha=0.10, color=color, linewidths=0)
        gx, gy = _lowess(fx, ys)
        slope = np.polyfit(fx, ys, 1)[0]
        axes[1].plot(gx, gy, color=color, linewidth=2.4,
                     label=f"{label}  (slope {slope:+.3f})")

    axes[1].set_xlabel("Foreign-currency share of debt")
    axes[1].set_ylabel("Final share of the mix")
    axes[1].set_title("Neither tracks fx once seeding is fair",
                      pad=10, loc="left", fontsize=11)
    axes[1].legend(loc="upper left")

    if asymmetric:
        labels_a = asymmetric[0]["final"][0]["mix_labels"]
        clt_a = float(np.mean([f["mix"][labels_a.index("clt")]
                               for r in asymmetric for f in r["final"]]))
        cheap_a = float(np.mean([f["mix"][labels_a.index("cash_cheap")]
                                 for r in asymmetric for f in r["final"]]))
        axes[0].annotate(
            f"seeding CLT alone instead:\nCLT {clt_a:.3f} vs control {cheap_a:.3f}",
            xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8.5,
            color=INK["secondary"])

    fig.suptitle("With both alternatives seeded, the equally cheap cash control "
                 "matches CLT and neither tracks fx",
                 fontsize=12.5, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig6_three_way.png")


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load()
    print("generating phase 3 figures...")
    fig1_scatter(runs, FIG)
    fig2_counterfactual(FIG)
    fig3_paths(runs, FIG)
    fig4_sourcing(FIG)
    fig5_channels(FIG)
    fig6_three_way(runs, FIG)


if __name__ == "__main__":
    main()
