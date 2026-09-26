"""Phase 4 figures: survival curves and severity distributions."""

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

RESULTS = ROOT / "results" / "phase4"
FIG = ROOT / "figures" / "phase4"

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
MOD_COLOR = {"none": INK["muted"], "ubi": PALETTE["violet"],
             "cash_t": PALETTE["blue"], "voucher": PALETTE["yellow"],
             "clt": PALETTE["aqua"]}
MOD_LABEL = {"none": "No transfer", "ubi": "UBI", "cash_t": "Targeted cash",
             "voucher": "Food voucher", "clt": "CLT"}
BLOCS = ("A", "B", "C", "D")
BLOC_LABEL = {"A": "A (reserve)", "B": "B (non-reserve adv.)",
              "C": "C (emerging)", "D": "D (developing)"}


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


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {path.name}.png/.pdf")


def fig1_survival(report: dict, out: Path, regime: str = "all") -> None:
    """Kaplan-Meier survival against insolvency, by bloc, modalities overlaid."""
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), sharey=True)
    any_event = False
    for ax, bloc in zip(axes, BLOCS):
        for mod in MOD_LABEL:
            key = f"{regime}|{mod}|{bloc}"
            c = report["survival"].get(key)
            if not c:
                continue
            if min(c["s"]) < 1.0:
                any_event = True
            ax.step(c["t"], c["s"], where="post", color=MOD_COLOR[mod],
                    label=MOD_LABEL[mod], linewidth=1.6)
        ax.set_title(BLOC_LABEL[bloc], color=INK["primary"])
        ax.set_xlabel("Years since start")
        ax.set_xlim(0, 30)
    axes[0].set_ylabel("Survival (not insolvent)")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(loc="lower left")
    note = ("" if any_event else
            "  No path reaches insolvency in any bloc; all curves sit at 1.0.")
    fig.suptitle(f"Time to insolvency, {regime} shocks{note}",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def fig2_severity(report: dict, out: Path) -> None:
    """Severity quantiles by modality and shock regime, bloc D."""
    regimes = ["none", "rate_only", "food_only", "stop_only", "all"]
    mods = [m for m in MOD_LABEL]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), sharey=True)
    for ax, q in zip(axes, ("p50", "p90", "p99")):
        width = 0.15
        xs = np.arange(len(regimes))
        for i, mod in enumerate(mods):
            vals = []
            for reg in regimes:
                u = 0.03 if mod == "none" else 0.05
                c = report["cells"].get(f"{reg}|{mod}|{u}|staggered|D")
                vals.append(c[f"severity_{q}"] if c else np.nan)
            ax.bar(xs + (i - 2) * width, vals, width, color=MOD_COLOR[mod],
                   label=MOD_LABEL[mod])
        ax.set_xticks(xs)
        ax.set_xticklabels(regimes, rotation=20)
        ax.set_title(f"Severity, {q}", color=INK["primary"])
    axes[0].set_ylabel("Severity (bloc D)")
    axes[-1].legend(loc="upper left")
    fig.suptitle("Crisis severity by shock regime — bloc D, u=0.05, staggered",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def fig3_paired(report: dict, out: Path) -> None:
    """CLT minus each alternative, paired on the shock path, by regime."""
    regimes = ["rate_only", "food_only", "stop_only", "all"]
    others = ["cash_t", "voucher", "ubi"]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), sharey=True)
    for ax, bloc in zip(axes, BLOCS):
        xs = np.arange(len(regimes))
        for i, other in enumerate(others):
            vals, errs = [], []
            for reg in regimes:
                d = report["paired"].get(
                    f"{reg}|u0.05|staggered|clt-{other}|{bloc}|severity", {})
                vals.append(d.get("mean", np.nan))
                errs.append(0.0)
            ax.bar(xs + (i - 1) * 0.26, vals, 0.26,
                   color=[PALETTE["blue"], PALETTE["yellow"], PALETTE["violet"]][i],
                   label=f"CLT − {MOD_LABEL[other]}")
        ax.axhline(0, color=INK["secondary"], linewidth=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(regimes, rotation=20)
        ax.set_title(BLOC_LABEL[bloc], color=INK["primary"])
    axes[0].set_ylabel("Mean paired severity difference")
    axes[-1].legend(loc="best")
    fig.suptitle("CLT against each alternative on the same shock paths "
                 "(negative = CLT milder)", color=INK["primary"], y=1.04)
    _finish(fig, out)


def main() -> None:
    style()
    report = json.loads((RESULTS / "analysis.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    fig1_survival(report, FIG / "fig1_survival")
    fig2_severity(report, FIG / "fig2_severity")
    fig3_paired(report, FIG / "fig3_paired")


if __name__ == "__main__":
    main()
