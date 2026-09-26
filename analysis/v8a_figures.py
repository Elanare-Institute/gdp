"""Phase A/B figures: the world price responds to world demand."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "v8a"
FIG = ROOT / "figures" / "v8a"

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


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {path.name}.png/.pdf")


def _annual_inflation(path) -> list[float]:
    """Year-on-year world inflation, in percent.

    The path is sampled annually, so this is the simple year-on-year change.
    The first point has no predecessor and is reported as zero rather than
    dropped, so the series lines up with the years axis.
    """
    prices = [p for _, p in path]
    out = [0.0]
    for prev, now in zip(prices, prices[1:]):
        out.append(100.0 * (now / prev - 1.0) if prev > 0 else 0.0)
    return out


def fig1_coalitions(report: dict, out: Path) -> None:
    """World price path as more blocs run the same programme."""
    labels = {"none": "No transfer", "D": "D only", "C+D": "C and D",
              "B+C+D": "B, C and D", "A+B+C+D": "All four"}
    colors = [INK["muted"], PALETTE["yellow"], PALETTE["aqua"],
              PALETTE["blue"], PALETTE["orange"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    for (key, label), color in zip(labels.items(), colors):
        cell = report["coalitions"][key]
        years = [y for y, _ in cell["world_path"]]
        axes[0].plot(years, _annual_inflation(cell["world_path"]), color=color,
                     label=label, linewidth=1.8)
        axes[1].plot(years, [100 * gp for _, gp in cell["gap_path"]], color=color,
                     linewidth=1.8)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("World inflation (% per year)")
    axes[0].set_title("World inflation", color=INK["primary"])
    axes[0].legend(loc="upper left")
    axes[1].axhline(0, color=INK["secondary"], linewidth=0.9)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("World output gap (%)")
    axes[1].set_title("World demand against capacity", color=INK["primary"])
    fig.suptitle(f"Cash leaves a country but not the world "
                 f"(universal cash, u = {report['u']:g})",
                 color=INK["primary"], y=1.03)
    _finish(fig, out)


def fig2_monotone(report: dict, out: Path) -> None:
    """Final world price against coalition size and against transfer size."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))

    order = ["none", "D", "C+D", "B+C+D", "A+B+C+D"]
    counts = [0, 1, 2, 3, 4]
    prices = [report["coalitions"][k]["world_price"] for k in order]
    axes[0].plot(counts, prices, "o-", color=PALETTE["blue"], linewidth=1.8,
                 markersize=7)
    axes[0].set_xticks(counts)
    axes[0].set_xlabel("Number of blocs running the programme")
    axes[0].set_ylabel("World price level, year 30")
    axes[0].set_title("More payers, higher world price", color=INK["primary"])

    sizes = sorted(report["sizes"], key=float)
    axes[1].plot([float(s) * 100 for s in sizes],
                 [report["sizes"][s]["world_price"] for s in sizes],
                 "o-", color=PALETTE["orange"], linewidth=1.8, markersize=7)
    axes[1].set_xlabel("Transfer size (% of GDP), every bloc")
    axes[1].set_ylabel("World price level, year 30")
    axes[1].set_title("Bigger transfers, higher world price", color=INK["primary"])
    fig.suptitle("The world price is monotone in world demand",
                 color=INK["primary"], y=1.03)
    _finish(fig, out)


def fig3_modalities(report: dict, out: Path) -> None:
    """World price by modality at a matched welfare target."""
    labels = {"none": "No transfer", "ubi": "UBI", "cash_t": "Targeted cash",
              "voucher": "Food voucher", "clt": "CLT"}
    colors = {"none": INK["muted"], "ubi": PALETTE["violet"],
              "cash_t": PALETTE["blue"], "voucher": PALETTE["yellow"],
              "clt": PALETTE["aqua"]}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    for key, label in labels.items():
        cell = report["modalities"][key]
        axes[0].plot([y for y, _ in cell["world_path"]],
                     [p for _, p in cell["world_path"]],
                     color=colors[key], label=label, linewidth=1.8)
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("World price level $P_w$")
    axes[0].set_title("World price by modality", color=INK["primary"])
    axes[0].legend(loc="upper left")

    # Plotted as the rise above the no-transfer baseline. On a zero-based axis
    # every bar is dominated by the baseline level the modalities share, and
    # the differences between them — which is what the panel is about —
    # collapse into the top few percent of the bars.
    keys = [k for k in labels if k != "none"]
    base = report["modalities"]["none"]["world_price"]
    axes[1].bar(range(len(keys)),
                [report["modalities"][k]["world_price"] - base for k in keys],
                color=[colors[k] for k in keys])
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels([labels[k] for k in keys], rotation=20)
    axes[1].set_ylabel(f"World price above no-transfer ({base:.3f})")
    axes[1].set_title("Same welfare target, different world pressure",
                      color=INK["primary"])
    fig.suptitle(f"What is handed out changes how much reaches the world market "
                 f"(u = {report['u']:g})", color=INK["primary"], y=1.03)
    _finish(fig, out)


def main() -> None:
    style()
    report = json.loads((RESULTS / "world.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating phase A/B figures...")
    fig1_coalitions(report, FIG / "fig1_coalitions")
    fig2_monotone(report, FIG / "fig2_monotone")
    fig3_modalities(report, FIG / "fig3_modalities")


if __name__ == "__main__":
    main()
