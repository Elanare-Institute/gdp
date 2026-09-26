"""post_employment figures: what falling labour demand does to the trap."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "v8pe"
FIG = ROOT / "figures" / "v8pe"

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "violet": "#4a3aa7"}
CATEGORY_COLOR = {"already": "#7a5195", "trapped": "#c1121f",
                  "no_fixed_point": "#f08a00", "clear": "#f5efe6"}
CATEGORY_LABEL = {
    "already": "Already constrained\n(cannot pay for imports with no transfer)",
    "trapped": "Trapped\n(needs a transfer it cannot settle)",
    "no_fixed_point": "No fixed point\n(each transfer raises the need by more)",
}


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


def _category(cell, tolerance: str = "0") -> str:
    if cell["already_constrained"]:
        return "already"
    if cell["no_fixed_point"]:
        return "no_fixed_point"
    if cell["trapped_by_tolerance"][tolerance]:
        return "trapped"
    return "clear"


def fig1_automation_floor(report, out: Path) -> None:
    """Trapped positions across automation and the subsistence floor."""
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    autos = report["automation_levels"]
    floors = report["floor_shares"]
    for ax, indexation in zip(axs, report["indexations"]):
        grid = np.array([[report["scans"][f"{a:g}|{f:g}|{indexation}"]
                          ["trapped_by_tolerance"]["0"]
                          for f in floors] for a in autos], dtype=float)
        im = ax.imshow(grid, origin="lower", cmap="Reds", aspect="auto",
                       vmin=0, vmax=max(6, grid.max()),
                       extent=(floors[0] - 0.05, floors[-1] + 0.05,
                               autos[0] - 0.1, autos[-1] + 0.1))
        ax.set_xticks(floors); ax.set_yticks(autos)
        ax.set_xlabel("Subsistence floor (share of income)")
        ax.set_ylabel("Automation (share of labour income lost)")
        ax.set_title(f"{indexation}", color=INK["primary"])
        ax.grid(False)
        for i, a in enumerate(autos):
            for j, f in enumerate(floors):
                ax.text(f, a, f"{int(grid[i, j])}", ha="center", va="center",
                        fontsize=9,
                        color="white" if grid[i, j] > grid.max() * 0.6 else INK["primary"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Trapped positions (of 25)")
    fig.suptitle("Neither automation nor the subsistence floor gets a single "
                 "value: both are swept.\nLeft panel is the decisive one — "
                 "indexation maintained.", color=INK["primary"], y=1.05)
    _finish(fig, out)


def fig2_need_vs_afford(report, out: Path) -> None:
    """What the worst-placed bloc needs, against what *it* can settle.

    Taking the maximum affordable transfer across the plane would report the
    most comfortable bloc's headroom against the most exposed bloc's need,
    which is not a comparison of anything. Both are read off the same
    position: the one that needs the most.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    autos = report["automation_levels"]

    for ax, indexation in zip(axs, report["indexations"]):
        needs, affords = [], {t: [] for t in ("0", "0.05", "0.1")}
        for a in autos:
            cells = report["scans"][f"{a:g}|0.85|{indexation}"]["cells"]
            worst = max(cells.values(), key=lambda c: c["u_needed_static"])
            needs.append(worst["u_needed_static"])
            for tol in affords:
                value = worst["u_affordable"][tol]
                affords[tol].append(0.0 if value is None else value)

        ax.plot(autos, needs, "o-", color=PALETTE["orange"], linewidth=2.0,
                markersize=7, label="Transfer needed")
        for tol, colour in zip(("0", "0.05", "0.1"),
                               (PALETTE["violet"], PALETTE["blue"], PALETTE["aqua"])):
            ax.plot(autos, affords[tol], "s--", color=colour, linewidth=1.6,
                    markersize=5, label=f"Can settle (tolerance {float(tol):.0%})")
        ax.fill_between(autos, needs,
                        [max(affords[t][i] for t in affords)
                         for i in range(len(autos))],
                        where=[needs[i] > max(affords[t][i] for t in affords)
                               for i in range(len(autos))],
                        color=PALETTE["orange"], alpha=0.12)
        ax.set_xlabel("Automation (share of labour income lost)")
        ax.set_ylabel("Transfer, share of GDP")
        ax.set_title(indexation, color=INK["primary"])
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("The most exposed position: what it needs against what it can "
                 "settle, at a floor of 85%", color=INK["primary"], y=1.04)
    _finish(fig, out)


def fig3_planes(report, out: Path) -> None:
    """The plane at four automation levels, indexation maintained."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    autos = [a for a in report["automation_levels"] if a in (0.0, 0.2, 0.4, 0.8)]
    order = ["clear", "trapped", "already", "no_fixed_point"]
    cmap = ListedColormap([CATEGORY_COLOR[k] for k in order])

    fig, axs = plt.subplots(1, len(autos), figsize=(4.2 * len(autos), 4.5))
    for ax, a in zip(axs, autos):
        scan = report["scans"][f"{a:g}|0.85|cpi_indexed"]
        cells = scan["cells"]
        axes = sorted({c["credit"] for c in cells.values()})
        grid = np.zeros((len(axes), len(axes)))
        for c in cells.values():
            grid[axes.index(c["credit"]), axes.index(c["self_sufficiency"])] = \
                order.index(_category(c))
        ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=len(order) - 1,
                  aspect="auto", extent=(-0.125, 1.125, -0.125, 1.125))
        ax.set_xticks(axes); ax.set_yticks(axes)
        ax.set_xlabel("Self-sufficiency")
        ax.grid(False)
        n = scan["trapped_by_tolerance"]["0"]
        label = "baseline" if a == 0 else f"automation {a:.0%}"
        ax.set_title(f"{label} — {n} trapped", color=INK["primary"])
        for key, (c, s) in report["archetype_positions"].items():
            if 0.0 <= c <= 1.0 and 0.0 <= s <= 1.0:
                ax.plot(s, c, "o", color=INK["primary"], markersize=4)
                ax.annotate(key, (s, c), textcoords="offset points",
                            xytext=(5, 3), fontsize=8, color=INK["primary"])
    axs[0].set_ylabel("Creditworthiness")
    handles = [Patch(facecolor=CATEGORY_COLOR[k], label=CATEGORY_LABEL[k])
               for k in ("trapped", "already", "no_fixed_point")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.suptitle("Indexation maintained (cpi_indexed), floor 85%, no rationing "
                 "tolerated", color=INK["primary"], y=1.04)
    _finish(fig, out)


def main() -> None:
    style()
    report = json.loads((RESULTS / "postemp.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating post_employment figures...")
    fig1_automation_floor(report, FIG / "fig1_automation_floor")
    fig2_need_vs_afford(report, FIG / "fig2_need_vs_afford")
    fig3_planes(report, FIG / "fig3_planes")


if __name__ == "__main__":
    main()
