"""Where the transfer a bloc needs exceeds the transfer it can pay for."""

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

RESULTS = ROOT / "results" / "v8u"
FIG = ROOT / "figures" / "v8u"

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "violet": "#4a3aa7"}
LINK_LABEL = {"growth_linked": "Income tracks growth",
              "cpi_linked": "Income tracks prices (main)",
              "fixed_nominal": "Income fixed in nominal terms"}
LINK_COLOR = {"growth_linked": PALETTE["aqua"],
              "cpi_linked": PALETTE["orange"],
              "fixed_nominal": PALETTE["violet"]}


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


CATEGORY_COLOR = {"already": "#7a5195", "trapped": "#c1121f",
                  "no_fixed_point": "#f08a00", "clear": "#f5efe6"}
CATEGORY_LABEL = {
    "already": "Already constrained\n(cannot pay for imports with no transfer)",
    "trapped": "Trapped\n(needs a transfer it cannot settle)",
    "no_fixed_point": "No fixed point\n(each transfer raises the need by more)",
}


def _category(cell, tolerance: str = "0") -> str:
    """Which of the three situations a position is in.

    They are different problems and were conflated in the first version of
    this scan. A bloc rationed with no programme at all is not one that cannot
    afford its transfer; it cannot afford its imports, which no transfer
    design addresses.
    """
    if cell["already_constrained"]:
        return "already"
    if cell["no_fixed_point"]:
        return "no_fixed_point"
    if cell["trapped_by_tolerance"][tolerance]:
        return "trapped"
    return "clear"


def _draw_plane(ax, cells, tolerance: str, title: str) -> None:
    from matplotlib.colors import ListedColormap
    order = ["clear", "trapped", "already", "no_fixed_point"]
    cmap = ListedColormap([CATEGORY_COLOR[k] for k in order])
    axes = sorted({c["credit"] for c in cells.values()})
    grid = np.zeros((len(axes), len(axes)))
    for c in cells.values():
        i = axes.index(c["credit"])
        j = axes.index(c["self_sufficiency"])
        grid[i, j] = order.index(_category(c, tolerance))
    ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=len(order) - 1,
              aspect="auto", extent=(-0.125, 1.125, -0.125, 1.125))
    ax.set_xticks(axes); ax.set_yticks(axes)
    ax.set_xlabel("Self-sufficiency")
    ax.set_title(title, color=INK["primary"], fontsize=11)
    ax.grid(False)


def fig1_trap_regions(report, out: Path) -> None:
    """The plane by category, across floors and rationing tolerances."""
    from matplotlib.patches import Patch
    floors = report["floor_shares"]
    tolerances = ["0", "0.05", "0.1"]
    tol_label = {"0": "no rationing tolerated", "0.05": "5% average rationing",
                 "0.1": "10% average rationing"}

    fig, axs = plt.subplots(len(tolerances), len(floors),
                            figsize=(3.9 * len(floors), 3.9 * len(tolerances)))
    for row, tol in enumerate(tolerances):
        for col, floor in enumerate(floors):
            scan = report["scans"][f"{floor:g}|cpi_linked"]
            n = sum(1 for c in scan["cells"].values()
                    if _category(c, tol) == "trapped")
            _draw_plane(axs[row][col], scan["cells"], tol,
                        f"Floor {floor:.0%} — {n} trapped")
        axs[row][0].set_ylabel(f"{tol_label[tol]}\n\nCreditworthiness")

    for ax in axs.flat:
        for key, (c, s) in report["archetype_positions"].items():
            if 0.0 <= c <= 1.0 and 0.0 <= s <= 1.0:
                ax.plot(s, c, "o", color=INK["primary"], markersize=4)
                ax.annotate(key, (s, c), textcoords="offset points",
                            xytext=(5, 3), fontsize=8, color=INK["primary"])

    handles = [Patch(facecolor=CATEGORY_COLOR[k], label=CATEGORY_LABEL[k])
               for k in ("trapped", "already", "no_fixed_point")]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle("Three different situations, kept apart. Income tracks prices "
                 "(cpi_linked).\nThe region depends on both the subsistence "
                 "floor and on how much rationing counts as unaffordable.",
                 color=INK["primary"], y=1.02)
    _finish(fig, out)


def fig2_sweep(report, out: Path) -> None:
    """How each category responds to the floor and to the linkage rule."""
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.4))
    floors = report["floor_shares"]
    xs = [100 * f for f in floors]

    for linkage in report["linkages"]:
        counts = [report["scans"][f"{f:g}|{linkage}"]["trapped_by_tolerance"]["0"]
                  for f in floors]
        axs[0].plot(xs, counts, "o-", color=LINK_COLOR[linkage],
                    label=LINK_LABEL[linkage], linewidth=1.8, markersize=6)
    axs[0].set_ylabel("Trapped positions (of 25)")
    axs[0].set_title("Trapped: needs a transfer it cannot settle",
                     color=INK["primary"])
    axs[0].legend(loc="upper left", fontsize=8)

    for linkage in report["linkages"]:
        counts = [report["scans"][f"{f:g}|{linkage}"]["no_fixed_point_count"]
                  for f in floors]
        axs[1].plot(xs, counts, "o-", color=LINK_COLOR[linkage],
                    linewidth=1.8, markersize=6)
    axs[1].set_ylabel("Positions with no fixed point (of 25)")
    axs[1].set_title("No fixed point: only where income is not price-linked",
                     color=INK["primary"])

    for tol, colour in zip(("0", "0.05", "0.1"),
                           (PALETTE["violet"], PALETTE["blue"], PALETTE["aqua"])):
        counts = [report["scans"][f"{f:g}|cpi_linked"]["trapped_by_tolerance"][tol]
                  for f in floors]
        axs[2].plot(xs, counts, "o-", color=colour, linewidth=1.8, markersize=6,
                    label=f"tolerance {float(tol):.0%}")
    axs[2].set_ylabel("Trapped positions (of 25)")
    axs[2].set_title("How much rationing counts as unaffordable",
                     color=INK["primary"])
    axs[2].legend(loc="upper left", fontsize=8)

    for ax in axs:
        ax.set_xlabel("Subsistence floor (% of income)")

    fig.suptitle("No single number: the region depends on the floor, on the "
                 "income linkage, and on the tolerance",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def main() -> None:
    style()
    report = json.loads((RESULTS / "trap.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating phase U figures...")
    fig1_trap_regions(report, FIG / "fig1_trap_regions")
    fig2_sweep(report, FIG / "fig2_sweep")


if __name__ == "__main__":
    main()
