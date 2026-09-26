"""Phase D figures: the plane, and what contagion does to it."""

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

RESULTS = ROOT / "results" / "v8d"
FIG = ROOT / "figures" / "v8d"

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "violet": "#4a3aa7"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
MOD_LABEL = {"none": "No transfer", "ubi": "UBI",
             "cash_t": "Targeted cash", "clt": "In-kind housing"}
MOD_COLOR = {"none": INK["muted"], "ubi": PALETTE["violet"],
             "cash_t": PALETTE["blue"], "clt": PALETTE["aqua"]}


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


def _matrix(cells: dict, field: str, missing: float = np.nan):
    axes = sorted({c["credit"] for c in cells.values()})
    grid = np.full((len(axes), len(axes)), missing, dtype=float)
    for c in cells.values():
        i = axes.index(c["credit"])
        j = axes.index(c["self_sufficiency"])
        value = c[field]
        grid[i, j] = missing if value is None else float(value)
    return axes, grid


def _heat(ax, axes, grid, title, cmap, label, vmin=None, vmax=None):
    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax,
                   extent=(-0.125, 1.125, -0.125, 1.125))
    ax.set_xticks(axes); ax.set_yticks(axes)
    ax.set_xlabel("Self-sufficiency")
    ax.set_title(title, color=INK["primary"])
    ax.grid(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label)
    return im


def fig1_plane(report, out: Path) -> None:
    """Where the constraint binds, with and without contagion.

    The two rows are the same plane. What the second adds is the region that
    only fails once a neighbour's default takes its lenders with it.
    """
    u = report["main_u"]
    fig, axs = plt.subplots(2, 3, figsize=(16.5, 9.0))

    for row, contagion in enumerate((0.0, 1.0)):
        cells = report["plane"][f"ubi|{contagion:g}"]["cells"]
        axes, rationed = _matrix(cells, "first_rationed")
        _heat(axs[row][0], axes, rationed, "First year imports are rationed",
              "YlOrRd_r", "Year (blank = never)", vmin=0, vmax=30)
        axes, default = _matrix(cells, "insolvent_year")
        _heat(axs[row][1], axes, default, "Year the bloc stops being sustainable",
              "YlOrRd_r", "Year (blank = not within 30)", vmin=0, vmax=30)
        axes, housing = _matrix(cells, "housing")
        _heat(axs[row][2], axes, housing, "Housing secured at year 30",
              "YlGnBu", "Start = 100", vmin=110, vmax=200)
        label = "No contagion" if contagion == 0 else "With contagion"
        for ax in axs[row]:
            ax.set_ylabel("Creditworthiness")
        axs[row][0].set_ylabel(f"{label}\n\nCreditworthiness", labelpad=8)

    # Mark the cells that only fail once contagion is on: the positions that
    # are swept in by somebody else's default rather than by their own
    # position.
    plain = report["plane"]["ubi|0"]["cells"]
    spread = report["plane"]["ubi|1"]["cells"]
    for name, cell in spread.items():
        if cell["insolvent_year"] is not None and plain[name]["insolvent_year"] is None:
            axs[1][1].plot(cell["self_sufficiency"], cell["credit"], "s",
                           markerfacecolor="none", markeredgecolor="#c1121f",
                           markersize=16, markeredgewidth=2.0)

    for ax in axs.flat:
        for key, (c, s_) in report["archetype_positions"].items():
            if 0.0 <= c <= 1.0 and 0.0 <= s_ <= 1.0:
                ax.plot(s_, c, "o", color=INK["primary"], markersize=5)
                ax.annotate(key, (s_, c), textcoords="offset points",
                            xytext=(6, 4), fontsize=9, color=INK["primary"])

    fig.suptitle(f"The settlement constraint on the credit x self-sufficiency "
                 f"plane (UBI, u = {u:g}). Red squares: positions swept in by "
                 f"contagion. Black dots: the four archetypes, on the diagonal.",
                 color=INK["primary"], y=1.00)
    _finish(fig, out)


def fig2_contagion(report, out: Path) -> None:
    """What a default taking its neighbours with it does to the world."""
    fig, axs = plt.subplots(1, 3, figsize=(16.0, 4.4))

    for modality in MOD_LABEL:
        for contagion, style_ in ((0.0, "--"), (1.0, "-")):
            run = report["plane"][f"{modality}|{contagion:g}"]
            years = [y for y, _ in run["world_path"]]
            prices = [p for _, p in run["world_path"]]
            axs[0].plot(years, prices, style_, color=MOD_COLOR[modality],
                        linewidth=1.8,
                        label=f"{MOD_LABEL[modality]}"
                              f"{' + contagion' if contagion else ''}")
    axs[0].set_xlabel("Year"); axs[0].set_ylabel("World price level")
    axs[0].set_title("World price: dashed = no contagion", color=INK["primary"])
    axs[0].legend(loc="upper left", fontsize=7.5)

    mods = list(MOD_LABEL)
    width = 0.36
    xs = np.arange(len(mods))
    for k, contagion in enumerate((0.0, 1.0)):
        counts = [len(report["plane"][f"{m}|{contagion:g}"]["defaults"]) for m in mods]
        axs[1].bar(xs + (k - 0.5) * width, counts, width,
                   color=PALETTE["blue"] if k == 0 else PALETTE["orange"],
                   label="No contagion" if k == 0 else "With contagion")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels([MOD_LABEL[m] for m in mods], rotation=15)
    axs[1].set_ylabel("Blocs that stop being sustainable")
    axs[1].set_title("Out of 25 blocs on the plane", color=INK["primary"])
    axs[1].legend(loc="upper right")

    us = sorted({float(k.split("|")[0]) for k in report["by_u"]})
    for contagion, color, label in ((0.0, PALETTE["blue"], "No contagion"),
                                    (1.0, PALETTE["orange"], "With contagion")):
        counts = [len(report["by_u"][f"{u:g}|{contagion:g}"]["defaults"]) for u in us]
        axs[2].plot([100 * u for u in us], counts, "o-", color=color,
                    label=label, linewidth=1.8, markersize=6)
    axs[2].set_xlabel("Transfer size (% of GDP)")
    axs[2].set_ylabel("Blocs that stop being sustainable")
    axs[2].set_title("Contagion lowers the level a world can carry",
                     color=INK["primary"])
    axs[2].legend(loc="upper left")

    fig.suptitle("A default does not stay with the borrower: lenders withdraw "
                 "from everything that resembles it",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def fig3_modality_plane(report, out: Path) -> None:
    """Which modality leaves recipients with most, across the plane."""
    mods = ["ubi", "cash_t", "clt"]
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    grids = {}
    for modality in mods:
        axes, grids[modality] = _matrix(
            report["plane"][f"{modality}|1"]["cells"], "housing")
    lo = min(np.nanmin(g) for g in grids.values())
    hi = max(np.nanmax(g) for g in grids.values())
    for ax, modality in zip(axs, mods):
        _heat(ax, axes, grids[modality], MOD_LABEL[modality],
              "YlGnBu", "Housing at year 30 (start = 100)", vmin=lo, vmax=hi)
    fig.suptitle(f"What recipients end up with, by modality and position "
                 f"(u = {report['main_u']:g}, with contagion)",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def main() -> None:
    style()
    report = json.loads((RESULTS / "plane.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating phase D figures...")
    fig1_plane(report, FIG / "fig1_plane")
    fig2_contagion(report, FIG / "fig2_contagion")
    fig3_modality_plane(report, FIG / "fig3_modality_plane")


if __name__ == "__main__":
    main()
