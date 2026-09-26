"""The final phase diagram: which provisioning forms are usable where."""

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

RESULTS = ROOT / "results" / "v8final"
FIG = ROOT / "figures" / "v8final"

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
MOD_LABEL = {"ubi": "UBI", "cash_t": "Targeted cash",
             "voucher": "Food voucher", "clt": "In-kind housing"}
MOD_SHORT = {"ubi": "U", "cash_t": "C", "voucher": "V", "clt": "H"}
#: How many of the four are open at a position.
COUNT_COLOURS = ["#c1121f", "#e36414", "#f6bd60", "#84a98c", "#2a6f5f"]


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
        "axes.titlesize": 11.5, "axes.labelsize": 10.5, "legend.fontsize": 9,
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


def _draw(ax, cells, modalities, title, archetypes) -> None:
    """One plane: how many forms are open, and which."""
    from matplotlib.colors import ListedColormap
    axes = sorted({c["credit"] for c in cells.values()})
    counts = np.zeros((len(axes), len(axes)))
    labels: dict[tuple[int, int], str] = {}
    for c in cells.values():
        i, j = axes.index(c["credit"]), axes.index(c["self_sufficiency"])
        open_now = [m for m in modalities if c["modalities"].get(m)]
        counts[i, j] = len(open_now)
        labels[(i, j)] = "".join(MOD_SHORT[m] for m in open_now) or "—"
    ax.imshow(counts, origin="lower", cmap=ListedColormap(COUNT_COLOURS),
              vmin=0, vmax=len(modalities), aspect="auto",
              extent=(-0.125, 1.125, -0.125, 1.125))
    for (i, j), text in labels.items():
        ax.text(axes[j], axes[i], text, ha="center", va="center", fontsize=8.5,
                color="white" if counts[i, j] <= 1 else INK["primary"])
    ax.set_xticks(axes); ax.set_yticks(axes)
    ax.set_xlabel("Self-sufficiency")
    ax.set_title(title, color=INK["primary"])
    ax.grid(False)
    for key, (c, s) in archetypes.items():
        if 0.0 <= c <= 1.0 and 0.0 <= s <= 1.0:
            ax.plot(s, c, "o", color=INK["primary"], markersize=4)
            ax.annotate(key, (s, c), textcoords="offset points",
                        xytext=(5, 3), fontsize=8, color=INK["primary"])


def fig1_base(report, out: Path) -> None:
    """The main plane, baseline against automation."""
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, a in zip(axs, report["automation_levels"]):
        label = "baseline" if a == 0 else f"automation {a:.0%}"
        _draw(ax, report["base"][f"{a:g}"]["cells"], report["modalities"],
              label, report["archetype_positions"])
    axs[0].set_ylabel("Creditworthiness")
    fig.suptitle("Which provisioning forms a bloc can actually run\n"
                 "U = UBI, C = targeted cash, V = food voucher, "
                 "H = in-kind housing; — none",
                 color=INK["primary"], y=1.06)
    _finish(fig, out)


def fig2_sections(report, out: Path, dimension: str, label: str) -> None:
    """One structural dimension at both ends, at both automation levels."""
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.4))
    for row, a in enumerate(report["automation_levels"]):
        for col, end in enumerate(("low", "high")):
            key = f"{dimension}_{end}"
            if key not in report["sections"]:
                continue
            auto = "baseline" if a == 0 else f"automation {a:.0%}"
            _draw(axs[row][col], report["sections"][key][f"{a:g}"]["cells"],
                  report["modalities"], f"{label} {end} — {auto}",
                  report["archetype_positions"])
        axs[row][0].set_ylabel("Creditworthiness")
    fig.suptitle(f"Section: {label}", color=INK["primary"], y=1.02)
    _finish(fig, out)


def main() -> None:
    style()
    report = json.loads((RESULTS / "phase_diagram.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating phase diagram...")
    fig1_base(report, FIG / "fig1_plane")
    for dimension, label in (("land_share", "Land rent share"),
                             ("eps_supply", "Housing supply elasticity"),
                             ("m_H", "Construction import share"),
                             ("clt_build_rate", "Build capacity")):
        fig2_sections(report, FIG / f"fig2_{dimension}", dimension, label)


if __name__ == "__main__":
    main()
