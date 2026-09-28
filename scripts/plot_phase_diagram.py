"""Phase diagram: institutional feasibility on the credit x self-sufficiency plane.

One colour per position, decided by which modalities are open there:

  green   every form works
  amber   housing in kind works where universal cash does not
  red     nothing works — the need has no fixed point, or no transfer that
          reaches it can be settled
  grey    already constrained: rationed with no transfer at all, so its
          imports are unaffordable whatever is handed out. A different problem
          from the one the diagram is about, and kept separate for that reason

Reads output/phase_diagram.json; writes the figure as PDF and PNG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "output"

PANELS = (("automation_0.0", "(a) Baseline"),
          ("automation_0.4", "(b) Automation ceiling = 0.4"))
#: Order matters: it is the colormap's index order.
CATEGORIES = ("all", "housing_only", "none", "constrained")
COLOURS = {"all": "#2a6f5f", "housing_only": "#eda100",
           "none": "#c1121f", "constrained": "#9a9a95"}
LABELS = {
    "all": "All four forms feasible",
    "housing_only": "In-kind housing feasible, universal cash not",
    "none": "No form feasible",
    "constrained": "Already constrained (rationed with no transfer)",
}
INK = {"primary": "#0b0b0b", "muted": "#898781", "axis": "#c3c2b7"}


def style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.edgecolor": INK["axis"], "xtick.color": INK["muted"],
        "ytick.color": INK["muted"], "savefig.bbox": "tight",
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _index(cells) -> dict[tuple[float, float], dict]:
    return {(c["credit"], c["self_sufficiency"]): c for c in cells}


def categorise(panel: dict[str, list[dict]],
               position: tuple[float, float]) -> str:
    """Which of the four states this position is in.

    `already_constrained` is read off any modality — it is a property of the
    bloc, not of what it hands out — and takes precedence, because a bloc that
    cannot pay for its imports without a programme is not being told anything
    by a diagram about programmes.
    """
    status = {name: _index(cells)[position]["status"]
              for name, cells in panel.items() if cells}
    if any(s == "already_constrained" for s in status.values()):
        return "constrained"
    workable = {name for name, s in status.items() if s == "feasible"}
    if len(workable) == len(status):
        return "all"
    if "clt" in workable and "ubi" not in workable:
        return "housing_only"
    if not workable:
        return "none"
    # Anything else — some cash form works, housing does not, or a mixture —
    # is reported as "no form feasible" only when nothing works, so this falls
    # to the residual: partial feasibility that the four-colour scheme does not
    # separate. Coloured as housing_only would misstate it; as all, likewise.
    return "none" if "clt" not in workable else "housing_only"


def draw(ax, panel: dict[str, list[dict]], title: str, cells) -> None:
    axes = sorted({c["credit"] for c in cells})
    grid = np.zeros((len(axes), len(axes)))
    for c in cells:
        position = (c["credit"], c["self_sufficiency"])
        category = categorise(panel, position)
        # Rows are self-sufficiency (y), columns are credit (x).
        grid[axes.index(c["self_sufficiency"]),
             axes.index(c["credit"])] = CATEGORIES.index(category)

    ax.imshow(grid, origin="lower",
              cmap=ListedColormap([COLOURS[k] for k in CATEGORIES]),
              vmin=0, vmax=len(CATEGORIES) - 1, aspect="equal",
              extent=(-0.125, 1.125, -0.125, 1.125))
    ax.set_xticks(axes)
    ax.set_yticks(axes)
    ax.set_xlabel("Creditworthiness")
    ax.set_title(title, color=INK["primary"], pad=10)
    ax.grid(False)

    for c in cells:
        if c["reserve"]:
            ax.plot(c["credit"], c["self_sufficiency"], marker="*",
                    markersize=18, color="white", markeredgecolor=INK["primary"],
                    markeredgewidth=0.8, zorder=5)


def main() -> None:
    style()
    report = json.loads((OUTPUT / "phase_diagram.json").read_text())

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 5.2))
    for ax, (key, title) in zip(axs, PANELS):
        panel = {name: cells for name, cells in report[key].items()
                 if isinstance(cells, list)}
        reference = next(cells for cells in panel.values() if cells)
        draw(ax, panel, title, reference)
    axs[0].set_ylabel("Self-sufficiency")

    # Only the states that actually occur. A legend entry for a colour no
    # cell carries invites the reader to hunt for it.
    present = {categorise(panel, (c["credit"], c["self_sufficiency"]))
               for key, _ in PANELS
               for panel in [{n: cells for n, cells in report[key].items()
                              if isinstance(cells, list)}]
               for c in next(cells for cells in panel.values() if cells)}
    handles = [Patch(facecolor=COLOURS[k], label=LABELS[k])
               for k in CATEGORIES if k in present]
    handles.append(plt.Line2D([], [], marker="*", linestyle="none",
                              markersize=13, color="white",
                              markeredgecolor=INK["primary"],
                              label="Reserve-currency issuer"))
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.16))
    # No figure title inside the image: the manuscript's `\caption` carries it,
    # and elsarticle numbers the figure itself. A title drawn here would both
    # duplicate the caption and fix a number that LaTeX is entitled to change.

    for suffix in (".pdf", ".png"):
        path = OUTPUT / f"figure2_phase_diagram{suffix}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
