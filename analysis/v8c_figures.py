"""Phase C figures: what indexation does, measured three ways."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.core import build, simulate  # noqa: E402
from model.params import G  # noqa: E402

FIG = ROOT / "figures" / "v8c"
U = 0.02
RULES = ("gdp_linked", "fixed_nominal", "cpi_indexed")
RULE_LABEL = {"gdp_linked": "GDP-linked (v7)", "fixed_nominal": "Fixed nominal",
              "cpi_indexed": "CPI-indexed"}
MODALITIES = ("none", "ubi", "cash_t", "voucher", "clt")
MOD_LABEL = {"none": "No transfer", "ubi": "UBI", "cash_t": "Targeted cash",
             "voucher": "Food voucher", "clt": "In-kind housing"}
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "violet": "#4a3aa7"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
MOD_COLOR = {"none": INK["muted"], "ubi": PALETTE["violet"],
             "cash_t": PALETTE["blue"], "voucher": PALETTE["yellow"],
             "clt": PALETTE["aqua"]}
BLOCS = {0: "A (reserve)", 1: "B (non-reserve adv.)",
         2: "C (emerging)", 3: "D (developing)"}


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


def _quantities(rule: str, modality: str, good: str, bloc: int):
    g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02, indexation=rule)
    blocs = build(g, modality, U, (1, 1, 1, 1)) if modality != "none" else build(g, "none")
    rows = simulate(blocs, g)["quantities"][1:]
    base = rows[0][f"{good}_{bloc}"]
    return ([r["year"] for r in rows],
            [100.0 * r[f"{good}_{bloc}"] / base for r in rows])


def fig1_quantity_paths(out: Path, good: str = "H", bloc: int = 3) -> None:
    """What a constrained household actually secures, by rule."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.4), sharey=True)
    for ax, rule in zip(axes, RULES):
        for modality in MODALITIES:
            years, values = _quantities(rule, modality, good, bloc)
            ax.plot(years, values, color=MOD_COLOR[modality],
                    label=MOD_LABEL[modality], linewidth=1.8,
                    linestyle="--" if modality == "none" else "-")
        ax.set_title(RULE_LABEL[rule], color=INK["primary"])
        ax.set_xlabel("Year")
    axes[0].set_ylabel(f"Quantity secured, {good} (start = 100)")
    axes[-1].legend(loc="upper left")
    name = {"H": "housing", "F": "food"}[good]
    fig.suptitle(f"What a constrained household in {BLOCS[bloc]} ends up with — "
                 f"{name}, welfare-equivalent programmes at u = {U:g}",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def fig2_three_measures(out: Path, bloc: int = 3) -> None:
    """The same programmes read in CPI, in housing, and in quantities."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.4))
    mods = [m for m in MODALITIES if m != "none"]
    width = 0.26
    xs = range(len(RULES))

    def bars(ax, values, title, ylabel):
        for i, modality in enumerate(mods):
            ax.bar([x + (i - 1.5) * width for x in xs],
                   [values[rule][modality] for rule in RULES], width,
                   color=MOD_COLOR[modality], label=MOD_LABEL[modality])
        ax.set_xticks(list(xs))
        ax.set_xticklabels([RULE_LABEL[r] for r in RULES], rotation=15)
        ax.set_title(title, color=INK["primary"])
        ax.set_ylabel(ylabel)

    cpi, housing, quantity = {}, {}, {}
    for rule in RULES:
        g = G(n_types=5, clt_capacity=1.0, clt_build_rate=0.02, indexation=rule)
        cpi[rule], housing[rule], quantity[rule] = {}, {}, {}
        for modality in mods:
            result = simulate(build(g, modality, U, (1, 1, 1, 1)), g)
            name = BLOCS[bloc]
            cpi[rule][modality] = result["real_value"][name]
            housing[rule][modality] = result["housing_value"][name]
            rows = result["quantities"][1:]
            quantity[rule][modality] = rows[-1][f"H_{bloc}"] / rows[0][f"H_{bloc}"]

    bars(axes[0], cpi, "Transfer value in CPI terms", "Year 30 / start")
    bars(axes[1], housing, "Transfer value in housing terms", "Year 30 / start")
    bars(axes[2], quantity, "Housing actually secured", "Year 30 / start")
    axes[0].axhline(1.0, color=INK["secondary"], linewidth=0.9)
    axes[2].legend(loc="upper left")
    fig.suptitle(f"Three readings of the same programmes — {BLOCS[bloc]}, u = {U:g}. "
                 "Only the third compares cash with in-kind on the same terms.",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def fig3_by_bloc(out: Path, good: str = "H") -> None:
    """The divergence, bloc by bloc, under CPI indexation."""
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), sharey=True)
    for ax, (bloc, label) in zip(axes, BLOCS.items()):
        for modality in MODALITIES:
            years, values = _quantities("cpi_indexed", modality, good, bloc)
            ax.plot(years, values, color=MOD_COLOR[modality],
                    label=MOD_LABEL[modality], linewidth=1.8,
                    linestyle="--" if modality == "none" else "-")
        ax.set_title(label, color=INK["primary"])
        ax.set_xlabel("Year")
    axes[0].set_ylabel(f"Quantity secured, {good} (start = 100)")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Housing secured under CPI-indexed transfers, by bloc — "
                 "the gap opens where the external constraint binds",
                 color=INK["primary"], y=1.04)
    _finish(fig, out)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    print("generating phase C figures...")
    fig1_quantity_paths(FIG / "fig1_housing_paths", "H")
    fig1_quantity_paths(FIG / "fig2_food_paths", "F")
    fig2_three_measures(FIG / "fig3_three_measures")
    fig3_by_bloc(FIG / "fig4_by_bloc")


if __name__ == "__main__":
    main()
