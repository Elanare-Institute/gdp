"""Phase 0 figures: the v6 baseline and the reserve-flight robustness check."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from legacy.v7.core import run_scenarios, simulate, build  # noqa: E402
from legacy.v7.params import G  # noqa: E402

FIG = ROOT / "figures" / "phase0"

# Colorblind-safe categorical palette.
PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "magenta": "#e87ba4"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
SHORT = {"A (reserve)": "A", "B (non-reserve adv.)": "B",
         "C (emerging)": "C", "D (developing)": "D"}


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


def fig1_leakage(res: dict, out: Path) -> None:
    """Leakage difference (CLT - cash) by bloc and comparison basis.

    Bars above zero are sign reversals: CLT leaks MORE than cash there.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    blocs = list(res["runs"]["welfare:cash_t"]["leakage"])
    width = 0.36

    for offset, (basis, color, label) in enumerate((
        ("welfare", PALETTE["blue"], "Welfare-equivalent"),
        ("cost", PALETTE["orange"], "Equal fiscal cost"),
    )):
        diffs = [
            res["runs"][f"{basis}:clt"]["leakage"][b]["external_leak_per_fiscal"]
            - res["runs"][f"{basis}:cash_t"]["leakage"][b]["external_leak_per_fiscal"]
            for b in blocs
        ]
        xs = [i + (offset - 0.5) * width for i in range(len(blocs))]
        bars = ax.bar(xs, diffs, width * 0.92, color=color, label=label, linewidth=0)
        for bar, value in zip(bars, diffs):
            ax.annotate(f"{value:+.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 4 if value >= 0 else -12),
                        textcoords="offset points", ha="center",
                        fontsize=8, color=INK["secondary"])

    ax.axhline(0, color=INK["secondary"], linewidth=1.1, zorder=3)
    ax.set_xticks(range(len(blocs)))
    ax.set_xticklabels([SHORT[b] for b in blocs])
    ax.set_xlabel("Bloc")
    ax.set_ylabel("External leakage per fiscal unit\n(CLT − cash)")
    ax.set_title("CLT leaks less than cash in some blocs, more in others",
                 pad=12, loc="left")
    ax.annotate("above zero = CLT leaks MORE", xy=(0.015, 0.93),
                xycoords="axes fraction", fontsize=8.5, color=INK["secondary"])
    # Headroom so the tallest bar's label clears the legend.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "fig1_leakage_difference.png")
    plt.close(fig)
    print("  wrote fig1_leakage_difference.png")


def fig2_reserve_flight(out: Path) -> None:
    """Bloc A's exchange rate under the three reserve-flight modes."""
    fig, (ax_e, ax_sev) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    colors = {"v6": PALETTE["orange"], "off": PALETTE["blue"], "capped": PALETTE["aqua"]}

    for mode, color in colors.items():
        g = G(reserve_flight_mode=mode)
        run = simulate(build(g, "cash_t", 0.05, (1, 5, 10, 15), "welfare"), g)
        years = [h["year"] for h in run["history"]]
        e_path = [h["e_0"] for h in run["history"]]
        ax_e.plot(years, e_path, color=color, linewidth=2.2, label=f"mode = {mode}")
        ax_e.annotate(f"{e_path[-1]:.2f}", xy=(years[-1], e_path[-1]), xytext=(5, 0),
                      textcoords="offset points", color=color, fontsize=9, va="center")

    ax_e.axhline(0.3, color=INK["secondary"], linewidth=1.1, linestyle="-", zorder=0)
    ax_e.annotate("appreciation crisis threshold (0.3)", xy=(1, 0.32),
                  fontsize=8.5, color=INK["secondary"])
    ax_e.set_xlabel("Year")
    ax_e.set_ylabel("Exchange rate, bloc A")
    ax_e.set_title("Safe-haven inflow drives bloc A's exchange rate", pad=10, loc="left")
    ax_e.legend(loc="upper right")

    scenarios = ["welfare:ubi", "cost:cash_t", "cost:voucher", "cost:clt"]
    width = 0.36
    for offset, (mode, color) in enumerate((("v6", PALETTE["orange"]), ("off", PALETTE["blue"]))):
        res = run_scenarios(G(reserve_flight_mode=mode), 0.05)
        vals = [res["runs"][s]["crisis"]["A (reserve)"]["severity"] for s in scenarios]
        xs = [i + (offset - 0.5) * width for i in range(len(scenarios))]
        ax_sev.bar(xs, vals, width * 0.92, color=color, label=f"mode = {mode}", linewidth=0)

    ax_sev.set_xticks(range(len(scenarios)))
    ax_sev.set_xticklabels([s.replace(":", "\n") for s in scenarios], fontsize=8.5)
    ax_sev.set_ylabel("Crisis severity, bloc A")
    ax_sev.set_title("Most of bloc A's severity is the inflow artifact", pad=10, loc="left")
    ax_sev.legend(loc="upper right")
    ax_sev.grid(axis="x", visible=False)

    fig.savefig(out / "fig2_reserve_flight.png")
    plt.close(fig)
    print("  wrote fig2_reserve_flight.png")


def fig3_rent_incidence(res: dict, out: Path) -> None:
    """Rent capture per fiscal unit: cash is absorbed, CLT returns rent."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    blocs = list(res["runs"]["welfare:cash_t"]["leakage"])
    width = 0.36

    for offset, (mod, color, label) in enumerate((
        ("cash_t", PALETTE["orange"], "Targeted cash"),
        ("clt", PALETTE["aqua"], "Community land trust"),
    )):
        vals = [res["runs"][f"welfare:{mod}"]["leakage"][b]["rent_capture_per_fiscal"]
                for b in blocs]
        xs = [i + (offset - 0.5) * width for i in range(len(blocs))]
        ax.bar(xs, vals, width * 0.92, color=color, label=label, linewidth=0)

    ax.axhline(0, color=INK["secondary"], linewidth=1.1, zorder=3)
    ax.set_xticks(range(len(blocs)))
    ax.set_xticklabels([SHORT[b] for b in blocs])
    ax.set_xlabel("Bloc")
    ax.set_ylabel("Rent capture per fiscal unit")
    ax.set_title("Cash is captured as rent; CLT lowers the market rent",
                 pad=12, loc="left")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "fig3_rent_incidence.png")
    plt.close(fig)
    print("  wrote fig3_rent_incidence.png")


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    res = run_scenarios(G(), 0.05)
    print("generating phase 0 figures...")
    fig1_leakage(res, FIG)
    fig2_reserve_flight(FIG)
    fig3_rent_incidence(res, FIG)


if __name__ == "__main__":
    main()
