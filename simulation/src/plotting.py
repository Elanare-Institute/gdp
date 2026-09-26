"""Publication-quality figures.

Design rules applied throughout:
  * Colorblind-safe categorical palette (validated: all checks pass, light mode).
  * A legend whenever two or more series are present, plus selective direct
    labels at the line ends, so identity never rests on color alone — these
    figures must survive grayscale printing.
  * No dual axes anywhere. Where series differ by orders of magnitude they are
    indexed to a common base and drawn on one logarithmic axis.
  * Thin marks, hairline solid gridlines, recessive axes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless rendering; no display required
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .config import (
    CHANNEL_COLORS,
    CHANNEL_LABELS,
    FIGURE_DPI,
    INK,
    PALETTE,
    PHI_HIGH,
    PHI_LOW,
)

LINE_WIDTH = 1.8
EMPHASIS_WIDTH = 2.6


def _apply_style() -> None:
    """Set the shared figure style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": INK["axis"],
            "axes.labelcolor": INK["primary"],
            "text.color": INK["primary"],
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "grid.color": INK["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.grid": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def _finish(fig: plt.Figure, path: Path) -> None:
    """Save and close a figure."""
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def _label_end(ax: plt.Axes, x: float, y: float, text: str, color: str, dx: float = 0.008) -> None:
    """Direct-label a line at its right end."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, 0),
        textcoords="offset fontsize",
        color=color,
        fontsize=8.5,
        va="center",
        fontweight="normal",
    )


def fig1_theta_vs_phi(scenarios: Sequence, out: Path) -> None:
    """Labor share against fugacity, both parameter regimes."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = (PALETTE["blue"], PALETTE["orange"])

    for result, color in zip(scenarios, colors):
        ax.plot(
            result.phi_grid,
            result.theta_path,
            color=color,
            linewidth=EMPHASIS_WIDTH,
            label=result.scenario.label,
            solid_capstyle="round",
        )
        _label_end(ax, result.phi_grid[-1], result.theta_path[-1],
                   f"{result.theta_path[-1]:.3f}", color)

    ax.set_xlabel("Fugacity $\\varphi$  (1970s $\\rightarrow$ 2020s)")
    ax.set_ylabel("Aggregate labor share $\\theta_L$")
    ax.set_title("Rising fugacity lowers the labor share", pad=12, loc="left")
    ax.set_xlim(PHI_LOW, PHI_HIGH + 0.05)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    # Reference band: the observed US decline, for scale.
    ax.axhspan(0.58, 0.66, color=INK["grid"], alpha=0.55, zorder=0, linewidth=0)
    ax.annotate(
        "observed US range (~0.66 $\\rightarrow$ ~0.58)",
        xy=(PHI_LOW + 0.02, 0.62), fontsize=8, color=INK["secondary"], va="center",
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02))
    _finish(fig, out / "fig1_theta_vs_phi.png")


def fig2_channel_decomposition(result, out: Path) -> None:
    """Individual channels versus all four moving together."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for channel, path in result.channel_paths.items():
        color = CHANNEL_COLORS[channel]
        ax.plot(result.phi_grid, path, color=color, linewidth=LINE_WIDTH,
                label=CHANNEL_LABELS[channel], solid_capstyle="round")
        _label_end(ax, result.phi_grid[-1], path[-1], CHANNEL_LABELS[channel].split()[0], color)

    ax.plot(result.phi_grid, result.joint_path, color=PALETTE["blue"],
            linewidth=EMPHASIS_WIDTH, label="All channels jointly", solid_capstyle="round")
    _label_end(ax, result.phi_grid[-1], result.joint_path[-1], "Joint", PALETTE["blue"])

    ax.set_xlabel("Fugacity $\\varphi$")
    ax.set_ylabel("Aggregate labor share $\\theta_L$")
    ax.set_title(
        "Each channel alone falls short of the joint effect",
        pad=12, loc="left",
    )
    ax.set_xlim(PHI_LOW, PHI_HIGH + 0.09)
    ax.legend(loc="lower left", ncol=2)
    _finish(fig, out / "fig2_channel_decomposition.png")


def fig3_overcounting(scenarios: Sequence, out: Path) -> None:
    """The convention-dependence of over-counting — the central result."""
    fig, (ax_bar, ax_ratio) = plt.subplots(
        1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    main = scenarios[0]

    # Left: per-channel contributions under each convention.
    conventions = [d.convention for d in main.decompositions]
    channels = list(main.decompositions[0].contributions.keys())
    n_conv, width = len(conventions), 0.2
    xs = range(n_conv)

    for i, channel in enumerate(channels):
        offsets = [x + (i - (len(channels) - 1) / 2) * width for x in xs]
        values = [d.contributions[channel] for d in main.decompositions]
        ax_bar.bar(offsets, values, width * 0.92, color=CHANNEL_COLORS[channel],
                   label=CHANNEL_LABELS[channel], linewidth=0)

    ax_bar.set_xticks(list(xs))
    ax_bar.set_xticklabels(["one-at-a-time\n(A)", "leave-one-out\n(B)", "Shapley\n(C)"])
    ax_bar.set_ylabel("Contribution to $\\theta_L$ decline")
    ax_bar.set_title("Per-channel contribution by convention", pad=10, loc="left")
    ax_bar.legend(loc="upper right", ncol=2)
    ax_bar.grid(axis="x", visible=False)

    # Right: sum-to-joint ratio, the headline number.
    labels = ["one-at-a-time (A)", "leave-one-out (B)", "Shapley (C)"]
    ratios = [d.ratio for d in main.decompositions]
    bar_colors = [PALETTE["orange"], PALETTE["aqua"], PALETTE["blue"]]

    bars = ax_ratio.barh(labels, ratios, height=0.55, color=bar_colors, linewidth=0)
    ax_ratio.axvline(1.0, color=INK["secondary"], linewidth=1.2, zorder=3)
    ax_ratio.annotate(
        "exact (1.0)",
        xy=(1.0, 0.015), xycoords=("data", "axes fraction"),
        xytext=(5, 0), textcoords="offset points",
        fontsize=8.5, color=INK["secondary"], va="bottom",
    )

    for bar, ratio in zip(bars, ratios):
        ax_ratio.annotate(
            f"{ratio:.3f}×",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=9.5, fontweight="normal", color=INK["primary"],
        )

    ax_ratio.set_xlabel("Sum of individual effects ÷ joint effect")
    ax_ratio.set_title("Over-counting depends on the convention", pad=10, loc="left")
    ax_ratio.set_xlim(0, max(ratios) * 1.22)
    ax_ratio.invert_yaxis()
    ax_ratio.grid(axis="y", visible=False)

    fig.suptitle(
        "Over-counting is an artifact of the decomposition convention, not of the economy",
        fontsize=12.5, x=0.008, ha="left", y=1.02,
    )
    _finish(fig, out / "fig3_overcounting.png")


def fig4_wage_comparison(wage_series: Sequence, out: Path) -> None:
    """Three wage representations, indexed to 100 and on a single log axis."""
    fig, axes = plt.subplots(1, len(wage_series), figsize=(11.0, 4.7), sharey=True)
    if len(wage_series) == 1:
        axes = [axes]

    series_spec = (
        ("nominal", "Nominal wage", PALETTE["blue"]),
        ("cpi_real", "CPI-real wage", PALETTE["aqua"]),
        ("gold", "Gold-denominated wage", PALETTE["orange"]),
    )

    for ax, wages in zip(axes, wage_series):
        for attr, label, color in series_spec:
            values = getattr(wages, attr)
            ax.plot(wages.years, values, color=color, linewidth=EMPHASIS_WIDTH,
                    label=label, solid_capstyle="round")
            _label_end(ax, wages.years[-1], values[-1], f"{values[-1]:.0f}", color)

        ax.set_yscale("log")
        ax.axhline(100.0, color=INK["axis"], linewidth=1.0, zorder=0)
        ax.set_xlabel("Year")
        ax.set_title(wages.label, pad=10, loc="left")
        ax.set_xlim(wages.years[0], wages.years[-1] + 3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    axes[0].set_ylabel("Index (1970 = 100, log scale)")
    axes[0].legend(loc="lower left")

    fig.suptitle(
        "One wage path, three representations",
        fontsize=12.5, x=0.008, ha="left", y=1.02,
    )
    _finish(fig, out / "fig4_wage_comparison.png")


def fig5_sensitivity(sensitivity, out: Path) -> None:
    """Sensitivity to sigma and to the channel sensitivities."""
    fig, (ax_sigma, ax_pert) = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)

    sigma_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["yellow"]]
    # The paths nearly coincide. Stepping the width down lets each remain
    # visible under the next, so the reader can confirm all four are drawn
    # rather than seeing only the last one plotted.
    n_sigma = len(sensitivity.sigma_paths)
    for i, ((name, path), color) in enumerate(zip(sensitivity.sigma_paths.items(), sigma_colors)):
        ax_sigma.plot(
            sensitivity.phi_grid, path, color=color,
            linewidth=EMPHASIS_WIDTH + 1.6 * (n_sigma - 1 - i),
            label=f"$\\sigma$ = {name}", solid_capstyle="round",
        )

    ax_sigma.set_xlabel("Fugacity $\\varphi$")
    ax_sigma.set_ylabel("Aggregate labor share $\\theta_L$")
    ax_sigma.set_title("Elasticity of substitution: nearly no effect", pad=10, loc="left")
    ax_sigma.legend(loc="upper right")
    ax_sigma.annotate(
        "all four lines are drawn, at stepped widths —\nthey coincide to within 0.0005. "
        "Expected, not a bug.",
        xy=(0.135, 0.285), fontsize=8.5, color=INK["secondary"],
    )

    pert_colors = [
        PALETTE["blue"], PALETTE["blue"], PALETTE["orange"],
        PALETTE["orange"], PALETTE["aqua"], PALETTE["aqua"],
    ]
    for (name, path), color in zip(sensitivity.perturbation_paths.items(), pert_colors):
        style = "--" if "+50%" in name else "-"
        ax_pert.plot(sensitivity.phi_grid, path, color=color, linewidth=LINE_WIDTH,
                     linestyle=style, label=name, solid_capstyle="round")

    ax_pert.set_xlabel("Fugacity $\\varphi$")
    ax_pert.set_title("Channel sensitivities $\\pm$50%: this is where the leverage is",
                      pad=10, loc="left")
    ax_pert.legend(loc="upper right", ncol=2)

    fig.suptitle(
        "Sensitivity analysis", fontsize=12.5, x=0.008, ha="left", y=1.02,
    )
    _finish(fig, out / "fig5_sensitivity.png")


def make_all_figures(scenarios: Sequence, wage_series: Sequence, sensitivity, out: Path) -> None:
    """Render every figure into `out`."""
    _apply_style()
    out.mkdir(parents=True, exist_ok=True)
    fig1_theta_vs_phi(scenarios, out)
    fig2_channel_decomposition(scenarios[0], out)
    fig3_overcounting(scenarios, out)
    fig4_wage_comparison(wage_series, out)
    fig5_sensitivity(sensitivity, out)
