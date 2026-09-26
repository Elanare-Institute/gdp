"""Phase 2 figures: incidence, calibration modes, and the build-rate path."""

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

RESULTS = ROOT / "results" / "phase2"
FIG = ROOT / "figures" / "phase2"

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
           "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}
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


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (RESULTS / f"{name}.jsonl").read_text().splitlines() if l.strip()]


def _finish(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def fig1_incidence(rows: list[dict], out: Path) -> None:
    """Tenant versus market-renter gains as tenancy becomes scarce."""
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.4), sharey=True)
    caps = sorted({r["capacity"] for r in rows}, reverse=True)

    for ax, bloc in zip(axes, ("A", "B", "C", "D")):
        tenant, renter = [], []
        for cap in caps:
            sub = [r for r in rows if r["bloc"] == bloc and r["u"] == 0.05
                   and r["capacity"] == cap and r["rule"] == "priority_low_income"
                   and r["modality"] == "clt"]
            for bucket, status in ((tenant, "tenant"), (renter, "renter")):
                sel = [r for r in sub if r["status"] == status]
                w = sum(r["pop_share"] for r in sel)
                bucket.append(sum(r["d_utility"] * r["pop_share"] for r in sel) / w
                              if w else np.nan)

        x = np.arange(len(caps))
        ax.bar(x - 0.2, tenant, 0.38, color=PALETTE["blue"], label="CLT tenants", linewidth=0)
        ax.bar(x + 0.2, renter, 0.38, color=PALETTE["aqua"], label="Market renters", linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.0%}" for c in caps])
        ax.set_xlabel("CLT capacity")
        ax.set_title(BLOC_LABEL[bloc], pad=10, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Mean utility gain vs no transfer")
    axes[0].legend(loc="upper left")
    fig.suptitle("Scarcer tenancy concentrates the gain; the rent externality stays flat",
                 fontsize=12.5, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig1_incidence_by_capacity.png")


def fig2_calibration(rows: list[dict], out: Path) -> None:
    """Fiscal cost of welfare equivalence under the two calibration targets."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    caps = sorted({r["capacity"] for r in rows}, reverse=True)
    width = 0.36

    for offset, (mode, color, label) in enumerate((
        ("tenant", PALETTE["orange"], "Tenant-only target"),
        ("utilitarian", PALETTE["blue"], "Utilitarian target"),
    )):
        vals = []
        for cap in caps:
            sel = [r["fiscal_clt"] for r in rows
                   if r["capacity"] == cap and r["mode"] == mode and r["u"] == 0.05]
            vals.append(float(np.mean(sel)))
        xs = [i + (offset - 0.5) * width for i in range(len(caps))]
        bars = ax.bar(xs, vals, width * 0.92, color=color, label=label, linewidth=0)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK["secondary"])

    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels([f"{c:.0%}" for c in caps])
    ax.set_xlabel("CLT capacity")
    ax.set_ylabel("Fiscal cost of the CLT (% of GDP)")
    ax.set_title("Ignoring the renters' gain makes the programme look dearer",
                 pad=12, loc="left")
    ax.annotate("averaged over blocs, u = 0.05", xy=(0.015, 0.94),
                xycoords="axes fraction", fontsize=8.5, color=INK["secondary"])
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    _finish(fig, out / "fig2_calibration_modes.png")


def fig3_build_rate(rows: list[dict], out: Path) -> None:
    """Coverage, rent and leakage when construction is capacity-constrained."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    colors = {None: PALETTE["blue"], 0.02: PALETTE["orange"], 0.005: PALETTE["magenta"]}
    labels = {None: "no cap (v6)", 0.02: "2% of stock / yr", 0.005: "0.5% of stock / yr"}
    bloc = "C"

    for rate, color in colors.items():
        sel = sorted([r for r in rows if r["bloc"] == bloc and r["build_rate"] == rate],
                     key=lambda r: r["year"])
        years = [r["year"] for r in sel]
        axes[0].plot(years, [100 * r["clt_units_frac"] for r in sel],
                     color=color, linewidth=2.2, label=labels[rate])
        axes[1].plot(years, [r["pH"] for r in sel], color=color, linewidth=2.2,
                     label=labels[rate])
        axes[2].plot(years, [r["leak_pct_gdp"] for r in sel], color=color,
                     linewidth=2.2, label=labels[rate])

    base_pH = sel[0]["pH_base"]
    axes[1].axhline(base_pH, color=INK["secondary"], linewidth=1.1, zorder=0)
    axes[1].annotate("no-transfer rent", xy=(1, base_pH * 1.004), fontsize=8.5,
                     color=INK["secondary"])

    for ax, title, ylab in zip(
        axes,
        ("CLT programme in place", "Market rent", "External leakage"),
        ("% of target stock", "clearing rent $p_H$", "% of GDP")):
        ax.set_xlabel("Years since introduction")
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=10, loc="left", fontsize=11)
    axes[0].legend(loc="lower right")

    fig.suptitle(f"Build-rate cap delays every channel — bloc {BLOC_LABEL[bloc]}",
                 fontsize=12.5, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig3_build_rate_path.png")


def fig4_type_detail(rows: list[dict], out: Path) -> None:
    """Per-quantile gains, tenants and renters side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)

    for ax, cap in zip(axes, (0.6, 0.3)):
        sel = [r for r in rows if r["bloc"] == "C" and r["u"] == 0.05
               and r["capacity"] == cap and r["rule"] == "priority_low_income"
               and r["modality"] == "clt"]
        order = sorted({r["type"] for r in sel})
        x = np.arange(len(order))
        for offset, (status, color, label) in enumerate((
            ("tenant", PALETTE["blue"], "CLT tenant"),
            ("renter", PALETTE["aqua"], "Market renter"),
        )):
            vals = []
            for name in order:
                hit = [r for r in sel if r["type"] == name and r["status"] == status]
                vals.append(hit[0]["d_utility"] if hit else 0.0)
            ax.bar(x + (offset - 0.5) * 0.36, vals, 0.33, color=color,
                   label=label, linewidth=0)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_xlabel("Income quantile (q1 = poorest)")
        ax.set_title(f"capacity {cap:.0%}", pad=10, loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Utility gain vs no transfer")
    axes[0].legend(loc="upper right")
    fig.suptitle("Within a quantile, tenancy decides how much of the gain you get "
                 "(bloc C, low-income priority)",
                 fontsize=12.5, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig4_type_detail.png")


def fig5_pv_boundary(out: Path) -> None:
    """Where a present-value equivalent exists, by horizon treatment."""
    rows = load("pv_variants")
    if not rows:
        print("  [skip] pv_variants not available")
        return

    builds = [None, 0.05, 0.02, 0.01, 0.005]
    discounts = sorted({r["discount"] for r in rows})
    variants = [("30y", "30-year horizon"),
                ("60y", "60-year horizon"),
                ("30y+term0.020", "30 years + terminal value")]

    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 4.4),
                             sharey=True)
    cmap = matplotlib.colors.ListedColormap([PALETTE["orange"], PALETTE["blue"]])

    for ax, (key, title) in zip(axes, variants):
        grid = np.full((len(builds), len(discounts)), np.nan)
        for i, build in enumerate(builds):
            for j, disc in enumerate(discounts):
                hit = [r for r in rows if r["build_rate"] == build
                       and r["discount"] == disc and r["variant"] == key]
                if hit:
                    grid[i, j] = 1.0 if hit[0]["solved"] else 0.0
        ax.pcolormesh(np.arange(len(discounts) + 1), np.arange(len(builds) + 1),
                      grid, cmap=cmap, vmin=0, vmax=1, shading="flat")
        for i in range(len(builds)):
            for j in range(len(discounts)):
                if not np.isnan(grid[i, j]):
                    ax.annotate("solves" if grid[i, j] > 0.5 else "none",
                                xy=(j + 0.5, i + 0.5), ha="center", va="center",
                                fontsize=8.5,
                                color="white" if grid[i, j] > 0.5 else INK["primary"])
        ax.set_xticks(np.arange(len(discounts)) + 0.5)
        ax.set_xticklabels([f"{d:.0%}" for d in discounts])
        ax.set_yticks(np.arange(len(builds)) + 0.5)
        ax.set_yticklabels(["no cap" if b is None else f"{b:.1%}/yr" for b in builds])
        ax.set_xlabel("Discount rate")
        ax.set_title(title, pad=10, loc="left", fontsize=11)
        ax.grid(False)

    axes[0].set_ylabel("CLT build rate")
    fig.suptitle("A longer horizon rescues the 2%/yr case; below 1%/yr the cap "
                 "binds regardless (bloc C)",
                 fontsize=12, x=0.008, ha="left", y=1.03)
    _finish(fig, out / "fig5_pv_boundary.png")


def fig6_instrument(out: Path) -> None:
    """Whether `size` still moves welfare while the build cap binds."""
    rows = load("pv_instrument")
    if not rows:
        print("  [skip] pv_instrument not available")
        return

    builds = [None, 0.05, 0.02, 0.01, 0.005]
    years = sorted({r["year"] for r in rows})
    grid = np.full((len(builds), len(years)), np.nan)
    for i, build in enumerate(builds):
        for j, year in enumerate(years):
            hit = [r for r in rows if r["build_rate"] == build and r["year"] == year]
            if hit:
                grid[i, j] = 1.0 if hit[0]["monotone_increasing"] else 0.0

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    cmap = matplotlib.colors.ListedColormap([PALETTE["orange"], PALETTE["blue"]])
    ax.pcolormesh(np.arange(len(years) + 1), np.arange(len(builds) + 1),
                  grid, cmap=cmap, vmin=0, vmax=1, shading="flat")
    for i in range(len(builds)):
        for j in range(len(years)):
            if not np.isnan(grid[i, j]):
                ax.annotate("monotone" if grid[i, j] > 0.5 else "non-mono.",
                            xy=(j + 0.5, i + 0.5), ha="center", va="center",
                            fontsize=8,
                            color="white" if grid[i, j] > 0.5 else INK["primary"])
    ax.set_xticks(np.arange(len(years)) + 0.5)
    ax.set_xticklabels([f"y{y}" for y in years])
    ax.set_yticks(np.arange(len(builds)) + 0.5)
    ax.set_yticklabels(["no cap" if b is None else f"{b:.1%}/yr" for b in builds])
    ax.set_xlabel("Years since introduction")
    ax.set_ylabel("CLT build rate")
    ax.set_title("While the cap binds, programme size stops moving welfare",
                 pad=12, loc="left")
    ax.grid(False)
    _finish(fig, out / "fig6_size_instrument.png")


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    incidence = load("incidence")
    print("generating phase 2 figures...")
    fig1_incidence(incidence, FIG)
    fig2_calibration(load("calibration_modes"), FIG)
    fig3_build_rate(load("build_rate_path"), FIG)
    fig4_type_detail(incidence, FIG)
    fig5_pv_boundary(FIG)
    fig6_instrument(FIG)


if __name__ == "__main__":
    main()
