"""Phase 1 figures: Sobol bars and sign-reversal phase diagrams."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.legacy.phase1_spec import PARAMS, unit_to_value  # noqa: E402
from legacy.v7.params import archetypes  # noqa: E402

RESULTS = ROOT / "results" / "phase1"
FIG = ROOT / "figures" / "phase1"

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


def load_rows(name: str) -> list[dict[str, Any]]:
    path = RESULTS / name
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fig_sobol(report: dict, out: Path) -> None:
    """First-order and total Sobol indices for the headline outputs."""
    keys = [k for k in ("leak_abs|welfare|D", "leak_per_fiscal|cost|D",
                        "severity_noapp|welfare|D") if k in report.get("sobol", {})]
    if not keys:
        print("  [skip] no Sobol indices available")
        return

    # No sharey: each panel sorts its own parameters, so labels must be per-axis.
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 5.2))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        idx = report["sobol"][key]
        order = sorted(idx["ST"], key=lambda n: idx["ST"][n], reverse=True)
        ys = np.arange(len(order))
        ax.barh(ys - 0.2, [idx["S1"][n] for n in order], 0.38,
                color=PALETTE["blue"], label="First-order $S_1$", linewidth=0)
        ax.barh(ys + 0.2, [idx["ST"][n] for n in order], 0.38,
                color=PALETTE["orange"], label="Total $S_T$", linewidth=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(order, fontsize=8.5)
        ax.tick_params(labelleft=True)
        ax.invert_yaxis()
        ax.set_xlabel("Sobol index")
        metric, basis, bloc = key.split("|")
        ax.set_title(f"{metric}\n{basis} basis, bloc {bloc}", pad=10, loc="left")
        # Flag outputs whose confidence intervals exceed the estimates.
        worst = max(idx["ST"], key=lambda n: idx["ST"][n])
        if idx["ST_conf"][worst] > idx["ST"][worst]:
            ax.annotate("confidence intervals exceed the estimates —\n"
                        "not resolved at this sample size",
                        xy=(0.03, 0.03), xycoords="axes fraction",
                        fontsize=8.5, color=INK["secondary"], va="bottom")
        ax.grid(axis="y", visible=False)

    axes[0].legend(loc="lower right")
    fig.suptitle("What drives the CLT − cash difference", fontsize=12.5,
                 x=0.008, ha="left", y=1.02)
    fig.savefig(out / "fig1_sobol_indices.png")
    plt.close(fig)
    print("  wrote fig1_sobol_indices.png")


def _phase_panel(ax, rows, xname, yname, zkey, bloc_key, bins=16):
    """Bin-average Δ on a 2-D grid and shade by sign."""
    base = archetypes()[bloc_key]
    spec = {p.name: p for p in PARAMS}

    def to_value(name, unit):
        holder = base if hasattr(base, name) else None
        b = getattr(holder, name, 0.0) if holder is not None else 0.0
        return unit_to_value(spec[name], unit, b)

    xs, ys, zs = [], [], []
    for r in rows:
        z = r.get(zkey)
        if z is None:
            continue
        xs.append(to_value(xname, r[xname]))
        ys.append(to_value(yname, r[yname]))
        zs.append(z)
    if not zs:
        return None

    xs, ys, zs = np.asarray(xs), np.asarray(ys), np.asarray(zs)
    logx = spec[xname].kind == "logmult"
    xb = (np.logspace(np.log10(xs.min()), np.log10(xs.max()), bins + 1) if logx
          else np.linspace(xs.min(), xs.max(), bins + 1))
    yb = np.linspace(ys.min(), ys.max(), bins + 1)

    grid = np.full((bins, bins), np.nan)
    for i in range(bins):
        for j in range(bins):
            m = ((xs >= xb[i]) & (xs < xb[i + 1]) & (ys >= yb[j]) & (ys < yb[j + 1]))
            if m.sum() >= 3:
                grid[j, i] = np.median(zs[m])

    lim = np.nanpercentile(np.abs(grid), 95) or 1.0
    mesh = ax.pcolormesh(xb, yb, grid, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="flat")
    # Zero contour = the sign-reversal boundary.
    xc = 0.5 * (xb[:-1] + xb[1:])
    yc = 0.5 * (yb[:-1] + yb[1:])
    if np.nanmin(grid) < 0 < np.nanmax(grid):
        ax.contour(xc, yc, np.nan_to_num(grid), levels=[0.0],
                   colors=[INK["primary"]], linewidths=1.6)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    return mesh


def fig_phase(rows: list[dict], out: Path) -> None:
    """Phase diagrams of the sign of Δleak.

    Drawn on the cell where reversals actually occur. On the welfare basis the
    sign almost never flips (99.5-100% of the sampled space favours CLT), so a
    diagram there would be uniformly one colour and uninformative. Bloc A on
    the equal-cost basis reverses in about 35% of the space and is where the
    boundary can be seen.
    """
    panels = [
        ("eps_supply", "land_discount", "A", "Prior: supply elasticity x land discount"),
        ("land_share", "land_discount", "A", "Land: share x acquisition discount"),
        ("m_H", "clt_domestic_sourcing", "A", "Policy: imports x domestic sourcing"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    zkey = "cost.A.d_leak_abs"
    mesh = None
    for ax, (xn, yn, bloc, title) in zip(axes, panels):
        mesh = _phase_panel(ax, rows, xn, yn, zkey, bloc) or mesh
        ax.set_title(title, pad=10, loc="left", fontsize=11)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("median Δ leak_abs (CLT − cash), % of GDP")
    fig.suptitle(
        "Blue = CLT leaks less; red = CLT leaks more (bloc A, equal-cost basis). "
        "Black line is the sign reversal.",
        fontsize=11.5, x=0.008, ha="left", y=1.04)
    fig.savefig(out / "fig2_phase_diagrams.png")
    plt.close(fig)
    print("  wrote fig2_phase_diagrams.png")


def fig_phase_welfare(rows: list[dict], out: Path) -> None:
    """The same cuts on the welfare basis, where the sign is near-uniform.

    Shown so the contrast with the equal-cost basis is visible rather than
    asserted: the comparison basis, not the structural parameters, is what
    decides whether a reversal region exists at all.
    """
    panels = [
        ("eps_supply", "land_discount", "D", "Prior: supply elasticity x land discount"),
        ("m_H", "land_share", "D", "Prior: construction imports x land share"),
        ("m_H", "clt_domestic_sourcing", "D", "Policy: imports x domestic sourcing"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    zkey = "welfare.D.d_leak_abs"
    mesh = None
    for ax, (xn, yn, bloc, title) in zip(axes, panels):
        mesh = _phase_panel(ax, rows, xn, yn, zkey, bloc) or mesh
        ax.set_title(title, pad=10, loc="left", fontsize=11)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("median Δ leak_abs (CLT − cash), % of GDP")
    fig.suptitle(
        "Welfare-equivalent basis, bloc D: no reversal region — the whole "
        "sampled space favours CLT (99.5%).",
        fontsize=11.5, x=0.008, ha="left", y=1.04)
    fig.savefig(out / "fig4_phase_welfare.png")
    plt.close(fig)
    print("  wrote fig4_phase_welfare.png")


def fig_sign_shares(report: dict, out: Path) -> None:
    """Share of parameter space with Δ<0, by metric, basis and bloc."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    metrics = ("leak_abs", "leak_per_fiscal", "leak_per_welfare")
    blocs = ("A", "B", "C", "D")
    width = 0.13

    for m_i, metric in enumerate(metrics):
        for b_i, basis in enumerate(("welfare", "cost")):
            shares = []
            for bloc in blocs:
                cell = report["sign_shares"].get(f"{metric}|{basis}|{bloc}", {})
                shares.append(100 * (cell.get("share_negative") or 0.0))
            offset = (m_i * 2 + b_i - 2.5) * width
            color = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"]][m_i]
            ax.bar([i + offset for i in range(len(blocs))], shares, width * 0.92,
                   color=color, alpha=1.0 if basis == "welfare" else 0.55,
                   label=f"{metric} ({basis})", linewidth=0)

    ax.axhline(50, color=INK["secondary"], linewidth=1.1, linestyle="-", zorder=3)
    ax.annotate("50% — sign is a coin flip", xy=(-0.42, 52), fontsize=8.5,
                color=INK["secondary"])
    ax.set_xticks(range(len(blocs)))
    ax.set_xticklabels(blocs)
    ax.set_xlabel("Bloc")
    ax.set_ylabel("Share of sampled space with\nCLT leaking less than cash (%)")
    ax.set_ylim(0, 108)
    ax.set_title("How often does the CLT advantage hold?", pad=12, loc="left")
    ax.legend(loc="lower left", ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "fig3_sign_shares.png")
    plt.close(fig)
    print("  wrote fig3_sign_shares.png")




def fig_scale_free(report: dict, out: Path) -> None:
    """Sobol indices for the scale-free and sign outputs.

    The headline `d_leak_abs` decomposition is dominated by the transfer level
    u, which rescales both modalities at once. `ratio` divides that scale out;
    `sign` asks only which way the comparison goes. Together they separate
    "how big is the gap" from "which direction is it".
    """
    keys = [k for k in ("ratio_leak_abs|welfare|D", "sign_leak_abs|cost|A",
                        "sign_leak_abs|cost|B")
            if k in report.get("sobol", {})]
    if not keys:
        print("  [skip] no scale-free Sobol indices available")
        return

    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 5.2))
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        idx = report["sobol"][key]
        order = sorted(idx["ST"], key=lambda n: idx["ST"][n], reverse=True)
        ys = np.arange(len(order))
        colors = [PALETTE["aqua"] if n in ("eps_supply", "land_share")
                  else PALETTE["blue"] for n in order]
        ax.barh(ys - 0.2, [idx["S1"][n] for n in order], 0.38,
                color=colors, label="First-order $S_1$", linewidth=0)
        ax.barh(ys + 0.2, [idx["ST"][n] for n in order], 0.38,
                color=PALETTE["orange"], label="Total $S_T$", linewidth=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(order, fontsize=8.5)
        ax.tick_params(labelleft=True)
        ax.invert_yaxis()
        ax.set_xlabel("Sobol index")
        target, basis, bloc = key.split("|")
        ax.set_title(f"{target}\n{basis} basis, bloc {bloc}", pad=10, loc="left")
        ax.grid(axis="y", visible=False)

    axes[0].legend(loc="lower right")
    fig.suptitle("Scale-free targets: teal marks the two parameters named in the prior",
                 fontsize=12.5, x=0.008, ha="left", y=1.02)
    fig.savefig(out / "fig5_scale_free_sobol.png")
    plt.close(fig)
    print("  wrote fig5_scale_free_sobol.png")


def fig_matched(out: Path) -> None:
    """Headline vs matched-range indices for m_H and the sourcing lever."""
    base = RESULTS / "analysis.json"
    matched = RESULTS / "analysis_matched.json"
    if not matched.exists():
        print("  [skip] matched-range analysis not available")
        return

    a = json.loads(base.read_text()).get("sobol", {}).get("leak_abs|welfare|D")
    b = json.loads(matched.read_text()).get("sobol", {}).get("leak_abs|welfare|D")
    if not a or not b:
        print("  [skip] matched-range comparison unavailable")
        return

    names = ["m_H", "clt_domestic_sourcing"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    width = 0.36
    for offset, (idx, label, color) in enumerate((
        (a, "Headline ranges", PALETTE["blue"]),
        (b, "Matched ranges", PALETTE["orange"]),
    )):
        vals = [idx["ST"].get(n, 0.0) for n in names]
        xs = [i + (offset - 0.5) * width for i in range(len(names))]
        bars = ax.bar(xs, vals, width * 0.92, color=color, label=label, linewidth=0)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK["secondary"])

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["m_H\n(structure)", "clt_domestic_sourcing\n(policy)"])
    ax.set_ylabel("Total-order Sobol index $S_T$")
    ax.set_title("Does the policy-over-structure gap survive equal ranges?",
                 pad=12, loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "fig6_matched_ranges.png")
    plt.close(fig)
    print("  wrote fig6_matched_ranges.png")


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    report = json.loads((RESULTS / "analysis.json").read_text())
    rows = load_rows("lhs.jsonl")
    print("generating phase 1 figures...")
    fig_sign_shares(report, FIG)
    fig_sobol(report, FIG)
    fig_phase(rows, FIG)
    fig_phase_welfare(rows, FIG)
    fig_scale_free(report, FIG)
    fig_matched(FIG)


if __name__ == "__main__":
    main()
