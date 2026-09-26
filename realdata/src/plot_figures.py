"""Figures 4a-4d, in the same style as the simulation figures.

Design rules, matching `simulation/src/plotting.py`:
  * Colorblind-safe categorical palette.
  * A legend for two or more series, plus direct end labels, so identity never
    rests on color alone — these must survive grayscale printing.
  * No dual axes. Series spanning orders of magnitude are indexed to a common
    base and drawn on a single logarithmic axis.
  * Thin marks, hairline solid gridlines, recessive axes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .config import (
    BASE_YEAR,
    CASE_SHILLER_START,
    COUNTRY_COLORS,
    END_YEAR,
    FIGURE_DPI,
    INK,
    INTL_BASE_YEAR,
    OUTPUT_DIR,
    PALETTE,
    ROBUSTNESS_BASE_YEARS,
    SERIES_COLORS,
    SERIES_LABELS,
)
from .construct_indices import (
    USIndices,
    build_us_indices,
    gold_robustness,
    housing_comparison,
    summarize,
)
from .construct_intl import CountrySeries, build_international
from .fetch_data import (
    fetch_fx_annual,
    fetch_ilo_country,
    fetch_oecd_wages,
    fetch_sp500_dividends,
    fetch_us_annual,
)

LINE_WIDTH = 1.8
EMPHASIS_WIDTH = 2.6


def _apply_style() -> None:
    """Shared figure style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
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
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def _label_end(
    ax: plt.Axes, x: float, y: float, text: str, color: str, dy: float = 0.0
) -> None:
    """Direct-label a line at its right end.

    `dy` offsets the label vertically in points, to separate labels whose
    series converge to nearly the same value.
    """
    ax.annotate(
        text, xy=(x, y), xytext=(6, dy * 60), textcoords="offset points",
        color=color, fontsize=9.5, va="center",
    )


def _log_axis(ax: plt.Axes) -> None:
    """Log scale with plain (non-scientific) tick labels."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def fig4a(indices: USIndices, out: Path) -> None:
    """The six asset-relative wage series on one indexed log axis."""
    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    order = ["nominal", "cpi_real", "commodity_ppi", "housing", "sp500", "gold"]
    for name in order:
        series = indices.series[name].loc[BASE_YEAR:END_YEAR]
        style = "--" if name == "commodity_ppi" else "-"
        ax.plot(
            series.index, series.values, color=SERIES_COLORS[name],
            linewidth=EMPHASIS_WIDTH if name in ("gold", "nominal") else LINE_WIDTH,
            linestyle=style, label=SERIES_LABELS[name], solid_capstyle="round",
        )
        # gold and equity converge near 14-15; nudge apart so labels stay legible.
        nudge = {"gold": -0.14, "sp500": 0.14}.get(name, 0.0)
        _label_end(ax, series.index[-1], series.iloc[-1], f"{series.iloc[-1]:.0f}",
                   SERIES_COLORS[name], dy=nudge)

    if indices.total_return is not None:
        tr = indices.total_return.loc[BASE_YEAR:]
        ax.plot(tr.index, tr.values, color=SERIES_COLORS["sp500"], linewidth=1.3,
                linestyle=":", label="Equity, dividends reinvested")
        _label_end(ax, tr.index[-1], tr.iloc[-1],
                   f"{tr.iloc[-1]:.0f} ({tr.index[-1]})", SERIES_COLORS["sp500"])

    ax.axhline(100.0, color=INK["axis"], linewidth=1.1, zorder=0)
    _log_axis(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Index ({BASE_YEAR} = 100, log scale)")
    ax.set_title(
        "The same wage, measured against different things",
        pad=12, loc="left",
    )
    ax.set_xlim(BASE_YEAR, END_YEAR + 4)
    ax.legend(loc="lower left", ncol=2)
    _finish(fig, out / "fig4a_asset_relative_wages.png")


def fig4b(robustness: dict[int, pd.Series], out: Path) -> None:
    """Gold-denominated wage from four base years.

    The 1980 base reverses the sign of the result, because 1980 was the
    Hunt-brothers gold spike. The figure says so rather than omitting it.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    colors = [PALETTE["blue"], PALETTE["aqua"], PALETTE["orange"], PALETTE["violet"]]

    for (base, series), color in zip(sorted(robustness.items()), colors):
        visible = series.loc[base:END_YEAR]
        reversed_sign = visible.loc[END_YEAR] > 100
        ax.plot(
            visible.index, visible.values, color=color,
            linewidth=EMPHASIS_WIDTH if reversed_sign else LINE_WIDTH,
            label=f"{base} = 100", solid_capstyle="round",
        )
        _label_end(ax, visible.index[-1], visible.iloc[-1], f"{visible.iloc[-1]:.0f}", color)

    ax.axhline(100.0, color=INK["axis"], linewidth=1.1, zorder=0)
    _log_axis(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gold-denominated wage index (log scale)")
    ax.set_title("The decline depends on where you start", pad=12, loc="left")
    ax.set_xlim(min(robustness), END_YEAR + 4)
    ax.annotate(
        "1980 was the Hunt-brothers gold spike (annual mean $608).\n"
        "Anchoring on that peak reverses the sign: the wage ends above 100.",
        xy=(0.5, -0.155), xycoords="axes fraction", ha="center",
        fontsize=9.5, color=INK["secondary"], va="top",
    )
    ax.legend(loc="upper right", ncol=2)
    _finish(fig, out / "fig4b_gold_robustness.png")


def fig4c(comparison: dict[str, pd.Series], out: Path) -> None:
    """Housing-denominated wage under both house-price measures."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    spec = (
        ("median_price", "Median sales price (MSPUS)", PALETTE["violet"]),
        ("case_shiller", "Case-Shiller national index", PALETTE["orange"]),
    )
    for key, label, color in spec:
        series = comparison[key].loc[CASE_SHILLER_START:END_YEAR]
        ax.plot(series.index, series.values, color=color, linewidth=EMPHASIS_WIDTH,
                label=label, solid_capstyle="round")
        _label_end(ax, series.index[-1], series.iloc[-1], f"{series.iloc[-1]:.0f}", color)

    ax.axhline(100.0, color=INK["axis"], linewidth=1.1, zorder=0)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Index ({CASE_SHILLER_START} = 100)")
    ax.set_title("Two house-price measures agree on direction, not level",
                 pad=12, loc="left")
    ax.set_xlim(CASE_SHILLER_START, END_YEAR + 3)

    gap = abs(comparison["median_price"].loc[END_YEAR] - comparison["case_shiller"].loc[END_YEAR])
    ax.annotate(
        f"{END_YEAR} gap: {gap:.0f} index points.\n"
        "Case-Shiller is quality-adjusted repeat-sales; the median price\n"
        "drifts with the size and mix of houses actually sold.",
        xy=(0.035, 0.05), xycoords="axes fraction",
        fontsize=9.5, color=INK["secondary"], va="bottom",
    )
    ax.legend(loc="upper right")
    _finish(fig, out / "fig4c_housing_detail.png")


def fig4d(countries: Sequence[CountrySeries], skipped: dict[str, str], out: Path) -> None:
    """Gold-denominated wages across countries, indexed to a common base."""
    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    for country in countries:
        series = country.index[country.index.index <= END_YEAR]
        color = COUNTRY_COLORS.get(country.code, INK["muted"])
        if country.spec.marker_only:
            suffix = "sparse" if country.on_common_base else f"sparse, {country.anchor_year}=100"
            ax.plot(series.index, series.values, marker="o", markersize=5.5,
                    linestyle="none", color=color,
                    label=f"{country.spec.label} ({suffix})")
        else:
            # Observations before the common base are context, drawn faintly —
            # but only when they form a continuous run up to the base year. A
            # gapped pre-period (Brazil has no 2000 observation) would render as
            # a stray disconnected segment and is dropped instead.
            pre = series[series.index <= INTL_BASE_YEAR]
            post = series[series.index >= INTL_BASE_YEAR]
            continuous = (
                len(pre) > 1
                and int(pre.index.max()) == INTL_BASE_YEAR
                and len(pre) == INTL_BASE_YEAR - int(pre.index.min()) + 1
            )
            if continuous:
                ax.plot(pre.index, pre.values, color=color, linewidth=1.2, alpha=0.45)
            ax.plot(post.index, post.values, color=color, linewidth=EMPHASIS_WIDTH,
                    label=country.spec.label, solid_capstyle="round")
        if len(series):
            _label_end(ax, series.index[-1], series.iloc[-1], f"{series.iloc[-1]:.0f}", color)

    ax.axhline(100.0, color=INK["axis"], linewidth=1.1, zorder=0)
    _log_axis(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Gold-denominated wage ({INTL_BASE_YEAR} = 100, log scale)")
    ax.set_title("Wages measured in gold fall almost everywhere", pad=12, loc="left")

    notes = ["Japan's series starts in 1990, at the bubble peak — this maximizes its decline."]
    off_base = [c for c in countries if not c.on_common_base]
    for country in off_base:
        notes.append(
            f"{country.spec.label} has no data at {INTL_BASE_YEAR}; it is indexed to "
            f"{country.anchor_year} and its level is not directly comparable."
        )
    if "CHN" in skipped:
        notes.append("China omitted: no clean total-economy series after 1997.")
    ax.annotate(
        "\n".join(notes),
        xy=(0.035, 0.05), xycoords="axes fraction",
        fontsize=9.5, color=INK["secondary"], va="bottom",
    )
    ax.legend(loc="upper right", ncol=2)
    _finish(fig, out / "fig4d_international_gold_wages.png")


# --- reporting --------------------------------------------------------------

CAPTIONS = {
    "fig4a": (
        "US average hourly earnings deflated by six different denominators, each "
        "indexed to 100 in 1971 and drawn on one logarithmic axis. Nominal wages rise "
        "more than eightfold and CPI-deflated wages are roughly flat, but wages measured "
        "against wealth-storing assets collapse. The commodity series is the control: "
        "wages outpace consumption goods, so the collapse is specific to assets."
    ),
    "fig4b": (
        "Gold-denominated wages indexed from four different base years. Three of the four "
        "show declines of 54-86 percent. The 1980 base reverses the sign because 1980 was "
        "the Hunt-brothers gold spike; anchoring on a bubble peak flatters the wage. The "
        "direction of the result is robust, but its magnitude is not base-year independent."
    ),
    "fig4c": (
        "Housing-denominated wages under both available house-price measures, 1987 onward. "
        "The two agree that wages lost roughly a fifth to a third of their housing purchasing "
        "power but differ in level by about 14 index points, because Case-Shiller is a "
        "quality-adjusted repeat-sales index while the median sale price moves with the mix "
        "of houses transacted."
    ),
    "fig4d": (
        "Gold-denominated wages for the countries with obtainable data, indexed to 100 in "
        "2000. Japan falls furthest, though its 1990 start at the bubble peak exaggerates "
        "the magnitude. India is plotted as markers because its series has fewer than a "
        "dozen observations across a survey-methodology break."
    ),
}


def build_report(
    indices: USIndices,
    robustness: dict[int, pd.Series],
    housing: dict[str, pd.Series],
    countries: Sequence[CountrySeries],
    skipped: dict[str, str],
    annual: dict[str, pd.Series],
) -> dict:
    """Assemble the numerical report."""
    from .config import US_SOURCES

    values = summarize(indices, END_YEAR)
    declines = {k: round(v - 100.0, 1) for k, v in values.items()}

    intl = {}
    for country in countries:
        series = country.index[country.index.index <= END_YEAR]
        if not len(series):
            continue
        last_year = int(series.index[-1])
        intl[country.code] = {
            "label": country.spec.label,
            "index_last": round(float(series.iloc[-1]), 1),
            "last_year": last_year,
            "anchor_year": country.anchor_year,
            "on_common_base": country.on_common_base,
            "change_pct": round(float(series.iloc[-1]) - 100.0, 1),
            "coverage": [int(series.index[0]), last_year],
            "caveat": country.spec.caveat,
        }

    tr_note = None
    if indices.total_return is not None:
        end = int(indices.total_return.index[-1])
        tr_note = {
            "index": round(float(indices.total_return.iloc[-1]), 1),
            "end_year": end,
            "note": (
                "Dividend-reinvested equity. The specification assumed this was only "
                "available from 1988; the Shiller data carries dividends back to 1871. "
                f"Truncated at {end} because dividends are published with a lag."
            ),
        }

    return {
        "base_year": BASE_YEAR,
        "end_year": END_YEAR,
        "wage_1971_nominal": round(float(annual["wage"].loc[BASE_YEAR]), 3),
        "wage_2024_nominal": round(float(annual["wage"].loc[END_YEAR]), 2),
        "index_values_2024": {k: round(v, 1) for k, v in values.items()},
        "decline_pct": declines,
        "sp500_total_return": tr_note,
        "robustness": {
            f"gold_{base}_base_index_2024": round(float(series.loc[END_YEAR]), 1)
            for base, series in sorted(robustness.items())
        },
        "housing_measures_1987_base": {
            key: round(float(series.loc[END_YEAR]), 1) for key, series in housing.items()
        },
        "international_gold_wages": {
            "base_year": INTL_BASE_YEAR,
            "countries": intl,
            "omitted": skipped,
        },
        "data_sources": {
            role: {"url": source.url, "note": source.note}
            for role, source in US_SOURCES.items()
        },
        "spec_deviations": [
            "FRED GOLDPMGBD228NLBM is discontinued (returns HTML); replaced with the "
            "datahub gold-prices CSV mirror. Verified: 1971 mean $40.92 against the "
            "specification's stated ~$41.",
            "FRED SP500 retains only a rolling 10-year window and cannot reach 1971; "
            "replaced with the Shiller dataset CSV mirror. Verified: 1971-01 = 93.49 "
            "against the specification's stated ~92.",
            "The specification's JSON template gives wage_1971_nominal = 3.45, which is "
            "the January value; the 1971 annual mean used here is 3.63.",
            "The specification expects the gold decline to be base-year independent. It "
            "is not: a 1980 base reverses the sign. Three of four bases still decline.",
            "The specification expects the two housing measures to agree; they agree on "
            "direction but differ by about 14 index points in 2024.",
            "China is omitted from the international comparison: its clean total-economy "
            "series ends in 1997, leaving the 2000-2024 window empty.",
        ],
        "figure_captions": CAPTIONS,
        "key_findings": [
            f"Nominal wages rose to {values['nominal']:.0f} while CPI-deflated wages "
            f"reached only {values['cpi_real']:.0f} — roughly flat across 53 years.",
            f"Against wealth-storing assets wages collapsed: gold {values['gold']:.0f}, "
            f"equities {values['sp500']:.0f}, housing {values['housing']:.0f}.",
            f"Against consumption goods they did not: the commodity index is "
            f"{values['commodity_ppi']:.0f}, so wages outpaced commodities. This is the "
            "paper's central distinction, and the data support it.",
        ],
    }


def main(refresh: bool = False) -> dict:
    """Build every figure and the numerical report."""
    from .config import COUNTRIES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    print("Loading data...")
    annual = fetch_us_annual(refresh=refresh)
    dividends = fetch_sp500_dividends(refresh=refresh)
    fx = fetch_fx_annual(refresh=refresh)
    oecd = fetch_oecd_wages(refresh=refresh)
    ilo = {
        spec.code: fetch_ilo_country(spec.code, spec.ilo_dataflow, refresh=refresh)
        for spec in COUNTRIES
        if spec.wage_source == "ilo"
    }

    print("Constructing indices...")
    indices = build_us_indices(annual, dividends)
    robustness = gold_robustness(annual, ROBUSTNESS_BASE_YEARS)
    housing = housing_comparison(annual, CASE_SHILLER_START)
    countries, skipped = build_international(
        annual["wage"], annual["gold"], fx, oecd, ilo
    )

    for name, value in summarize(indices).items():
        print(f"  {name:22s} {value:8.1f}")
    print(f"  countries built: {[c.code for c in countries]}")
    print(f"  countries skipped: {list(skipped)}")

    print("Generating figures...")
    fig4a(indices, OUTPUT_DIR)
    fig4b(robustness, OUTPUT_DIR)
    fig4c(housing, OUTPUT_DIR)
    fig4d(countries, skipped, OUTPUT_DIR)

    report = build_report(indices, robustness, housing, countries, skipped, annual)
    path = OUTPUT_DIR / "results_realdata.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return report


if __name__ == "__main__":
    import sys

    main(refresh="--refresh" in sys.argv)
