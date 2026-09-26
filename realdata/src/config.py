"""Configuration: data sources, series identifiers, and figure parameters.

Every URL, series ID, and parameter lives here; nothing is hard-coded in the
fetch, construction, or plotting layers.

Two sources named in the specification are no longer usable and are replaced
here (see README for the verification trail):

  * ``GOLDPMGBD228NLBM`` (LBMA gold via FRED) has been discontinued and now
    returns an HTML error page rather than CSV.
  * ``SP500`` via FRED retains only a rolling 10-year window, so it cannot
    reach 1971.

The replacements are plain-CSV mirrors requiring no API key. Both were
cross-checked against benchmark values stated independently in the
specification (gold 1971 ~= $41/oz; S&P January 1971 ~= 92), and both agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output"

BASE_YEAR: Final[int] = 1971
END_YEAR: Final[int] = 2024

# Alternate base years for the fig4b robustness check. 1980 is deliberately
# included: it was the Hunt-brothers gold spike, and anchoring there reverses
# the sign of the result. See README.
ROBUSTNESS_BASE_YEARS: Final[tuple[int, ...]] = (1971, 1975, 1980, 1985)

CASE_SHILLER_START: Final[int] = 1987  # CSUSHPINSA begins 1987-01
INTL_BASE_YEAR: Final[int] = 2000


@dataclass(frozen=True)
class Source:
    """A downloadable time series.

    Attributes:
        key: Local cache filename stem.
        url: Fully-formed download URL.
        date_col: Column holding the observation date.
        value_col: Column holding the observation value.
        note: Why this source was chosen, especially where it departs from the
            specification.
    """

    key: str
    url: str
    date_col: str = "DATE"
    value_col: str = ""
    note: str = ""

    @property
    def cache_path(self) -> Path:
        return DATA_DIR / f"{self.key}.csv"


def _fred(series_id: str, start: str = "1960-01-01", end: str = "2025-12-31") -> str:
    """Build a FRED CSV download URL (no API key required)."""
    return (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )


# --- United States series ---------------------------------------------------

US_SOURCES: Final[dict[str, Source]] = {
    "wage": Source(
        key="wage_AHETPI",
        url=_fred("AHETPI"),
        value_col="AHETPI",
        note="Average hourly earnings, production and nonsupervisory employees.",
    ),
    "cpi": Source(
        key="cpi_CPIAUCSL",
        url=_fred("CPIAUCSL"),
        value_col="CPIAUCSL",
        note="CPI-U, all items, seasonally adjusted.",
    ),
    "housing_median": Source(
        key="housing_MSPUS",
        url=_fred("MSPUS", start="1963-01-01"),
        value_col="MSPUS",
        note="Median sales price of houses sold; quarterly.",
    ),
    "housing_cs": Source(
        key="housing_CSUSHPINSA",
        url=_fred("CSUSHPINSA", start="1987-01-01"),
        value_col="CSUSHPINSA",
        note="Case-Shiller national home price index; monthly from 1987.",
    ),
    "commodities": Source(
        key="ppi_PPIACO",
        url=_fred("PPIACO"),
        value_col="PPIACO",
        note="PPI all commodities — the control series for consumption goods.",
    ),
    "gold": Source(
        key="gold_datahub",
        url="https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv",
        date_col="Date",
        value_col="Price",
        note=(
            "REPLACEMENT for FRED GOLDPMGBD228NLBM, which is discontinued. "
            "Monthly USD/troy oz, 1833-present. Verified: 1971 mean $40.92, "
            "matching the specification's stated ~$41."
        ),
    ),
    "sp500": Source(
        key="sp500_shiller",
        url="https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv",
        date_col="Date",
        value_col="SP500",
        note=(
            "REPLACEMENT for FRED SP500, which retains only 10 years. This is "
            "the Shiller dataset as plain CSV, monthly 1871-present, and it "
            "carries a Dividend column that permits a total-return series. "
            "Verified: 1971-01 = 93.49, matching the specification's ~92."
        ),
    ),
}

# The Shiller file also supplies dividends, used for the total-return variant.
SP500_DIVIDEND_COL: Final[str] = "Dividend"

# Shiller publishes dividends with a lag; recent months are zero-filled, so the
# total-return series is truncated at this year.
TOTAL_RETURN_END_YEAR: Final[int] = 2024


# --- International series ---------------------------------------------------

@dataclass(frozen=True)
class CountrySpec:
    """A country in the international gold-denominated wage comparison."""

    code: str
    label: str
    wage_source: str  # "oecd" | "ilo" | "fred"
    fx_series: str | None  # FRED series, local-currency-per-USD; None for USA
    ilo_dataflow: str = ""
    start_year: int = 1971
    marker_only: bool = False  # plot as points, not a line
    caveat: str = ""


OECD_WAGE_URL: Final[str] = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ELS.SAE,DSD_EARNINGS@AV_AN_WAGE,1.0/all"
    "?startPeriod=1970&format=csvfilewithlabels"
)
"""OECD average annual wages.

The key must be ``all``: a dimension-filtered key such as ``JPN.A.....``
returns HTTP 404 against this dataflow. Filter after download on
``REF_AREA``, ``UNIT_MEASURE`` (currency) and ``PRICE_BASE`` == "V"
(current prices; "Q" is constant prices and must not be used, since a
gold-denominated calculation needs nominal local currency).
"""

ILO_BASE_URL: Final[str] = "https://sdmx.ilo.org/rest/data/ILO"
ILO_ACCEPT: Final[str] = "application/vnd.sdmx.data+csv;version=1.0.0"
"""ILOSTAT SDMX.

The bulk-download paths named in the specification are dead
(``rplumber.ilo.org`` returns HTTP 200 with an empty body;
``webapps.ilo.org/ilostat-files/`` returns 404). This REST service works, but
an unfiltered query times out with HTTP 504 — always filter by country.
"""

COUNTRIES: Final[tuple[CountrySpec, ...]] = (
    CountrySpec(
        code="USA", label="United States", wage_source="fred", fx_series=None,
        start_year=1971,
    ),
    CountrySpec(
        code="JPN", label="Japan", wage_source="oecd", fx_series="DEXJPUS",
        start_year=1990,
        caveat=(
            "OECD coverage begins in 1990, at the peak of the asset bubble. "
            "Anchoring there maximizes the apparent decline; 1971-1989 is "
            "unavailable without a registered API key."
        ),
    ),
    CountrySpec(
        code="BRA", label="Brazil", wage_source="ilo", fx_series="DEXBZUS",
        ilo_dataflow="DF_EAR_EMTM_SEX_ECO_CUR_NB", start_year=1995,
        caveat=(
            "The nominal local-currency series spans the 1994 Real "
            "redenomination; gold denomination cancels the currency switch in "
            "principle, but continuity is checked visually."
        ),
    ),
    CountrySpec(
        code="IND", label="India", wage_source="ilo", fx_series="DEXINUS",
        ilo_dataflow="DF_EAR_EMTA_SEX_ECO_CUR_NB", start_year=2005,
        marker_only=True,
        caveat=(
            "Roughly 11 observations with survey-methodology breaks "
            "(NSS to PLFS). Plotted as markers rather than a line."
        ),
    ),
)

OMITTED_COUNTRIES: Final[dict[str, str]] = {
    "CHN": (
        "Omitted. ILOSTAT appears to cover 1971-2022, but that span mixes "
        "sub-sector series. Restricting to a clean total-economy series "
        "(ECO=*_TOTAL, CUR_TYPE_LCU, SEX_T) leaves only about 10 observations "
        "ending in 1997, so the 2000-2024 target window is empty. Per the "
        "specification, countries whose data cannot be obtained are not forced in."
    ),
}

FX_SOURCES: Final[dict[str, Source]] = {
    spec.fx_series: Source(
        key=f"fx_{spec.fx_series}",
        url=_fred(spec.fx_series, start="1971-01-01"),
        value_col=spec.fx_series,
        note=f"Local currency per USD, for {spec.label}.",
    )
    for spec in COUNTRIES
    if spec.fx_series
}

# ILO dimension filters for a clean total-economy series.
ILO_FILTERS: Final[dict[str, str]] = {
    "CUR": "CUR_TYPE_LCU",
    "SEX": "SEX_T",
}
ILO_ECO_SUFFIX: Final[str] = "_TOTAL"


# --- Presentation -----------------------------------------------------------

# Colorblind-safe categorical palette, shared with the simulation figures.
PALETTE: Final[dict[str, str]] = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "violet": "#4a3aa7",
}

SERIES_COLORS: Final[dict[str, str]] = {
    "nominal": PALETTE["blue"],
    "cpi_real": PALETTE["aqua"],
    "commodity_ppi": "#898781",
    "housing": PALETTE["violet"],
    "sp500": PALETTE["magenta"],
    "gold": PALETTE["orange"],
}

SERIES_LABELS: Final[dict[str, str]] = {
    "nominal": "Nominal wage",
    "cpi_real": "CPI-real wage",
    "commodity_ppi": "Commodity-denominated (PPI)",
    "housing": "Housing-denominated",
    "sp500": "Equity-denominated (S&P 500)",
    "gold": "Gold-denominated",
}

COUNTRY_COLORS: Final[dict[str, str]] = {
    "USA": PALETTE["blue"],
    "JPN": PALETTE["aqua"],
    "BRA": PALETTE["yellow"],
    "IND": PALETTE["violet"],
}

INK: Final[dict[str, str]] = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

FIGURE_DPI: Final[int] = 300

# Benchmark values used by the tests to detect a source silently changing.
EXPECTED_BENCHMARKS: Final[dict[str, tuple[float, float]]] = {
    "gold_1971_mean": (40.92, 0.5),
    "sp500_1971_01": (93.49, 0.5),
    "wage_1971_mean": (3.63, 0.02),
}
