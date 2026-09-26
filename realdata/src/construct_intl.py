"""Gold-denominated wages for the international comparison.

For each country the wage is expressed in troy ounces of gold::

    gold_wage(t) = wage_local(t) / (gold_usd(t) * local_per_usd(t))

Gold trades in London dollars, so the local wage is converted through the
USD exchange rate. Denominating in gold also cancels currency redenominations
(such as Brazil's 1994 Real) in principle, since both numerator and
denominator move together.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import (
    COUNTRIES,
    ILO_ECO_SUFFIX,
    ILO_FILTERS,
    INTL_BASE_YEAR,
    OMITTED_COUNTRIES,
    CountrySpec,
)


@dataclass(frozen=True)
class CountrySeries:
    """One country's gold-denominated wage index."""

    spec: CountrySpec
    wage_local: pd.Series
    gold_wage: pd.Series
    index: pd.Series
    anchor_year: int

    @property
    def code(self) -> str:
        return self.spec.code

    @property
    def on_common_base(self) -> bool:
        """Whether this series is indexed to the shared base year."""
        return self.anchor_year == INTL_BASE_YEAR


def extract_oecd_wage(frame: pd.DataFrame, code: str, currency: str) -> pd.Series:
    """Pull one country's nominal annual wage from the OECD table.

    ``PRICE_BASE == "V"`` selects current prices. The constant-price variant
    ("Q") must not be used: a gold-denominated calculation needs nominal
    local currency.
    """
    subset = frame[
        (frame["REF_AREA"] == code)
        & (frame["UNIT_MEASURE"] == currency)
        & (frame["PRICE_BASE"] == "V")
    ]
    if subset.empty:
        raise ValueError(f"OECD: no current-price {currency} rows for {code}")

    series = (
        subset.assign(
            year=pd.to_numeric(subset["TIME_PERIOD"], errors="coerce"),
            value=pd.to_numeric(subset["OBS_VALUE"], errors="coerce"),
        )
        .dropna(subset=["year", "value"])
        .groupby("year")["value"]
        .mean()
    )
    series.index = series.index.astype(int)
    return series.sort_index()


def extract_ilo_wage(frame: pd.DataFrame, code: str) -> pd.Series:
    """Pull a clean total-economy wage series from an ILOSTAT table.

    The raw download mixes currencies, sexes, and sub-sector breakdowns, which
    is what makes an unfiltered series look longer than it really is. Filtering
    to local currency, both sexes, and a total-economy activity code gives the
    comparable series.
    """
    mask = pd.Series(True, index=frame.index)
    for column, value in ILO_FILTERS.items():
        if column in frame.columns:
            mask &= frame[column] == value
    if "ECO" in frame.columns:
        mask &= frame["ECO"].astype(str).str.upper().str.endswith(ILO_ECO_SUFFIX)

    subset = frame[mask]
    if subset.empty:
        raise ValueError(f"ILO: no clean total-economy rows for {code}")

    series = (
        subset.assign(
            year=pd.to_numeric(subset["TIME_PERIOD"], errors="coerce"),
            value=pd.to_numeric(subset["OBS_VALUE"], errors="coerce"),
        )
        .dropna(subset=["year", "value"])
        .groupby("year")["value"]
        .mean()
    )
    series.index = series.index.astype(int)
    return series.sort_index()


def gold_denominated_wage(
    wage_local: pd.Series,
    gold_usd: pd.Series,
    fx_local_per_usd: pd.Series | None,
) -> pd.Series:
    """Convert a local-currency wage into troy ounces of gold.

    A ``None`` exchange rate means the wage is already in USD.
    """
    gold_local = gold_usd if fx_local_per_usd is None else gold_usd * fx_local_per_usd
    return (wage_local / gold_local).dropna()


def build_country_series(
    spec: CountrySpec,
    wage_local: pd.Series,
    gold_usd: pd.Series,
    fx: pd.Series | None,
    base_year: int = INTL_BASE_YEAR,
) -> CountrySeries | None:
    """Build one country's indexed gold-wage series, or None if unusable."""
    gold_wage = gold_denominated_wage(wage_local, gold_usd, fx)
    # start_year is a hard floor, not a display preference: Brazil's pre-1994
    # observations are denominated in retired currencies (cruzado/cruzeiro) and
    # are not comparable to Real-era values, so they must not appear at all.
    gold_wage = gold_wage[gold_wage.index >= spec.start_year]
    if gold_wage.empty:
        return None

    anchor_value, anchor_year = _anchor(gold_wage, base_year)
    if anchor_value is None:
        return None
    index = gold_wage / anchor_value * 100.0

    return CountrySeries(
        spec=spec,
        wage_local=wage_local,
        gold_wage=gold_wage,
        index=index,
        anchor_year=anchor_year,
    )


def _anchor(gold_wage: pd.Series, base_year: int) -> tuple[float | None, int]:
    """Resolve the value to index against, at the common base year.

    Countries whose series has a gap at the base year are interpolated to it,
    so every country is genuinely on the same base. Only when the base year
    lies outside a country's coverage entirely does the anchor fall back to
    that country's own first year — which the figure must then disclose,
    because such a series is not comparable on level.
    """
    if base_year in gold_wage.index:
        return float(gold_wage.loc[base_year]), base_year

    before = gold_wage.index[gold_wage.index < base_year]
    after = gold_wage.index[gold_wage.index > base_year]
    if len(before) and len(after):
        lo, hi = int(before.max()), int(after.min())
        weight = (base_year - lo) / (hi - lo)
        value = gold_wage.loc[lo] + weight * (gold_wage.loc[hi] - gold_wage.loc[lo])
        return float(value), base_year

    if not len(gold_wage):
        return None, base_year
    first = int(gold_wage.index.min())
    return float(gold_wage.loc[first]), first


def build_international(
    us_wage: pd.Series,
    gold_usd: pd.Series,
    fx: dict[str, pd.Series],
    oecd_frame: pd.DataFrame,
    ilo_frames: dict[str, pd.DataFrame],
    base_year: int = INTL_BASE_YEAR,
) -> tuple[list[CountrySeries], dict[str, str]]:
    """Build every country series, returning the built ones and any skipped.

    Countries whose data cannot be obtained are reported rather than forced in.
    """
    built: list[CountrySeries] = []
    skipped: dict[str, str] = dict(OMITTED_COUNTRIES)

    currency_by_country = {"JPN": "JPY"}

    for spec in COUNTRIES:
        try:
            if spec.wage_source == "fred":
                wage = us_wage
            elif spec.wage_source == "oecd":
                wage = extract_oecd_wage(
                    oecd_frame, spec.code, currency_by_country[spec.code]
                )
            else:
                wage = extract_ilo_wage(ilo_frames[spec.code], spec.code)
        except (ValueError, KeyError) as exc:
            skipped[spec.code] = f"wage series unavailable: {exc}"
            continue

        rate = fx.get(spec.fx_series) if spec.fx_series else None
        series = build_country_series(spec, wage, gold_usd, rate, base_year)
        if series is None or len(series.index) < 3:
            skipped[spec.code] = "fewer than three usable observations after alignment"
            continue
        built.append(series)

    return built, skipped
