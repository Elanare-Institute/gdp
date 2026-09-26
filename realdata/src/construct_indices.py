"""Construct the US asset-relative wage indices.

The whole construction reduces to one pure function, :func:`index_series`:
divide the wage by a denominator, then rebase so the base year equals 100.
Every one of the six series is that same operation with a different
denominator (a denominator of 1 gives the nominal series).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BASE_YEAR, END_YEAR, TOTAL_RETURN_END_YEAR


class IndexError_(ValueError):
    """Raised when an index cannot be constructed."""


def index_series(
    numerator: pd.Series,
    denominator: pd.Series | None,
    base_year: int,
) -> pd.Series:
    """Index ``numerator / denominator`` so that ``base_year`` equals 100.

    Args:
        numerator: The wage series, indexed by year.
        denominator: Price series to deflate by, indexed by year. ``None``
            leaves the numerator undeflated (the nominal series).
        base_year: Year set to 100.

    Returns:
        The indexed ratio over the years present in both inputs.

    Raises:
        IndexError_: If the base year is missing from either input.
    """
    ratio = numerator if denominator is None else (numerator / denominator).dropna()

    if base_year not in ratio.index:
        raise IndexError_(
            f"base year {base_year} not available (range "
            f"{ratio.index.min()}-{ratio.index.max()})"
        )

    base_value = ratio.loc[base_year]
    if base_value == 0 or pd.isna(base_value):
        raise IndexError_(f"base year {base_year} value is {base_value}")

    return (ratio / base_value * 100.0).dropna()


def total_return_index(price: pd.Series, dividend: pd.Series) -> pd.Series:
    """Build a dividend-reinvested S&P 500 level series.

    The specification assumes a total-return series is only available from
    1988; the Shiller data carries dividends back to 1871, so it can be
    constructed for the full period. Dividends are published with a lag and
    recent entries are zero-filled, so the series is truncated at
    ``TOTAL_RETURN_END_YEAR``.

    Each year's gross return is the price change plus that year's dividend
    yield, compounded from the first available year.
    """
    joined = pd.DataFrame({"price": price, "dividend": dividend}).dropna()
    joined = joined[(joined["dividend"] > 0) & (joined.index <= TOTAL_RETURN_END_YEAR)]
    if joined.empty:
        raise IndexError_("no overlapping price and dividend observations")

    price_return = joined["price"] / joined["price"].shift(1)
    dividend_yield = joined["dividend"].shift(1) / joined["price"].shift(1)
    gross = (price_return + dividend_yield).dropna()

    level = gross.cumprod()
    first_year = joined.index[0]
    level.loc[first_year] = 1.0
    return level.sort_index() * joined["price"].iloc[0]


@dataclass(frozen=True)
class USIndices:
    """The six headline series plus the total-return variant."""

    base_year: int
    series: dict[str, pd.Series]
    total_return: pd.Series | None

    def value_at(self, name: str, year: int) -> float | None:
        """Index value for `name` at `year`, or None if unavailable."""
        s = self.series.get(name)
        if s is None or year not in s.index:
            return None
        return float(s.loc[year])


def build_us_indices(
    annual: dict[str, pd.Series],
    dividends: pd.Series | None = None,
    base_year: int = BASE_YEAR,
) -> USIndices:
    """Build all six US indices from the annual source series.

    The commodity series is the analytical control: the paper's claim is that
    wages collapse against wealth-storing assets but not against consumption
    goods, so this one is expected to sit above 100.
    """
    wage = annual["wage"]

    series: dict[str, pd.Series] = {
        "nominal": index_series(wage, None, base_year),
        "cpi_real": index_series(wage, annual["cpi"], base_year),
        "commodity_ppi": index_series(wage, annual["commodities"], base_year),
        "housing": index_series(wage, annual["housing_median"], base_year),
        "sp500": index_series(wage, annual["sp500"], base_year),
        "gold": index_series(wage, annual["gold"], base_year),
    }

    total_return: pd.Series | None = None
    if dividends is not None:
        try:
            tr_level = total_return_index(annual["sp500"], dividends)
            total_return = index_series(wage, tr_level, base_year)
        except IndexError_ as exc:
            print(f"  [skip] total-return series: {exc}")

    return USIndices(base_year=base_year, series=series, total_return=total_return)


def gold_robustness(
    annual: dict[str, pd.Series], base_years: tuple[int, ...]
) -> dict[int, pd.Series]:
    """Gold-denominated wage indexed from several base years.

    This is the robustness check, and it does not come out the way the
    specification anticipates: anchoring on 1980 — the Hunt-brothers gold
    spike — reverses the sign of the result. Three of the four base years
    still show a large decline. See README.
    """
    return {
        year: index_series(annual["wage"], annual["gold"], year) for year in base_years
    }


def housing_comparison(
    annual: dict[str, pd.Series], base_year: int
) -> dict[str, pd.Series]:
    """Housing-denominated wage under both available house-price measures.

    The two disagree in level by as much as 14 index points. That is expected:
    Case-Shiller is a quality-adjusted repeat-sales index, while the median
    sale price drifts with the size and mix of houses actually transacted.
    They agree on direction, not magnitude.
    """
    return {
        "median_price": index_series(annual["wage"], annual["housing_median"], base_year),
        "case_shiller": index_series(annual["wage"], annual["housing_cs"], base_year),
    }


def summarize(indices: USIndices, year: int = END_YEAR) -> dict[str, float]:
    """Index values at `year` for every series."""
    out = {name: indices.value_at(name, year) for name in indices.series}
    if indices.total_return is not None and year in indices.total_return.index:
        out["sp500_total_return"] = float(indices.total_return.loc[year])
    return {k: v for k, v in out.items() if v is not None}
