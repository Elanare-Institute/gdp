"""Consistency tests for the asset-relative wage indices.

These test the *claims*, not the implementation. Several encode findings that
contradict the specification's expectations; they are pinned here so the
findings cannot regress silently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    BASE_YEAR,
    CASE_SHILLER_START,
    END_YEAR,
    EXPECTED_BENCHMARKS,
    OMITTED_COUNTRIES,
    ROBUSTNESS_BASE_YEARS,
)
from src.construct_indices import (
    IndexError_,
    build_us_indices,
    gold_robustness,
    housing_comparison,
    index_series,
    summarize,
)
from src.fetch_data import fetch_sp500_dividends, fetch_us_annual


@pytest.fixture(scope="module")
def annual() -> dict[str, pd.Series]:
    """Annual source series, served from the local cache."""
    return fetch_us_annual()


@pytest.fixture(scope="module")
def indices(annual):
    return build_us_indices(annual, fetch_sp500_dividends())


# --- 1. definition of an index ---------------------------------------------

def test_base_year_is_exactly_100() -> None:
    """An index equals 100 in its base year by construction."""
    numerator = pd.Series({2000: 10.0, 2001: 12.0, 2002: 15.0})
    denominator = pd.Series({2000: 2.0, 2001: 3.0, 2002: 2.5})
    result = index_series(numerator, denominator, 2000)
    assert result.loc[2000] == pytest.approx(100.0)
    assert result.loc[2001] == pytest.approx(80.0)


def test_nominal_index_needs_no_denominator() -> None:
    """Passing None leaves the numerator undeflated."""
    wage = pd.Series({1971: 4.0, 2024: 32.0})
    assert index_series(wage, None, 1971).loc[2024] == pytest.approx(800.0)


def test_missing_base_year_is_rejected() -> None:
    """A base year outside the data is an error, not a silent rebase."""
    wage = pd.Series({2000: 1.0, 2001: 2.0})
    with pytest.raises(IndexError_, match="base year"):
        index_series(wage, None, 1971)


def test_every_series_starts_at_100(indices) -> None:
    for name, series in indices.series.items():
        assert series.loc[BASE_YEAR] == pytest.approx(100.0), name


# --- 2. source benchmarks ---------------------------------------------------

def test_source_benchmarks_match(annual) -> None:
    """Guard against a data source silently changing underneath us.

    These values were cross-checked against figures stated independently in the
    specification (gold ~$41, S&P ~92), which is what established that the
    replacement sources are the right series.
    """
    expected, tol = EXPECTED_BENCHMARKS["gold_1971_mean"]
    assert annual["gold"].loc[1971] == pytest.approx(expected, abs=tol)

    expected, tol = EXPECTED_BENCHMARKS["wage_1971_mean"]
    assert annual["wage"].loc[1971] == pytest.approx(expected, abs=tol)


def test_wage_1971_is_annual_mean_not_january(annual) -> None:
    """The specification's template gives 3.45, which is the January value.

    The annual mean is 3.63, and annual means are what this project uses.
    """
    assert annual["wage"].loc[1971] == pytest.approx(3.63, abs=0.02)
    assert annual["wage"].loc[1971] > 3.5


# --- 3. the paper's central claim ------------------------------------------

def test_wages_collapse_against_wealth_assets(indices) -> None:
    """Gold, equities and housing all end far below the 1971 base."""
    values = summarize(indices, END_YEAR)
    for asset in ("gold", "sp500", "housing"):
        assert values[asset] < 60.0, f"{asset} = {values[asset]}"


def test_wages_outpace_consumption_goods(indices) -> None:
    """The commodity control ends ABOVE 100 — the paper's key distinction.

    If this ever failed, the claim that the collapse is specific to
    wealth-storing assets would not hold.
    """
    assert summarize(indices, END_YEAR)["commodity_ppi"] > 100.0


def test_asset_collapse_is_larger_than_any_commodity_move(indices) -> None:
    """The asset decline dwarfs the commodity series' deviation from 100."""
    values = summarize(indices, END_YEAR)
    commodity_gap = abs(values["commodity_ppi"] - 100.0)
    gold_gap = abs(values["gold"] - 100.0)
    assert gold_gap > 3 * commodity_gap


# --- 4. CPI-real is roughly flat -------------------------------------------

def test_cpi_real_wage_is_roughly_flat(indices) -> None:
    """53 years of CPI-deflated wages land close to where they started."""
    assert 90.0 <= summarize(indices, END_YEAR)["cpi_real"] <= 120.0


def test_nominal_wage_rises_severalfold(indices) -> None:
    assert summarize(indices, END_YEAR)["nominal"] > 500.0


# --- 5. base-year fragility (contradicts the specification) ----------------

def test_gold_decline_holds_for_most_base_years(annual) -> None:
    """Three of the four base years show a large decline."""
    results = gold_robustness(annual, ROBUSTNESS_BASE_YEARS)
    declining = [b for b, s in results.items() if s.loc[END_YEAR] < 60.0]
    assert len(declining) >= 3


def test_1980_base_reverses_the_sign(annual) -> None:
    """Anchoring on the 1980 gold spike inverts the conclusion.

    The specification predicts the pattern is base-year independent. It is not.
    This is pinned deliberately: it is a real property of the data, and the
    figure reports it rather than hiding it.
    """
    results = gold_robustness(annual, ROBUSTNESS_BASE_YEARS)
    assert results[1980].loc[END_YEAR] > 100.0
    assert results[1971].loc[END_YEAR] < 20.0


def test_1980_was_a_gold_price_spike(annual) -> None:
    """The mechanism behind the reversal: gold's 1980 mean dwarfs 1975's."""
    assert annual["gold"].loc[1980] > 3 * annual["gold"].loc[1975]


# --- 6. housing measures disagree in level ---------------------------------

def test_housing_measures_agree_on_direction(annual) -> None:
    """Both house-price measures put wages below the 1987 base."""
    comparison = housing_comparison(annual, CASE_SHILLER_START)
    for name, series in comparison.items():
        assert series.loc[END_YEAR] < 100.0, name


def test_housing_measures_differ_in_level(annual) -> None:
    """They disagree by roughly 14 index points — expected, not a bug.

    Case-Shiller is quality-adjusted repeat-sales; the median sale price moves
    with the mix of houses transacted. The specification expected agreement.
    """
    comparison = housing_comparison(annual, CASE_SHILLER_START)
    gap = abs(comparison["median_price"].loc[END_YEAR] - comparison["case_shiller"].loc[END_YEAR])
    assert gap > 5.0


# --- 7. international coverage ---------------------------------------------

def test_china_is_omitted_with_a_reason() -> None:
    """China cannot be built for the target window and is documented as such."""
    assert "CHN" in OMITTED_COUNTRIES
    assert "1997" in OMITTED_COUNTRIES["CHN"]


def test_international_series_are_built():
    """The obtainable countries build and all decline against gold."""
    from src.config import COUNTRIES
    from src.construct_intl import build_international
    from src.fetch_data import fetch_fx_annual, fetch_ilo_country, fetch_oecd_wages

    annual = fetch_us_annual()
    countries, skipped = build_international(
        annual["wage"],
        annual["gold"],
        fetch_fx_annual(),
        fetch_oecd_wages(),
        {
            spec.code: fetch_ilo_country(spec.code, spec.ilo_dataflow)
            for spec in COUNTRIES
            if spec.wage_source == "ilo"
        },
    )
    assert {c.code for c in countries} >= {"USA", "JPN"}
    assert "CHN" in skipped
    for country in countries:
        assert country.index.iloc[-1] < 100.0, country.code


def test_brazil_excludes_pre_real_currency():
    """Brazil's series must not include pre-1994 redenominated observations.

    The 1989-1990 values are in cruzado/cruzeiro and are not comparable to
    Real-era figures; including them produces a spurious spike.
    """
    from src.config import COUNTRIES
    from src.construct_intl import build_international
    from src.fetch_data import fetch_fx_annual, fetch_ilo_country, fetch_oecd_wages

    annual = fetch_us_annual()
    countries, _ = build_international(
        annual["wage"], annual["gold"], fetch_fx_annual(), fetch_oecd_wages(),
        {
            spec.code: fetch_ilo_country(spec.code, spec.ilo_dataflow)
            for spec in COUNTRIES
            if spec.wage_source == "ilo"
        },
    )
    brazil = next((c for c in countries if c.code == "BRA"), None)
    if brazil is not None:
        assert brazil.index.index.min() >= 1995


# --- 8. caching -------------------------------------------------------------

def test_sources_are_cached_locally() -> None:
    """Every US source has a local cache file, so reruns need no network."""
    from src.config import US_SOURCES

    for role, source in US_SOURCES.items():
        assert source.cache_path.exists(), f"{role} not cached"
