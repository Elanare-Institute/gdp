"""Download and cache the source time series.

All downloads are cached under ``data/``; a cached file is reused unless
``refresh=True``. Failures raise rather than returning partial data, so a
broken source can never quietly produce a half-empty figure.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    FX_SOURCES,
    ILO_ACCEPT,
    ILO_BASE_URL,
    OECD_WAGE_URL,
    US_SOURCES,
    Source,
)

TIMEOUT = 90
# FRED rejects some custom user-agent strings (returning no response at all),
# so curl's default UA is used. Do not set -A here.
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 3.0


class DataFetchError(RuntimeError):
    """Raised when a source cannot be downloaded or parsed."""


def _download(url: str, dest: Path, headers: dict[str, str] | None = None) -> None:
    """Fetch `url` into `dest`, failing loudly on error or on an HTML body.

    Transport is ``curl`` rather than ``requests``. FRED reliably serves curl
    in well under a second while hanging on urllib3 connections from this
    environment, so curl is the dependable path. Retries with linear backoff
    cover transient drops.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    command = ["curl", "-s", "-L", "--max-time", str(TIMEOUT)]
    for key, value in (headers or {}).items():
        command += ["-H", f"{key}: {value}"]
    command += ["-w", "%{http_code}", "-o", str(dest), url]

    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = subprocess.run(command, capture_output=True, text=True)
        status = result.stdout.strip()[-3:]
        if result.returncode == 0 and status == "200":
            break
        last_error = f"exit={result.returncode} http={status or 'n/a'}"
        if attempt < MAX_ATTEMPTS:
            print(f"    retry {attempt}/{MAX_ATTEMPTS - 1} ({last_error})")
            time.sleep(BACKOFF_SECONDS * attempt)
    else:
        dest.unlink(missing_ok=True)
        raise DataFetchError(f"download failed after {MAX_ATTEMPTS} attempts: {url}\n  {last_error}")

    text = dest.read_text(encoding="utf-8", errors="replace")
    # A discontinued FRED series returns an HTML error page with HTTP 200.
    if text.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        dest.unlink(missing_ok=True)
        raise DataFetchError(
            f"source returned HTML rather than CSV (likely discontinued): {url}"
        )
    if not text.strip():
        dest.unlink(missing_ok=True)
        raise DataFetchError(f"source returned an empty body: {url}")


def fetch_source(source: Source, refresh: bool = False) -> pd.DataFrame:
    """Return a source as a DataFrame, downloading it if not already cached."""
    if refresh or not source.cache_path.exists():
        _download(source.url, source.cache_path)

    try:
        frame = pd.read_csv(source.cache_path)
    except Exception as exc:
        raise DataFetchError(f"could not parse {source.cache_path}: {exc}") from exc

    if frame.empty:
        raise DataFetchError(f"{source.key} parsed to zero rows")
    return frame


def to_annual(
    frame: pd.DataFrame, date_col: str, value_col: str, name: str
) -> pd.Series:
    """Collapse a dated series to annual means, indexed by integer year.

    Annual means are used throughout: the sources mix monthly, quarterly, and
    annual frequencies, and averaging is both the most robust reconciliation
    and what the specification asks for (e.g. the 1971 gold price is the
    annual mean).
    """
    if date_col not in frame.columns:
        # FRED renamed its date column from DATE to observation_date.
        alternatives = [c for c in frame.columns if c.lower() in ("date", "observation_date")]
        if not alternatives:
            raise DataFetchError(f"{name}: no date column among {list(frame.columns)}")
        date_col = alternatives[0]

    if value_col not in frame.columns:
        raise DataFetchError(f"{name}: no column '{value_col}' among {list(frame.columns)}")

    values = pd.to_numeric(frame[value_col], errors="coerce")
    years = pd.to_datetime(frame[date_col], errors="coerce").dt.year

    series = (
        pd.DataFrame({"year": years, "value": values})
        .dropna()
        .groupby("year")["value"]
        .mean()
    )
    series.index = series.index.astype(int)
    series.name = name

    if series.empty:
        raise DataFetchError(f"{name}: no usable observations after aggregation")
    return series


def fetch_us_annual(refresh: bool = False) -> dict[str, pd.Series]:
    """Fetch every US series and return annual means keyed by role."""
    out: dict[str, pd.Series] = {}
    for role, source in US_SOURCES.items():
        frame = fetch_source(source, refresh=refresh)
        out[role] = to_annual(frame, source.date_col, source.value_col, role)
    return out


def fetch_sp500_dividends(refresh: bool = False) -> pd.Series:
    """Annual mean of the Shiller dividend column, for the total-return series."""
    from .config import SP500_DIVIDEND_COL

    source = US_SOURCES["sp500"]
    frame = fetch_source(source, refresh=refresh)
    return to_annual(frame, source.date_col, SP500_DIVIDEND_COL, "sp500_dividend")


def fetch_fx_annual(refresh: bool = False) -> dict[str, pd.Series]:
    """Fetch exchange rates (local currency per USD) as annual means."""
    out: dict[str, pd.Series] = {}
    for series_id, source in FX_SOURCES.items():
        frame = fetch_source(source, refresh=refresh)
        out[series_id] = to_annual(frame, source.date_col, source.value_col, series_id)
    return out


def fetch_oecd_wages(refresh: bool = False) -> pd.DataFrame:
    """Download the OECD average-annual-wages table.

    The key must be ``all``; a dimension-filtered key returns HTTP 404 against
    this dataflow, so filtering happens after download.
    """
    dest = DATA_DIR / "oecd_av_an_wage.csv"
    if refresh or not dest.exists():
        _download(OECD_WAGE_URL, dest)
    return pd.read_csv(dest, low_memory=False)


def fetch_ilo_country(code: str, dataflow: str, refresh: bool = False) -> pd.DataFrame:
    """Download one country's ILOSTAT earnings table.

    Filtering by country is mandatory: an unfiltered query times out (HTTP 504).
    """
    dest = DATA_DIR / f"ilo_{code}.csv"
    if refresh or not dest.exists():
        url = f"{ILO_BASE_URL},{dataflow},1.0/{code}.A.....?startPeriod=1971"
        _download(url, dest, headers={"Accept": ILO_ACCEPT})
    return pd.read_csv(dest, low_memory=False)


def fetch_all(refresh: bool = False) -> None:
    """Warm the cache for every source, reporting what succeeded."""
    from .config import COUNTRIES

    print("Fetching US series...")
    us = fetch_us_annual(refresh=refresh)
    for role, series in us.items():
        print(f"  {role:16s} {series.index.min()}-{series.index.max()} ({len(series)} yrs)")

    print("Fetching exchange rates...")
    for name, series in fetch_fx_annual(refresh=refresh).items():
        print(f"  {name:16s} {series.index.min()}-{series.index.max()}")

    print("Fetching international wages...")
    fetch_oecd_wages(refresh=refresh)
    print("  OECD average annual wages: cached")
    for spec in COUNTRIES:
        if spec.wage_source == "ilo":
            fetch_ilo_country(spec.code, spec.ilo_dataflow, refresh=refresh)
            print(f"  ILO {spec.code}: cached")


if __name__ == "__main__":
    import sys

    fetch_all(refresh="--refresh" in sys.argv)
