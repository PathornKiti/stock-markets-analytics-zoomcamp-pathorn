"""Data-source loaders with a transparent on-disk parquet cache.

yfinance and FRED are rate-limited and occasionally flaky. Every loader here
writes its result to `data/cache/` and reuses it on the next call, so re-running
a notebook top-to-bottom does not re-hit the network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import pandas as pd

from smaz.config import CACHE_DIR


def _cache_key(name: str, **parts: object) -> str:
    blob = "|".join(f"{k}={parts[k]!r}" for k in sorted(parts))
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return f"{name}__{digest}.parquet"


def cached(
    name: str, loader: Callable[[], pd.DataFrame], *, refresh: bool = False, **key: object
) -> pd.DataFrame:
    """Return `loader()`, memoised to a parquet file under `data/cache/`.

    Pass `refresh=True` to bypass and overwrite the cached copy.
    """
    path = CACHE_DIR / _cache_key(name, **key)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = loader()
    df.to_parquet(path)
    return df


def load_ohlcv(
    tickers: str | Sequence[str],
    start: str = "2010-01-01",
    end: str | None = None,
    interval: str = "1d",
    *,
    auto_adjust: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV bars from Yahoo Finance.

    Returns a long/tidy frame with a `Ticker` column, which is far easier to
    `groupby` in later modules than yfinance's wide MultiIndex output.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)

    def _fetch() -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by="ticker",
            progress=False,
        )
        if raw.empty:
            raise ValueError(f"yfinance returned no rows for {tickers}")

        frames = []
        for t in tickers:
            sub = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            sub["Ticker"] = t
            frames.append(sub.reset_index())
        return pd.concat(frames, ignore_index=True).dropna(subset=["Close"])

    return cached(
        "ohlcv",
        _fetch,
        refresh=refresh,
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
    )


def load_fred(
    series_ids: str | Sequence[str], start: str = "2000-01-01", *, refresh: bool = False
) -> pd.DataFrame:
    """Download macro series from FRED (e.g. `CPILFESL`, `DGS10`, `FEDFUNDS`)."""
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    series_ids = list(series_ids)

    def _fetch() -> pd.DataFrame:
        from pandas_datareader import data as pdr

        return pdr.DataReader(series_ids, "fred", start=start).reset_index()

    return cached("fred", _fetch, refresh=refresh, series_ids=series_ids, start=start)


def load_sp500_constituents(*, refresh: bool = False) -> pd.DataFrame:
    """Scrape the current S&P 500 membership table from Wikipedia.

    Used in module 1's homework (index additions / tenure questions).
    """

    def _fetch() -> pd.DataFrame:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]
        return df

    return cached("sp500_constituents", _fetch, refresh=refresh, url="wikipedia")
