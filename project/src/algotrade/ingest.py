"""Step 1 - pull raw data from every source and land it in `project/data/raw/`.

Supports the rubric's "dual regime" requirement: `mode="file"` replays whatever
is already on disk (fast, offline, reproducible for a peer reviewer), while
`mode="live"` re-hits the APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from smaz import data as smaz_data

PROJECT_DATA = Path(__file__).resolve().parents[2] / "data"
RAW = PROJECT_DATA / "raw"

Mode = Literal["file", "live"]

# TODO: replace with the universe your project actually trades.
TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]
BENCHMARK = "SPY"
MACRO_SERIES: list[str] = ["DGS10", "FEDFUNDS", "CPILFESL", "UNRATE"]


def _path(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    return RAW / f"{name}.parquet"


def ingest_prices(mode: Mode = "file", start: str = "2010-01-01") -> pd.DataFrame:
    """OHLCV bars for the trading universe plus the benchmark."""
    path = _path("prices")
    if mode == "file" and path.exists():
        return pd.read_parquet(path)

    df = smaz_data.load_ohlcv([*TICKERS, BENCHMARK], start=start, refresh=(mode == "live"))
    df.to_parquet(path)
    return df


def ingest_macro(mode: Mode = "file", start: str = "2010-01-01") -> pd.DataFrame:
    """Macro context from FRED."""
    path = _path("macro")
    if mode == "file" and path.exists():
        return pd.read_parquet(path)

    df = smaz_data.load_fred(MACRO_SERIES, start=start, refresh=(mode == "live"))
    df.to_parquet(path)
    return df


def ingest_all(mode: Mode = "file") -> dict[str, pd.DataFrame]:
    return {"prices": ingest_prices(mode), "macro": ingest_macro(mode)}
