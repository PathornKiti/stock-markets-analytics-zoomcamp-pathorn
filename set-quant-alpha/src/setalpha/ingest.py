"""Step 1 - ingest raw data from four sources into an incremental parquet store.

Sources
-------
1. Yahoo Finance screener  -> the list of every Thai common stock (SET + mai)
2. Yahoo Finance prices    -> daily OHLCV for ~880 Thai stocks + global market series
3. FRED                    -> US rates, credit spread, THB/USD
4. Cboe CSV feed           -> VIX term structure (VIX9D, VIX, VIX3M), VVIX, SKEW
5. Kenneth French library  -> Fama-French Emerging Markets 5 factors + momentum (monthly, USD)

Two regimes (rubric: "run from a file on drive, or download from the sources"):
* ``mode="file"`` - read the parquet store as it is (no network). The store can live on a synced drive folder
  (``SETALPHA_DATA_DIR``).
* ``mode="live"`` - update the store incrementally: only the days after the last
  stored date are downloaded (with a small overlap to pick up revisions).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

import pandas as pd

from setalpha import config

log = logging.getLogger(__name__)
Mode = Literal["file", "live"]

PRICES = config.RAW_DIR / "prices.parquet"
UNIVERSE = config.RAW_DIR / "universe.parquet"
MARKET = config.RAW_DIR / "market.parquet"
FRED = config.RAW_DIR / "fred.parquet"
CBOE = config.RAW_DIR / "cboe.parquet"
SECTORS = config.RAW_DIR / "sectors.parquet"
FAMA_FRENCH = config.RAW_DIR / "ff_em_factors.parquet"
EARNINGS = config.RAW_DIR / "earnings.parquet"
DIVIDENDS = config.RAW_DIR / "dividends.parquet"
EVENTS_REFRESH_DAYS = 7

FUND_WORDS = r"REIT|INFRA|FUND|LEASEHOLD|PROPERTY|TRUST|\bIF\b|ETF"


# ----------------------------------------------------------------------------- helpers
def _yf_download(tickers: list[str], start: str, batch: int = 100) -> pd.DataFrame:
    """yfinance in batches -> long frame [Date, Ticker, Open, High, Low, Close, Volume]."""
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        raw = yf.download(
            chunk, start=start, auto_adjust=True, group_by="ticker", progress=False, threads=True
        )
        if raw.empty:
            continue
        for t in chunk:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t]
            else:
                sub = raw
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            frames.append(sub.reset_index().assign(Ticker=t))
        log.info("    prices: %d/%d tickers requested", min(i + batch, len(tickers)), len(tickers))
    cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)[cols]
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out


def _upsert(path, new: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    """Append `new` to the parquet at `path`, newest version of each key wins."""
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new = (
        new.drop_duplicates(subset=list(keys), keep="last")
        .sort_values(list(keys))
        .reset_index(drop=True)
    )
    new.to_parquet(path, index=False)
    return new


def _incremental_start(path, date_col: str = "Date", overlap_days: int = 10) -> str | None:
    if not path.exists():
        return None
    last = pd.read_parquet(path, columns=[date_col])[date_col].max()
    return (pd.Timestamp(last) - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------- sources
def discover_universe(mode: Mode = "file") -> pd.DataFrame:
    """Every Thai common stock listed on SET / mai, from Yahoo's equity screener.

    Depositary receipts (``*80.BK``), warrants / NVDR / foreign-board lines (``-W``, ``-R``,
    ``-F``) and property / infrastructure funds are removed.
    """
    if mode == "file" and UNIVERSE.exists():
        return pd.read_parquet(UNIVERSE)
    import yfinance as yf
    from yfinance import EquityQuery

    query, rows, offset = EquityQuery("eq", ["region", "th"]), [], 0
    while True:
        res = yf.screen(
            query, size=250, offset=offset, sortField="intradaymarketcap", sortAsc=False
        )
        quotes = res.get("quotes", [])
        rows += quotes
        offset += len(quotes)
        if not quotes or offset >= res.get("total", 0):
            break
    df = pd.DataFrame(rows)
    base = df["symbol"].str.removesuffix(".BK")
    name = df["shortName"].fillna("")
    keep = (
        df["symbol"].str.endswith(".BK")
        & (df["quoteType"] == "EQUITY")
        & ~base.str.contains("-")
        & ~name.str.contains("_DR|DR ", regex=True)
        & ~name.str.upper().str.contains(FUND_WORDS, regex=True)
    )
    out = df.loc[keep, ["symbol", "shortName", "marketCap"]].rename(
        columns={"symbol": "Ticker", "shortName": "name"}
    )
    out.to_parquet(UNIVERSE, index=False)
    log.info("    universe: %d Thai common stocks", len(out))
    return out


def _only(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Keep the requested universe only (the store may hold rows written by older runs)."""
    return df[df["Ticker"].isin(set(tickers))].reset_index(drop=True)


def ingest_prices(tickers: list[str], mode: Mode = "file") -> pd.DataFrame:
    """Daily OHLCV for every Thai stock, updated incrementally in live mode."""
    if mode == "file":
        return _only(pd.read_parquet(PRICES), tickers)
    start = _incremental_start(PRICES)
    known = (
        set(pd.read_parquet(PRICES, columns=["Ticker"])["Ticker"].unique())
        if PRICES.exists()
        else set()
    )
    new_names = [t for t in tickers if t not in known]
    parts = []
    if start:  # incremental top-up of names already stored
        log.info("    prices: incremental update from %s", start)
        parts.append(_yf_download(sorted(known & set(tickers)), start))
    if new_names:  # full history for names seen for the first time
        log.info("    prices: full history for %d new tickers", len(new_names))
        parts.append(_yf_download(new_names, config.START))
    new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    store = _upsert(PRICES, new, ["Ticker", "Date"]) if len(new) else pd.read_parquet(PRICES)
    return _only(store, tickers)


def ingest_market(mode: Mode = "file") -> pd.DataFrame:
    """Global market series from Yahoo (USD/THB, EEM, S&P 500, dollar, oil, gold, benchmark)."""
    if mode == "file":
        return pd.read_parquet(MARKET)
    start = _incremental_start(MARKET) or "2009-01-01"
    new = _yf_download(list(config.MACRO_YF), start)
    return _upsert(MARKET, new, ["Ticker", "Date"])


def ingest_fred(mode: Mode = "file") -> pd.DataFrame:
    """US 10y and 3m yields, high-yield spread and THB/USD from FRED (small: full refresh)."""
    if mode == "file":
        return pd.read_parquet(FRED)
    from pandas_datareader import data as pdr

    df = pdr.DataReader(list(config.FRED_SERIES), "fred", start="2009-01-01").rename(
        columns=config.FRED_SERIES
    )
    df.index.name = "Date"
    df = df.reset_index()
    df.to_parquet(FRED, index=False)
    return df


def ingest_cboe(mode: Mode = "file") -> pd.DataFrame:
    """VIX term structure, VVIX and SKEW from Cboe's free daily CSV files."""
    if mode == "file":
        return pd.read_parquet(CBOE)
    out = None
    for sym in config.CBOE_SERIES:
        raw = pd.read_csv(
            f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
        )
        value = "CLOSE" if "CLOSE" in raw.columns else raw.columns[-1]
        s = pd.DataFrame(
            {
                "Date": pd.to_datetime(raw["DATE"], format="%m/%d/%Y"),
                sym.lower(): raw[value].astype(float),
            }
        )
        out = s if out is None else out.merge(s, on="Date", how="outer")
    out = out.sort_values("Date").reset_index(drop=True)
    out.to_parquet(CBOE, index=False)
    return out


def ingest_sectors(tickers: list[str], mode: Mode = "file") -> pd.DataFrame:
    """Sector / industry per stock from Yahoo profiles (fetched once, only for missing names)."""
    have = (
        pd.read_parquet(SECTORS)
        if SECTORS.exists()
        else pd.DataFrame(columns=["Ticker", "sector", "industry"])
    )
    if mode == "file":
        return have
    import yfinance as yf

    missing = [t for t in tickers if t not in set(have["Ticker"])]
    rows = []
    for i, t in enumerate(missing, 1):
        try:
            info = yf.Ticker(t).info
            rows.append(
                {"Ticker": t, "sector": info.get("sector"), "industry": info.get("industry")}
            )
        except Exception as exc:  # noqa: BLE001 - one bad profile must not stop the run
            log.warning("    sector lookup failed for %s: %s", t, exc)
        if i % 100 == 0:
            log.info("    sectors: %d/%d", i, len(missing))
    if rows:
        have = _upsert(SECTORS, pd.DataFrame(rows), ["Ticker"])
    return have


def ingest_fama_french(mode: Mode = "file") -> pd.DataFrame | None:
    """Monthly Fama-French Emerging Markets factors (Mkt-RF, SMB, HML, RMW, CMA, RF) + momentum WML, in %."""
    if mode == "file":
        return pd.read_parquet(FAMA_FRENCH) if FAMA_FRENCH.exists() else None
    from pandas_datareader import data as pdr

    five = pdr.DataReader("Emerging_5_Factors", "famafrench", start="2005-01-01")[0]
    mom = pdr.DataReader("Emerging_MOM_Factor", "famafrench", start="2005-01-01")[0]
    df = five.join(mom, how="inner")
    df.index = df.index.to_timestamp(how="end").normalize()
    df.index.name = "Date"
    df = df.reset_index()
    df.to_parquet(FAMA_FRENCH, index=False)
    return df


def load_external_flows() -> pd.DataFrame | None:
    """Optional: SET 'trading by investor type' exported by hand from set.or.th.

    Drop a CSV with columns ``Date, foreign_net, institution_net, proprietary_net, retail_net``
    (million THB) into ``data/external/set_investor_type.csv`` and it is joined as features.
    SET blocks automated downloads, so this file is never scraped.
    """
    path = config.EXTERNAL_DIR / "set_investor_type.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    log.info("    external investor-type flows: %d days", len(df))
    return df


def _events_one(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    try:
        ed = tk.get_earnings_dates(limit=60)
    except Exception:  # noqa: BLE001 - many small caps have no earnings calendar
        ed = None
    earn = pd.DataFrame(columns=["Ticker", "Date", "eps_est", "eps_rep", "surprise_pct"])
    if ed is not None and len(ed):
        idx = pd.DatetimeIndex(ed.index)
        idx = idx.tz_convert("Asia/Bangkok") if idx.tz is not None else idx
        earn = pd.DataFrame(
            {
                "Ticker": ticker,
                "Date": idx.tz_localize(None).normalize(),
                "eps_est": ed.get("EPS Estimate"),
                "eps_rep": ed.get("Reported EPS"),
                "surprise_pct": ed.get("Surprise(%)"),
            }
        ).reset_index(drop=True)
    div = pd.DataFrame(columns=["Ticker", "Date", "dividend"])
    try:
        d = tk.dividends
        if len(d):
            div = pd.DataFrame(
                {
                    "Ticker": ticker,
                    "Date": pd.DatetimeIndex(d.index).tz_localize(None).normalize(),
                    "dividend": d.to_numpy(),
                }
            )
    except Exception:  # noqa: BLE001
        pass
    return earn, div


def ingest_events(
    tickers: list[str], mode: Mode = "file", workers: int = 8
) -> dict[str, pd.DataFrame | None]:
    """Earnings dates (with EPS estimate / reported / surprise %) and dividends per stock from Yahoo.

    Research data: Yahoo's earnings calendar for Thai stocks starts around 2020 and has gaps. Dates are converted to
    Bangkok calendar days. Not part of the daily run: `scripts/fetch_events.py` refreshes the store on demand
    (at most once every EVENTS_REFRESH_DAYS days).
    """
    fresh = (
        EARNINGS.exists()
        and (pd.Timestamp.now() - pd.Timestamp(EARNINGS.stat().st_mtime, unit="s")).days
        < EVENTS_REFRESH_DAYS
    )
    if mode == "live" and not fresh:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as pool:
            out = list(pool.map(_events_one, tickers))
        earn = [e for e, _ in out if len(e)]
        div = [d for _, d in out if len(d)]
        if earn:  # a Yahoo outage leaves the existing store untouched
            _upsert(EARNINGS, pd.concat(earn, ignore_index=True), ["Ticker", "Date"])
        if div:
            _upsert(DIVIDENDS, pd.concat(div, ignore_index=True), ["Ticker", "Date"])
        log.info(
            "    events: earnings for %d tickers, dividends for %d tickers", len(earn), len(div)
        )
    return {
        "earnings": pd.read_parquet(EARNINGS) if EARNINGS.exists() else None,
        "dividends": pd.read_parquet(DIVIDENDS) if DIVIDENDS.exists() else None,
    }


def load_external_nvdr() -> pd.DataFrame | None:
    """Optional: SET NVDR trading by security, collected by hand from set.or.th (never scraped).

    CSV ``data/external/nvdr_by_stock.csv`` with columns ``Date, Ticker, nvdr_buy, nvdr_sell`` (THB; Ticker without
    the .BK suffix is accepted). NVDR holders are mostly foreign investors, so this is a per-stock foreign-flow proxy.
    """
    path = config.EXTERNAL_DIR / "nvdr_by_stock.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df.loc[~df["Ticker"].str.endswith(".BK"), "Ticker"] += ".BK"
    log.info(
        "    external NVDR flows: %d rows, %d tickers, %s -> %s",
        len(df),
        df["Ticker"].nunique(),
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    return df


def ingest_all(mode: Mode = "file") -> dict[str, pd.DataFrame]:
    if mode == "file" and not PRICES.exists():
        raise FileNotFoundError(f"No local data in {config.RAW_DIR}. Run once with --mode live.")
    universe = discover_universe(mode)
    tickers = universe["Ticker"].tolist()
    return {
        "universe": universe,
        "prices": ingest_prices(tickers, mode),
        "market": ingest_market(mode),
        "fred": ingest_fred(mode),
        "cboe": ingest_cboe(mode),
        "sectors": ingest_sectors(tickers, mode),
        "fama_french": ingest_fama_french(mode),
        "flows": load_external_flows(),
        "nvdr": load_external_nvdr(),
        **ingest_events(
            tickers, "file"
        ),  # research data: read the store only; refresh with scripts/fetch_events.py
    }
