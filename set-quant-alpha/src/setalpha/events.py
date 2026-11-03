"""Research features from corporate events (Yahoo earnings / dividends) and NVDR flows.

These are NOT production-model features (see config.USE_FLOW_FEATURES): a new feature would silently change the
signal of the frozen paper-trading shadows. Experiments use them; a feature that earns its place is promoted through
a pre-registered experiment and a new paper shadow.

Point-in-time rules
* An earnings date E is mapped to the first trading day on or after it. Thai companies often report after the close,
  so the announcement reaction is measured over days E and E+1 and is known only at the close of E+1.
* `exp_earn_next5` uses last year's report date + 364 days, so it needs no future information. The actual next-event
  flag `diag_earn_in_hold` looks ahead and is for diagnostics only (prefix `diag_`).
* NVDR flows are lagged one trading day (publication time is not guaranteed before the evening run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from setalpha import config

DECAY_DAYS = 60  # an earnings reaction stays a feature for 60 trading days


def _calendar(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["Date"].unique()))


def _event_matrix(
    events: pd.DataFrame, dates: pd.DatetimeIndex, tickers, col: str | None = None
) -> pd.DataFrame:
    """Wide (date x ticker) matrix: True (or `col` value) on the first trading day on or after each event."""
    e = events[events["Ticker"].isin(tickers)].copy()
    pos = dates.searchsorted(e["Date"].to_numpy())
    e = e[pos < len(dates)]
    e["Date"] = dates[pos[pos < len(dates)]]
    vals = e[col] if col else True
    out = e.assign(v=vals).pivot_table(index="Date", columns="Ticker", values="v", aggfunc="last")
    return out.reindex(index=dates, columns=tickers)


def daily_alpha(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """r_t - beta_{t-1} x EW-liquid-market_t, the daily version of the training target."""
    C = prices.pivot_table(index="Date", columns="Ticker", values="Close")
    r = C.pct_change(fill_method=None)
    r = r.where(r.abs() < 0.45)
    beta = df.pivot_table(index="Date", columns="Ticker", values="beta_mkt").reindex_like(r)
    liquid = df.pivot_table(
        index="Date", columns="Ticker", values="liquid", aggfunc="first"
    ).reindex_like(r)
    mkt = r.where(liquid.fillna(False).astype(bool)).mean(axis=1)
    return r - beta.shift(1).mul(mkt, axis=0)


def earnings_features(
    df: pd.DataFrame, earnings: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Long frame (Date, Ticker) with earnings-event features."""
    dates = _calendar(df)
    tickers = sorted(df["Ticker"].unique())
    ar = np.arange(len(dates), dtype=float)[:, None]
    past = earnings[earnings["Date"] <= dates[-1]]
    ev = _event_matrix(past, dates, tickers).notna()
    idx = pd.DataFrame(np.where(ev, ar, np.nan), index=dates, columns=tickers)
    days_since = ar - idx.ffill().to_numpy()
    nxt = idx.shift(-1).bfill().to_numpy() - ar  # look-ahead: diagnostics only

    expected = past.assign(Date=past["Date"] + pd.Timedelta(days=364))
    ex = _event_matrix(expected, dates, tickers).notna()
    ex_idx = pd.DataFrame(np.where(ex, ar, np.nan), index=dates, columns=tickers)
    exp_next = ex_idx.shift(-1).bfill().to_numpy() - ar  # known: built from events a year earlier

    A = daily_alpha(df, prices).reindex(index=dates, columns=tickers)
    ear = (A + A.shift(-1)).where(ev).shift(1).ffill(limit=DECAY_DAYS)  # known at the close of E+1
    sur = _event_matrix(past.dropna(subset=["surprise_pct"]), dates, tickers, "surprise_pct")
    sur = sur.shift(1).ffill(limit=DECAY_DAYS)

    wide = {
        "days_since_earn": pd.DataFrame(days_since, index=dates, columns=tickers),
        "post_earn_5": pd.DataFrame((days_since <= 5).astype(float), index=dates, columns=tickers),
        "exp_earn_next5": pd.DataFrame((exp_next <= 5).astype(float), index=dates, columns=tickers),
        "diag_earn_in_hold": pd.DataFrame(
            (nxt <= config.HORIZON).astype(float), index=dates, columns=tickers
        ),
        "last_ear": ear,
        "last_surprise": sur,
    }
    long = pd.concat({k: v.stack(future_stack=True) for k, v in wide.items()}, axis=1)
    long.index.names = ["Date", "Ticker"]
    has = past.groupby("Ticker")["Date"].min()  # coverage starts at a ticker's first record
    long = long.reset_index()
    first = long["Ticker"].map(has)
    return long[first.notna() & (long["Date"] >= first)].reset_index(drop=True)


def nvdr_features(df: pd.DataFrame, nvdr: pd.DataFrame) -> pd.DataFrame:
    """Net NVDR buying scaled by the stock's traded value, lagged one day; 1-, 5- and 20-day windows."""
    dates = _calendar(df)
    net = (
        nvdr.assign(net=nvdr["nvdr_buy"] - nvdr["nvdr_sell"])
        .pivot_table(index="Date", columns="Ticker", values="net", aggfunc="sum")
        .reindex(dates)
    )
    dvol = df.pivot_table(index="Date", columns="Ticker", values="dvol60").reindex(
        index=dates, columns=net.columns
    )
    inten = (net / dvol).shift(1)
    out = pd.concat(
        {
            "nvdr_int_1": inten,
            "nvdr_int_5": inten.rolling(5, min_periods=3).sum(),
            "nvdr_int_20": inten.rolling(20, min_periods=10).sum(),
        },
        axis=1,
    ).stack(future_stack=True)
    out.index.names = ["Date", "Ticker"]
    return out.dropna(how="all").reset_index()
