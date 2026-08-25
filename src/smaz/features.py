"""Feature engineering shared between module 2 (features) and modules 3-4 (models)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_returns(
    df: pd.DataFrame,
    price_col: str = "Close",
    group_col: str | None = "Ticker",
    horizons: Sequence[int] = (1, 5, 21, 63, 252),
) -> pd.DataFrame:
    """Add trailing (`ret_{h}d`) and forward (`fwd_ret_{h}d`) simple returns.

    The forward columns are the prediction targets for module 3. They are built
    with `.shift(-h)` so row `t` holds the return realised *after* `t` -- keeping
    the target strictly out of the feature set avoids look-ahead leakage.
    """
    out = df.sort_values([group_col, "Date"] if group_col else ["Date"]).copy()
    grp = out.groupby(group_col)[price_col] if group_col else out[price_col]

    for h in horizons:
        out[f"ret_{h}d"] = grp.pct_change(h)
        fwd = grp.shift(-h) / out[price_col] - 1
        out[f"fwd_ret_{h}d"] = fwd
    return out


def add_moving_averages(
    df: pd.DataFrame,
    price_col: str = "Close",
    group_col: str | None = "Ticker",
    windows: Sequence[int] = (10, 20, 50, 200),
) -> pd.DataFrame:
    """Simple moving averages plus the price/SMA ratio (the scale-free version)."""
    out = df.copy()
    grp = out.groupby(group_col)[price_col] if group_col else out[price_col]

    for w in windows:
        sma = (
            grp.transform(lambda s, w=w: s.rolling(w).mean())
            if group_col
            else out[price_col].rolling(w).mean()
        )
        out[f"sma_{w}"] = sma
        out[f"price_to_sma_{w}"] = out[price_col] / sma
    return out


def add_volatility(
    df: pd.DataFrame,
    ret_col: str = "ret_1d",
    group_col: str | None = "Ticker",
    windows: Sequence[int] = (21, 63),
    trading_days: int = 252,
) -> pd.DataFrame:
    """Annualised rolling realised volatility."""
    out = df.copy()
    grp = out.groupby(group_col)[ret_col] if group_col else out[ret_col]

    for w in windows:
        std = (
            grp.transform(lambda s, w=w: s.rolling(w).std())
            if group_col
            else out[ret_col].rolling(w).std()
        )
        out[f"vol_{w}d"] = std * np.sqrt(trading_days)
    return out


def add_ta_indicators(df: pd.DataFrame, group_col: str | None = "Ticker") -> pd.DataFrame:
    """RSI, MACD and Bollinger %B from the `ta` library used in the lectures."""
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands

    def _one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").copy()
        close = g["Close"]
        g["rsi_14"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        g["macd"] = macd.macd()
        g["macd_signal"] = macd.macd_signal()
        g["macd_diff"] = macd.macd_diff()
        bb = BollingerBands(close, window=20, window_dev=2)
        g["bb_pct_b"] = bb.bollinger_pband()
        g["bb_width"] = bb.bollinger_wband()
        return g

    if not group_col:
        return _one(df)

    # Iterate the groups explicitly rather than using `groupby(...).apply(...)`:
    # since pandas 2.2 the grouping column is withheld from the frame handed to
    # the callable, which silently drops `Ticker` from the result.
    parts = [_one(g) for _, g in df.groupby(group_col, sort=False)]
    return pd.concat(parts).sort_index()


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Month / weekday / quarter dummies-in-waiting for seasonality checks."""
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["year"] = d.dt.year
    out["month"] = d.dt.month
    out["weekday"] = d.dt.weekday
    out["quarter"] = d.dt.quarter
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    return out


def make_binary_target(
    df: pd.DataFrame, fwd_col: str = "fwd_ret_5d", threshold: float = 0.0
) -> pd.DataFrame:
    """Binary "does it go up by more than `threshold`?" label for module 3."""
    out = df.copy()
    out[f"target_{fwd_col}"] = (out[fwd_col] > threshold).astype("Int8")
    out.loc[out[fwd_col].isna(), f"target_{fwd_col}"] = pd.NA
    return out


def build_feature_set(
    df: pd.DataFrame,
    group_col: str | None = "Ticker",
    *,
    with_ta: bool = True,
) -> pd.DataFrame:
    """The whole module-2 pipeline in one call."""
    out = add_returns(df, group_col=group_col)
    out = add_moving_averages(out, group_col=group_col)
    out = add_volatility(out, group_col=group_col)
    if with_ta:
        out = add_ta_indicators(out, group_col=group_col)
    return add_calendar_features(out)
