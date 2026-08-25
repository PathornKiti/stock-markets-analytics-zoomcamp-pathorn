"""Step 2 - join the sources and engineer features into one modelling frame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from smaz import features as smaz_features

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

TARGET = "target_fwd_ret_5d"


def build_dataset(prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Combine price features with forward-filled macro context.

    Macro series publish on their own (monthly/weekly) calendars, so they are
    merged as-of and forward-filled onto trading days -- a plain join would drop
    most rows or, worse, leak a value before its release date.
    """
    feat = smaz_features.build_feature_set(prices, group_col="Ticker")
    feat = smaz_features.make_binary_target(feat, fwd_col="fwd_ret_5d")

    macro = macro.copy()
    date_col = "DATE" if "DATE" in macro.columns else macro.columns[0]
    macro = macro.rename(columns={date_col: "Date"})

    # yfinance hands back datetime64[s] while FRED hands back datetime64[us];
    # merge_asof refuses to join keys of differing resolution, so pin both to
    # nanoseconds before merging.
    macro["Date"] = pd.to_datetime(macro["Date"]).astype("datetime64[ns]")
    feat["Date"] = pd.to_datetime(feat["Date"]).astype("datetime64[ns]")

    macro = macro.sort_values("Date").ffill()

    out = pd.merge_asof(
        feat.sort_values("Date"),
        macro,
        on="Date",
        direction="backward",
    )
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def save(df: pd.DataFrame, name: str = "dataset") -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path)
    return path


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every numeric column that is not an identifier, a raw price or a target.

    Anything starting with `fwd_` or `target_` is excluded by construction so a
    future-looking column cannot accidentally become a feature.
    """
    drop_prefixes = ("fwd_", "target_")
    drop_exact = {"Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    return [
        c
        for c in df.columns
        if c not in drop_exact
        and not c.startswith(drop_prefixes)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
