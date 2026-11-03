"""dividends_v1 - free quick-win check: ex-dividend timing and dividend yield (Yahoo dividends already stored).

    uv run set-quant-alpha/experiments/dividends_v1/dividends_v1.py

Diagnostics only. 2013-19 and 2020-21 decide; 2022-26 is for information.
A. Pre-ex-date window: do stocks with an ex-dividend date in the next 5 trading days earn different 5-day alpha
   (all tradable names and top-decile picks)? Thai ex-dates are announced weeks ahead, so treating them as known
   5 days before is a mild assumption (flagged in the report).
B. Trailing 12-month dividend yield: IC vs 5-day alpha, correlation with the model score, and the Fama-MacBeth slope
   added to the score. Prices are dividend-adjusted, which inflates older yields; cross-sectional ranks are used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import ingest, signal_eval, transform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "events_v1"))
import events_v1 as ev  # noqa: E402  (reuse periods, Fama-MacBeth and markdown helpers)

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def features(df: pd.DataFrame, div: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(df["Date"].unique()))
    tickers = sorted(df["Ticker"].unique())
    d = div[div["Ticker"].isin(tickers)].copy()
    pos = dates.searchsorted(d["Date"].to_numpy())
    d = d[pos < len(dates)].assign(Date=dates[pos[pos < len(dates)]])
    amt = d.pivot_table(index="Date", columns="Ticker", values="dividend", aggfunc="sum").reindex(
        index=dates, columns=tickers
    )
    ar = np.arange(len(dates), dtype=float)[:, None]
    idx = pd.DataFrame(np.where(amt.notna(), ar, np.nan), index=dates, columns=tickers)
    to_next = idx.shift(-1).bfill().to_numpy() - ar  # trading days to the next ex-date
    close = prices.pivot_table(index="Date", columns="Ticker", values="Close").reindex(
        index=dates, columns=tickers
    )
    ttm = amt.fillna(0.0).rolling(245, min_periods=200).sum()  # about 12 months of trading days
    wide = {
        "pre_xd_5": pd.DataFrame((to_next <= 5).astype(float), index=dates, columns=tickers),
        "div_yield_ttm": ttm / close,
    }
    long = pd.concat({k: v.stack(future_stack=True) for k, v in wide.items()}, axis=1)
    long.index.names = ["Date", "Ticker"]
    return long.reset_index()


def window_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, (lo, hi) in ev.PERIODS.items():
        s = ev._period(d, lo, hi)
        for scope, g0 in (("all tradable", s), ("top-decile picks", s[s["top"]])):
            for flag, g in g0.groupby(g0["pre_xd_5"] > 0):
                rows[(name, scope, f"pre_xd_5={int(flag)}")] = {
                    "rows": len(g),
                    "share %": 100 * len(g) / len(g0),
                    "mean 5d alpha %": 100 * g["fwd_alpha_5"].mean(),
                }
    return pd.DataFrame(rows).T


def yield_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, (lo, hi) in ev.PERIODS.items():
        g = ev._period(d, lo, hi).dropna(subset=["div_yield_ttm"])
        summ = signal_eval.ic_summary(signal_eval.daily_ic(g, "div_yield_ttm"))
        rows[name] = {
            "IC": summ["IC mean"],
            "IC t (non-overlap)": summ["t (non-overlapping)"],
            "corr with score": signal_eval.daily_ic(g, "div_yield_ttm", "pred_ens").mean(),
        }
    return pd.DataFrame(rows).T


def main() -> None:
    df = transform.load()
    raw = ingest.ingest_all("file")
    feat = features(df, raw["dividends"], raw["prices"])
    p = pd.read_parquet(ev.PREDS, columns=["Date", "Ticker", "pred_ens"])
    d = p.merge(df[["Date", "Ticker", "tradable", "fwd_alpha_5"]], on=["Date", "Ticker"]).merge(
        feat, on=["Date", "Ticker"], how="left"
    )
    d = d[d["tradable"]].dropna(subset=["fwd_alpha_5", "pred_ens"])
    d["top"] = d.groupby("Date")["pred_ens"].rank(pct=True) >= 0.9
    tables = {
        "A_pre_ex_dividend_window": window_table(d),
        "B_dividend_yield_ic": yield_table(d),
        "B_fama_macbeth_score_plus_yield": ev.fama_macbeth(d, ["pred_ens", "div_yield_ttm"]),
    }
    md = []
    for k, t in tables.items():
        t.to_csv(OUT / f"{k}.csv")
        md.append(f"## {k}\n\n{ev.to_md(t)}\n")
    (OUT / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
