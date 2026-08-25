"""Step 4 - turn model probabilities into positions and simulate the P&L."""

from __future__ import annotations

import pandas as pd

from smaz import backtest


def threshold_signal(proba: pd.Series, threshold: float = 0.55) -> pd.Series:
    """Long when the model is confident enough, flat otherwise."""
    return (proba >= threshold).astype(float)


def long_short_signal(proba: pd.Series, long_at: float = 0.55, short_at: float = 0.45) -> pd.Series:
    """Long above `long_at`, short below `short_at`, flat in the uncertain middle."""
    sig = pd.Series(0.0, index=proba.index)
    sig[proba >= long_at] = 1.0
    sig[proba <= short_at] = -1.0
    return sig


def top_n_signal(
    df: pd.DataFrame, proba_col: str = "proba", n: int = 3, date_col: str = "Date"
) -> pd.Series:
    """Each day, hold an equal-weight basket of the `n` highest-probability names."""
    rank = df.groupby(date_col)[proba_col].rank(ascending=False, method="first")
    return (rank <= n).astype(float) / n


def portfolio_returns(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str = "ret_1d",
    date_col: str = "Date",
    *,
    fee_bps: float = 5.0,
) -> pd.Series:
    """Aggregate per-ticker positions into one daily portfolio return series.

    Positions are **summed**, not averaged. With a 0/1 signal per ticker that
    means holding N tickers is N times levered -- fine for `top_n_signal`, which
    already divides by N, but it will inflate both the return and the volatility
    of `threshold_signal`. Normalise the signal (e.g. divide by the number of
    concurrent positions) before claiming a Sharpe ratio in the write-up.
    """
    # Explicit group iteration -- see the note in `smaz.features.add_ta_indicators`
    # about pandas withholding the grouping column from `.apply()`.
    per_ticker = [
        backtest.simulate_signal(
            g.set_index(date_col)[ret_col],
            g.set_index(date_col)[signal_col],
            fee_bps=fee_bps,
        )
        for _, g in df.groupby("Ticker", sort=False)
    ]
    if not per_ticker:
        return pd.Series(dtype=float)
    return pd.concat(per_ticker).groupby(level=0).sum().sort_index()


def benchmark_returns(
    df: pd.DataFrame, ticker: str = "SPY", ret_col: str = "ret_1d", date_col: str = "Date"
) -> pd.Series:
    """Buy-and-hold the benchmark, for the rubric's "beat a realistic benchmark" test."""
    b = df[df["Ticker"] == ticker].set_index(date_col)[ret_col]
    return b.sort_index()


def report(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    """Rubric-shaped comparison table: CAGR, Sharpe, max drawdown, and friends."""
    return backtest.compare(strategies)
