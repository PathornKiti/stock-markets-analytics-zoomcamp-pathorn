"""Vectorised trading simulation and the performance metrics the project rubric asks for.

The project is graded partly on reporting CAGR, Sharpe ratio and max drawdown,
and on comparing a strategy against a realistic benchmark -- `summary()` and
`compare()` below produce exactly that table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# A constant return series has a floating-point stdev around 1e-18 rather than
# exactly 0, so ratio metrics must guard on a tolerance. Testing `if not sd`
# lets that residue through and returns a Sharpe of ~1e16.
_EPS = 1e-12


def equity_curve(returns: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    """Compound a series of periodic simple returns into an equity curve."""
    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by the return series."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    total = float((1.0 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def sharpe(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised Sharpe ratio. `risk_free` is an annual rate."""
    r = returns.dropna() - risk_free / periods_per_year
    sd = r.std()
    if len(r) < 2 or not np.isfinite(sd) or sd < _EPS:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    """Like Sharpe but penalising downside deviation only."""
    r = returns.dropna() - risk_free / periods_per_year
    downside = r[r < 0].std()
    if not np.isfinite(downside) or downside < _EPS:
        return float("nan")
    return float(r.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the equity curve, as a negative fraction."""
    eq = equity_curve(returns)
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by the absolute max drawdown."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or abs(mdd) < _EPS:
        return float("nan")
    return cagr(returns, periods_per_year) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    """Share of periods with a positive return."""
    r = returns.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def summary(
    returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> pd.Series:
    """One-column performance report for a single strategy."""
    r = returns.dropna()
    return pd.Series(
        {
            "total_return": float((1.0 + r).prod() - 1.0) if len(r) else np.nan,
            "cagr": cagr(r, periods_per_year),
            "volatility": float(r.std() * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan,
            "sharpe": sharpe(r, risk_free, periods_per_year),
            "sortino": sortino(r, risk_free, periods_per_year),
            "max_drawdown": max_drawdown(r),
            "calmar": calmar(r, periods_per_year),
            "hit_rate": hit_rate(r),
            "n_periods": len(r),
        }
    )


def compare(strategies: dict[str, pd.Series], risk_free: float = 0.0) -> pd.DataFrame:
    """Side-by-side `summary()` for several strategies, e.g. strategy vs buy-and-hold."""
    return pd.DataFrame({name: summary(r, risk_free) for name, r in strategies.items()})


def simulate_signal(
    returns: pd.Series,
    signal: pd.Series,
    *,
    fee_bps: float = 0.0,
    lag: int = 1,
) -> pd.Series:
    """Apply a position `signal` to a `returns` series and net out trading fees.

    `signal` is a position size (1 = long, 0 = flat, -1 = short). It is shifted
    by `lag` periods first: a signal computed from today's close can only be
    traded from tomorrow, and skipping this shift is the single most common way
    a backtest silently reports look-ahead profits.

    `fee_bps` is charged in basis points on the *change* in position, so holding
    a constant position costs nothing.
    """
    pos = signal.shift(lag).fillna(0.0)
    gross = pos * returns.fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return gross - turnover * (fee_bps / 10_000.0)
