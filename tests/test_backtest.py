"""Tests for the metric helpers -- these numbers end up in the project report,
so they are worth pinning down against hand-computable cases."""

import numpy as np
import pandas as pd
import pytest

from smaz import backtest


def const_returns(r: float, n: int) -> pd.Series:
    return pd.Series([r] * n, index=pd.bdate_range("2020-01-01", periods=n))


def test_equity_curve_compounds():
    eq = backtest.equity_curve(const_returns(0.01, 3))
    assert eq.iloc[-1] == pytest.approx(1.01**3)


def test_cagr_of_flat_series_is_zero():
    assert backtest.cagr(const_returns(0.0, 252)) == pytest.approx(0.0)


def test_cagr_recovers_known_annual_growth():
    # 252 daily periods compounding to exactly 10% over one year
    daily = 1.10 ** (1 / 252) - 1
    assert backtest.cagr(const_returns(daily, 252)) == pytest.approx(0.10, abs=1e-6)


def test_max_drawdown_is_negative_and_exact():
    r = pd.Series([0.5, -0.5, 0.0])  # 1.0 -> 1.5 -> 0.75: a 50% drawdown from the peak
    assert backtest.max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_zero_when_monotonically_rising():
    assert backtest.max_drawdown(const_returns(0.01, 10)) == pytest.approx(0.0)


def test_sharpe_positive_for_upward_drift():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert backtest.sharpe(r) > 0


def test_sharpe_nan_on_zero_variance():
    assert np.isnan(backtest.sharpe(const_returns(0.01, 10)))


def test_simulate_signal_lags_by_one_period():
    """A signal known at t can only be traded at t+1. Without the lag the
    strategy would 'earn' the very return that produced the signal."""
    returns = pd.Series([0.10, 0.20, 0.30])
    signal = pd.Series([1.0, 0.0, 0.0])
    out = backtest.simulate_signal(returns, signal, fee_bps=0)
    assert out.tolist() == pytest.approx([0.0, 0.20, 0.0])


def test_simulate_signal_charges_fees_on_turnover_only():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    signal = pd.Series([1.0, 1.0, 1.0, 1.0])  # enter once, then hold
    out = backtest.simulate_signal(returns, signal, fee_bps=100)  # 1%
    assert out.iloc[1] == pytest.approx(-0.01)  # entry costs
    assert out.iloc[2:].tolist() == pytest.approx([0.0, 0.0])  # holding is free


def test_compare_returns_one_column_per_strategy():
    out = backtest.compare({"a": const_returns(0.01, 50), "b": const_returns(-0.01, 50)})
    assert list(out.columns) == ["a", "b"]
    assert "max_drawdown" in out.index


def test_sortino_nan_when_there_is_no_downside():
    assert np.isnan(backtest.sortino(const_returns(0.01, 50)))


def test_calmar_nan_without_a_drawdown():
    assert np.isnan(backtest.calmar(const_returns(0.01, 50)))


def test_summary_on_flat_series_has_no_infinities():
    """Ratio metrics on a zero-variance series must be NaN, never ~1e16."""
    out = backtest.summary(const_returns(0.0, 100))
    assert not np.isinf(out.astype(float)).any()
    assert np.isnan(out["sharpe"])


def test_empty_series_does_not_raise():
    out = backtest.summary(pd.Series(dtype=float))
    assert out["n_periods"] == 0
