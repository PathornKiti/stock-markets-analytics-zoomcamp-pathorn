"""Tests for feature engineering -- mainly guarding against look-ahead leakage."""

import numpy as np
import pandas as pd
import pytest

from smaz import features


@pytest.fixture
def prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=300)
    frames = []
    for i, ticker in enumerate(["AAA", "BBB"]):
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "Close": np.linspace(100 + i * 10, 200 + i * 10, len(dates)),
                    "High": np.linspace(101 + i * 10, 201 + i * 10, len(dates)),
                    "Low": np.linspace(99 + i * 10, 199 + i * 10, len(dates)),
                    "Volume": 1_000_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_forward_return_looks_forward(prices):
    out = features.add_returns(prices, horizons=(1,))
    aaa = out[out.Ticker == "AAA"].reset_index(drop=True)
    expected = aaa.Close.iloc[1] / aaa.Close.iloc[0] - 1
    assert aaa["fwd_ret_1d"].iloc[0] == pytest.approx(expected)


def test_trailing_return_looks_backward(prices):
    out = features.add_returns(prices, horizons=(1,))
    aaa = out[out.Ticker == "AAA"].reset_index(drop=True)
    expected = aaa.Close.iloc[1] / aaa.Close.iloc[0] - 1
    assert aaa["ret_1d"].iloc[1] == pytest.approx(expected)
    assert pd.isna(aaa["ret_1d"].iloc[0])


def test_returns_do_not_bleed_across_tickers(prices):
    out = features.add_returns(prices, horizons=(1,))
    # The first row of each ticker has no prior bar, so its trailing return is NaN.
    firsts = out.groupby("Ticker").head(1)
    assert firsts["ret_1d"].isna().all()


def test_forward_return_is_nan_at_the_end(prices):
    out = features.add_returns(prices, horizons=(5,))
    lasts = out.groupby("Ticker").tail(5)
    assert lasts["fwd_ret_5d"].isna().all()


def test_moving_average_ratio(prices):
    out = features.add_moving_averages(prices, windows=(10,))
    row = out.dropna(subset=["sma_10"]).iloc[0]
    assert row["price_to_sma_10"] == pytest.approx(row["Close"] / row["sma_10"])


def test_binary_target_is_nullable_where_forward_return_is_unknown(prices):
    out = features.add_returns(prices, horizons=(5,))
    out = features.make_binary_target(out, "fwd_ret_5d")
    col = out["target_fwd_ret_5d"]
    assert col[out["fwd_ret_5d"].isna()].isna().all()
    # This series is monotonically increasing, so every known label is 1.
    assert (col.dropna() == 1).all()


def test_build_feature_set_adds_expected_columns(prices):
    out = features.build_feature_set(prices, with_ta=True)
    for col in ["ret_1d", "fwd_ret_5d", "sma_50", "vol_21d", "rsi_14", "macd_diff", "month"]:
        assert col in out.columns, col


def test_build_feature_set_preserves_the_grouping_column(prices):
    """pandas >=2.2 withholds the grouping column inside `groupby.apply`, which
    silently dropped `Ticker` from the TA step. Guard against a regression."""
    out = features.build_feature_set(prices, with_ta=True)
    assert "Ticker" in out.columns
    assert set(out["Ticker"].unique()) == {"AAA", "BBB"}
    assert len(out) == len(prices)


def test_ta_indicators_computed_per_ticker(prices):
    out = features.add_ta_indicators(prices)
    counts = out.groupby("Ticker")["rsi_14"].apply(lambda s: s.notna().sum())
    # Each ticker warms up independently; neither is starved by the other.
    assert (counts > 0).all()
    assert counts["AAA"] == counts["BBB"]
