"""Step 2 - one combined dataset: factor model, features, dummies and targets.

Every row is one (Date, Ticker). Features use information available at the Thai
close of `Date`; anything printed later than that (US markets, FRED, Cboe) is lagged
one day. Targets look forward from the NEXT open, which is when an order placed after
the close can actually be filled (this also removes the bid-ask bounce in the close).

Quant layer built here
----------------------
Single-index factor model, re-estimated every day on a rolling 120-day window:

    r_i,t = a_i + b_i * m_t + e_i,t        m_t = equal-weight return of liquid Thai stocks

The residual e (the stock's own, market-free return) drives three features (residual
momentum, residual reversal, idiosyncratic volatility) and the main target: the 5-day
forward *alpha*  r_i,t+1..t+5 - b_i * m_t+1..t+5.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from setalpha import config, regime

log = logging.getLogger(__name__)
DATASET = config.PROCESSED_DIR / "dataset.parquet"

TO_PREDICT = ["fwd_ret_5", "fwd_alpha_5", "is_alpha_5", "is_positive_5"]
TARGET = "is_alpha_5"  # main classification target
ID_COLS = ["Date", "Ticker"]
FLAG_COLS = ["liquid", "tradable", "sector", "dvol60", "beta_mkt", "idio_vol"]
CATEGORICAL = [
    "month",
    "weekday",
    "sector",
    "size_q",
    "mom_q",
    "rsi_bin",
    "vix_regime",
    "thb_regime",
    "hmm_state",
]

# Features documented in the README "Data sources" table.
LECTURE_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ret_120",
    "vol_20",
    "vol_60",
    "sma10_ratio",
    "sma20_ratio",
    "sma50_ratio",
    "sma200_ratio",
    "rsi_14",
    "macd_hist",
    "bb_pctb",
    "atr_14",
    "us10y_chg_5",
    "tbill3m",
]
NEW_FEATURES = [
    # factor model (quant layer)
    "beta_mkt",
    "beta_thb",
    "beta_eem",
    "idio_vol",
    "resid_rev_5",
    "resid_mom_20",
    "resid_mom_120",
    # liquidity / microstructure
    "log_dvol60",
    "vol_z20",
    "amihud_20",
    "zero_ret_20",
    "gap_1",
    "range_1",
    "max_ret_20",
    # position in range / relative strength
    "dist_52w_high",
    "dist_52w_low",
    "rel_ret_5",
    "rel_ret_20",
    # cross-sectional ranks
    "cs_ret_5",
    "cs_ret_20",
    "cs_resid_mom_20",
    "cs_vol_20",
    "cs_dvol",
    # Thai market state
    "mkt_ret_1",
    "mkt_ret_5",
    "breadth",
    "dispersion",
    # Gaussian HMM market regime (filtered, walk-forward)
    "hmm_p_calm",
    "hmm_p_stressed",
    # foreign-flow / global risk proxies (lagged 1 day)
    "usdthb_ret_1",
    "usdthb_ret_5",
    "usdthb_vol_20",
    "eem_ret_1",
    "eem_ret_5",
    "eem_vol_z",
    "spx_ret_5",
    "dxy_ret_5",
    "oil_ret_5",
    "gold_ret_5",
    "credit_spread",
    "credit_spread_chg_5",
    # Cboe volatility term structure (lagged 1 day)
    "vix",
    "vix_slope",
    "vix9d_ratio",
    "vvix",
    "skew",
    "vix_chg_5",
]


# ----------------------------------------------------------------------------- helpers
def _wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return prices.pivot_table(index="Date", columns="Ticker", values=field).astype("float64")


def _lag_to(index: pd.DatetimeIndex, frame: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Shift `frame` `lag` rows on its own calendar, then carry it onto the Thai calendar."""
    shifted = frame.sort_index().shift(lag)
    return shifted.reindex(shifted.index.union(index)).ffill().reindex(index)


def _rolling_beta(
    r: pd.DataFrame, f: pd.Series, window: int = 120, min_periods: int = 60
) -> pd.DataFrame:
    cov = r.apply(lambda col: col.rolling(window, min_periods=min_periods).cov(f))
    return cov.div(f.rolling(window, min_periods=min_periods).var(), axis=0)


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder RSI, the same definition as `ta.momentum.RSIIndicator` used in the lectures."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def _cs_rank(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank across liquid stocks on each date (0 = lowest, 1 = highest)."""
    return frame.where(mask).rank(axis=1, pct=True)


# ----------------------------------------------------------------------------- build
def build_dataset(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prices = raw["prices"]
    counts = prices.groupby("Ticker").size()
    prices = prices[prices["Ticker"].isin(counts[counts >= 250].index)]

    O, H, L, C, V = (_wide(prices, f) for f in ["Open", "High", "Low", "Close", "Volume"])  # noqa: E741 - OHLCV naming
    dates = C.index
    log.info("    panel: %d dates x %d tickers", *C.shape)

    # --- cleaning: Thai daily limit is +-30%; larger moves are bad prints / unadjusted actions
    ret1 = C.pct_change(fill_method=None)
    ret1 = ret1.where(ret1.abs() < 0.45)
    dvol = (C * V).replace(0, np.nan)

    # --- universe flags (point-in-time)
    dvol60 = dvol.rolling(60, min_periods=40).median()
    liquid = (dvol60 >= config.MIN_DVOL) & (C >= config.MIN_PRICE)
    liq_rank = dvol60.where(liquid).rank(axis=1, ascending=False)
    tradable = liq_rank <= config.TOP_N

    # --- Thai market factor: equal-weight return of liquid stocks
    mkt = ret1.where(liquid).mean(axis=1)

    # --- global series (lagged one day: they close after Bangkok)
    mk = raw["market"].pivot_table(index="Date", columns="Ticker", values=["Close", "Volume"])
    close_g = mk["Close"].rename(columns=config.MACRO_YF)
    vol_g = mk["Volume"].rename(columns=config.MACRO_YF)
    glob = pd.DataFrame(index=close_g.index)
    for name in ["usdthb", "eem", "spx", "dxy", "oil", "gold"]:
        glob[f"{name}_ret_1"] = close_g[name].pct_change(fill_method=None)
        glob[f"{name}_ret_5"] = close_g[name].pct_change(5, fill_method=None)
    glob["usdthb_vol_20"] = glob["usdthb_ret_1"].rolling(20).std() * np.sqrt(252)
    eem_lv = np.log(vol_g["eem"].replace(0, np.nan))
    glob["eem_vol_z"] = (eem_lv - eem_lv.rolling(20).mean()) / eem_lv.rolling(20).std()
    glob = _lag_to(dates, glob)

    fred = raw["fred"].set_index("Date").sort_index()
    fr = pd.DataFrame(
        {
            "tbill3m": fred["tbill3m"].ffill(),
            "us10y_chg_5": fred["us10y"].ffill().diff(5),
            "credit_spread": fred["credit_spread"].ffill(),
            "credit_spread_chg_5": fred["credit_spread"].ffill().diff(5),
        }
    )
    fr = _lag_to(dates, fr)

    cb = raw["cboe"].set_index("Date").sort_index()
    vx = pd.DataFrame(
        {
            "vix": cb["vix"],
            "vix_slope": cb["vix"] / cb["vix3m"],
            "vix9d_ratio": cb["vix9d"] / cb["vix"],
            "vvix": cb["vvix"],
            "skew": cb["skew"],
            "vix_chg_5": cb["vix"].diff(5),
        }
    )
    vx = _lag_to(dates, vx)

    # --- factor model (quant layer)
    beta_mkt = _rolling_beta(ret1, mkt)
    resid = ret1 - beta_mkt.shift(1).mul(mkt, axis=0)  # beta estimated before today
    beta_thb = _rolling_beta(ret1, glob["usdthb_ret_1"])
    beta_eem = _rolling_beta(ret1, glob["eem_ret_1"])

    # --- assemble the long table feature by feature (float32 keeps 2M+ rows in memory)
    base = C.stack(future_stack=True).dropna().index  # one row per (Date, Ticker) that traded
    out: dict[str, np.ndarray] = {}

    def put(name: str, frame: pd.DataFrame) -> None:
        out[name] = frame.stack(future_stack=True).reindex(base).astype("float32").to_numpy()

    def put_date(name: str, series: pd.Series) -> None:
        out[name] = series.reindex(base.get_level_values("Date")).astype("float32").to_numpy()

    # lecture-style price features
    for h in [1, 5, 10, 20, 60, 120]:
        put(f"ret_{h}", ret1 if h == 1 else C.pct_change(h, fill_method=None))
    put("vol_20", ret1.rolling(20, min_periods=15).std())
    put("vol_60", ret1.rolling(60, min_periods=40).std())
    for w in [10, 20, 50, 200]:
        put(f"sma{w}_ratio", C / C.rolling(w, min_periods=int(w * 0.8)).mean())
    put("rsi_14", _rsi(C))
    macd = C.ewm(span=12, adjust=False).mean() - C.ewm(span=26, adjust=False).mean()
    put("macd_hist", (macd - macd.ewm(span=9, adjust=False).mean()) / C)
    sma20, sd20 = C.rolling(20).mean(), C.rolling(20).std()
    put("bb_pctb", (C - (sma20 - 2 * sd20)) / (4 * sd20))
    tr = (
        pd.concat({"a": H - L, "b": (H - C.shift()).abs(), "c": (L - C.shift()).abs()})
        .groupby(level=1)
        .max()
    )
    put("atr_14", tr.reindex(dates).ewm(alpha=1 / 14, adjust=False).mean() / C)

    # factor-model features
    put("beta_mkt", beta_mkt)
    put("beta_thb", beta_thb)
    put("beta_eem", beta_eem)
    idio = resid.rolling(60, min_periods=40).std()
    put("idio_vol", idio)
    put("resid_rev_5", resid.rolling(5).sum())
    put("resid_mom_20", resid.rolling(20).sum())
    put("resid_mom_120", resid.rolling(120, min_periods=80).sum().shift(5))  # skip the last week

    # liquidity / microstructure
    put("dvol60", dvol60)
    put("log_dvol60", np.log(dvol60))
    lv = np.log(V.replace(0, np.nan))
    put("vol_z20", (lv - lv.rolling(20).mean()) / lv.rolling(20).std())
    put("amihud_20", np.log1p((ret1.abs() / dvol * 1e9).rolling(20, min_periods=10).mean()))
    put("zero_ret_20", (ret1 == 0).astype(float).where(ret1.notna()).rolling(20).mean())
    put("gap_1", O / C.shift() - 1)
    put("range_1", (H - L) / C)
    put("max_ret_20", ret1.rolling(20).max())

    # range position / relative strength
    put("dist_52w_high", C / C.rolling(252, min_periods=200).max() - 1)
    put("dist_52w_low", C / C.rolling(252, min_periods=200).min() - 1)
    mkt5, mkt20 = (
        (1 + mkt).rolling(5).apply(np.prod, raw=True) - 1,
        (1 + mkt).rolling(20).apply(np.prod, raw=True) - 1,
    )
    put("rel_ret_5", C.pct_change(5, fill_method=None).sub(mkt5, axis=0))
    put("rel_ret_20", C.pct_change(20, fill_method=None).sub(mkt20, axis=0))

    # cross-sectional ranks
    put("cs_ret_5", _cs_rank(C.pct_change(5, fill_method=None), liquid))
    put("cs_ret_20", _cs_rank(C.pct_change(20, fill_method=None), liquid))
    put("cs_resid_mom_20", _cs_rank(resid.rolling(20).sum(), liquid))
    put("cs_vol_20", _cs_rank(ret1.rolling(20, min_periods=15).std(), liquid))
    put("cs_dvol", _cs_rank(dvol60, liquid))

    # Thai market state
    put_date("mkt_ret_1", mkt)
    put_date("mkt_ret_5", mkt5)
    put_date("breadth", (ret1.where(liquid) > 0).sum(axis=1) / liquid.sum(axis=1))
    put_date("dispersion", ret1.where(liquid).std(axis=1))
    regimes = regime.walk_forward_regimes(mkt)
    put_date("hmm_p_calm", regimes["hmm_p_calm"])
    put_date("hmm_p_stressed", regimes["hmm_p_stressed"])
    put_date("hmm_state", regimes["hmm_state"])

    # global / flow proxies, rates, Cboe
    for frame in (glob, fr, vx):
        for col in frame.columns:
            if col in LECTURE_FEATURES + NEW_FEATURES:
                put_date(col, frame[col])

    # --- targets: buy next open, sell close in 5 days
    h = config.HORIZON
    fwd_ret = C.shift(-h) / O.shift(-1) - 1
    fwd_ret = fwd_ret.where(fwd_ret.abs() < 1.0)
    fwd_mkt = fwd_ret.where(liquid).mean(axis=1)
    put("fwd_ret_5", fwd_ret)
    put("fwd_alpha_5", fwd_ret - beta_mkt.mul(fwd_mkt, axis=0))
    put("liquid", liquid.astype(float))
    put("tradable", tradable.astype(float))

    df = pd.DataFrame(out, index=base).reset_index()
    df["liquid"] = df["liquid"].eq(1)
    df["tradable"] = df["tradable"].eq(1)

    # optional hand-exported SET investor-type flows (same-day, published after the close)
    if raw.get("flows") is not None:
        fl = raw["flows"].set_index("Date").sort_index()
        roll = fl.rolling(5).sum().add_suffix("_5")
        df = df.merge(pd.concat([fl, roll], axis=1).reset_index(), on="Date", how="left")

    # --- sector and categorical transformations (binned features -> dummies later)
    sectors = (
        raw["sectors"].set_index("Ticker")["sector"]
        if len(raw["sectors"])
        else pd.Series(dtype=str)
    )
    df["sector"] = df["Ticker"].map(sectors).replace("", np.nan).fillna("Unknown")
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.weekday
    df["size_q"] = pd.cut(
        df["cs_dvol"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=False, include_lowest=True
    )
    df["mom_q"] = pd.cut(
        df["cs_ret_20"], [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=False, include_lowest=True
    )
    df["rsi_bin"] = pd.cut(
        df["rsi_14"],
        [0, 30, 45, 55, 70, 100],
        labels=["oversold", "weak", "neutral", "strong", "overbought"],
    )
    df["vix_regime"] = pd.cut(
        df["vix"], [0, 15, 20, 30, 200], labels=["calm", "normal", "stressed", "panic"]
    )
    df["hmm_state"] = df["hmm_state"].map(dict(enumerate(regime.STATE_NAMES))).fillna("unknown")
    df["thb_regime"] = np.select(
        [df["usdthb_ret_5"] > 0.01, df["usdthb_ret_5"] < -0.01],
        ["baht_weak", "baht_strong"],
        "neutral",
    )

    df["is_alpha_5"] = (df["fwd_alpha_5"] > 0).astype("float32").where(df["fwd_alpha_5"].notna())
    df["is_positive_5"] = (df["fwd_ret_5"] > 0).astype("float32").where(df["fwd_ret_5"].notna())
    df = df[df["Date"] >= config.START].reset_index(drop=True)
    log.info(
        "    dataset: %s rows x %d columns (%s liquid, %s tradable)",
        f"{len(df):,}",
        df.shape[1],
        f"{df['liquid'].sum():,}",
        f"{df['tradable'].sum():,}",
    )
    return df


def add_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the categorical / binned columns (kept as int8 to save memory)."""
    dummies = pd.get_dummies(df[CATEGORICAL].astype(str), prefix=CATEGORICAL, dtype="int8")
    return pd.concat([df, dummies], axis=1)


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """The feature sets the rubric asks for: TO_PREDICT, NUMERIC, DUMMIES."""
    numeric = [c for c in LECTURE_FEATURES + NEW_FEATURES if c in df.columns]
    if (
        config.USE_FLOW_FEATURES
    ):  # optional SET flows; off by default so the frozen paper shadows stay fixed
        numeric += [c for c in df.columns if c.endswith("_net") or c.endswith("_net_5")]
    dummies = [
        c
        for c in df.columns
        if any(c.startswith(f"{p}_") for p in CATEGORICAL) and df[c].dtype == "int8"
    ]
    return {"TO_PREDICT": TO_PREDICT, "NUMERIC": numeric, "DUMMIES": dummies}


def save(df: pd.DataFrame) -> str:
    df.to_parquet(DATASET, index=False)
    return str(DATASET)


def load() -> pd.DataFrame:
    return pd.read_parquet(DATASET)
