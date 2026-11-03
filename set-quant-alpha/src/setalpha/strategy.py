"""Step 4 - trading simulations.

Vector simulations (course style): every prediction row can be a THB 100 trade held
5 days (buy next open, sell close t+5). `sim{k}_invest` / `sim{k}_net` columns are
generated for each strategy and summarised with an approximate CAGR on the capital
that overlapping trades tie up.

Exact simulation: a day-by-day loop over the calendar with real portfolio accounting.
Weekly rebalance at the next open, 100-share board lots, fees on every trade, per-name
stop-loss on the daily low, profits reinvested, optional beta hedge with the SET50 ETF and
optional HMM regime de-risking. Weights come from `portfolio.py` (equal or mean-variance).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from setalpha import config, portfolio

log = logging.getLogger(__name__)
LOT = 100  # SET board lot
HEDGE_FEE = 0.0005  # per side, SET50 futures / ETF hedge


# ----------------------------------------------------------------------------- inputs
def wide_prices(prices: pd.DataFrame, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    p = prices if tickers is None else prices[prices["Ticker"].isin(tickers)]
    return {
        f: p.pivot_table(index="Date", columns="Ticker", values=f)
        for f in ["Open", "High", "Low", "Close"]
    }


def benchmark_series(market: pd.DataFrame, ticker: str = config.BENCHMARK) -> pd.DataFrame:
    """Daily return and 5-day next-open-to-close return of the benchmark ETF."""
    b = market[market["Ticker"] == ticker].set_index("Date").sort_index()
    out = pd.DataFrame(index=b.index)
    out["bench_ret_1"] = b["Close"].pct_change(fill_method=None)
    out["bench_fwd_5"] = b["Close"].shift(-config.HORIZON) / b["Open"].shift(-1) - 1
    return out


def sim_frame(
    df: pd.DataFrame, preds: pd.DataFrame, market: pd.DataFrame, split: str = "test"
) -> pd.DataFrame:
    cols = [
        "Date",
        "Ticker",
        "tradable",
        "fwd_ret_5",
        "fwd_alpha_5",
        "is_alpha_5",
        "idio_vol",
        "beta_mkt",
        "vix_slope",
        "usdthb_ret_5",
        "hmm_p_stressed",
        "sector",
        "size_q",
        "month",
        "vix_regime",
        "hmm_state",
    ]
    f = preds[preds["split"] == split].merge(df[cols], on=["Date", "Ticker"], how="left")
    f = f[
        f["tradable"].fillna(False).astype(bool)
    ]  # no filter on a known fwd_ret_5: that would drop soon-suspended names (look-ahead)
    return f.merge(
        benchmark_series(market)[["bench_fwd_5"]], left_on="Date", right_index=True, how="left"
    )


# ----------------------------------------------------------------------------- vector sims
def vector_sims(
    f: pd.DataFrame, thr: dict[str, float], fee: float = config.FEE_PER_SIDE, stake: float = 100.0
) -> pd.DataFrame:
    """Add sim{k}_invest / sim{k}_net columns for six strategies."""
    f = f.copy()
    rank = f.groupby("Date")["pred_ens"].rank(ascending=False, method="first")
    top5 = rank <= 5
    calm = (f["hmm_p_stressed"] < 0.5) & (f["vix_slope"] < 1.0)
    m1, m2 = list(thr)[:2]  # the two ensemble members
    rules = {
        "sim1_threshold": f[f"pred_{m1}"]
        >= thr[m1],  # custom rule: best model's prob in its top 5%
        "sim2_top5": top5,  # 5 best ensemble scores per day
        "sim3_top5_regime": top5 & calm,  # ...only in a calm HMM / VIX regime
        "sim4_agree": (f[f"pred_{m1}"] >= thr[m1])
        & (f[f"pred_{m2}"] >= thr[m2]),  # both models confident
    }
    for name, mask in rules.items():
        f[f"{name}_invest"] = np.where(mask, stake, 0.0)
        f[f"{name}_net"] = f[f"{name}_invest"] * (f["fwd_ret_5"] - 2 * fee)
    # long top-5, short the SET50 ETF for the same amount: earns stock-specific alpha only
    f["sim5_longshort_invest"] = np.where(top5, stake, 0.0)
    f["sim5_longshort_net"] = f["sim5_longshort_invest"] * (
        f["fwd_ret_5"] - f["bench_fwd_5"] - 2 * fee - 2 * HEDGE_FEE
    )
    # bigger stake for higher conviction: 1.5x for the top-2, 1x for ranks 3-5
    f["sim6_conviction_invest"] = np.where(rank <= 2, 1.5 * stake, np.where(top5, stake, 0.0))
    f["sim6_conviction_net"] = f["sim6_conviction_invest"] * (f["fwd_ret_5"] - 2 * fee)
    return f


def summarize_vector(f: pd.DataFrame) -> pd.DataFrame:
    years = (f["Date"].max() - f["Date"].min()).days / 365.25
    rows = {}
    for col in [c for c in f.columns if c.endswith("_invest")]:
        name = col.removesuffix("_invest")
        trades = f[f[col] > 0]
        daily_inv = f.groupby("Date")[col].sum()
        capital = float(
            daily_inv.rolling(config.HORIZON, min_periods=1).sum().quantile(0.95)
        )  # capital tied up
        net = float(trades[f"{name}_net"].sum())
        rows[name] = {
            "trades": len(trades),
            "trades/day": len(trades) / f["Date"].nunique(),
            "hit rate %": 100 * float((trades[f"{name}_net"] > 0).mean()),
            "avg net / trade %": 100 * float((trades[f"{name}_net"] / trades[col]).mean()),
            "profit factor": float(
                trades.loc[trades[f"{name}_net"] > 0, f"{name}_net"].sum()
                / -trades.loc[trades[f"{name}_net"] < 0, f"{name}_net"].sum()
            ),
            "payoff ratio": float(
                trades.loc[trades[f"{name}_net"] > 0, f"{name}_net"].mean()
                / -trades.loc[trades[f"{name}_net"] < 0, f"{name}_net"].mean()
            ),
            "net PnL (THB)": net,
            "capital (THB)": capital,
            "approx CAGR %": 100 * (((capital + net) / capital) ** (1 / years) - 1)
            if capital > 0
            else np.nan,
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- exact sim
@dataclass
class ExactConfig:
    name: str
    method: str = "equal"  # "equal" | "mvo"
    score: str = "pred_ens"
    n_candidates: int = 20
    max_positions: int = config.MAX_POSITIONS
    stop_loss: float | None = None
    regime: bool = False  # halve exposure when the HMM says "stressed" or VIX is inverted
    hedge: bool = False  # short the SET50 ETF against the book's market beta
    vol_target: bool = True  # mvo only: scale the book to the ex-ante volatility target
    risk_aversion: float = config.RISK_AVERSION  # mvo only
    max_weight: float = config.MAX_WEIGHT  # mvo only
    min_trade_frac: float = 0.01  # skip rebalancing trades smaller than 1% of equity
    buffer: int | None = None  # equal only: keep a holding while it ranks within the top `buffer`


@dataclass
class ExactResult:
    name: str
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    weights: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def target_weights(
    cfg: ExactConfig,
    day: pd.DataFrame,
    rets: pd.DataFrame,
    ic: float,
    held: list[str] | None = None,
) -> pd.Series:
    """Weights for one rebalance day: top candidates by score -> equal or mean-variance weights.

    With a `buffer`, names already held stay in the book while they rank inside the top
    `buffer`; only the freed slots are filled with new names. This hysteresis rule cuts
    turnover (and costs) without changing which stocks the model likes.
    """
    cand = day.nlargest(cfg.n_candidates, cfg.score).set_index("Ticker")
    if cfg.method == "equal":
        names = list(cand.head(cfg.max_positions).index)
        if cfg.buffer and held:
            ranked = list(day.sort_values(cfg.score, ascending=False)["Ticker"])
            keep = [t for t in held if t in ranked[: cfg.buffer]][: cfg.max_positions]
            names = keep + [t for t in ranked if t not in keep][: cfg.max_positions - len(keep)]
        return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)
    alpha = portfolio.alpha_from_scores(
        cand[cfg.score], cand["idio_vol"].fillna(cand["idio_vol"].median()), ic
    )
    window = rets[[t for t in cand.index if t in rets.columns]].tail(120)
    cov = portfolio.ledoit_wolf_cov(window)
    w = portfolio.mean_variance(
        alpha, cov, risk_aversion=cfg.risk_aversion, max_weight=cfg.max_weight
    )
    w = w.nlargest(cfg.max_positions)
    return portfolio.vol_target(w, cov) if cfg.vol_target else w


def exact_sim(
    cfg: ExactConfig,
    sims: pd.DataFrame,
    px: dict[str, pd.DataFrame],
    bench_px: pd.DataFrame,
    ic: float,
    capital: float = config.INITIAL_CAPITAL,
    fee: float = config.FEE_PER_SIDE,
) -> ExactResult:
    O, L, C = px["Open"], px["Low"], px["Close"].ffill()  # noqa: E741 - OHLC naming
    rets = px["Close"].pct_change(fill_method=None)
    bO = bench_px["Open"].reindex(C.index)  # ETF has gaps on Yahoo: align to stock calendar
    bC = bench_px["Close"].reindex(C.index).ffill()
    dates = [d for d in C.index if d >= sims["Date"].min()]
    by_day = {d: g for d, g in sims.groupby("Date")}

    cash, shares, entry, hedge_units = capital, {}, {}, 0.0
    pending, trades, equity, exposure = None, [], {}, {}

    def trade(tkr, qty, price, day, reason):
        nonlocal cash
        cash -= qty * price + abs(qty * price) * fee
        trades.append({"Date": day, "Ticker": tkr, "qty": qty, "price": price, "reason": reason})

    for i, d in enumerate(dates):
        # 1) execute yesterday's decisions at today's open
        if pending is not None:
            targets, hedge_target = pending
            eq_open = (
                cash
                + sum(
                    q * (O.at[d, t] if pd.notna(O.at[d, t]) else C.at[d, t])
                    for t, q in shares.items()
                )
                + hedge_units * (bO.get(d, np.nan) if pd.notna(bO.get(d, np.nan)) else bC.get(d, 0))
            )
            for t in list(shares):  # sells first
                if t not in targets.index and pd.notna(O.at[d, t]):
                    trade(t, -shares.pop(t), O.at[d, t], d, "rebalance-out")
                    entry.pop(t, None)
            for t, w in targets.items():  # then buys / resizes
                price = O.at[d, t] if t in O.columns else np.nan
                if pd.isna(price) or price <= 0:
                    continue
                want = int(w * eq_open / price / LOT) * LOT
                diff = want - shares.get(t, 0)
                if diff != 0 and abs(diff * price) >= cfg.min_trade_frac * eq_open:
                    trade(t, diff, price, d, "rebalance")
                    shares[t] = shares.get(t, 0) + diff
                    entry[t] = price if diff > 0 and t not in entry else entry.get(t, price)
                    if shares[t] == 0:
                        shares.pop(t), entry.pop(t, None)
            if cfg.hedge and pd.notna(bO.get(d, np.nan)):
                units = hedge_target * eq_open / bO[d]
                cash -= (units - hedge_units) * bO[d] + abs(
                    (units - hedge_units) * bO[d]
                ) * HEDGE_FEE
                hedge_units = units
            pending = None

        # 2) stop-loss on the daily low
        if cfg.stop_loss:
            for t in list(shares):
                stop_px = entry[t] * (1 - cfg.stop_loss)
                if pd.notna(L.at[d, t]) and L.at[d, t] <= stop_px:
                    fill = min(O.at[d, t], stop_px) if pd.notna(O.at[d, t]) else stop_px
                    trade(t, -shares.pop(t), fill, d, "stop-loss")
                    entry.pop(t)

        # 3) mark to market at the close
        long_val = sum(q * C.at[d, t] for t, q in shares.items())
        equity[d] = cash + long_val + hedge_units * bC.get(d, 0)
        exposure[d] = long_val / equity[d] if equity[d] else 0.0

        # 4) weekly decision after the close, executed tomorrow
        if i % config.HORIZON == 0 and i < len(dates) - 1 and d in by_day:
            day = by_day[d]
            w = target_weights(cfg, day, rets.loc[:d], ic, held=list(shares))
            if cfg.regime and (
                day["hmm_p_stressed"].iloc[0] > 0.5 or day["vix_slope"].iloc[0] > 1.0
            ):
                w = w * 0.5
            hedge = (
                -float((w * day.set_index("Ticker").reindex(w.index)["beta_mkt"].fillna(1.0)).sum())
                if cfg.hedge
                else 0.0
            )
            pending = (w, hedge)

    eq = pd.Series(equity, name=cfg.name)
    log.info(
        "    exact %-28s final equity %12s, trades %d", cfg.name, f"{eq.iloc[-1]:,.0f}", len(trades)
    )
    return ExactResult(cfg.name, eq, pd.DataFrame(trades), pd.Series(exposure, name=cfg.name))


def default_exact_configs() -> list[ExactConfig]:
    return [
        ExactConfig("E1 equal-weight top-10"),
        ExactConfig(
            "E2 equal top-10 + 30-rank buffer + stop", buffer=30, stop_loss=config.STOP_LOSS
        ),
        ExactConfig("E3 MVO + stop", method="mvo", vol_target=False, stop_loss=config.STOP_LOSS),
        ExactConfig(
            "E4 E3 + HMM regime filter",
            method="mvo",
            vol_target=False,
            stop_loss=config.STOP_LOSS,
            regime=True,
        ),
        ExactConfig(
            "E5 E3 beta-hedged (long-short)",
            method="mvo",
            vol_target=False,
            stop_loss=config.STOP_LOSS,
            hedge=True,
        ),
        ExactConfig(
            "E6 MVO conservative + 15% vol target",
            method="mvo",
            stop_loss=config.STOP_LOSS,
            risk_aversion=config.CONSERVATIVE_RISK_AVERSION,
            max_weight=0.15,
        ),
    ]


def config_by_name(name: str | None) -> ExactConfig:
    cfgs = {c.name: c for c in default_exact_configs()}
    return cfgs.get(name, cfgs["E3 MVO + stop"])


def universe_benchmark(px_close: pd.DataFrame, sims: pd.DataFrame) -> pd.Series:
    """Equal-weight return of the tradable top-100 (rebalanced daily, no costs)."""
    r = px_close.pct_change(fill_method=None)
    members = sims.pivot_table(
        index="Date", columns="Ticker", values="tradable", aggfunc="first"
    ).reindex(r.index)
    return (
        r.where(members.shift(1).fillna(False).astype(bool)).mean(axis=1).loc[sims["Date"].min() :]
    )
