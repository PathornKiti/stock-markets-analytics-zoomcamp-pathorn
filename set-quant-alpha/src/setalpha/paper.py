"""Live paper trading: pre-registered shadow portfolios and a signal-decay monitor.

The 2022-26 test period has been used by every experiment, so the only clean out-of-sample evidence left is the
future. Four candidates were frozen on 2026-11-02 (REGISTERED) and are traded on paper from the first trading day
on or after that date, by the daily Airflow run:

* S1 - production E2 (top-10 equal weight, 30-rank buffer, 8% stop)
* S2 - E2 without the stop (strategy_v2: the buffer is the active ingredient, the stop is not robust)
* S3 - TDEX + factor-neutral, cost-aware tilts (strategy_v3 F2: fixes the cost and size leaks)
* S4 - E2 on a slower score: the average rank of the 5-day ensemble and a 20-day-target model
       (model_v2 V8: alpha persists past day 5, and a slower score trades less)
* S5 - registered later (2026-11-02T10:56:17Z) with its own timestamp: the experiments/integrated_v1 walk-forward
       pick for 2026 (C2 N10 R5 B30 E) - the same blended score, top-10, 30-rank buffer, weekly, equal weight, no stop.
       Its 2016-26 walk-forward record (7.8% vs TDEX 7.2%, alpha t about 1) is its pre-registration.
* S6 - registered 2026-11-02T11:02:29Z: the integrated_v1 HINDSIGHT-BEST design (C2 N10 R5 B T), i.e. S5 with
       score-tilted weights (1.5x for the best name down to 0.5x for the 10th). It beat CAPM in the backtest
       (2016-26 net alpha +9.9% a year, t 2.36) but was chosen after seeing 2013-26 (DSR 0.55), so that record is
       NOT evidence; only its forward record counts.

Rules for an honest track record:
* point-in-time - each day's scores are logged when published (`scores.parquet`, first version kept), decisions
  use only data up to that close, and orders fill at the next open;
* append-only - NAV, trades and scores are only ever appended; a re-run of the same day is a no-op;
* frozen - the shadows, their parameters and the alert thresholds below must not be changed after REGISTERED.
  A new idea becomes a new shadow with its own registration date.

Execution model (same as the backtests): decisions every 5 trading days after the close, fills at the next open,
100-share lots, config.FEE_PER_SIDE per side on stocks (0.168% all-in since 2026-11-02, before any live trade) and 0.05% on TDEX, stops checked on the daily low. THB 20M per shadow, so
100-share lots do not block S3's small tilts.

Signal monitor: the daily IC of the published score is measured once its 5-day forward alpha is known, and the
rolling 60-day mean is compared with fixed thresholds (amber < 0.02, red < 0). History before the paper start comes
from the walk-forward predictions, so the chart shows 2013-2026 decay and the live continuation on one axis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from setalpha import config, factors, model, portfolio, report, signal_eval, strategy

log = logging.getLogger(__name__)

REGISTERED = "2026-11-02T05:33:08Z"
START = "2026-11-02"  # first trading day on or after this date is day 1
PAPER_DIR = config.DATA_DIR / "paper"
STATE, NAV, TRADES, SCORES = (
    PAPER_DIR / f for f in ("state.json", "nav.parquet", "trades.parquet", "scores.parquet")
)
SUMMARY = PAPER_DIR / "summary.joblib"
SLOW_MODEL = config.MODELS_DIR / "slow.joblib"
EXT_PREDICTIONS = (
    config.PROJECT_DIR / "experiments" / "strategy_v3" / "output" / "predictions_ext.parquet"
)
CAPITAL = 20e6
SLOW_REFIT_DAYS = 7
IC_WINDOW, IC_AMBER, IC_RED = 60, 0.02, 0.0


@dataclass(frozen=True)
class Shadow:
    name: str
    kind: str  # "rule" (E2-style equal weight) | "enhanced" (TDEX + neutral tilts)
    score: str = "pred_ens"
    stop: float | None = None
    buffer: int = 30
    registered: str = REGISTERED
    start: str | None = (
        None  # first trading day on or after this date is day 1 (None = the ledger START)
    )
    tilt: bool = (
        False  # score-tilted weights (integrated_v1 "T"): 1.5x best ... 0.5x worst, normalised
    )


SHADOWS = [
    Shadow("S1 E2 production", "rule", stop=config.STOP_LOSS),
    Shadow("S2 E2 without stop", "rule"),
    Shadow("S3 TDEX + neutral cost-aware tilts", "enhanced"),
    Shadow("S4 E2 on blended 5d+20d score", "rule", score="pred_blend", stop=config.STOP_LOSS),
    Shadow(
        "S5 integrated pick: blend, no stop",
        "rule",
        score="pred_blend",
        registered="2026-11-02T10:56:17Z",
        start="2026-11-02",
    ),
    Shadow(
        "S6 hindsight best: blend, score-tilted",
        "rule",
        score="pred_blend",
        registered="2026-11-02T11:02:29Z",
        start="2026-11-02",
        tilt=True,
    ),
]


DEPLOYED = "S5 integrated pick: blend, no stop"  # the production strategy behind reports/latest_signals.csv
DEPLOYED_DESC = "integrated walk-forward pick: ML + 20-day score, top-10, 30-rank buffer, weekly, equal weight, no stop"


def deployed() -> Shadow:
    return next(s for s in SHADOWS if s.name == DEPLOYED)


def deployed_holdings() -> list[str]:
    """Names the deployed paper book holds after its pending orders fill (empty before the ledger starts)."""
    if not STATE.exists():
        return []
    st = json.loads(STATE.read_text())["shadows"].get(DEPLOYED)
    if not st:
        return []
    return (
        list(st["pending"]["w"]) if st.get("pending") else [t for t, q in st["shares"].items() if q]
    )


def slow_features() -> list[str]:
    return joblib.load(SLOW_MODEL)["features"]


# ----------------------------------------------------------------------------- slow (20-day) model
def fit_slow(df: pd.DataFrame, features: list[str], prices: pd.DataFrame) -> dict:
    """XGB regression on the per-date rank of the 20-day forward alpha (model_v2 V8), all labelled liquid rows."""
    fwd = signal_eval.forward_alphas(prices, df, start=config.START)[
        ["Date", "Ticker", "fwd_alpha_20"]
    ]
    lab = df[df["liquid"]].merge(fwd, on=["Date", "Ticker"]).dropna(subset=["fwd_alpha_20"])
    y = lab.groupby("Date")["fwd_alpha_20"].rank(pct=True)
    p = json.loads((config.MODELS_DIR / "best_params.json").read_text())["xgb"]
    m = XGBRegressor(
        subsample=0.8, colsample_bytree=0.7, tree_method="hist", n_jobs=-1, random_state=0, **p
    )
    m.fit(lab[features], y)
    bundle = {
        "model": m,
        "features": features,
        "trained_through": str(lab["Date"].max().date()),
        "fitted_on": str(df["Date"].max().date()),
    }
    joblib.dump(bundle, SLOW_MODEL)
    log.info(
        "paper: slow model refit on %s rows, labels through %s",
        f"{len(lab):,}",
        bundle["trained_through"],
    )
    return bundle


def slow_model(df: pd.DataFrame, features: list[str], prices: pd.DataFrame) -> dict:
    if SLOW_MODEL.exists():
        b = joblib.load(SLOW_MODEL)
        if (df["Date"].max() - pd.Timestamp(b["fitted_on"])).days < SLOW_REFIT_DAYS:
            return b
    return fit_slow(df, features, prices)


def day_scores(day: pd.DataFrame, slow: dict, members: list[str]) -> pd.DataFrame:
    """Production ensemble, slow score and their blend for one date's liquid names."""
    out = day[["Date", "Ticker", "tradable"]].copy()
    for name in members:
        b = model.load_final(name)
        out[f"pred_{name}"] = model.predict(b["model"], day, b["features"])
    out["pred_ens"] = sum(out[f"pred_{m}"].rank(pct=True) for m in members) / len(members)
    out["pred_slow"] = slow["model"].predict(day[slow["features"]])
    out["pred_blend"] = 0.5 * out["pred_ens"].rank(pct=True) + 0.5 * out["pred_slow"].rank(pct=True)
    return out


# ----------------------------------------------------------------------------- ledger mechanics
def new_state() -> dict:
    return {
        "cash": CAPITAL,
        "shares": {},
        "entry": {},
        "tdex": 0.0,
        "pending": None,
        "n": 0,
        "last": None,
    }


def _px(frame: pd.DataFrame, d, t) -> float:
    return (
        float(frame.at[d, t])
        if t in frame.columns and d in frame.index and pd.notna(frame.at[d, t])
        else np.nan
    )


def execute(
    st: dict,
    d,
    O: pd.DataFrame,
    C: pd.DataFrame,
    b_open: float,
    trades: list,
    name: str,  # noqa: E741
    fee: float = config.FEE_PER_SIDE,
    tdex_fee: float = strategy.HEDGE_FEE,
    min_trade: float = 0.01,
) -> None:
    """Fill yesterday's pending targets at today's open: sells first, 100-share lots, costs per side."""
    targets, t_w = st["pending"]["w"], st["pending"]["tdex"]
    mark = {t: _px(O, d, t) if pd.notna(_px(O, d, t)) else _px(C, d, t) for t in st["shares"]}
    value = (
        st["cash"]
        + sum(q * mark[t] for t, q in st["shares"].items() if pd.notna(mark[t]))
        + st["tdex"] * b_open
    )

    def trade(t, qty, price, reason, cost):
        st["cash"] -= qty * price + abs(qty * price) * cost
        trades.append(
            {"Date": d, "shadow": name, "Ticker": t, "qty": qty, "price": price, "reason": reason}
        )

    orders = []
    for t in set(st["shares"]) | set(targets):
        price = _px(O, d, t)
        if pd.isna(price) or price <= 0:
            continue  # suspended: keep / skip until it trades
        dq = int(targets.get(t, 0.0) * value / price / strategy.LOT) * strategy.LOT - st[
            "shares"
        ].get(t, 0)
        if dq and (abs(dq * price) >= min_trade * value or t not in targets):
            orders.append((dq, t, price))
    for dq, t, price in sorted(orders):  # negative quantities (sells) first
        trade(t, dq, price, "rebalance", fee)
        q = st["shares"].get(t, 0) + dq
        if q:
            st["shares"][t] = q
            st["entry"].setdefault(t, price)
        else:
            st["shares"].pop(t, None), st["entry"].pop(t, None)
    units = t_w * value / b_open if b_open > 0 else st["tdex"]
    if abs(units - st["tdex"]) * b_open > 1e-6 * value:
        trade(config.BENCHMARK, units - st["tdex"], b_open, "tdex", tdex_fee)
        st["tdex"] = units
    st["pending"] = None


def stops(
    st: dict,
    d,
    O: pd.DataFrame,
    L: pd.DataFrame,
    stop: float,
    trades: list,
    name: str,  # noqa: E741
    fee: float = config.FEE_PER_SIDE,
) -> None:
    for t in list(st["shares"]):
        stop_px = st["entry"][t] * (1 - stop)
        low = _px(L, d, t)
        if pd.notna(low) and low <= stop_px:
            o = _px(O, d, t)
            fill = min(o, stop_px) if pd.notna(o) else stop_px
            q = st["shares"].pop(t)
            st["entry"].pop(t)
            st["cash"] += q * fill - abs(q * fill) * fee
            trades.append(
                {
                    "Date": d,
                    "shadow": name,
                    "Ticker": t,
                    "qty": -q,
                    "price": fill,
                    "reason": "stop-loss",
                }
            )


def mark(st: dict, d, C: pd.DataFrame, b_close: float) -> dict:
    stock = sum(q * _px(C, d, t) for t, q in st["shares"].items() if pd.notna(_px(C, d, t)))
    nav = st["cash"] + stock + st["tdex"] * b_close
    return {"Date": d, "nav": nav, "stock_w": stock / nav, "tdex_w": st["tdex"] * b_close / nav}


def tilt_weights(names: list[str], score: pd.Series, n: int) -> pd.Series:
    """integrated_v1 score tilt: names sorted by score get 1.5x ... 0.5x of an equal 1/n weight."""
    names = sorted(names, key=lambda t: -score[t])
    lin = 1.5 - np.arange(len(names)) / (len(names) - 1)
    return pd.Series(lin / lin.sum() * len(names) / n, index=names)


def targets(sh: Shadow, st: dict, day: pd.DataFrame, nav: dict, ctx: dict) -> tuple[dict, float]:
    """Next weights for one shadow from the day's tradable scores (decided at the close)."""
    held_value = {t: q * _px(ctx["C"], nav["Date"], t) for t, q in st["shares"].items()}
    if sh.kind == "rule":
        cfg = strategy.ExactConfig(sh.name, score=sh.score, buffer=sh.buffer, stop_loss=sh.stop)
        w = strategy.target_weights(
            cfg, day, ctx["rets"].loc[: nav["Date"]], ctx["ic"], held=list(st["shares"])
        )
        if sh.tilt and len(w) > 1:
            w = tilt_weights(list(w.index), day.set_index("Ticker")[sh.score], config.MAX_POSITIONS)
        return {t: float(x) for t, x in w.items()}, 0.0
    tr = day.dropna(subset=["idio_vol"]).set_index("Ticker")
    a = portfolio.alpha_from_scores(tr["pred_ens"], tr["idio_vol"], ctx["ic"])
    held = pd.Series(held_value, dtype=float).fillna(0.0) / nav["nav"]
    cand = sorted(set(a.index[a > 0]) | (set(held.index[held > 0]) & set(a.index)))
    L = portfolio.factor_loadings(ctx["rets"], ctx["fac"], nav["Date"], cand)
    cand = list(L.index)
    w, wt, ok = portfolio.enhanced_index(
        a.reindex(cand).to_numpy(),
        (L["s_eps"] ** 2 * config.HORIZON).to_numpy(),
        L[["b_MKT", "b_SMB", "b_WML"]].to_numpy(),
        held.reindex(cand).fillna(0.0).to_numpy(),
        nav["tdex_w"],
    )
    if not ok:
        log.warning("paper: %s optimiser did not converge on %s", sh.name, nav["Date"].date())
    return {t: float(x) for t, x in zip(cand, w, strict=True) if x > 0}, wt


# ----------------------------------------------------------------------------- daily step
def _bench(market: pd.DataFrame) -> pd.DataFrame:
    return (
        market[market["Ticker"] == config.BENCHMARK]
        .set_index("Date")
        .sort_index()[["Open", "Close"]]
    )


def _append(path, new: pd.DataFrame, keys: list[str]) -> None:
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True).drop_duplicates(
            keys, keep="first"
        )
    new.to_parquet(path, index=False)


def run_day(
    df: pd.DataFrame, features: list[str], prices: pd.DataFrame, market: pd.DataFrame
) -> pd.DataFrame:
    """Process every trading day after the last one in the ledger (normally one). Returns the new NAV rows."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if not state:
        state = {
            "registered": REGISTERED,
            "shadows": {s.name: new_state() for s in SHADOWS},
            "registry": [asdict(s) for s in SHADOWS],
        }
    known = {r["name"] for r in state["registry"]}
    state["registry"] += [
        asdict(s) for s in SHADOWS if s.name not in known
    ]  # append later registrations
    last = max((s["last"] for s in state["shadows"].values() if s["last"]), default=None)
    days = sorted(
        d
        for d in df["Date"].unique()
        if d >= pd.Timestamp(START) and (last is None or d > pd.Timestamp(last))
    )
    if not days:
        log.info(
            "paper: nothing to do (ledger at %s, data to %s, start %s)",
            last,
            df["Date"].max().date(),
            START,
        )
        return pd.DataFrame()

    px = strategy.wide_prices(
        prices[prices["Date"] >= pd.Timestamp(START) - pd.Timedelta(days=400)]
    )
    O, L, C = px["Open"], px["Low"], px["Close"].ffill()  # noqa: E741 - OHLC naming
    bench = _bench(market)
    ctx = {
        "C": C,
        "rets": px["Close"].pct_change(fill_method=None),
        "ic": report.load_meta()["ic_valid"],
        "fac": factors.local_factors(df, prices, bench["Close"]),
    }
    members = report.load_meta()["ensemble"]
    slow = slow_model(df, features, prices)
    navs, trades, scores = [], [], []
    for d in days:
        b_open = (
            float(bench["Open"].get(d, np.nan))
            if pd.notna(bench["Open"].get(d, np.nan))
            else float(bench["Close"].ffill().loc[:d].iloc[-1])
        )
        b_close = float(bench["Close"].ffill().loc[:d].iloc[-1])
        sc = day_scores(df[(df["Date"] == d) & df["liquid"]], slow, members)
        scores.append(sc[["Date", "Ticker", "tradable", "pred_ens", "pred_slow", "pred_blend"]])
        day = sc[sc["tradable"]].merge(df.loc[df["Date"] == d, ["Ticker", "idio_vol"]], on="Ticker")
        for sh in SHADOWS:
            if d < pd.Timestamp(sh.start or START):
                continue
            st = state["shadows"].setdefault(
                sh.name, new_state()
            )  # shadows registered later join here
            if st["pending"]:
                execute(st, d, O, C, b_open, trades, sh.name)
            if sh.stop:
                stops(st, d, O, L, sh.stop, trades, sh.name)
            nav = mark(st, d, C, b_close)
            navs.append({"shadow": sh.name, **nav})
            if st["n"] % config.HORIZON == 0:
                w, wt = targets(sh, st, day, nav, ctx)
                st["pending"] = {"w": w, "tdex": wt}
            st["n"] += 1
            st["last"] = str(pd.Timestamp(d).date())
    new_nav = pd.DataFrame(navs, columns=["shadow", "Date", "nav", "stock_w", "tdex_w"])
    if len(new_nav):
        _append(NAV, new_nav, ["Date", "shadow"])
    if trades:
        _append(TRADES, pd.DataFrame(trades), ["Date", "shadow", "Ticker", "reason"])
    _append(SCORES, pd.concat(scores, ignore_index=True), ["Date", "Ticker"])
    STATE.write_text(
        json.dumps(state, indent=1, default=str)
    )  # written last: a crash leaves the old state
    log.info(
        "paper: processed %d day(s) to %s\n%s",
        len(days),
        days[-1].date(),
        new_nav.groupby("shadow")["nav"].last().map("{:,.0f}".format).to_string(),
    )
    return new_nav


# ----------------------------------------------------------------------------- monitoring + summary
def ic_monitor(df: pd.DataFrame) -> dict:
    """Daily IC of the published score vs realised 5-day alpha: walk-forward history + live paper scores."""
    real = df[["Date", "Ticker", "tradable", "fwd_alpha_5"]]
    src = EXT_PREDICTIONS if EXT_PREDICTIONS.exists() else model.PREDICTIONS
    hist = model.daily_ic(
        pd.read_parquet(src, columns=["Date", "Ticker", "pred_ens"]).merge(
            real, on=["Date", "Ticker"]
        ),
        "pred_ens",
    ).rename("walk-forward")
    live = pd.Series(dtype=float, name="live")
    if SCORES.exists():
        s = pd.read_parquet(SCORES, columns=["Date", "Ticker", "pred_ens"]).merge(
            real, on=["Date", "Ticker"]
        )
        if s["fwd_alpha_5"].notna().any():
            live = model.daily_ic(s, "pred_ens").rename("live")
    ic = pd.concat([hist[hist.index < pd.Timestamp(START)], live]).sort_index()
    roll = ic.rolling(IC_WINDOW, min_periods=IC_WINDOW // 2).mean()
    last = float(roll.dropna().iloc[-1]) if roll.notna().any() else np.nan
    recent = ic.tail(IC_WINDOW).iloc[:: config.HORIZON]  # non-overlapping 5-day targets
    t = float(recent.mean() / recent.std() * np.sqrt(len(recent))) if len(recent) > 2 else np.nan
    status = "red" if last < IC_RED else "amber" if last < IC_AMBER else "green"
    return {
        "ic": ic,
        "rolling": roll,
        "live_days": int(live.notna().sum()),
        "last_rolling_ic": last,
        "t_last_window": t,
        "status": status,
        "thresholds": {"amber": IC_AMBER, "red": IC_RED, "window": IC_WINDOW},
    }


def summary(market: pd.DataFrame, monitor: dict) -> dict:
    reg = pd.DataFrame([asdict(s) for s in SHADOWS]).set_index("name")
    out = {
        "registered": REGISTERED,
        "start": START,
        "registry": reg,
        "monitor": monitor,
        "nav": pd.DataFrame(),
        "perf": pd.DataFrame(),
        "trades": pd.DataFrame(),
    }
    if NAV.exists():
        nav = pd.read_parquet(NAV).pivot_table(index="Date", columns="shadow", values="nav")
        bench = _bench(market)["Close"].ffill().reindex(nav.index).ffill()
        tr = pd.read_parquet(TRADES) if TRADES.exists() else pd.DataFrame(columns=["shadow"])
        rows = {}
        for c in nav.columns:
            r = nav[c] / nav[c].iloc[0] - 1
            rows[c] = {
                "since": nav.index[0].date(),
                "days": len(nav),
                "return %": 100 * r.iloc[-1],
                "TDEX same window %": 100 * (bench.iloc[-1] / bench.iloc[0] - 1),
                "active %": 100 * (r.iloc[-1] - (bench.iloc[-1] / bench.iloc[0] - 1)),
                "max DD %": 100 * (nav[c] / nav[c].cummax() - 1).min(),
                "trades": int((tr["shadow"] == c).sum()),
            }
        out.update(
            nav=nav.assign(**{f"{config.BENCHMARK} (scaled)": CAPITAL * bench / bench.iloc[0]}),
            perf=pd.DataFrame(rows).T,
            trades=tr.tail(30),
        )
    return out


def step(df: pd.DataFrame, features: list[str], prices: pd.DataFrame, market: pd.DataFrame) -> dict:
    run_day(df, features, prices, market)
    mon = ic_monitor(df)
    if mon["status"] != "green":
        log.warning(
            "paper: signal monitor %s - rolling %d-day IC %.3f (amber < %.2f, red < %.2f)",
            mon["status"].upper(),
            IC_WINDOW,
            mon["last_rolling_ic"],
            IC_AMBER,
            IC_RED,
        )
    res = summary(market, mon)
    joblib.dump(res, SUMMARY)
    return res
