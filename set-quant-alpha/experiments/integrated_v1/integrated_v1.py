"""integrated_v1 - every available signal, a portfolio-parameter grid and WALK-FORWARD OPTIMISATION (2016-2026).

    uv run python set-quant-alpha/experiments/integrated_v1/make_v8_wf.py      # V8 walk-forward 2013-26 (~1 min)
    uv run python set-quant-alpha/experiments/integrated_v1/integrated_v1.py panel   # signals + wide panels
    uv run python set-quant-alpha/experiments/integrated_v1/integrated_v1.py grid    # fast sim of every config
    uv run python set-quant-alpha/experiments/integrated_v1/integrated_v1.py wf      # WFO, exact confirmation, stats

Pre-registered before any simulation (2026-11-02):

Signals (all known at the close of t; ranks are per-date percentiles inside the tradable top-100):
  ml      walk-forward ML ensemble score pred_ens (strategy_v3 predictions_ext)
  v8      walk-forward 20-day-target XGB score (model_v2 V8, refit each January, make_v8_wf.py)
  lowbeta -beta_mkt (120d)                 lowvol  -250d volatility
  resmom  12-1m residual momentum / vol     hi52    price / 250d high      (literature_v1.add_scores)
  eps     last EPS surprise %, 60-day decay (setalpha.events)
  divy    trailing-12-month dividend yield (Yahoo dividends, ex-date based)
Missing signal values are neutral (rank 0.5).

Composites (8):
  C1 ML       ml                           C2 ML2     mean rank(ml, v8)
  C3 LIT4     mean rank(lowbeta, lowvol, resmom, hi52)
  C4 EQ8      mean rank of all 8           C7 ML+EV   mean rank(ml, v8, eps, divy)
  C5 ICW-X    sum_k w_k (rank_k - .5), w_k = max(IC_k, 0) / sum, IC = mean daily rank IC vs forward alpha
              at the horizon matching the rebalance interval (5/10/20d), EXPANDING window of prior years
              (2013 .. Y-1, forward windows ending before Y); equal weights in 2013
  C6 ICW-3Y   as C5 with a rolling 3-year IC window
  C8 ICW-N    C5 residualised each day on size (log 60d traded value), beta and 12-1 momentum

Portfolio grid (8 x 36 = 288 configs): N in {10, 20, 30}; rebalance every {5, 10, 21} trading days (counted
from the first trading day of each year); hold buffer {none, 3N capped at 80}; weighting {equal, score-tilted
(linear 1.5 -> 0.5 by rank within the book)}. Long-only, tradable top-100, decision at the close, execution at
the next open, 0.15%/side, a name is resized only if its weight moves >= 1% of equity.
Not in the selection grid (pre-declared ablations): 8% stop-loss (strategy_v2/v4: not robust) -> an extended
576-config WFO; TDEX core 50/70% (CAPM-alpha t and IR are invariant to a TDEX core, so it cannot be chosen by the
objective; reported as overlays); IR objective instead of alpha t.

Walk-forward optimisation: for each year Y = 2016..2026 choose the config with the highest NET CAPM-alpha t-stat
vs TDEX (daily, Newey-West 5 lags) over 2013-01-01 .. Y-1-12-31 among configs with two-way turnover <= 40x/yr in
that window, then trade year Y with it. One continuous simulation follows the schedule, so the switching trades
are paid. Confirmed by an exact simulator (THB 1M, 100-share lots).
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from setalpha import config, evaluate, events, factors, validity

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
for sub in ("strategy_v3", "strategy_v2", "literature_v1"):
    sys.path.insert(0, str(HERE.parent / sub))
import literature_v1 as lit  # noqa: E402
import sv2  # noqa: E402
import v3  # noqa: E402

PANEL = OUT / "panel.joblib"
GRID = OUT / "grid.joblib"
V8 = OUT / "predictions_v8_wf.parquet"
FEE, TFEE = config.FEE_PER_SIDE, sv2.TDEX_FEE
DAYS = 252
SIGNALS = ["ml", "v8", "lowbeta", "lowvol", "resmom", "hi52", "eps", "divy"]
COMPOSITES = [
    "C1 ML",
    "C2 ML2",
    "C3 LIT4",
    "C4 EQ8",
    "C5 ICW-X",
    "C6 ICW-3Y",
    "C7 ML+EV",
    "C8 ICW-N",
]
HORIZON_OF = {5: 5, 10: 10, 21: 20}
WF_YEARS = list(range(2016, 2027))
FIRST_YEAR = 2013
MAX_TURNOVER = 40.0  # two-way x/yr sanity bound in the selection window
MIN_TRADE = 0.01  # resize a held name only if its weight moves >= 1% of equity
PRIOR_TRIALS = (
    validity.N_TRIALS
    - validity.TRIALS.get("integrated_v1", 0)
    - validity.TRIALS.get("retrain_v1", 0)
)  # 89: trials before this experiment


@dataclass(frozen=True)
class Cfg:
    comp: str
    n: int
    reb: int
    buf: bool
    tilt: bool
    stop: float = 0.0
    core: float = 0.0

    @property
    def key(self) -> str:
        return (
            f"{self.comp.split()[0]}|N{self.n}|R{self.reb}|{'B' if self.buf else 'b'}|{'T' if self.tilt else 'E'}"
            + (f"|S{self.stop:.2f}" if self.stop else "")
            + (f"|K{self.core:.2f}" if self.core else "")
        )

    @property
    def buffer(self) -> int:
        return min(3 * self.n, 80) if self.buf else self.n

    @property
    def score(self) -> str:
        c = self.comp.split()[0]
        return f"{c}_h{HORIZON_OF[self.reb]}" if c in ("C5", "C6", "C8") else c


def grid(stop: float = 0.0) -> list[Cfg]:
    return [
        Cfg(c, n, r, b, t, stop)
        for c, n, r, b, t in itertools.product(
            COMPOSITES, (10, 20, 30), (5, 10, 21), (False, True), (False, True)
        )
    ]


# ============================================================================ 1. signals + panels
def _wide(f: pd.DataFrame, col: str, dates, tickers) -> np.ndarray:
    return (
        f.pivot_table(index="Date", columns="Ticker", values=col)
        .reindex(index=dates, columns=tickers)
        .to_numpy()
    )


def _div_yield(df: pd.DataFrame, div: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Trailing-12-month dividends (by ex-date, mapped to the first trading day on/after it) / close."""
    dates, tickers = close.index, close.columns
    d = div[div["Ticker"].isin(tickers)].copy()
    pos = dates.searchsorted(d["Date"].to_numpy())
    d = d[pos < len(dates)].assign(Date=dates[pos[pos < len(dates)]])
    amt = d.pivot_table(index="Date", columns="Ticker", values="dividend", aggfunc="sum").reindex(
        index=dates, columns=tickers
    )
    return amt.fillna(0.0).rolling(245, min_periods=200).sum() / close


def build_panel() -> None:
    t0 = time.time()
    inp = sv2.load_inputs(v3.PREDS_EXT, v8_path=None)
    f = pd.concat([sv2.make_frame(inp, s) for s in ("sel", "test")], ignore_index=True)
    f = lit.add_scores(f, inp)
    f = f.merge(pd.read_parquet(V8), on=["Date", "Ticker"], how="left")
    df, prices = inp["df"], inp["prices"]
    earn = pd.read_parquet(config.RAW_DIR / "earnings.parquet")
    div = pd.read_parquet(config.RAW_DIR / "dividends.parquet")
    ev = events.earnings_features(df, earn, prices)[["Date", "Ticker", "last_surprise"]]
    f = f.merge(ev, on=["Date", "Ticker"], how="left")
    print(f"frame {len(f):,} rows, {time.time() - t0:.0f}s", flush=True)

    f = f[f["tradable"].fillna(False).astype(bool)].copy()  # the tradable top-100 is the universe
    dates_all = pd.DatetimeIndex(sorted(prices["Date"].unique()), name="Date")
    dates = dates_all[(dates_all >= f["Date"].min()) & (dates_all <= f["Date"].max())]
    tickers = sorted(f["Ticker"].unique())
    W = lambda c: prices.pivot_table(index="Date", columns="Ticker", values=c).reindex(  # noqa: E731
        index=dates_all, columns=pd.Index(tickers, name="Ticker")
    )
    O, C, L = W("Open"), W("Close"), W("Low")
    dy = _div_yield(df, div, C)
    dy = dy.stack(future_stack=True).rename("divy").reset_index()
    f = f.merge(dy, on=["Date", "Ticker"], how="left")
    mom = (C.shift(21) / C.shift(252) - 1).stack(future_stack=True).rename("mom121").reset_index()
    f = f.merge(mom, on=["Date", "Ticker"], how="left")
    f["logsize"] = np.log(f["dvol60"])

    # forward beta-adjusted alphas (buy next open, sell close t+h) vs the EW liquid market; IC targets only
    liq = df.pivot_table(index="Date", columns="Ticker", values="liquid", aggfunc="first")
    liq = liq.reindex(index=dates_all, columns=C.columns).fillna(False).astype(bool)
    Call = prices.pivot_table(index="Date", columns="Ticker", values="Close").reindex(dates_all)
    Oall = prices.pivot_table(index="Date", columns="Ticker", values="Open").reindex(dates_all)
    liqa = (
        df.pivot_table(index="Date", columns="Ticker", values="liquid", aggfunc="first")
        .reindex(index=dates_all, columns=Call.columns)
        .fillna(False)
        .astype(bool)
    )
    beta_all = df.pivot_table(index="Date", columns="Ticker", values="beta_mkt").reindex(
        index=dates_all, columns=Call.columns
    )
    fwd = {}
    for h in (5, 10, 20):
        fr = Call.shift(-h) / Oall.shift(-1) - 1
        fr = fr.where(fr.abs() < 1.0)
        fm = fr.where(liqa).mean(axis=1)
        fwd[h] = (fr - beta_all.mul(fm, axis=0)).reindex(index=dates, columns=tickers).to_numpy()
    del liq

    src = {
        "ml": "pred_ens",
        "v8": "pred_v8",
        "lowbeta": "s_lowbeta",
        "lowvol": "s_lowvol",
        "resmom": "s_resmom",
        "hi52": "s_hi52",
        "eps": "last_surprise",
        "divy": "divy",
    }
    g = f.groupby("Date")
    for k, c in src.items():
        f[f"r_{k}"] = g[c].rank(pct=True)
    cover = {k: float(f[f"r_{k}"].notna().mean()) for k in SIGNALS}
    print(
        "signal coverage in the tradable universe:",
        {k: round(v, 3) for k, v in cover.items()},
        flush=True,
    )

    sig = {k: _wide(f, f"r_{k}", dates, tickers) for k in SIGNALS}
    aux = {c: _wide(f, c, dates, tickers) for c in ("logsize", "beta_mkt", "mom121")}
    tradable = ~np.isnan(_wide(f, "r_ml", dates, tickers))
    bench = inp["bench"]
    panel = {
        "dates": dates,
        "tickers": tickers,
        "O": O.reindex(dates).to_numpy(),
        "C": C.reindex(dates).to_numpy(),
        "L": L.reindex(dates).to_numpy(),
        "C_prev0": C.shift(1).reindex(dates).to_numpy()[0],
        "sig": sig,
        "aux": aux,
        "fwd": fwd,
        "tradable": tradable,
        "cover": cover,
        "bench": bench[["Open", "Close"]],
        "local": inp["local"],
        "ff": inp["ff"],
        "usdthb": inp["usdthb"],
        "ew": sv2.universe_ew(C.reindex(dates_all), f).reindex(dates),
    }
    panel["local_net"] = local_factors_net(df, prices, bench["Close"])
    joblib.dump(panel, PANEL, compress=3)
    print(f"panel saved: {len(dates)} days x {len(tickers)} tickers, {time.time() - t0:.0f}s")


def local_factors_net(df, prices, bench_close, fee=FEE) -> pd.DataFrame:
    """Local SMB / WML (same construction as setalpha.factors) charged their own membership turnover costs."""
    close = prices.pivot_table(index="Date", columns="Ticker", values="Close")
    ret = close.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 0.45)
    liquid = df.pivot_table(index="Date", columns="Ticker", values="liquid", aggfunc="first")
    liquid = liquid.reindex(index=ret.index, columns=ret.columns).fillna(False).astype(bool)
    size = df.pivot_table(index="Date", columns="Ticker", values="dvol60").reindex(
        index=ret.index, columns=ret.columns
    )
    size_lag = size.shift(1).where(liquid.shift(1, fill_value=False))
    med = size_lag.median(axis=1)
    mom = (close.shift(21) / close.shift(252) - 1).shift(1).where(liquid.shift(1, fill_value=False))
    hi, lo = mom.quantile(0.7, axis=1), mom.quantile(0.3, axis=1)

    def leg(mask):
        m = mask & ret.notna()
        w = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        r = (ret.fillna(0.0) * w).sum(axis=1)
        drift = w.mul(1 + ret.fillna(0.0)).div((1 + r), axis=0)
        to = (w - drift.shift(1)).abs().sum(axis=1)  # buys + sells to get back to equal weight
        return r, to

    out = {}
    for name, (long, short) in {
        "SMB": (size_lag.le(med, axis=0), size_lag.gt(med, axis=0)),
        "WML": (mom.ge(hi, axis=0), mom.le(lo, axis=0)),
    }.items():
        rl, tl = leg(long)
        rs, ts = leg(short)
        out[name] = (rl - rs) - fee * (tl + ts)
        out[name + "_cost_pa"] = fee * (tl + ts)
    mkt = bench_close.reindex(ret.index).ffill().pct_change()
    res = pd.DataFrame({"MKT": mkt, "SMB": out["SMB"], "WML": out["WML"]}).dropna()
    res.attrs["cost_pa"] = {k: float(out[k + "_cost_pa"].mean() * DAYS) for k in ("SMB", "WML")}
    return res


# ============================================================================ 2. composites (past-only IC weights)
def daily_ic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row rank correlation of x and y (x already pct ranks; y re-ranked on the common rows)."""
    out = np.full(x.shape[0], np.nan)
    for t in range(x.shape[0]):
        m = ~np.isnan(x[t]) & ~np.isnan(y[t])
        if m.sum() < 20:
            continue
        a = pd.Series(x[t, m]).rank().to_numpy()
        b = pd.Series(y[t, m]).rank().to_numpy()
        out[t] = np.corrcoef(a, b)[0, 1]
    return out


def ic_tables(panel: dict) -> dict[int, pd.DataFrame]:
    """Daily IC of every signal at h = 5, 10, 20 (on the tradable universe)."""
    idx = panel["dates"]
    return {
        h: pd.DataFrame({k: daily_ic(panel["sig"][k], panel["fwd"][h]) for k in SIGNALS}, index=idx)
        for h in (5, 10, 20)
    }


def ic_weights(ic: pd.DataFrame, h: int, window: int | None) -> pd.DataFrame:
    """Weights for each year Y from daily ICs whose forward window ends before Y (past-only)."""
    dates = ic.index
    rows = {}
    for Y in range(FIRST_YEAR, 2027):
        first = dates.searchsorted(pd.Timestamp(f"{Y}-01-01"))
        lo = dates.searchsorted(pd.Timestamp(f"{Y - window}-01-01")) if window else 0
        hist = ic.iloc[lo : max(first - h, 0)]
        m = hist.mean() if len(hist) >= 120 else pd.Series(np.nan, index=ic.columns)
        w = m.clip(lower=0).fillna(0.0)
        rows[Y] = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(SIGNALS), index=ic.columns)
    return pd.DataFrame(rows).T


def _neutralise(s: np.ndarray, aux: dict) -> np.ndarray:
    out = np.full_like(s, np.nan)
    X3 = np.stack([aux["logsize"], aux["beta_mkt"], aux["mom121"]], axis=-1)
    for t in range(s.shape[0]):
        m = ~np.isnan(s[t])
        if m.sum() < 20:
            continue
        X = X3[t, m]
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        X = np.column_stack([np.ones(m.sum()), X])
        b, *_ = np.linalg.lstsq(X, s[t, m], rcond=None)
        out[t, m] = s[t, m] - X @ b
    return out


def build_scores(panel: dict, ics: dict) -> tuple[dict[str, np.ndarray], dict]:
    trad = panel["tradable"]
    R = {
        k: np.where(trad, np.nan_to_num(v - 0.5, nan=0.0), np.nan) for k, v in panel["sig"].items()
    }
    mean = lambda ks: sum(R[k] for k in ks) / len(ks)  # noqa: E731
    S = {
        "C1": R["ml"],
        "C2": mean(["ml", "v8"]),
        "C3": mean(["lowbeta", "lowvol", "resmom", "hi52"]),
        "C4": mean(SIGNALS),
        "C7": mean(["ml", "v8", "eps", "divy"]),
    }
    years = panel["dates"].year.to_numpy()
    weights = {}
    for h in (5, 10, 20):
        for tag, win in (("C5", None), ("C6", 3)):
            w = ic_weights(ics[h], h, win)
            weights[f"{tag}_h{h}"] = w
            s = np.zeros_like(R["ml"])
            for k in SIGNALS:
                s += w[k].reindex(years).to_numpy()[:, None] * R[k]
            S[f"{tag}_h{h}"] = s
        S[f"C8_h{h}"] = _neutralise(S[f"C5_h{h}"], panel["aux"])
    return S, weights


# ============================================================================ 3. fast weight-based simulator
def decision_days(dates: pd.DatetimeIndex, reb: int) -> np.ndarray:
    y = dates.year.to_numpy()
    pos = np.zeros(len(dates), dtype=int)
    for yr in np.unique(y):
        m = y == yr
        pos[m] = np.arange(m.sum())
    return pos % reb == 0


def pick(score_row: np.ndarray, held: np.ndarray, n: int, buffer: int, tilt: bool) -> np.ndarray:
    """Target weights from one day's scores (NaN = not eligible); held = boolean of names owned."""
    w = np.zeros_like(score_row)
    elig = np.where(~np.isnan(score_row))[0]
    if len(elig) == 0:
        return w
    order = elig[np.argsort(-score_row[elig], kind="stable")]
    keep = [i for i in order[:buffer] if held[i]][:n]
    ks = set(keep)
    names = keep + [i for i in order if i not in ks][: n - len(keep)]
    names = sorted(names, key=lambda i: -score_row[i])
    if tilt and len(names) > 1:
        lin = 1.5 - np.arange(len(names)) / (len(names) - 1)
        w[names] = lin / lin.sum() * len(names) / n
    else:
        w[names] = 1.0 / n
    return w


def fast_sim(
    panel: dict, scores, sched: list[tuple[np.ndarray, Cfg]] | Cfg, fee: float = FEE, start: int = 0
) -> pd.DataFrame:
    """Proportional-cost book. Returns per-day gross return, turnover (sum |dw|, buys + sells) and net return.

    `sched` is one config or a list of (day mask, config) segments (walk-forward schedule). Decisions at the
    close of decision days, execution at the next open; prices ffilled (no trade at a stale open).
    """
    O, C, L = panel["O"], panel["C"], panel["L"]
    T, K = C.shape
    Cf = pd.DataFrame(C).ffill().to_numpy()
    Of = np.where(np.isnan(O), Cf, O)
    prevC = np.r_[[panel["C_prev0"]], Cf[:-1]]
    prevC = pd.DataFrame(prevC).ffill().to_numpy()
    if isinstance(sched, Cfg):
        sched = [(np.ones(T, bool), sched)]
    cfg_at = np.empty(T, dtype=object)
    for mask, c in sched:
        cfg_at[mask] = c
    dd = {}
    w = np.zeros(K)  # weights at the previous close
    entry = np.full(K, np.nan)
    pending = None
    gross, turn = np.zeros(T), np.zeros(T)
    bC = (
        panel["bench"]["Close"]
        .reindex(panel["dates"].union(panel["bench"].index))
        .ffill()
        .reindex(panel["dates"])
        .to_numpy()
    )
    bprev = np.r_[np.nan, bC[:-1]]
    for t in range(T):
        c = cfg_at[t]
        if c is None:
            continue
        on = np.where(prevC[t] > 0, Of[t] / prevC[t], 1.0)
        on = np.where(np.isfinite(on), on, 1.0)
        intra = np.where(Of[t] > 0, Cf[t] / Of[t], 1.0)
        intra = np.where(np.isfinite(intra), intra, 1.0)
        v = w * on
        cash = 1.0 - w.sum()
        tv = 0.0
        if pending is not None:
            eq = v.sum() + cash
            cur = v / eq
            tgt = pending.copy()
            stale = np.isnan(O[t])
            tgt[stale] = cur[stale]  # no trade at a missing open
            small = (np.abs(tgt - cur) < MIN_TRADE) & (cur > 0) & (tgt > 0)
            tgt[small] = cur[small]
            new = (tgt > 0) & (cur == 0)
            entry[new] = Of[t, new]
            entry[tgt == 0] = np.nan
            tv = np.abs(tgt - cur).sum()
            v, cash = tgt * eq, eq - (tgt * eq).sum()
            pending = None
        v1 = v * intra
        if c.stop:
            hit = (v1 > 0) & (L[t] <= entry * (1 - c.stop))
            if hit.any():  # exit at min(open, stop price)
                px = np.minimum(Of[t, hit], entry[hit] * (1 - c.stop))
                proceeds = v[hit] * px / Of[t, hit]
                tv += proceeds.sum() / (v.sum() + cash)
                cash += proceeds.sum()
                v1[hit] = 0.0
                entry[hit] = np.nan
        eq0 = w.sum() + (1.0 - w.sum())
        eq1 = v1.sum() + cash
        r = eq1 / eq0 - 1.0
        if c.core:
            rb = bC[t] / bprev[t] - 1 if np.isfinite(bprev[t]) else 0.0
            r = (1 - c.core) * r + c.core * rb
            tv = (1 - c.core) * tv
        gross[t], turn[t] = r, tv
        w = v1 / eq1
        if c.reb not in dd:
            dd[c.reb] = decision_days(panel["dates"], c.reb)
        if dd[c.reb][t] and t < T - 1:
            nxt = cfg_at[t + 1]
            cc = (
                nxt
                if (
                    nxt is not None
                    and nxt is not c
                    and panel["dates"][t + 1].year != panel["dates"][t].year
                )
                else c
            )
            s = scores[cc.score][t]
            pending = pick(s, w > 0, cc.n, cc.buffer, cc.tilt)
    out = pd.DataFrame({"gross": gross, "turn": turn}, index=panel["dates"])
    first = next(t for t in range(T) if cfg_at[t] is not None)
    out = out.iloc[first:]
    out["net"] = (1 + out["gross"]) * (1 - fee * out["turn"]) - 1
    return out


def net_at(g: pd.DataFrame, fee: float) -> pd.Series:
    return (1 + g["gross"]) * (1 - fee * g["turn"]) - 1


# ============================================================================ 4. metrics
def capm_t(r: pd.Series, b: pd.Series) -> tuple[float, float]:
    c = evaluate.capm(r, b)
    return c["alpha % p.a."], c["alpha t-stat"]


def ir(r: pd.Series, b: pd.Series) -> float:
    a = (r - b.reindex(r.index)).dropna()
    return float(a.mean() / a.std() * np.sqrt(DAYS))


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    return 100 * ((1 + r).prod() ** (DAYS / len(r)) - 1)


def maxdd(r: pd.Series) -> float:
    c = (1 + r.fillna(0)).cumprod()
    return 100 * float((c / c.cummax() - 1).min())


def full_stats(
    net: pd.Series,
    gross: pd.Series | None,
    turn: pd.Series | None,
    panel: dict,
    trial_srs: list[float] | None = None,
) -> dict:
    b = (
        panel["bench"]["Close"]
        .reindex(net.index.union(panel["dates"]))
        .ffill()
        .pct_change()
        .reindex(net.index)
    )
    out = {
        "CAGR %": cagr(net),
        "TDEX CAGR %": cagr(b),
        "excess CAGR %": cagr(net) - cagr(b),
        "vol %": 100 * net.std() * np.sqrt(DAYS),
        "Sharpe": net.mean() / net.std() * np.sqrt(DAYS),
        "TDEX Sharpe": b.mean() / b.std() * np.sqrt(DAYS),
        "max DD %": maxdd(net),
        "TDEX max DD %": maxdd(b),
        "TE %": 100 * (net - b).std() * np.sqrt(DAYS),
        "IR vs TDEX": ir(net, b),
    }
    a, t = capm_t(net, b)
    out.update({"CAPM alpha %": a, "CAPM alpha t": t, "CAPM beta": evaluate.capm(net, b)["beta"]})
    if gross is not None:
        out["gross CAGR %"] = cagr(gross)
        out["cost drag %"] = out["gross CAGR %"] - out["CAGR %"]
        ga, gt = capm_t(gross, b)
        out.update({"gross CAPM alpha %": ga, "gross CAPM alpha t": gt})
    if turn is not None:
        out["turnover 2-way x/yr"] = float(turn.sum() / (len(turn) / DAYS))
    for tag, r in (("", net), ("gross ", gross)):
        if r is None:
            continue
        ff3 = sv2.ff3_em(r, panel["ff"], panel["usdthb"])
        c4 = factors.carhart_em(r, panel["ff"], panel["usdthb"])
        th = factors.thai_local(r, panel["local"])
        out.update(
            {
                f"{tag}FF3-EM alpha %": ff3["alpha % p.a."],
                f"{tag}FF3-EM alpha t": ff3["alpha t"],
                f"{tag}Carhart-EM alpha %": c4["alpha % p.a."],
                f"{tag}Carhart-EM alpha t": c4["alpha t"],
                f"{tag}TH-Carhart alpha %": th["alpha % p.a."],
                f"{tag}TH-Carhart alpha t": th["alpha t"],
            }
        )
        if not tag:
            out.update(
                {
                    "C4 b_Mkt": c4["b_Mkt-RF"],
                    "C4 b_SMB": c4["b_SMB"],
                    "C4 b_HML": c4["b_HML"],
                    "C4 b_WML": c4["b_WML"],
                    "TH b_MKT": th["b_MKT"],
                    "TH b_SMB": th["b_SMB"],
                    "TH t_SMB": th["t_SMB"],
                    "TH b_WML": th["b_WML"],
                    "TH t_WML": th["t_WML"],
                }
            )
    thn = factors.thai_local(net, panel["local_net"])
    out.update(
        {
            "TH-Carhart (net factors) alpha %": thn["alpha % p.a."],
            "TH-Carhart (net factors) alpha t": thn["alpha t"],
        }
    )
    out["vs EW top-100 % p.a."] = 100 * (net - panel["ew"].reindex(net.index)).mean() * DAYS
    out["PSR"] = evaluate.psr(net)
    if trial_srs:
        out["DSR"] = evaluate.dsr(net, trial_srs)
    out["Sharpe t"] = validity.sharpe_tstat(net)
    out["MinTRL yrs"] = validity.min_track_record(net)
    out["years"] = len(net) / DAYS
    return out


# ============================================================================ stage: grid
def stage_grid() -> None:
    panel = joblib.load(PANEL)
    t0 = time.time()
    ics = ic_tables(panel)
    scores, weights = build_scores(panel, ics)
    print(f"ICs + scores {time.time() - t0:.0f}s", flush=True)
    res = {}
    for stop in (0.0, 0.08):
        for c in grid(stop):
            res[c.key] = fast_sim(panel, scores, c)[["gross", "turn"]].astype("float32")
        print(f"grid stop={stop}: {len(res)} configs, {time.time() - t0:.0f}s", flush=True)
    joblib.dump({"ics": ics, "weights": weights, "res": res}, GRID, compress=3)
    ic_year = pd.concat({h: ics[h].groupby(ics[h].index.year).mean() for h in ics}, axis=1)
    ic_year.to_csv(OUT / "signal_ic_by_year.csv")
    pd.concat(weights, axis=0).to_csv(OUT / "ic_weights_by_year.csv")
    print(ic_year.round(3).to_string())


# ============================================================================ stage: walk-forward
def select(
    cfgs: list[Cfg],
    res: dict,
    bench_r: pd.Series,
    Y: int,
    objective: str = "alpha_t",
    fee: float = FEE,
) -> tuple[Cfg, pd.DataFrame]:
    end = pd.Timestamp(f"{Y - 1}-12-31")
    rows = {}
    for c in cfgs:
        g = res[c.key].loc[:end]
        n = net_at(g, fee)
        to = float(g["turn"].sum() / (len(g) / DAYS))
        b = bench_r.reindex(n.index)
        if objective == "alpha_t":
            val = capm_t(n, b)[1]
        else:
            val = ir(n, b)
        rows[c.key] = {"score": val, "turnover": to, "eligible": to <= MAX_TURNOVER}
    t = pd.DataFrame(rows).T
    el = t[t["eligible"].astype(bool)]
    best = el["score"].astype(float).idxmax()
    return next(c for c in cfgs if c.key == best), t


def walk_forward(panel, scores, res, cfgs, objective="alpha_t", fee=FEE):
    dates = panel["dates"]
    bench_r = panel["bench"]["Close"].reindex(dates).ffill().pct_change()
    picks, tables = {}, {}
    for Y in WF_YEARS:
        c, t = select(cfgs, res, bench_r, Y, objective, fee)
        picks[Y], tables[Y] = c, t
    sched = [(np.asarray(dates.year == Y), picks[Y]) for Y in WF_YEARS]
    return picks, tables, sched


# ---------------------------------------------------------------- exact lot-based simulator
def exact_sim(
    panel: dict,
    scores: dict,
    sched: list[tuple[np.ndarray, Cfg]],
    fee: float = FEE,
    tfee: float = TFEE,
    capital: float = config.INITIAL_CAPITAL,
    lot: int = sv2.LOT,
) -> pd.DataFrame:
    """THB `capital`, 100-share lots, decisions at the close, next-open fills, sells first, costs per side, TDEX
    core in whole lots, stop on the daily low. Returns equity, stock traded value and TDEX traded value per day."""
    dates, O, C, L = panel["dates"], panel["O"], panel["C"], panel["L"]
    T, K = C.shape
    Cf = pd.DataFrame(C).ffill().to_numpy()
    bO = panel["bench"]["Open"].reindex(dates).to_numpy()
    bC = pd.Series(panel["bench"]["Close"].reindex(dates).to_numpy()).ffill().to_numpy()
    cfg_at = np.empty(T, dtype=object)
    for mask, c in sched:
        cfg_at[mask] = c
    dd = {}
    cash, sh, tu = capital, np.zeros(K), 0
    entry = np.full(K, np.nan)
    pending = None
    eq, traded, ttraded = np.full(T, np.nan), np.zeros(T), np.zeros(T)
    for t in range(T):
        c = cfg_at[t]
        if c is None:
            continue
        if pending is not None:
            tgt, core = pending
            po = np.where(np.isnan(O[t]), np.nan, O[t])
            pb = bO[t] if np.isfinite(bO[t]) else bC[t]
            mark = np.where(np.isfinite(po), po, Cf[t])
            value = cash + np.nansum(sh * mark) + tu * pb
            budget = value * 0.998
            want = np.where(
                np.isfinite(po) & (po > 0),
                np.floor(budget * tgt / np.where(po > 0, po, 1) / lot) * lot,
                sh,
            )
            dq = want - sh
            notional = np.abs(dq) * np.nan_to_num(po)
            cur_w = sh * mark / value
            small = (notional < MIN_TRADE * value) & (want > 0) & (sh > 0)
            dq[small] = 0
            dq[~np.isfinite(po)] = 0
            # sells first (stocks), then TDEX sells, then buys, then TDEX buys with the cash left
            for side in (-1, 1):
                idx = np.where(np.sign(dq) == side)[0]
                if side == -1:
                    cash += (-dq[idx] * po[idx]).sum() - (np.abs(dq[idx]) * po[idx]).sum() * fee
                    traded[t] += (np.abs(dq[idx]) * po[idx]).sum()
                    sh[idx] += dq[idx]
                    wu = int(core * budget / pb / lot) * lot if pb > 0 else tu
                    if wu < tu:
                        cash += (tu - wu) * pb * (1 - tfee)
                        ttraded[t] += (tu - wu) * pb
                        tu = wu
                else:
                    cost = dq[idx] * po[idx] * (1 + fee)
                    if cost.sum() > cash and cost.sum() > 0:  # scale buys down to available cash
                        f = max(cash, 0) / cost.sum()
                        dq[idx] = np.floor(dq[idx] * f / lot) * lot
                        cost = dq[idx] * po[idx] * (1 + fee)
                    cash -= cost.sum()
                    traded[t] += (dq[idx] * po[idx]).sum()
                    newb = idx[(sh[idx] == 0) & (dq[idx] > 0)]
                    entry[newb] = po[newb]
                    sh[idx] += dq[idx]
                    if core > 0 and pb > 0:
                        wu = int(core * budget / pb / lot) * lot
                        spare = int(max(cash - 0.002 * value, 0) / (pb * (1 + tfee)) / lot) * lot
                        wu = min(wu, tu + spare)
                        if wu > tu:
                            cash -= (wu - tu) * pb * (1 + tfee)
                            ttraded[t] += (wu - tu) * pb
                            tu = wu
            entry[sh == 0] = np.nan
            pending = None
        if c.stop:
            stop_px = entry * (1 - c.stop)
            hit = (sh > 0) & np.isfinite(L[t]) & (L[t] <= stop_px)
            if hit.any():
                fill = np.where(
                    np.isfinite(O[t, hit]), np.minimum(O[t, hit], stop_px[hit]), stop_px[hit]
                )
                cash += (sh[hit] * fill).sum() * (1 - fee)
                traded[t] += (sh[hit] * fill).sum()
                sh[hit] = 0
                entry[hit] = np.nan
        value = cash + np.nansum(sh * Cf[t]) + tu * bC[t]
        eq[t] = value
        if c.reb not in dd:
            dd[c.reb] = decision_days(dates, c.reb)
        if dd[c.reb][t] and t < T - 1:
            nxt = cfg_at[t + 1]
            cc = (
                nxt
                if (nxt is not None and nxt is not c and dates[t + 1].year != dates[t].year)
                else c
            )
            w_held = sh * Cf[t] / value
            core = cc.core
            tgt = pick(scores[cc.score][t], sh > 0, cc.n, cc.buffer, cc.tilt) * (1 - core)
            pending = (tgt, core)
            _ = w_held
    out = pd.DataFrame(
        {"equity": eq, "traded": traded, "tdex_traded": ttraded}, index=dates
    ).dropna(subset=["equity"])
    out["ret"] = out["equity"].pct_change()
    out.iloc[0, out.columns.get_loc("ret")] = out["equity"].iloc[0] / capital - 1
    return out


def prior_srs() -> list[float]:
    s = v3.prior_trial_srs()
    return s + [0.0] * max(0, PRIOR_TRIALS - len(s))


def cscv(R: pd.DataFrame, bench: pd.Series, n_blocks: int = 16, metric: str = "sharpe") -> dict:
    """CSCV PBO; metric 'sharpe' (validity.pbo_cscv) or 'alpha_t' (CAPM alpha / its IID s.e., from block sums)."""
    d = R.join(bench.rename("_b")).dropna()
    R, bench = d.drop(columns="_b"), d["_b"]
    if metric == "sharpe":
        return validity.pbo_cscv(R, n_blocks)
    X = R.to_numpy()
    b = bench.to_numpy()
    T, N = X.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    S = {k: [] for k in ("n", "x", "xx", "b", "bb", "xb")}
    for bl in blocks:
        S["n"].append(len(bl))
        S["x"].append(X[bl].sum(0))
        S["xx"].append((X[bl] ** 2).sum(0))
        S["b"].append(b[bl].sum())
        S["bb"].append((b[bl] ** 2).sum())
        S["xb"].append((X[bl] * b[bl, None]).sum(0))
    S = {k: np.array(v) for k, v in S.items()}

    def at(idx):
        n = S["n"][idx].sum()
        mx, mb = S["x"][idx].sum(0) / n, S["b"][idx].sum() / n
        vb = S["bb"][idx].sum() / n - mb**2
        cxb = S["xb"][idx].sum(0) / n - mx * mb
        beta = cxb / vb
        alpha = mx - beta * mb
        vx = S["xx"][idx].sum(0) / n - mx**2
        resid_var = np.maximum(vx - beta**2 * vb, 1e-18)
        return alpha / np.sqrt(resid_var / n * (1 + mb**2 / vb))

    logits, oos_best_pct = [], []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        ins = np.array(combo)
        oos = np.setdiff1d(np.arange(n_blocks), ins)
        a_is, a_oos = at(ins), at(oos)
        best = int(np.argmax(a_is))
        rank = (a_oos <= a_oos[best]).sum()
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        oos_best_pct.append(rank / N)
    logits = np.array(logits)
    return {
        "PBO": float((logits <= 0).mean()),
        "trials N": N,
        "days T": T,
        "splits": len(logits),
        "median logit": float(np.median(logits)),
        "mean OOS percentile of IS winner": float(np.mean(oos_best_pct)),
    }


def md(t: pd.DataFrame, digits: int = 2) -> str:
    return sv2.to_md(t, digits)


def stage_wf() -> None:
    t0 = time.time()
    panel = joblib.load(PANEL)
    G = joblib.load(GRID)
    ics, res = G["ics"], G["res"]
    scores, weights = build_scores(panel, ics)
    dates = panel["dates"]
    bench_r = panel["bench"]["Close"].reindex(dates).ffill().pct_change()
    cfgs = grid(0.0)
    cfgs_stop = cfgs + grid(0.08)
    oos = slice(pd.Timestamp("2016-01-01"), None)
    n_trials = PRIOR_TRIALS + len(cfgs_stop) + 6  # grid incl. stop extension + meta / overlays
    srs_grid = [
        float(net_at(res[c.key], FEE).loc[oos].pipe(lambda r: r.mean() / r.std()))
        for c in cfgs_stop
    ]
    trial_srs = prior_srs() + srs_grid + [0.0] * 6

    report, tables = {}, {}

    # ---------------- primary walk-forward
    picks, sel_tabs, sched = walk_forward(panel, scores, res, cfgs)
    wf = fast_sim(panel, scores, sched)
    wf_oos = wf.loc[oos]
    stitched = pd.concat([net_at(res[picks[Y].key], FEE).loc[str(Y)] for Y in WF_YEARS])
    # exact confirmation
    ex = exact_sim(panel, scores, sched)
    exg = exact_sim(panel, scores, sched, fee=0.0, tfee=0.0)
    ex_turn = ex["traded"] / ex["equity"].shift(1).fillna(config.INITIAL_CAPITAL)
    rows = {}
    rows["WF fast (net)"] = full_stats(
        wf_oos["net"], wf_oos["gross"], wf_oos["turn"], panel, trial_srs
    )
    rows["WF exact lots (net)"] = full_stats(
        ex["ret"].loc[oos], exg["ret"].loc[oos], ex_turn.loc[oos], panel, trial_srs
    )
    b_oos = bench_r.loc[oos].dropna()
    rows["TDEX buy & hold"] = full_stats(b_oos, None, None, panel)
    for k in (
        "CAPM alpha %",
        "CAPM alpha t",
        "TH-Carhart alpha %",
        "TH-Carhart alpha t",
        "IR vs TDEX",
        "TH-Carhart (net factors) alpha %",
        "TH-Carhart (net factors) alpha t",
        "TH b_SMB",
        "TH b_WML",
    ):
        rows["TDEX buy & hold"][k] = np.nan  # regression of TDEX on itself is degenerate
    sub = {}
    for per, sl in (
        ("2016-21", slice("2016-01-01", "2021-12-31")),
        ("2022-26", slice("2022-01-01", None)),
    ):
        for name, (n_, g_, t_) in {
            "WF fast": (wf["net"], wf["gross"], wf["turn"]),
            "WF exact lots": (ex["ret"], exg["ret"], ex_turn),
        }.items():
            sub[f"{name} {per}"] = full_stats(n_.loc[sl], g_.loc[sl], t_.loc[sl], panel)
        bb = bench_r.loc[sl].dropna()
        sub[f"TDEX {per}"] = {
            "CAGR %": cagr(bb),
            "Sharpe": bb.mean() / bb.std() * np.sqrt(DAYS),
            "max DD %": maxdd(bb),
        }
    sub_t = pd.DataFrame(sub).T
    sub_t.to_csv(OUT / "wf_subperiods.csv")
    rows["EW tradable top-100 (no costs)"] = full_stats(
        panel["ew"].loc[oos].dropna(), None, None, panel
    )
    print(
        f"WF done {time.time() - t0:.0f}s; fast vs exact daily corr "
        f"{wf_oos['net'].corr(ex['ret'].loc[oos]):.4f}; stitched-vs-continuous CAGR {cagr(stitched):.2f} vs "
        f"{cagr(wf_oos['net']):.2f}",
        flush=True,
    )

    # per-year picks and OOS returns
    yr = []
    for Y in WF_YEARS:
        r = wf.loc[str(Y), "net"]
        g = wf.loc[str(Y), "gross"]
        b = bench_r.loc[str(Y)]
        st = sel_tabs[Y]
        el = st[st["eligible"].astype(bool)]
        best = el["score"].astype(float)
        ex_r = ex["ret"].loc[str(Y)]
        yr.append(
            {
                "year": Y,
                "pick": picks[Y].key,
                "IS alpha t (2013..Y-1)": best.max(),
                "2nd best IS t": best.nlargest(2).iloc[-1],
                "eligible configs": len(el),
                "OOS net %": 100 * ((1 + r).prod() - 1),
                "OOS gross %": 100 * ((1 + g).prod() - 1),
                "OOS exact net %": 100 * ((1 + ex_r).prod() - 1),
                "TDEX %": 100 * ((1 + b.dropna()).prod() - 1),
                "OOS - TDEX pp": 100 * ((1 + r).prod() - (1 + b.dropna()).prod()),
                "turnover 2-way x": float(wf.loc[str(Y), "turn"].sum()),
                "pick's OOS percentile in grid": _oos_pct(res, cfgs, picks[Y], Y, bench_r),
            }
        )
    yrt = pd.DataFrame(yr).set_index("year")
    yrt.to_csv(OUT / "wf_per_year.csv")

    # ---------------- ablations / overlays (pre-declared)
    abl = {}
    for core in (0.5, 0.7):
        sc = [(m, replace(c, core=core)) for m, c in sched]
        f2 = fast_sim(panel, scores, sc).loc[oos]
        abl[f"WF + {int(100 * core)}% TDEX core"] = full_stats(
            f2["net"], f2["gross"], f2["turn"], panel, trial_srs
        )
    p_ir, _, s_ir = walk_forward(panel, scores, res, cfgs, objective="ir")
    f_ir = fast_sim(panel, scores, s_ir).loc[oos]
    abl["WF, objective = IR vs TDEX"] = full_stats(
        f_ir["net"], f_ir["gross"], f_ir["turn"], panel, trial_srs
    )
    p_st, _, s_st = walk_forward(panel, scores, res, cfgs_stop)
    f_st = fast_sim(panel, scores, s_st).loc[oos]
    abl["WF over 576 configs (incl. 8% stop)"] = full_stats(
        f_st["net"], f_st["gross"], f_st["turn"], panel, trial_srs
    )
    sched_same_stop = [(m, replace(c, stop=0.08)) for m, c in sched]
    f_ss = fast_sim(panel, scores, sched_same_stop).loc[oos]
    abl["WF picks + 8% stop"] = full_stats(
        f_ss["net"], f_ss["gross"], f_ss["turn"], panel, trial_srs
    )
    # composite restricted WFs (which signal family carries it?)
    for fam, comps in {
        "WF, ML composites only (C1, C2)": ("C1", "C2"),
        "WF, literature only (C3)": ("C3",),
        "WF, IC-weighted only (C5, C6, C8)": ("C5", "C6", "C8"),
    }.items():
        sub = [c for c in cfgs if c.comp.split()[0] in comps]
        _, _, s_f = walk_forward(panel, scores, res, sub)
        f_f = fast_sim(panel, scores, s_f).loc[oos]
        abl[fam] = full_stats(f_f["net"], f_f["gross"], f_f["turn"], panel, trial_srs)
    picks_alt = pd.DataFrame(
        {
            "IR objective": {Y: p_ir[Y].key for Y in WF_YEARS},
            "576 grid incl. stop": {Y: p_st[Y].key for Y in WF_YEARS},
        }
    )

    # ---------------- cost sensitivity: (a) same picks, other costs; (b) re-optimised at each cost
    cs = {}
    for fee in (0.0005, 0.0010, 0.0015):
        a = net_at(wf, fee).loc[oos]
        pk, _, sc = walk_forward(panel, scores, res, cfgs, fee=fee)
        fb = fast_sim(panel, scores, sc, fee=fee).loc[oos]
        for tag, r in (("same picks", a), ("re-optimised", fb["net"])):
            b = bench_r.reindex(r.index)
            al, at_ = capm_t(r, b)
            th = factors.thai_local(r, panel["local"])
            c4 = factors.carhart_em(r, panel["ff"], panel["usdthb"])
            cs[(f"{100 * fee:.2f}%/side", tag)] = {
                "CAGR %": cagr(r),
                "TDEX CAGR %": cagr(b),
                "IR vs TDEX": ir(r, b),
                "CAPM alpha %": al,
                "CAPM alpha t": at_,
                "TH-Carhart alpha %": th["alpha % p.a."],
                "TH-Carhart alpha t": th["alpha t"],
                "Carhart-EM alpha t": c4["alpha t"],
            }
        cs[(f"{100 * fee:.2f}%/side", "re-optimised")]["picks"] = "; ".join(
            f"{Y}:{pk[Y].key}" for Y in WF_YEARS
        )
    cst = pd.DataFrame(cs).T

    # ---------------- in-hindsight best (overfit, comparison only)
    full = {c.key: net_at(res[c.key], FEE) for c in cfgs}
    cg = pd.Series({k: cagr(v) for k, v in full.items()}).sort_values(ascending=False)
    hb = next(c for c in cfgs if c.key == cg.index[0])
    at_full = pd.Series(
        {k: capm_t(v, bench_r.reindex(v.index))[1] for k, v in full.items()}
    ).sort_values(ascending=False)
    hb_t = next(c for c in cfgs if c.key == at_full.index[0])
    hind = {}
    for tag, c in (
        (f"HINDSIGHT best net CAGR 2013-26: {hb.key}", hb),
        (f"HINDSIGHT best net alpha t 2013-26: {hb_t.key}", hb_t),
    ):
        g = res[c.key]
        for per, sl in (("2013-26", slice(None)), ("2016-26", oos)):
            gg = g.loc[sl]
            hind[f"{tag} [{per}]"] = full_stats(
                net_at(gg, FEE), gg["gross"], gg["turn"], panel, trial_srs
            )
        exh = exact_sim(panel, scores, [(np.ones(len(dates), bool), c)])
        exhg = exact_sim(panel, scores, [(np.ones(len(dates), bool), c)], fee=0.0, tfee=0.0)
        hind[f"{tag} [2016-26, exact lots]"] = full_stats(
            exh["ret"].loc[oos],
            exhg["ret"].loc[oos],
            (exh["traded"] / exh["equity"].shift(1)).loc[oos],
            panel,
            trial_srs,
        )
    top10 = pd.DataFrame(
        {
            "net CAGR 2013-26 %": cg.head(10),
            "net CAPM alpha t 2013-26": at_full.reindex(cg.head(10).index),
        }
    )

    # ---------------- PBO over the grid (CSCV) and grid summary
    R = pd.DataFrame({c.key: net_at(res[c.key], FEE) for c in cfgs}).dropna()
    pbo = {
        "Sharpe, 2013-26": cscv(R, bench_r, metric="sharpe"),
        "CAPM alpha t, 2013-26": cscv(R, bench_r, metric="alpha_t"),
        "CAPM alpha t, 2016-26": cscv(R.loc[oos], bench_r, metric="alpha_t"),
    }
    pbot = pd.DataFrame(pbo).T
    gsum = []
    for c in cfgs:
        g = res[c.key].loc[oos]
        n = net_at(g, FEE)
        gsum.append(
            {
                "key": c.key,
                "comp": c.comp,
                "N": c.n,
                "reb": c.reb,
                "buffer": c.buf,
                "tilt": c.tilt,
                "net CAGR 16-26 %": cagr(n),
                "gross CAGR 16-26 %": cagr(g["gross"]),
                "CAPM alpha t 16-26": capm_t(n, bench_r.reindex(n.index))[1],
                "turnover 2-way": float(g["turn"].sum() / (len(g) / DAYS)),
            }
        )
    gsum = pd.DataFrame(gsum).set_index("key")
    gsum.to_csv(OUT / "grid_2016_2026.csv")
    frac_beat = float((gsum["net CAGR 16-26 %"] > cagr(b_oos)).mean())
    t22 = bench_r.loc["2022-01-01":].dropna()
    g22 = pd.Series({c.key: cagr(net_at(res[c.key].loc["2022-01-01":], FEE)) for c in cfgs})
    gdist = pd.DataFrame(
        {
            "2016-26": gsum["net CAGR 16-26 %"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
            "2022-26": g22.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]),
        }
    ).T
    gdist["TDEX CAGR %"] = [cagr(b_oos), cagr(t22)]
    gdist["share beating TDEX"] = [frac_beat, float((g22 > cagr(t22)).mean())]
    gdist["WF net CAGR %"] = [cagr(wf_oos["net"]), cagr(wf["net"].loc["2022-01-01":])]
    gdist["WF percentile in grid"] = [
        float((gsum["net CAGR 16-26 %"] <= cagr(wf_oos["net"])).mean()),
        float((g22 <= cagr(wf["net"].loc["2022-01-01":])).mean()),
    ]
    by_comp = gsum.groupby("comp")[
        ["net CAGR 16-26 %", "gross CAGR 16-26 %", "CAPM alpha t 16-26", "turnover 2-way"]
    ].median()

    # DSR of the grid's best 2016-26 config (multiple-testing view)
    main = pd.DataFrame(rows).T
    abl_t = pd.DataFrame(abl).T
    hind_t = pd.DataFrame(hind).T
    main.to_csv(OUT / "wf_oos_2016_2026.csv")
    abl_t.to_csv(OUT / "wf_ablations.csv")
    hind_t.to_csv(OUT / "hindsight_overfit.csv")
    cst.to_csv(OUT / "cost_sensitivity.csv")
    pbot.to_csv(OUT / "pbo.csv")
    picks_alt.to_csv(OUT / "wf_alt_picks.csv")
    pd.DataFrame(
        {
            "wf_fast_net": wf["net"],
            "wf_fast_gross": wf["gross"],
            "wf_turn": wf["turn"],
            "wf_exact_net": ex["ret"],
            "wf_exact_gross": exg["ret"],
            "tdex": bench_r,
        }
    ).to_parquet(OUT / "wf_daily.parquet")
    weights_tab = pd.concat(
        {k: v for k, v in weights.items() if k in ("C5_h5", "C5_h20", "C6_h5")}, axis=1
    )
    report = {
        "n_trials": n_trials,
        "frac_grid_beating_tdex_cagr_16_26": frac_beat,
        "fast_exact_corr": float(wf_oos["net"].corr(ex["ret"].loc[oos])),
        "stitched_cagr": cagr(stitched),
        "local_factor_cost_pa": panel["local_net"].attrs.get("cost_pa"),
        "cover": panel["cover"],
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=1, default=str))

    KEY = [
        "CAGR %",
        "TDEX CAGR %",
        "gross CAGR %",
        "cost drag %",
        "turnover 2-way x/yr",
        "Sharpe",
        "max DD %",
        "IR vs TDEX",
        "CAPM alpha %",
        "CAPM alpha t",
        "gross CAPM alpha %",
        "gross CAPM alpha t",
        "CAPM beta",
        "FF3-EM alpha %",
        "FF3-EM alpha t",
        "gross FF3-EM alpha t",
        "Carhart-EM alpha %",
        "Carhart-EM alpha t",
        "gross Carhart-EM alpha t",
        "TH-Carhart alpha %",
        "TH-Carhart alpha t",
        "gross TH-Carhart alpha %",
        "gross TH-Carhart alpha t",
        "TH-Carhart (net factors) alpha %",
        "TH-Carhart (net factors) alpha t",
        "TH b_SMB",
        "TH b_WML",
        "vs EW top-100 % p.a.",
        "PSR",
        "DSR",
        "MinTRL yrs",
    ]
    txt = [
        "# integrated_v1 tables (auto-generated)",
        "",
        f"trials counted for DSR: {n_trials} (89 earlier + 576 grid incl. stop extension + 6 meta/overlays)",
        "",
        "## Walk-forward OOS 2016-01 -> 2026-10",
        md(main[[k for k in KEY if k in main.columns]].T.astype(float)),
        "",
        "## Sub-periods",
        md(sub_t[[k for k in KEY if k in sub_t.columns]].T.astype(float)),
        "",
        "## Per-year picks",
        md(yrt),
        "",
        "## Ablations / overlays (2016-26)",
        md(abl_t[[k for k in KEY if k in abl_t.columns]].T.astype(float)),
        "",
        "## Alternative WF picks",
        md(picks_alt),
        "",
        "## Cost sensitivity",
        md(cst),
        "",
        "## IN-SAMPLE / OVERFIT (hindsight, comparison only)",
        md(hind_t[[k for k in KEY if k in hind_t.columns]].T.astype(float)),
        "",
        "### top-10 configs by net CAGR 2013-26 (hindsight)",
        md(top10),
        "",
        "## PBO (CSCV, 16 blocks, 288 configs)",
        md(pbot.astype(float)),
        "",
        f"share of the 288 configs beating TDEX net CAGR 2016-26: {frac_beat:.2%}",
        "",
        "## distribution of net CAGR over the 288 configs (diagnostic, not selectable)",
        md(gdist),
        "",
        "## grid medians by composite (2016-26)",
        md(by_comp),
        "",
        "## IC weights (C5 expanding h5 / h20, C6 rolling-3y h5)",
        md(weights_tab),
        "",
        f"summary: {json.dumps(report, default=str)}",
    ]
    (OUT / "tables.md").write_text("\n".join(txt))
    print("\n".join(txt))


def _oos_pct(res, cfgs, c, Y, bench_r) -> float:
    vals = {}
    for k in cfgs:
        r = net_at(res[k.key].loc[str(Y)], FEE)
        vals[k.key] = (1 + r).prod()
    s = pd.Series(vals)
    return float((s <= s[c.key]).mean())


if __name__ == "__main__":
    {"panel": build_panel, "grid": stage_grid, "wf": stage_wf}[sys.argv[1]]()
