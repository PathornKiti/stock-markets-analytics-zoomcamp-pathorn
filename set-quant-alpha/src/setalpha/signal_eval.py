"""Signal research tear-sheet (alphalens-style) for the ensemble score.

What a buy-side research review asks of an alpha signal before it looks at any backtest:

* **IC** - daily Spearman rank correlation between the score at the close of t and the realised
  forward alpha; its mean, std, **ICIR** = mean / std, and t-stats that respect overlapping targets
  (Newey-West, and every h-th day only = non-overlapping);
* **IC decay** - IC against alpha over 1, 5, 10, 20 days, and over the later 5-day windows
  (days 6-10, 11-15, 16-20): how fast the information is used up;
* **quantile spread** - mean forward alpha of each daily score decile and the D10 - D1 spread;
* **factor-neutral IC / Fama-MacBeth** - does the score still predict alpha after controlling for
  size, 1-week and 1-month reversal, 12-1 momentum, idiosyncratic volatility and beta?
* **signal persistence** - rank autocorrelation and top-decile turnover (what the portfolio must trade);
* **fundamental law** (Grinold) - IR ~ IC x sqrt(breadth) x TC, with the transfer coefficient TC
  (Clarke, de Silva & Thorley, 2002) of the actual portfolio: corr(active weight x sigma, alpha / sigma).

No look-ahead: forward alphas are labels only; every control and score is known at the close of t,
and forward windows start at the next open (as the training target does).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from setalpha import config
from setalpha.validity import nw_tstat

HORIZONS = (1, 5, 10, 20)
LATER_WINDOWS = (5, 10, 15)  # alpha over days k+1 .. k+5 after the signal
CONTROLS = {
    "size": "log_dvol60",
    "reversal 1w": "ret_5",
    "reversal 1m": "ret_20",
    "momentum 12-1": "mom_12_1",
    "idio vol": "idio_vol",
    "beta": "beta_mkt",
}
MIN_NAMES = 20


# ----------------------------------------------------------------------------- forward alphas
def forward_alphas(
    prices: pd.DataFrame, df: pd.DataFrame, start: str = config.VALID_START
) -> pd.DataFrame:
    """Forward alphas r - beta x EW-liquid-market over several horizons + 12-1 momentum, long format.

    Same definition as the training target `fwd_alpha_5` (buy the next open, sell the close t+h).
    Later windows (days k+1..k+5) run close-to-close from t+k.
    """
    d = df[df["Date"] >= pd.Timestamp(start) - pd.Timedelta(days=400)]
    beta = d.set_index(["Date", "Ticker"])["beta_mkt"].unstack()
    liquid = d.set_index(["Date", "Ticker"])["liquid"].unstack().fillna(False).astype(bool)
    p = prices[prices["Date"].isin(beta.index) & prices["Ticker"].isin(beta.columns)]
    C = p.pivot_table(index="Date", columns="Ticker", values="Close").reindex(
        index=beta.index, columns=beta.columns
    )
    O = p.pivot_table(index="Date", columns="Ticker", values="Open").reindex(
        index=beta.index, columns=beta.columns
    )  # noqa: E741

    def alpha(ret: pd.DataFrame) -> pd.DataFrame:
        ret = ret.where(ret.abs() < 1.0)
        return ret - beta.mul(ret.where(liquid).mean(axis=1), axis=0)

    out = {f"fwd_alpha_{h}": alpha(C.shift(-h) / O.shift(-1) - 1) for h in HORIZONS}
    for k in LATER_WINDOWS:
        out[f"alpha_d{k + 1}_{k + 5}"] = alpha(C.shift(-(k + 5)) / C.shift(-k) - 1)
    out["mom_12_1"] = C.shift(21) / C.shift(252) - 1  # known at t (skips the last month)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in out.items()}, axis=1)
    long.index.names = ["Date", "Ticker"]
    return long[long.index.get_level_values("Date") >= start].reset_index()


def research_frame(
    preds: pd.DataFrame, df: pd.DataFrame, fwd: pd.DataFrame, split: str = "test"
) -> pd.DataFrame:
    """Tradable rows of one split: score + forward alphas + the Fama-MacBeth controls."""
    cols = ["Date", "Ticker", "tradable", "idio_vol", "beta_mkt", "log_dvol60", "ret_5", "ret_20"]
    f = preds.loc[preds["split"] == split, ["Date", "Ticker", "pred_ens"]].merge(
        df[cols], on=["Date", "Ticker"]
    )
    f = f[f["tradable"]].merge(fwd, on=["Date", "Ticker"], how="left")
    return f.drop(columns="tradable").reset_index(drop=True)


# ----------------------------------------------------------------------------- IC
def _cs_corr(d: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Per-date Pearson correlation of columns a and b (vectorised)."""
    g = d.groupby("Date")
    xa, xb = d[a] - g[a].transform("mean"), d[b] - g[b].transform("mean")
    num = (xa * xb).groupby(d["Date"]).sum()
    den = np.sqrt((xa**2).groupby(d["Date"]).sum() * (xb**2).groupby(d["Date"]).sum())
    return (num / den)[g.size() >= MIN_NAMES]


def daily_ic(f: pd.DataFrame, score: str = "pred_ens", target: str = "fwd_alpha_5") -> pd.Series:
    """Spearman IC per date = Pearson correlation of the within-date ranks."""
    d = f[["Date", score, target]].dropna()
    r = d.groupby("Date")[[score, target]].rank()
    r["Date"] = d["Date"]
    return _cs_corr(r, score, target)


def ic_summary(ic: pd.Series, horizon: int = config.HORIZON) -> dict:
    """IC mean / std / ICIR with overlap-robust t-stats (Newey-West and non-overlapping)."""
    ic = ic.dropna()
    non = ic.iloc[::horizon]
    icir_period = non.mean() / non.std()
    return {
        "IC mean": ic.mean(),
        "IC std": ic.std(),
        "ICIR (daily obs)": ic.mean() / ic.std(),
        f"ICIR ({horizon}d non-overlapping)": icir_period,
        "ICIR annualised": icir_period * np.sqrt(252 / horizon),
        "t (Newey-West)": nw_tstat(ic, lags=horizon),
        "t (non-overlapping)": icir_period * np.sqrt(len(non)),
        "t (naive, overlapping)": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "IC > 0 % of days": 100 * float((ic > 0).mean()),
        "days": len(ic),
    }


def ic_by_year(ic: pd.Series, horizon: int = config.HORIZON) -> pd.DataFrame:
    return pd.DataFrame({y: ic_summary(g, horizon) for y, g in ic.groupby(ic.index.year)}).T[
        [
            "IC mean",
            "IC std",
            f"ICIR ({horizon}d non-overlapping)",
            "t (Newey-West)",
            "IC > 0 % of days",
            "days",
        ]
    ]


def ic_decay(f: pd.DataFrame, score: str = "pred_ens") -> pd.DataFrame:
    """IC against cumulative horizons and later 5-day windows; t-stats use horizon-matched NW lags."""
    rows = {}
    for h in HORIZONS:
        s = ic_summary(daily_ic(f, score, f"fwd_alpha_{h}"), h)
        rows[f"days 1-{h}" if h > 1 else "day 1"] = {
            "horizon": h,
            **{k: s[k] for k in ["IC mean", "t (Newey-West)"]},
            "IC per day of horizon": s["IC mean"] / np.sqrt(h),
        }
    for k in LATER_WINDOWS:
        s = ic_summary(daily_ic(f, score, f"alpha_d{k + 1}_{k + 5}"), k + 5)
        rows[f"days {k + 1}-{k + 5}"] = {
            "horizon": k + 5,
            "IC mean": s["IC mean"],
            "t (Newey-West)": s["t (Newey-West)"],
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- quantiles
def quantile_alpha(
    f: pd.DataFrame, score: str = "pred_ens", target: str = "fwd_alpha_5", q: int = 10
) -> pd.DataFrame:
    """Mean forward alpha (%) of each daily score quantile; the last row is the top - bottom spread."""
    d = f[["Date", score, target]].dropna().copy()
    d["q"] = (
        np.ceil(d.groupby("Date")[score].rank(method="first", pct=True) * q).clip(1, q).astype(int)
    )
    by = d.groupby(["Date", "q"])[target].mean().unstack() * 100
    table = pd.DataFrame(
        {
            "mean alpha %": by.mean(),
            "t (NW)": [nw_tstat(by[c], 5) for c in by.columns],
            "hit rate %": 100 * (by > 0).mean(),
        }
    )
    spread = by[q] - by[1]
    table.loc[f"D{q}-D1"] = [spread.mean(), nw_tstat(spread, 5), 100 * float((spread > 0).mean())]
    table.index = [f"D{i}" if isinstance(i, int | np.integer) else i for i in table.index]
    return table


def monotonicity(qtable: pd.DataFrame) -> float:
    """Spearman correlation of decile number and decile mean alpha (1 = perfectly monotone)."""
    m = qtable.loc[[i for i in qtable.index if "-" not in i], "mean alpha %"]
    return float(pd.Series(range(len(m))).corr(pd.Series(m.to_numpy()), method="spearman"))


# ----------------------------------------------------------------------------- factor-neutral
def _z(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Within-date rank -> centred, unit-variance score; missing values set to the median (0)."""
    r = d.groupby("Date")[cols].rank(pct=True) - 0.5
    return (r / r.groupby(d["Date"]).transform("std")).fillna(0.0)


def fama_macbeth(
    f: pd.DataFrame,
    score: str = "pred_ens",
    target: str = "fwd_alpha_5",
    controls: dict[str, str] = CONTROLS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Daily cross-sectional OLS of forward alpha (%) on z-scored score + controls.

    Returns (slopes table: mean slope, NW t, with and without controls; neutral IC series).
    The neutral IC is the rank IC of the score's residual after regressing it on the controls.
    """
    cols = list(controls.values())
    d = f[["Date", score, target, *cols]].dropna(subset=[score, target]).reset_index(drop=True)
    z = _z(d, [score, *cols])
    y = d[target] * 100
    slopes, slopes_raw, neutral = {}, {}, []
    for day, idx in d.groupby("Date").indices.items():
        if len(idx) < MIN_NAMES:
            continue
        X = np.c_[np.ones(len(idx)), z.iloc[idx].to_numpy()]
        slopes[day] = np.linalg.lstsq(X, y.iloc[idx].to_numpy(), rcond=None)[0]
        slopes_raw[day] = np.linalg.lstsq(X[:, :2], y.iloc[idx].to_numpy(), rcond=None)[0]
        b = np.linalg.lstsq(X[:, [0, *range(2, X.shape[1])]], X[:, 1], rcond=None)[0]
        resid = X[:, 1] - X[:, [0, *range(2, X.shape[1])]] @ b
        neutral.append(pd.DataFrame({"Date": day, "resid": resid, "y": y.iloc[idx].to_numpy()}))
    names = ["const", "score", *controls]
    s = pd.DataFrame(slopes, index=names).T
    s_raw = pd.DataFrame(slopes_raw, index=["const", "score"]).T
    table = pd.DataFrame(
        {"slope (% 5d alpha per 1 sd)": s.mean(), "t (NW)": [nw_tstat(s[c], 5) for c in s.columns]}
    )
    table.loc["score (no controls)"] = [s_raw["score"].mean(), nw_tstat(s_raw["score"], 5)]
    n = pd.concat(neutral, ignore_index=True)
    return table.drop(index="const"), daily_ic(n, "resid", "y")


# ----------------------------------------------------------------------------- persistence
def score_panel(f: pd.DataFrame, score: str = "pred_ens") -> pd.DataFrame:
    """Date x Ticker panel of the within-date percentile rank."""
    return f.assign(pct=f.groupby("Date")[score].rank(pct=True)).pivot_table(
        index="Date", columns="Ticker", values="pct"
    )


def persistence(
    f: pd.DataFrame,
    score: str = "pred_ens",
    lags: tuple[int, ...] = (1, 5, 10, 20),
    top: float = 0.9,
) -> pd.DataFrame:
    """Rank autocorrelation and top-decile turnover (share of today's top decile that is new vs t-lag)."""
    S = score_panel(f, score)
    rows = {}
    for lag in lags:
        prev = S.shift(lag)
        ac = S.corrwith(prev, axis=1)
        now_top, old_top = S.ge(top), prev.ge(top)
        stay = (now_top & old_top).sum(axis=1) / now_top.sum(axis=1).replace(0, np.nan)
        stay = stay.where(prev.notna().any(axis=1))  # no earlier panel yet: undefined, not 100%
        rows[lag] = {
            "rank autocorrelation": ac.mean(),
            "top-decile turnover %": 100 * (1 - stay).mean(),
        }
    return pd.DataFrame(rows).T.rename_axis("lag (days)")


# ----------------------------------------------------------------------------- fundamental law
def holdings(trades: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    """Shares held at each close (Date x Ticker), rebuilt from the trade blotter."""
    if trades.empty:
        return pd.DataFrame(index=dates)
    q = trades.groupby(["Date", "Ticker"])["qty"].sum().unstack(fill_value=0)
    return q.reindex(dates, fill_value=0).cumsum()


def weights(trades: pd.DataFrame, close: pd.DataFrame, equity: pd.Series) -> pd.DataFrame:
    """Portfolio weights at each close = shares x close / equity."""
    sh = holdings(trades, equity.index)
    px = close.reindex(index=equity.index, columns=sh.columns).ffill()
    return (sh * px).div(equity, axis=0).fillna(0.0)


def transfer_coefficient(w: pd.DataFrame, f: pd.DataFrame, score: str = "pred_ens") -> pd.Series:
    """TC per decision date: corr over tradable names of (active weight x sigma, z-score), sigma = idio vol.

    Grinold alpha_i = IC sigma_i z_i, so alpha_i / sigma_i is proportional to z_i. Active weights are vs the
    equal-weight tradable universe (the benchmark of the alpha target); the book held after the
    next-open execution is matched to the decision date's scores.
    """
    out = {}
    dates = w.index
    for day, g in f.groupby("Date"):
        pos = dates.searchsorted(day, side="right")  # first close after the decision
        if pos >= len(dates):
            continue
        held = w.iloc[pos]
        g = g.dropna(subset=[score, "idio_vol"]).set_index("Ticker")
        if len(g) < MIN_NAMES or held.reindex(g.index).fillna(0).sum() <= 0:
            continue
        active = held.reindex(g.index).fillna(0.0) - held.reindex(g.index).fillna(0.0).sum() / len(
            g
        )
        zs = (g[score] - g[score].mean()) / g[score].std()
        out[day] = float(np.corrcoef(active * g["idio_vol"], zs)[0, 1])
    return pd.Series(out, name="TC")


def fundamental_law(
    ic_mean: float,
    breadth_names: float,
    tc: float,
    horizon: int = config.HORIZON,
    realised_ir: float | None = None,
) -> dict:
    """Ex-ante IR = IC x sqrt(BR) x TC with BR = names x independent rebalances per year."""
    br = breadth_names * 252 / horizon
    out = {
        "IC (per 5d bet)": ic_mean,
        "names per day": breadth_names,
        "breadth BR (bets / yr)": br,
        "IR full transfer (IC x sqrt BR)": ic_mean * np.sqrt(br),
        "transfer coefficient TC": tc,
        "expected IR (IC x sqrt BR x TC)": ic_mean * np.sqrt(br) * tc,
    }
    if realised_ir is not None:
        out["realised gross IR vs EW universe"] = realised_ir
        out["implied realised TC"] = realised_ir / (ic_mean * np.sqrt(br))
    return out


# ----------------------------------------------------------------------------- tear-sheet
def tearsheet(f: pd.DataFrame, score: str = "pred_ens") -> dict:
    """All signal tables for one split (see module docstring)."""
    ic = daily_ic(f, score)
    q = quantile_alpha(f, score)
    fm, neutral_ic = fama_macbeth(f, score)
    summary = pd.DataFrame(
        {"raw IC": ic_summary(ic), "factor-neutral IC": ic_summary(neutral_ic)}
    ).T
    return {
        "ic": ic,
        "ic_summary": summary,
        "ic_by_year": ic_by_year(ic),
        "ic_decay": ic_decay(f, score),
        "quantiles": q,
        "monotonicity": monotonicity(q),
        "fama_macbeth": fm,
        "neutral_ic": neutral_ic,
        "persistence": persistence(f, score),
        "names_per_day": float(f.groupby("Date").size().mean()),
    }


# ----------------------------------------------------------------------------- multi-signal correlation control
# Utilities for combining several alpha signals (experiments/alpha_portfolio_v1): wide (dates x names) rank ICs,
# symmetric orthogonalisation, mean-variance signal weights on the IC covariance, clustering of redundant signals,
# and the effective number of independent bets / diversification ratio of a signal set.
def rank_ic_panel(x: np.ndarray, y: np.ndarray, min_names: int = MIN_NAMES) -> np.ndarray:
    """Daily Spearman IC of two wide (T x K) arrays, rows = dates; NaN = missing. Ranks use the common names."""
    m = ~np.isnan(x) & ~np.isnan(y)
    xr = pd.DataFrame(np.where(m, x, np.nan)).rank(axis=1).to_numpy()
    yr = pd.DataFrame(np.where(m, y, np.nan)).rank(axis=1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        xc = xr - np.nanmean(xr, axis=1, keepdims=True)
        yc = yr - np.nanmean(yr, axis=1, keepdims=True)
        ic = np.nansum(xc * yc, 1) / np.sqrt(np.nansum(xc**2, 1) * np.nansum(yc**2, 1))
    ic[m.sum(1) < min_names] = np.nan
    return ic


def lowdin(X: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Symmetric (Loewdin) orthogonalisation of the columns of X (n names x k signals).

    Columns are z-scored, then X_orth = Z (Z'Z / n)^(-1/2): the orthonormal set closest to Z in least squares, so no
    signal is privileged (unlike Gram-Schmidt, which depends on the order). Constant columns stay zero.
    """
    Z = np.asarray(X, dtype=float)
    Z = Z - Z.mean(0)
    sd = Z.std(0)
    live = sd > 0
    out = np.zeros_like(Z)
    if live.sum() == 0:
        return out
    Zl = Z[:, live] / sd[live]
    M = Zl.T @ Zl / len(Zl)
    val, vec = np.linalg.eigh(M)
    inv_sqrt = vec @ np.diag(1.0 / np.sqrt(np.maximum(val, floor))) @ vec.T
    out[:, live] = Zl @ inv_sqrt
    return out


def mv_signal_weights(mu: pd.Series, cov: pd.DataFrame, long_only: bool = True) -> pd.Series:
    """'Factor portfolio' weights w ~ argmax w'mu - 1/2 w'S w (w >= 0 if long_only), normalised to sum |w| = 1."""
    from scipy.optimize import minimize

    m, S = mu.to_numpy(float), cov.loc[mu.index, mu.index].to_numpy(float)
    if not long_only:
        w = np.linalg.solve(S, m)
    else:
        res = minimize(
            lambda w: -(w @ m) + 0.5 * w @ S @ w,
            np.zeros(len(m)),
            jac=lambda w: -m + S @ w,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * len(m),
        )
        w = np.clip(res.x, 0.0, None)
    s = np.abs(w).sum()
    return pd.Series(w / s if s > 0 else np.full(len(m), 1.0 / len(m)), index=mu.index)


def cluster_signals(corr: pd.DataFrame, cut: float = 0.5) -> pd.Series:
    """Average-linkage hierarchical clustering on the distance 1 - corr; clusters cut at `cut` (labels 1..k)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(corr) == 1:
        return pd.Series([1], index=corr.index)
    d = np.clip(1.0 - corr.to_numpy(float), 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform((d + d.T) / 2, checks=False), method="average")
    return pd.Series(fcluster(z, t=cut, criterion="distance"), index=corr.index)


def effective_bets(corr: pd.DataFrame | np.ndarray) -> dict:
    """Effective number of independent signals from the eigenvalues l of a correlation matrix:
    entropy form exp(-sum p log p), p = l / sum l (Roy & Vetterli 2007; Meucci 2009) and the participation
    ratio (sum l)^2 / sum l^2. Both equal k for uncorrelated signals and 1 for perfectly correlated ones."""
    lam = np.clip(np.linalg.eigvalsh(np.asarray(corr, dtype=float)), 0.0, None)
    p = lam / lam.sum()
    nz = p[p > 0]
    return {
        "ENB entropy": float(np.exp(-(nz * np.log(nz)).sum())),
        "ENB participation": float(lam.sum() ** 2 / (lam**2).sum()),
    }


def diversification_ratio(w: pd.Series, cov: pd.DataFrame) -> float:
    """Choueifaty-Coignard diversification ratio sum |w_i| sd_i / sd(w'x) (1 = no diversification)."""
    w = w.reindex(cov.index).fillna(0.0).to_numpy(float)
    S = cov.to_numpy(float)
    den = np.sqrt(max(w @ S @ w, 1e-30))
    return float(np.abs(w) @ np.sqrt(np.diag(S)) / den)
