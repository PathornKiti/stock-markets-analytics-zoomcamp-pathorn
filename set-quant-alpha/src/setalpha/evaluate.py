"""Step 5 - performance statistics and diagnostics.

Beyond CAGR / Sharpe / drawdown, three quant checks guard against fooling ourselves:

* CAPM regression  r_p - rf = a + b (r_bench - rf) + e  with Newey-West (HAC) t-stats:
  is there return the benchmark does not explain?
* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012): probability that the true
  Sharpe is above 0, given track-record length, skewness and fat tails.
* Deflated Sharpe Ratio (2014): the same, but against the best Sharpe you would expect by
  luck after trying N strategies - we try many, so this is the honest number.

Lo-adjusted Sharpe, MinTRL, multiple-testing haircuts and PBO live in `validity.py`; Expected
Shortfall 97.5% (Basel FRTB) in `risk.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, norm, skew

from setalpha.risk import expected_shortfall
from setalpha.validity import lo_sharpe, min_track_record

DAYS = 252
EULER = 0.5772156649


def _sr(r: pd.Series) -> float:
    return float(r.mean() / r.std()) if r.std() > 0 else np.nan


def psr(r: pd.Series, sr_star: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_star)."""
    r = r.dropna()
    sr, n = _sr(r), len(r)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def dsr(r: pd.Series, trial_srs: list[float]) -> float:
    """Deflated Sharpe: PSR against the expected maximum Sharpe of N unskilled trials."""
    n, var = len(trial_srs), float(np.var(trial_srs, ddof=1)) if len(trial_srs) > 1 else 0.0
    if n < 2 or var == 0:
        return psr(r)
    sr_star = np.sqrt(var) * (
        (1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * np.e))
    )
    return psr(r, sr_star)


def capm(r: pd.Series, bench: pd.Series) -> dict:
    d = pd.concat([r, bench], axis=1, keys=["p", "b"]).dropna()
    fit = sm.OLS(d["p"], sm.add_constant(d["b"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "alpha % p.a.": 100 * fit.params["const"] * DAYS,
        "alpha t-stat": float(fit.tvalues["const"]),
        "beta": float(fit.params["b"]),
    }


def max_dd_duration(r: pd.Series) -> int:
    """Longest stretch (trading days) spent below a previous equity peak."""
    under = drawdown(r) < 0
    runs = under.groupby((~under).cumsum()).sum()
    return int(runs.max()) if len(runs) else 0


def ulcer_index(r: pd.Series) -> float:
    """sqrt(mean squared drawdown), in %: penalises both depth and duration of drawdowns."""
    return float(100 * np.sqrt((drawdown(r) ** 2).mean()))


def omega(r: pd.Series, threshold: float = 0.0) -> float:
    """Sum of gains above the threshold / sum of losses below it (whole return distribution)."""
    gain, loss = (r - threshold).clip(lower=0).sum(), (threshold - r).clip(lower=0).sum()
    return float(gain / loss) if loss > 0 else np.nan


def capture_ratios(r: pd.Series, bench: pd.Series) -> dict:
    """Monthly up / down capture and batting average vs the benchmark."""
    m = pd.concat([r, bench], axis=1, keys=["p", "b"]).dropna()
    m = (1 + m).resample("ME").prod() - 1
    up, down = m[m["b"] > 0], m[m["b"] < 0]
    return {
        "up capture %": 100 * up["p"].mean() / up["b"].mean() if len(up) else np.nan,
        "down capture %": 100 * down["p"].mean() / down["b"].mean() if len(down) else np.nan,
        "batting avg %": 100 * float((m["p"] > m["b"]).mean()),
    }


def bootstrap_sharpe_ci(
    r: pd.Series, n: int = 1000, block: int = 21, seed: int = 0
) -> tuple[float, float]:
    """5%-95% interval of the annualised Sharpe from a moving-block bootstrap (keeps autocorrelation)."""
    x, rng = r.dropna().to_numpy(), np.random.default_rng(seed)
    if len(x) < 2 * block:
        return np.nan, np.nan
    starts = rng.integers(0, len(x) - block, size=(n, len(x) // block + 1))
    samples = x[starts[:, :, None] + np.arange(block)].reshape(n, -1)[:, : len(x)]
    srs = samples.mean(1) / samples.std(1) * np.sqrt(DAYS)
    return float(np.percentile(srs, 5)), float(np.percentile(srs, 95))


def perf_stats(r: pd.Series, bench: pd.Series | None = None) -> dict:
    """Factsheet metrics: return, risk-adjusted, benchmark-relative, drawdown/tail, robustness."""
    r = r.dropna()
    curve = (1 + r).cumprod()
    years = len(r) / DAYS
    dd = curve / curve.cummax() - 1
    down = r[r < 0].std()
    cagr = curve.iloc[-1] ** (1 / years) - 1
    lo, hi = bootstrap_sharpe_ci(r)
    out = {
        # return
        "CAGR %": 100 * cagr,
        "best year %": 100 * float(((1 + r).resample("YE").prod() - 1).max()),
        "worst year %": 100 * float(((1 + r).resample("YE").prod() - 1).min()),
        # risk-adjusted
        "vol %": 100 * r.std() * np.sqrt(DAYS),
        "Sharpe": _sr(r) * np.sqrt(DAYS),
        "Sortino": float(r.mean() / down * np.sqrt(DAYS)) if down > 0 else np.nan,
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "Omega": omega(r),
        # drawdown and tail
        "max DD %": 100 * dd.min(),
        "max DD duration (days)": max_dd_duration(r),
        "Ulcer index": ulcer_index(r),
        "VaR 95% (1d) %": -100 * float(np.percentile(r, 5)),
        "CVaR 95% (1d) %": -100 * float(r[r <= np.percentile(r, 5)].mean()),
        "ES 97.5% (1d) %": expected_shortfall(r, 0.975),
        "ES 97.5% (10d) %": expected_shortfall(r, 0.975, 10),
        "skew": float(skew(r)),
        "kurtosis": float(kurtosis(r, fisher=False)),
        # trading
        "hit rate (days) %": 100 * float((r > 0).mean()),
        "profit factor (days)": float(r[r > 0].sum() / -r[r < 0].sum())
        if (r < 0).any()
        else np.nan,
        "payoff ratio (days)": float(r[r > 0].mean() / -r[r < 0].mean())
        if (r < 0).any()
        else np.nan,
        # robustness
        "PSR (SR>0)": psr(r),
        "Sharpe 5%": lo,
        "Sharpe 95%": hi,
        "Sharpe (Lo-adjusted)": lo_sharpe(r),
        "MinTRL (yrs)": min_track_record(r),
        "track record (yrs)": years,
    }
    if bench is not None:
        b = bench.reindex(r.index)
        active = (r - b).dropna()
        out["excess CAGR %"] = out["CAGR %"] - 100 * (((1 + b.dropna()).prod()) ** (1 / years) - 1)
        out["tracking error %"] = 100 * active.std() * np.sqrt(DAYS)
        out["info ratio"] = _sr(active) * np.sqrt(DAYS)
        out.update(capm(r, bench))
        out["Treynor %"] = 100 * r.mean() * DAYS / out["beta"] if out["beta"] else np.nan
        out.update(capture_ratios(r, bench))
    return out


# Column groups for factsheet-style tables (README / dashboard).
GROUPS = {
    "Return & risk-adjusted": [
        "CAGR %",
        "best year %",
        "worst year %",
        "vol %",
        "Sharpe",
        "Sortino",
        "Calmar",
        "Omega",
    ],
    "Benchmark-relative": [
        "excess CAGR %",
        "tracking error %",
        "info ratio",
        "alpha % p.a.",
        "alpha t-stat",
        "beta",
        "Treynor %",
        "up capture %",
        "down capture %",
        "batting avg %",
    ],
    "Drawdown & tail": [
        "max DD %",
        "max DD duration (days)",
        "Ulcer index",
        "VaR 95% (1d) %",
        "CVaR 95% (1d) %",
        "ES 97.5% (1d) %",
        "ES 97.5% (10d) %",
        "skew",
        "kurtosis",
    ],
    "Trading & implementation": [
        "avg exposure %",
        "turnover x/yr",
        "trades",
        "gross CAGR %",
        "cost drag %",
        "break-even cost vs TDEX (%/side)",
        "break-even cost vs EW universe (%/side)",
        "gross alpha % p.a.",
        "gross alpha per 1x turnover (bp)",
        "hit rate (days) %",
        "profit factor (days)",
        "payoff ratio (days)",
    ],
    "Robustness": [
        "PSR (SR>0)",
        "DSR",
        "Sharpe 5%",
        "Sharpe 95%",
        "Sharpe (Lo-adjusted)",
        "MinTRL (yrs)",
        "track record (yrs)",
    ],
}


def compare(
    returns: dict[str, pd.Series], bench: pd.Series, trials: list[str] | None = None
) -> pd.DataFrame:
    """Stats table; DSR deflates against the Sharpe ratios of the strategies in `trials` (default: all)."""
    table = pd.DataFrame({k: perf_stats(v, bench) for k, v in returns.items()}).T
    trial_srs = [_sr(returns[k].dropna()) for k in (trials or returns)]
    table["DSR"] = [dsr(v, trial_srs) for v in returns.values()]
    return table


def grouped(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a metrics table into the factsheet groups (columns that exist only)."""
    lead = [c for c in ["selected"] if c in metrics.columns]
    return {
        g: metrics[lead + [c for c in cols if c in metrics.columns]] for g, cols in GROUPS.items()
    }


def rolling_return(r: pd.Series, window: int = DAYS) -> pd.Series:
    return (1 + r).rolling(window).apply(np.prod, raw=True) - 1


def drawdown(r: pd.Series) -> pd.Series:
    curve = (1 + r).cumprod()
    return curve / curve.cummax() - 1


def by_group(
    f: pd.DataFrame, group: str, score: str = "pred_ens", top_pct: float = 0.9
) -> pd.DataFrame:
    """How well the daily top-decile picks did, split by `group` (sector, year, regime, ...)."""
    f = f.copy()
    f["top"] = f.groupby("Date")[score].rank(pct=True) >= top_pct
    g = f[f["top"]].groupby(group, observed=True)
    out = pd.DataFrame(
        {
            "picks": g.size(),
            "precision %": 100 * g["is_alpha_5"].mean(),
            "avg 5d alpha %": 100 * g["fwd_alpha_5"].mean(),
            "avg 5d return %": 100 * g["fwd_ret_5"].mean(),
        }
    )
    base = f.groupby(group, observed=True)["is_alpha_5"].mean() * 100
    out["base rate %"] = base
    out["lift (pp)"] = out["precision %"] - out["base rate %"]
    return out.sort_values("picks", ascending=False)


def worst_calls(f: pd.DataFrame, score: str = "pred_ens", n: int = 15) -> pd.DataFrame:
    """Most confident picks that lost the most: the debugging list."""
    f = f.copy()
    f["pct"] = f.groupby("Date")[score].rank(pct=True)
    cols = [
        "Date",
        "Ticker",
        "sector",
        score,
        "fwd_ret_5",
        "fwd_alpha_5",
        "hmm_state",
        "vix_regime",
    ]
    return f[f["pct"] >= 0.98].nsmallest(n, "fwd_alpha_5")[cols]
