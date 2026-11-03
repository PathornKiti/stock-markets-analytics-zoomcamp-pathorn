"""Statistical validity of a backtest: is the Sharpe ratio real, long enough, and not a selection artefact?

* Lo (2002): annualising a daily Sharpe with sqrt(252) assumes IID returns. With autocorrelation
  rho_k the right factor is  q / sqrt(q + 2 sum_{k<q} (q - k) rho_k),  q = 252.
* Minimum Track Record Length (Bailey & Lopez de Prado, 2012): the number of observations needed
  before the PSR of the observed Sharpe clears `conf`:
      MinTRL = 1 + [1 - g3 SR + (g4 - 1) / 4 SR^2] (z_conf / (SR - SR*))^2
* Harvey, Liu & Zhu (2016) / Harvey & Liu (2015): with M strategies tried, a t-stat of 2 is not
  significant. p-values are adjusted for multiple testing (Bonferroni, Holm, BHY) and the Sharpe is
  "haircut" to the value whose single-test p-value equals the adjusted one. HLZ's rule of thumb is a
  hurdle of t > 3 for a new factor.
* Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu, 2017) by combinatorially
  symmetric cross-validation (CSCV): split the T x N matrix of trial returns into S blocks; for each of
  the C(S, S/2) ways to pick half the blocks as in-sample, take the in-sample best trial and look up
  its out-of-sample rank. PBO = share of splits in which the IS winner ranks below the OOS median.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, norm, skew

DAYS = 252

# Trial ledger: every strategy design and model variant evaluated in this project (README "all trials").
TRIALS = {
    "pipeline E1-E6": 6,
    "cost_aware": 37,
    "strategy_v2": 17,
    "strategy_v3": 7,
    "strategy_v4": 4,
    "literature_v1": 6,
    "integrated_v1": 582,
    "model_v2": 12,
    "retrain_v1": 8,
    "alpha_portfolio_v1": 55,
    "survivorship_v1": 1,
    "foreign_v1": 6,
    "fundamentals_v1": 16,
    "lseg_signals_v1": 31,
    "adaptive_v1": 7,
}
N_TRIALS = sum(TRIALS.values())


def lo_sharpe(r: pd.Series, q: int = DAYS, lags: int = 10) -> float:
    """Annualised Sharpe with Lo's (2002) autocorrelation correction (rho_k beyond `lags` taken as 0)."""
    r = r.dropna()
    rho = [r.autocorr(k) for k in range(1, min(lags, q - 1) + 1)]
    denom = q + 2 * sum((q - k) * p for k, p in enumerate(rho, start=1))
    return float(r.mean() / r.std() * q / np.sqrt(denom)) if denom > 0 and r.std() > 0 else np.nan


def min_track_record(r: pd.Series, sr_star: float = 0.0, conf: float = 0.95) -> float:
    """MinTRL in years (DAYS obs per year): track record needed for P(true SR > sr_star) >= conf."""
    r = r.dropna()
    sr = r.mean() / r.std()
    if sr <= sr_star:
        return np.inf
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    n = 1 + (1 - g3 * sr + (g4 - 1) / 4 * sr**2) * (norm.ppf(conf) / (sr - sr_star)) ** 2
    return float(n / DAYS)


def nw_tstat(x: pd.Series, lags: int = 5) -> float:
    """t-stat of the mean of `x` with Newey-West (HAC) standard errors."""
    x = pd.Series(x).dropna()
    fit = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def p_value(t: float) -> float:
    """Two-sided normal p-value of a t-stat (Harvey & Liu use the same convention)."""
    return float(2 * (1 - norm.cdf(abs(t))))


def adjust_pvalue(p: float, others: list[float], n_trials: int, method: str) -> float:
    """Multiple-testing adjusted p-value of one test among `n_trials`.

    `others` are the observed p-values of the other trials; unobserved trials are padded with p = 1
    (they count in M but never rank ahead of ours), which is conservative for Holm and neutral for BHY.
    """
    m = max(n_trials, len(others) + 1)
    if method == "bonferroni":
        return min(1.0, m * p)
    ps = np.sort(np.r_[p, others, np.ones(m - len(others) - 1)])
    i = int(np.searchsorted(ps, p, side="left"))  # 0-based rank of our test (ties: best rank)
    if method == "holm":  # step-down
        adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * ps))
        return float(adj[i])
    if method == "bhy":  # Benjamini-Hochberg-Yekutieli step-up
        c = np.sum(1.0 / np.arange(1, m + 1))
        adj = np.minimum(1.0, ps * m * c / np.arange(1, m + 1))
        return float(np.minimum.accumulate(adj[::-1])[::-1][i])
    raise ValueError(method)


def haircut(
    t: float,
    years: float | None = None,
    others: list[float] | None = None,
    n_trials: int = N_TRIALS,
) -> dict:
    """Harvey-Liu haircut of a t-stat (and of the annual Sharpe t / sqrt(years) if `years` is given)."""
    p, others = p_value(t), [o for o in (others or []) if np.isfinite(o)]
    out = {
        "t": t,
        "p": p,
        "trials M": n_trials,
        "t hurdle (Bonferroni, 5%)": float(norm.ppf(1 - 0.05 / (2 * n_trials))),
    }
    for m in ["bonferroni", "holm", "bhy"]:
        pa = adjust_pvalue(p, others, n_trials, m)
        th = float(norm.ppf(1 - pa / 2)) * np.sign(t) if pa < 1 else 0.0
        out[f"p {m}"] = pa
        out[f"haircut t {m}"] = th
        if years:
            out[f"haircut SR {m}"] = th / np.sqrt(years)
            out[f"haircut % {m}"] = 100 * (1 - th / t) if t > 0 else 100.0
    if years:
        out["SR"] = t / np.sqrt(years)
    return out


def sharpe_tstat(r: pd.Series) -> float:
    """t-stat of the mean daily return = annual Sharpe x sqrt(years)."""
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else 0.0


def pbo_cscv(returns: pd.DataFrame, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting by CSCV, with the Sharpe ratio as the selection metric."""
    X = returns.dropna().to_numpy()
    T, N = X.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    s1 = np.array([X[b].sum(0) for b in blocks])  # per-block sums -> Sharpe of any union
    s2 = np.array([(X[b] ** 2).sum(0) for b in blocks])
    cnt = np.array([len(b) for b in blocks])[:, None]

    def sharpe(idx):
        n = cnt[idx].sum()
        mu = s1[idx].sum(0) / n
        return mu / np.sqrt(np.maximum(s2[idx].sum(0) / n - mu**2, 1e-18))

    logits, is_best, oos_best, oos_med = [], [], [], []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        ins = np.array(combo)
        oos = np.setdiff1d(np.arange(n_blocks), ins)
        sr_is, sr_oos = sharpe(ins), sharpe(oos)
        best = int(np.argmax(sr_is))
        rank = (sr_oos <= sr_oos[best]).sum()  # 1 = worst ... N = best out of sample
        w = rank / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_best.append(sr_is[best])
        oos_best.append(sr_oos[best])
        oos_med.append(np.median(sr_oos))
    logits, oos_best = np.array(logits), np.array(oos_best) * np.sqrt(DAYS)
    slope = np.polyfit(np.array(is_best) * np.sqrt(DAYS), oos_best, 1)[0]
    return {
        "PBO": float((logits <= 0).mean()),
        "trials N": N,
        "days T": T,
        "blocks S": n_blocks,
        "splits": len(logits),
        "median logit": float(np.median(logits)),
        "OOS Sharpe of IS winner (mean)": float(oos_best.mean()),
        "P(OOS Sharpe of IS winner < 0)": float((oos_best < 0).mean()),
        "OOS Sharpe of IS winner - OOS median": float(
            (oos_best - np.array(oos_med) * np.sqrt(DAYS)).mean()
        ),
        "degradation slope (OOS on IS Sharpe)": float(slope),
    }


def dedupe(returns: pd.DataFrame, max_corr: float = 0.995) -> pd.DataFrame:
    """Drop trials that are near-copies of an earlier column (e.g. the same design re-run elsewhere)."""
    r = returns.dropna(axis=1, how="all")
    c = r.corr().abs().to_numpy()
    keep = [j for j in range(r.shape[1]) if not any(c[i, j] > max_corr for i in range(j))]
    return r.iloc[:, keep]


def validity_table(
    returns: dict[str, pd.Series], others_p: list[float] | None = None
) -> pd.DataFrame:
    """Per-strategy Sharpe validity: IID vs Lo Sharpe, MinTRL vs track record, HLZ haircut."""
    rows = {}
    for name, r in returns.items():
        r = r.dropna()
        years, t = len(r) / DAYS, sharpe_tstat(r)
        h = haircut(t, years, [p for p in (others_p or []) if not np.isclose(p, p_value(t))])
        rows[name] = {
            "Sharpe": r.mean() / r.std() * np.sqrt(DAYS),
            "Sharpe (Lo 2002)": lo_sharpe(r),
            "track record (yrs)": years,
            "MinTRL SR>0 @95% (yrs)": min_track_record(r),
            "t (Sharpe)": t,
            "HLZ t hurdle": h["t hurdle (Bonferroni, 5%)"],
            "haircut SR Bonferroni": h["haircut SR bonferroni"],
            "haircut SR Holm": h["haircut SR holm"],
            "haircut SR BHY": h["haircut SR bhy"],
        }
    return pd.DataFrame(rows).T


def stationary_bootstrap_index(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One Politis & Romano (1994) stationary-bootstrap resample of 0..n-1: blocks start at random points and have
    geometric lengths with mean `mean_block`; the series wraps around."""
    new = rng.random(n) < 1.0 / mean_block
    new[0] = True
    pos = np.flatnonzero(new)  # where each block starts in the resample
    block = np.cumsum(new) - 1  # block number of each resampled day
    start = rng.integers(n, size=len(pos))  # random origin of each block
    return (start[block] + np.arange(n) - pos[block]) % n


def sharpe_diff_test(
    r1: pd.Series, r2: pd.Series, mean_block: float = 21, n_boot: int = 5000, seed: int = 0
) -> dict:
    """Paired stationary-bootstrap test of annualised Sharpe(r1) - Sharpe(r2) on common dates.

    The two series are resampled with the same indices, so their correlation and serial dependence are kept
    (Ledoit & Wolf 2008 use the same idea with a studentised statistic). p one-sided = share of centred bootstrap
    differences at least as large as the observed one, i.e. H0: Sharpe(r1) <= Sharpe(r2).
    """
    d = pd.concat([r1, r2], axis=1).dropna().to_numpy()
    n = len(d)

    def sr(x):
        s = x.std(axis=0, ddof=1)
        return np.where(s > 0, x.mean(axis=0) / s, 0.0) * np.sqrt(DAYS)

    obs = float(np.subtract(*sr(d)))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = np.subtract(*sr(d[stationary_bootstrap_index(n, mean_block, rng)]))
    return {
        "Sharpe 1": float(sr(d)[0]),
        "Sharpe 2": float(sr(d)[1]),
        "diff": obs,
        "CI 2.5%": float(np.quantile(boot, 0.025)),
        "CI 97.5%": float(np.quantile(boot, 0.975)),
        "boot sd": float(boot.std(ddof=1)),
        "P(boot diff <= 0)": float((boot <= 0).mean()),
        "p one-sided (H0 diff <= 0)": float((1 + ((boot - obs) >= obs).sum()) / (1 + n_boot)),
        "days": n,
        "mean block": mean_block,
        "draws": n_boot,
    }
