"""Quant layer - from ML scores to portfolio weights.

1. Alpha scaling (Grinold & Kahn, "fundamental law"):
       alpha_i = IC * sigma_i * z_i
   z = cross-sectional z-score of the model score, sigma = idiosyncratic volatility over
   the holding period, IC = the model's information coefficient measured on validation.
   This turns a classifier probability into an expected return with the right units.

2. Risk model: Ledoit-Wolf shrinkage covariance of daily returns (120-day window).
   A sample covariance of 20 noisy stocks is badly conditioned; shrinkage fixes that.

3. Mean-variance optimisation (Markowitz), long-only with a single-name cap:
       max_w  alpha'w - (lambda / 2) w' S w      s.t.  sum(w) <= 1,  0 <= w_i <= w_max

4. Volatility targeting: scale the book so its ex-ante volatility is at most the target;
   the rest stays in cash. Exposure falls automatically when markets get volatile.

5. Enhanced index (experiments/strategy_v3, design F2): TDEX core + stock tilts whose local
   MKT / SMB / WML loadings stay at TDEX's (1, 0, 0), with trading costs in the objective so a
   trade is only made when its expected alpha beats its cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from setalpha import config


def alpha_from_scores(
    scores: pd.Series, idio_vol: pd.Series, ic: float, horizon: int = config.HORIZON
) -> pd.Series:
    """Expected `horizon`-day residual return per stock (Grinold's rule)."""
    z = (scores - scores.mean()) / (scores.std() or 1.0)
    return ic * idio_vol * np.sqrt(horizon) * z


def ledoit_wolf_cov(returns: pd.DataFrame, min_obs: float = 0.8) -> pd.DataFrame:
    """Shrunk covariance of daily returns; names with too much missing history are dropped."""
    r = returns.loc[:, returns.notna().mean() >= min_obs]
    r = r.fillna(0.0)
    cov = LedoitWolf().fit(r.to_numpy()).covariance_
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)


def mean_variance(
    alpha: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = config.RISK_AVERSION,
    max_weight: float = config.MAX_WEIGHT,
    horizon: int = config.HORIZON,
) -> pd.Series:
    """Long-only Markowitz weights; returns an empty Series if no name has positive alpha."""
    names = [n for n in alpha.index if n in cov.index]
    a = alpha.loc[names].to_numpy()
    if len(names) == 0 or (a <= 0).all():
        return pd.Series(dtype=float)
    S = cov.loc[names, names].to_numpy() * horizon  # daily -> holding-period covariance

    def neg_utility(w):
        return -(a @ w - 0.5 * risk_aversion * w @ S @ w)

    def grad(w):
        return -(a - risk_aversion * S @ w)

    n = len(names)
    res = minimize(
        neg_utility,
        np.full(n, 1.0 / n),
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "ineq", "fun": lambda w: 1.0 - w.sum()}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    w = pd.Series(np.clip(res.x, 0, max_weight), index=names)
    return w[w > 1e-4]


def vol_target(
    weights: pd.Series, cov: pd.DataFrame, target: float = config.TARGET_VOL
) -> pd.Series:
    """Scale weights down (never up: no leverage) so ex-ante annual volatility <= target."""
    if weights.empty:
        return weights
    S = cov.loc[weights.index, weights.index].to_numpy()
    vol = float(np.sqrt(weights.to_numpy() @ S @ weights.to_numpy() * 252))
    return weights * min(1.0, target / vol) if vol > 0 else weights


def inverse_vol(vols: pd.Series) -> pd.Series:
    """Risk-parity-lite weights: each name gets weight proportional to 1 / volatility."""
    inv = 1.0 / vols.replace(0, np.nan).dropna()
    return inv / inv.sum()


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility (daily): sigma2_t = lam * sigma2_t-1 + (1 - lam) * r2_t-1."""
    return np.sqrt((returns**2).ewm(alpha=1 - lam, adjust=False).mean().shift(1))


def kelly_fraction(p: pd.Series, win: float, loss: float, fraction: float = 0.25) -> pd.Series:
    """Fractional Kelly stake for a bet won with probability p (payoff +win, loss -loss)."""
    f = (p * win - (1 - p) * loss) / (win * loss)
    return (fraction * f).clip(lower=0, upper=1)


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    S = cov.loc[weights.index, weights.index].to_numpy()
    return float(np.sqrt(weights.to_numpy() @ S @ weights.to_numpy() * 252))


def value_at_risk(
    weights: pd.Series, cov: pd.DataFrame, level: float = 0.99, horizon: int = config.HORIZON
) -> float:
    """Parametric (Gaussian) VaR of the book over the holding period, as a fraction of capital."""
    S = cov.loc[weights.index, weights.index].to_numpy() * horizon
    return float(norm.ppf(level) * np.sqrt(weights.to_numpy() @ S @ weights.to_numpy()))


# ----------------------------------------------------------------------------- enhanced index
LOADING_BAND = 0.05  # max |active loading| on MKT, SMB, WML vs TDEX
TDEX_LOADINGS = np.array([1.0, 0.0, 0.0])


def factor_loadings(
    rets: pd.DataFrame,
    fac: pd.DataFrame,
    day,
    names: list[str],
    window: int = 250,
    min_obs: int = 200,
) -> pd.DataFrame:
    """Rolling OLS of daily returns on local MKT / SMB / WML up to `day`, plus residual daily vol."""
    f = fac.loc[:day].tail(window)
    r = rets.reindex(index=f.index, columns=names)
    r = r.loc[:, r.notna().sum() >= min_obs].fillna(0.0)
    X = np.column_stack([np.ones(len(f)), f[["MKT", "SMB", "WML"]].to_numpy()])
    b, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    resid = r.to_numpy() - X @ b
    return pd.DataFrame(
        {"b_MKT": b[1], "b_SMB": b[2], "b_WML": b[3], "s_eps": resid.std(axis=0, ddof=4)},
        index=r.columns,
    )


def enhanced_index(
    alpha: np.ndarray,
    s2: np.ndarray,
    B: np.ndarray,
    w0: np.ndarray,
    t0: float,
    risk_aversion: float = 1.0,
    fee: float = config.FEE_PER_SIDE,
    tdex_fee: float = 0.0005,
    max_weight: float = 0.05,
) -> tuple[np.ndarray, float, bool]:
    """max a'w - lambda/2 sum s2 w^2 - fee |dw|   s.t. fully invested in stocks + TDEX, 0 <= w <= max_weight,
    |B'w + TDEX loadings * wT - TDEX loadings| <= LOADING_BAND.

    alpha: expected holding-period alpha; s2: residual variance over the same period; B: n x 3 loadings;
    w0 / t0: current stock / TDEX weights. Variables are buys u and sells v, so costs stay linear.
    Returns (stock weights, TDEX weight, solver success).
    """
    n = len(alpha)
    A = np.hstack([np.eye(n), -np.eye(n), np.zeros((n, 2))])  # w  = w0 + A x
    a_t = np.r_[np.zeros(2 * n), 1.0, -1.0]  # wT = t0 + a_t x
    jac_e = B.T @ A + np.outer(TDEX_LOADINGS, a_t)

    def obj(x):
        w = w0 + A @ x
        return (
            -(alpha @ w - 0.5 * risk_aversion * (s2 * w * w).sum())
            + fee * x[: 2 * n].sum()
            + tdex_fee * x[2 * n :].sum()
        )

    def grad(x):
        gw = -(alpha - risk_aversion * s2 * (w0 + A @ x))
        return np.concatenate([gw + fee, -gw + fee, [tdex_fee, tdex_fee]])

    def expo(x):
        return B.T @ (w0 + A @ x) + TDEX_LOADINGS * (t0 + a_t @ x) - TDEX_LOADINGS

    cons = [
        {
            "type": "eq",
            "fun": lambda x: (w0 + A @ x).sum() + t0 + a_t @ x - 1.0,
            "jac": lambda x: A.sum(0) + a_t,
        },
        {"type": "ineq", "fun": lambda x: w0 + A @ x, "jac": lambda x: A},
        {"type": "ineq", "fun": lambda x: max_weight - (w0 + A @ x), "jac": lambda x: -A},
        {
            "type": "ineq",
            "fun": lambda x: np.atleast_1d(t0 + a_t @ x),
            "jac": lambda x: a_t[None, :],
        },
        {"type": "ineq", "fun": lambda x: LOADING_BAND - expo(x), "jac": lambda x: -jac_e},
        {"type": "ineq", "fun": lambda x: LOADING_BAND + expo(x), "jac": lambda x: jac_e},
    ]
    bounds = (
        [(0.0, max_weight)] * n
        + [(0.0, max(float(w), 0.0)) for w in w0]
        + [(0.0, 1.0), (0.0, max(t0, 0.0))]
    )
    res = minimize(
        obj,
        np.zeros(2 * n + 2),
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    w = np.clip(w0 + A @ res.x, 0.0, max_weight)
    w[w < 1e-4] = 0.0
    return w, max(float(t0 + a_t @ res.x), 0.0), bool(res.success)


# ----------------------------------------------------------------------------- cost-aware long-only optimiser
def cost_aware_mvo(
    alpha: np.ndarray,
    cov: np.ndarray,
    w0: np.ndarray,
    fee: float = config.FEE_PER_SIDE,
    risk_aversion: float = 1.0,
    max_weight: float = 0.10,
    budget: float = 1.0,
    groups: np.ndarray | None = None,
    group_cap: float | None = None,
    exposures: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    band: float | None = None,
) -> tuple[np.ndarray, bool]:
    """Long-only, fully invested mean-variance rebalance with linear trading costs (experiments/alpha_portfolio_v1):

        max  a'w - lambda/2 w'S w - fee * sum|w - w0|
        s.t. sum w = budget,  0 <= w <= max_weight,  sum_{i in g} w_i <= group_cap  for each group g,
             |E'w - targets| <= band   (optional style-exposure bands, E = n x m exposures)

    The cost makes a trade happen only when its marginal alpha (net of risk) beats its cost, i.e. a no-trade band
    around the current book sized by the cost. alpha and S must be over the same horizon. Variables are buys u and
    sells v (w = w0 + u - v), so the cost stays linear. Returns (weights, solver success).
    """
    from scipy.optimize import minimize

    a, S, w0 = np.asarray(alpha, float), np.asarray(cov, float), np.asarray(w0, float)
    n = len(a)
    A = np.hstack([np.eye(n), -np.eye(n)])
    lam = risk_aversion

    def w_of(x):
        return w0 + x[:n] - x[n:]

    def obj(x):
        w = w_of(x)
        return -(a @ w) + 0.5 * lam * w @ S @ w + fee * x.sum()

    def grad(x):
        gw = -a + lam * S @ w_of(x)
        return np.concatenate([gw + fee, -gw + fee])

    cons = [
        {
            "type": "eq",
            "fun": lambda x: np.atleast_1d(w_of(x).sum() - budget),
            "jac": lambda x: A.sum(0)[None, :],
        },
        {"type": "ineq", "fun": lambda x: max_weight - w_of(x), "jac": lambda x: -A},
    ]
    if groups is not None and group_cap is not None:
        labels = np.unique(groups)
        G = np.stack([(groups == g).astype(float) for g in labels])  # groups x n
        cons.append(
            {"type": "ineq", "fun": lambda x: group_cap - G @ w_of(x), "jac": lambda x: -G @ A}
        )
    if exposures is not None and band is not None:
        E = np.asarray(exposures, float).reshape(n, -1)
        tg = np.asarray(targets, float).reshape(-1)
        cons += [
            {
                "type": "ineq",
                "fun": lambda x: band - (E.T @ w_of(x) - tg),
                "jac": lambda x: -E.T @ A,
            },
            {
                "type": "ineq",
                "fun": lambda x: band + (E.T @ w_of(x) - tg),
                "jac": lambda x: E.T @ A,
            },
        ]
    bounds = [(0.0, max(max_weight, 0.0))] * n + [(0.0, max(float(v), 0.0)) for v in w0]
    res = minimize(
        obj,
        np.zeros(2 * n),
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    w = np.clip(w_of(res.x), 0.0, max_weight)
    viol = abs(w.sum() - budget) > 1e-4
    return w, bool(res.success) and not viol
