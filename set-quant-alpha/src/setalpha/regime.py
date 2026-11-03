"""Market regimes with a Gaussian Hidden Markov Model.

The Thai market is assumed to switch between K hidden states (calm / normal / stressed),
each with its own mean and covariance of two daily observations:

    x_t = [ market return (%),  log 20-day realised volatility ]

Parameters (start probabilities, transition matrix A, Gaussian emissions) are fitted with
Baum-Welch (EM) by `hmmlearn`. For trading we need *filtered* probabilities
P(state_t | x_1..x_t), computed here with the forward recursion

    alpha_t = (alpha_t-1 @ A) * N(x_t | mu_k, Sigma_k),   normalised to sum to 1

`hmmlearn.predict_proba` returns *smoothed* probabilities P(state_t | x_1..x_T), which use
the future and would leak information into a backtest, so it is deliberately not used.
The model is refit every January on past data only (walk-forward), like the ML models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

log = logging.getLogger(__name__)
STATE_NAMES = ["calm", "normal", "stressed"]


def observations(mkt: pd.Series) -> pd.DataFrame:
    rv20 = mkt.rolling(20).std() * np.sqrt(252)
    return pd.DataFrame({"ret": 100 * mkt, "log_rv": np.log(rv20)}).dropna()


def fit_hmm(obs: pd.DataFrame, n_states: int = 3, seed: int = 0) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states, covariance_type="full", n_iter=200, random_state=seed
    )
    model.fit(obs.to_numpy())
    return model


def filtered_probs(model: GaussianHMM, obs: pd.DataFrame) -> pd.DataFrame:
    """Forward-filtered state probabilities, states ordered from lowest to highest volatility."""
    x = obs.to_numpy()
    k = model.n_components
    dens = np.column_stack(
        [
            multivariate_normal(model.means_[j], model.covars_[j], allow_singular=True).pdf(x)
            for j in range(k)
        ]
    )
    alpha = np.zeros((len(x), k))
    prev = model.startprob_
    for t in range(len(x)):
        a = (prev @ model.transmat_ if t else prev) * dens[t]
        s = a.sum()
        alpha[t] = a / s if s > 0 else np.full(k, 1.0 / k)
        prev = alpha[t]
    order = np.argsort(model.means_[:, 1])  # sort states by mean log-volatility
    names = STATE_NAMES if k == 3 else [f"state{i}" for i in range(k)]
    return pd.DataFrame(alpha[:, order], index=obs.index, columns=[f"hmm_p_{n}" for n in names])


def walk_forward_regimes(mkt: pd.Series, first_year: int = 2013, n_states: int = 3) -> pd.DataFrame:
    """Filtered regime probabilities with yearly refits; years before `first_year` use the first fit."""
    obs = observations(mkt)
    years = sorted(obs.index.year.unique())
    parts = []
    for yr in years:
        if yr < first_year:
            continue
        train = obs[obs.index < f"{yr}-01-01"]
        model = fit_hmm(train, n_states)
        probs = filtered_probs(model, obs[obs.index < f"{yr + 1}-01-01"])
        if yr == first_year:  # warm-up years: filtered with the first fit
            parts.append(probs[probs.index < f"{yr + 1}-01-01"])
        else:
            parts.append(probs[probs.index.year == yr])
    out = pd.concat(parts)
    out["hmm_state"] = out.to_numpy().argmax(axis=1)
    log.info(
        "    HMM regimes: %s", out["hmm_state"].value_counts(normalize=True).round(2).to_dict()
    )
    return out


def describe(model: GaussianHMM, obs: pd.DataFrame) -> pd.DataFrame:
    """Per-state mean return, volatility, persistence (expected duration in days)."""
    order = np.argsort(model.means_[:, 1])
    rows = []
    for name, j in zip(STATE_NAMES, order, strict=True):
        stay = model.transmat_[j, j]
        rows.append(
            {
                "state": name,
                "mean daily return %": model.means_[j, 0],
                "annual vol %": float(np.exp(model.means_[j, 1]) * 100),
                "P(stay)": stay,
                "expected duration (days)": 1 / (1 - stay),
            }
        )
    return pd.DataFrame(rows).set_index("state")
