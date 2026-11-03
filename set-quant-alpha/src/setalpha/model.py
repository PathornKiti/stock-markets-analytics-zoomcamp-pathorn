"""Step 3 - models: DecisionTree, RandomForest, Logistic regression, XGBoost.

Protocol (no look-ahead):
* train  < 2020-01-01, valid 2020-2021, test >= 2022-01-01, with a 5-day embargo before
  each boundary because the 5-day target windows overlap;
* hyperparameters are chosen on the validation years only;
* the test years are predicted walk-forward: refit every January on all earlier data.

Models are trained on *liquid* rows (all Thai stocks above the liquidity filter);
trading later uses only the tradable top-100.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from setalpha import config
from setalpha.transform import TARGET

log = logging.getLogger(__name__)
PREDICTIONS = config.PROCESSED_DIR / "predictions.parquet"
MODEL_NAMES = ["tree", "rf", "logit", "xgb"]
MAX_TRAIN_ROWS = {
    "tree": 600_000,
    "rf": 400_000,
    "logit": 600_000,
    "xgb": None,
}  # speed caps (random sample)

# Small, readable grids: tuned on 2020-2021 by daily top-decile alpha (what the strategy trades).
GRIDS = {
    "tree": {"max_depth": [4, 6, 8], "min_samples_leaf": [500, 2000]},
    "rf": {"max_depth": [8, 12], "min_samples_leaf": [200, 1000], "max_features": [0.3]},
    "logit": {"C": [0.01, 0.1, 1.0]},
    "xgb": {
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.08],
        "n_estimators": [300],
        "min_child_weight": [200],
    },
}


def make_model(name: str, **params):
    if name == "tree":
        return make_pipeline(
            SimpleImputer(strategy="median"), DecisionTreeClassifier(random_state=0, **params)
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=0, **params),
        )
    if name == "logit":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=500, **params),
        )
    if name == "xgb":
        return XGBClassifier(
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method="hist",
            n_jobs=-1,
            random_state=0,
            eval_metric="auc",
            **params,
        )
    raise ValueError(name)


@dataclass
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def _embargo(df: pd.DataFrame, boundary: str) -> pd.DataFrame:
    """Drop the last HORIZON trading days before `boundary` (their targets cross it)."""
    days = np.sort(df.loc[df["Date"] < boundary, "Date"].unique())
    return df[df["Date"] < days[-config.HORIZON]] if len(days) > config.HORIZON else df


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["liquid"] & df[TARGET].notna()]


def time_split(df: pd.DataFrame) -> Split:
    lab = labelled(df)
    train = _embargo(lab[lab["Date"] < config.VALID_START], config.VALID_START)
    valid = _embargo(
        lab[(lab["Date"] >= config.VALID_START) & (lab["Date"] < config.TEST_START)],
        config.TEST_START,
    )
    test = lab[lab["Date"] >= config.TEST_START]
    return Split(train, valid, test)


def _sample(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cap = MAX_TRAIN_ROWS[name]
    return df.sample(cap, random_state=0) if cap and len(df) > cap else df


def fit(name: str, train: pd.DataFrame, features: list[str], **params):
    tr = _sample(train, name)
    return make_model(name, **params).fit(tr[features], tr[TARGET].astype(int))


def predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]


# ----------------------------------------------------------------------------- metrics
def top_decile_alpha(df: pd.DataFrame, score: str) -> float:
    """Mean 5-day alpha of the stocks a model ranks in its daily top 10% (tradable names)."""
    d = df[df["tradable"]]
    pct = d.groupby("Date")[score].rank(pct=True)
    return float(d.loc[pct >= 0.9, "fwd_alpha_5"].mean())


def daily_ic(df: pd.DataFrame, score: str) -> pd.Series:
    """Spearman rank correlation between score and realised 5-day alpha, per day."""
    d = df[df["tradable"]].dropna(subset=["fwd_alpha_5"])
    return d.groupby("Date").apply(lambda g: g[score].corr(g["fwd_alpha_5"], method="spearman"))


def scorecard(df: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    rows = {}
    for s in scores:
        ic = daily_ic(df, s)
        y = df[TARGET].astype(int)
        rows[s] = {
            "AUC": roc_auc_score(y, df[s]),
            "precision@0.5": float(y[df[s] > 0.5].mean()) if (df[s] > 0.5).any() else np.nan,
            "IC mean": ic.mean(),
            "IC t-stat": ic.mean() / ic.std() * np.sqrt(ic.count()),
            "top-decile 5d alpha %": 100 * top_decile_alpha(df, s),
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- tuning
def tune(
    split: Split, features: list[str], names: list[str] = MODEL_NAMES
) -> tuple[dict, pd.DataFrame]:
    """Grid search on the validation years; returns best params per model and the full log."""
    best, logs = {}, []
    for name in names:
        grid = GRIDS[name]
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), combo, strict=True))
            model = fit(name, split.train, features, **params)
            v = split.valid.assign(score=predict(model, split.valid, features))
            row = {
                "model": name,
                **params,
                "valid_AUC": roc_auc_score(v[TARGET].astype(int), v["score"]),
                "valid_top_decile_alpha_%": 100 * top_decile_alpha(v, "score"),
            }
            logs.append(row)
            log.info(
                "    tune %s %s -> AUC %.4f, top-decile alpha %.3f%%",
                name,
                params,
                row["valid_AUC"],
                row["valid_top_decile_alpha_%"],
            )
        table = pd.DataFrame([r for r in logs if r["model"] == name])
        top = table.sort_values("valid_top_decile_alpha_%", ascending=False).iloc[0]
        best[name] = {k: type(grid[k][0])(top[k]) for k in grid}  # back to the grid's own type
    return best, pd.DataFrame(logs)


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(
    df: pd.DataFrame, features: list[str], best: dict, names: list[str] = MODEL_NAMES
) -> pd.DataFrame:
    """Validation predictions from the train-only fit + test predictions refit every January."""
    split = time_split(df)
    lab = labelled(df)
    out = [
        split.valid[["Date", "Ticker"]].copy(),
    ]
    out[0]["split"] = "valid"
    for name in names:
        out[0][f"pred_{name}"] = predict(
            fit(name, split.train, features, **best[name]), split.valid, features
        )

    years = sorted(split.test["Date"].dt.year.unique())
    for yr in years:
        start = f"{yr}-01-01"
        train = _embargo(lab[lab["Date"] < start], start)
        test = split.test[split.test["Date"].dt.year == yr]
        part = test[["Date", "Ticker"]].copy()
        part["split"] = "test"
        for name in names:
            part[f"pred_{name}"] = predict(fit(name, train, features, **best[name]), test, features)
        out.append(part)
        log.info(
            "    walk-forward %d: trained on %s rows, predicted %s",
            yr,
            f"{len(train):,}",
            f"{len(test):,}",
        )
    preds = pd.concat(out, ignore_index=True)
    preds = add_ensemble(preds, choose_ensemble(preds, df, names)[0])
    preds.to_parquet(PREDICTIONS, index=False)
    return preds


def choose_ensemble(
    preds: pd.DataFrame, df: pd.DataFrame, names=MODEL_NAMES, k: int = 2
) -> tuple[list[str], dict]:
    """The k model families with the best top-decile alpha on the VALIDATION years only."""
    v = preds[preds["split"] == "valid"].merge(
        df[["Date", "Ticker", "tradable", "fwd_alpha_5"]], on=["Date", "Ticker"]
    )
    scores = {n: top_decile_alpha(v, f"pred_{n}") for n in names}
    members = (
        list(config.ENSEMBLE)
        if config.ENSEMBLE
        else sorted(scores, key=scores.get, reverse=True)[:k]
    )
    log.info(
        "    ensemble members (%s; validation top-decile alpha): %s  %s",
        "frozen" if config.ENSEMBLE else "chosen",
        members,
        {n: round(100 * a, 3) for n, a in scores.items()},
    )
    return members, scores


def add_ensemble(preds: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """pred_ens = average daily percentile rank of the member models."""
    ranks = [preds.groupby("Date")[f"pred_{m}"].rank(pct=True) for m in members]
    preds["pred_ens"] = sum(ranks) / len(ranks)
    return preds


# ----------------------------------------------------------------------------- production
def fit_final(df: pd.DataFrame, features: list[str], best: dict, names: list[str]) -> dict:
    """Refit on every labelled row up to today and save, for the next-day signals."""
    lab = labelled(df)
    models = {}
    for name in names:
        models[name] = fit(name, lab, features, **best[name])
        joblib.dump(
            {"model": models[name], "features": features, "params": best[name]},
            config.MODELS_DIR / f"{name}.joblib",
        )
    return models


def load_final(name: str):
    return joblib.load(config.MODELS_DIR / f"{name}.joblib")


def feature_importance(model, features: list[str], top: int = 25) -> pd.Series:
    est = model[-1] if hasattr(model, "steps") else model
    return (
        pd.Series(est.feature_importances_, index=features).sort_values(ascending=False).head(top)
    )
