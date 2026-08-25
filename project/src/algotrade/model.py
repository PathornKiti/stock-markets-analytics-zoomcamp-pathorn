"""Step 3 - train and evaluate return-prediction models.

Splits are chronological, never random: a shuffled split would train on the
future and score on the past, which inflates every metric downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

MODELS = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def time_split(
    df: pd.DataFrame, valid_start: str, test_start: str, date_col: str = "Date"
) -> Split:
    """Chronological train/validation/test split on absolute dates."""
    d = pd.to_datetime(df[date_col])
    return Split(
        train=df[d < valid_start],
        valid=df[(d >= valid_start) & (d < test_start)],
        test=df[d >= test_start],
    )


def fit_classifier(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    kind: str = "xgboost",
    **kwargs,
):
    """Train one of the lecture models (or XGBoost) on the training slice."""
    X = train[features].astype(float)
    y = train[target].astype(int)

    if kind == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        params = {"max_depth": 6, "random_state": 42, **kwargs}
        clf = DecisionTreeClassifier(**params)
    elif kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = {"n_estimators": 300, "max_depth": 8, "n_jobs": -1, "random_state": 42, **kwargs}
        clf = RandomForestClassifier(**params)
    elif kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **kwargs))
    elif kind == "xgboost":
        from xgboost import XGBClassifier

        params = {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            **kwargs,
        }
        clf = XGBClassifier(**params)
    else:
        raise ValueError(f"unknown model kind: {kind!r}")

    return clf.fit(X.fillna(0.0), y)


def evaluate(clf, frame: pd.DataFrame, features: list[str], target: str) -> pd.Series:
    """Classification metrics on a held-out slice."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    X = frame[features].astype(float).fillna(0.0)
    y = frame[target].astype(int)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]
    return pd.Series(
        {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, proba),
            "n": len(frame),
            "base_rate": float(y.mean()),
        }
    )


def predict_proba(clf, frame: pd.DataFrame, features: list[str]) -> pd.Series:
    """Probability of the positive class, aligned to `frame`'s index."""
    X = frame[features].astype(float).fillna(0.0)
    return pd.Series(clf.predict_proba(X)[:, 1], index=frame.index, name="proba")


def save(clf, name: str) -> Path:
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{name}.joblib"
    joblib.dump(clf, path)
    return path


def load(name: str):
    return joblib.load(MODELS / f"{name}.joblib")
