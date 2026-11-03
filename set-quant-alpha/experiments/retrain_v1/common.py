"""Shared loaders for retrain_v1 (read-only use of setalpha)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import config, model, transform

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
FITS = OUT / "fits"
FITS.mkdir(exist_ok=True)
DECISION = ("2016-01-01", "2021-12-31")  # policy is chosen here only
INFO = ("2022-01-01", "2026-12-31")  # reported once, for information


def load_frame():
    df = transform.add_dummies(transform.load())
    sets = transform.feature_sets(df)
    feats = sets["NUMERIC"] + sets["DUMMIES"]
    best = json.loads((config.MODELS_DIR / "best_params.json").read_text())
    return df, feats, best


def labelled_frame():
    df, feats, best = load_frame()
    lab = model.labelled(df)
    keep = ["Date", "Ticker", "tradable", "fwd_alpha_5", model.TARGET] + feats
    lab = lab[keep].reset_index(drop=True)
    lab[feats] = lab[feats].astype("float32")
    return lab, feats, best


def ic_by_day(d: pd.DataFrame, score: str) -> pd.Series:
    """Daily Spearman IC on tradable names (fast: rank then pearson)."""
    d = d[d["tradable"]].dropna(subset=["fwd_alpha_5", score])
    r1 = d.groupby("Date")[score].rank()
    r2 = d.groupby("Date")["fwd_alpha_5"].rank()
    g = pd.DataFrame({"Date": d["Date"], "a": r1, "b": r2, "n": 1})
    g["aa"] = g.a * g.a
    g["bb"] = g.b * g.b
    g["ab"] = g.a * g.b
    s = g.groupby("Date")[["a", "b", "aa", "bb", "ab", "n"]].sum()
    cov = s.ab / s.n - (s.a / s.n) * (s.b / s.n)
    va = s.aa / s.n - (s.a / s.n) ** 2
    vb = s.bb / s.n - (s.b / s.n) ** 2
    return (cov / np.sqrt(va * vb)).where(s.n >= 20)


def top_decile(d: pd.DataFrame, score: str) -> pd.Series:
    """Daily mean fwd alpha of top-decile (by score) tradable names."""
    d = d[d["tradable"]].dropna(subset=["fwd_alpha_5", score])
    pct = d.groupby("Date")[score].rank(pct=True)
    return d[pct >= 0.9].groupby("Date")["fwd_alpha_5"].mean()
