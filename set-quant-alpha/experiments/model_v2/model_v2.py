"""model_v2: ML-layer improvement experiments (pre-registered, validation-only selection).

Phases (run both with no args, or one at a time):
  --phase valid : fit every variant on train (2010-19, embargoed), score on VALIDATION (2020-21),
                  write output/valid_comparison.csv + output/chosen.json (selection rule below, coded
                  before any result was seen).
  --phase test  : read chosen.json, walk-forward (refit each January) -> predictions_v2.parquet,
                  test comparison tables. Test is touched exactly once, here.

Pre-registered trial list (12 incl. the re-fit baseline classifier B0); all use the production XGB
hyper-parameters (depth 3, lr .08, 300 trees, min_child_weight 200) so only the formulation changes:
  B0  xgb classifier on is_alpha_5                       (reference re-fit)
  V1  xgb regression, fwd_alpha_5 winsorised 1/99
  V2  xgb regression, per-date cross-sectional rank of fwd_alpha_5 (centred)
  V3  xgb regression, vol-scaled alpha (alpha / (vol_60*sqrt5)), winsorised, per-date demeaned
  V4  xgb rank:pairwise, groups = Date, relevance = per-date quintile of fwd_alpha_5
  V5  V2 + stock-level features replaced by per-date cross-sectional ranks
  V6  V2 trained on tradable names only
  V7  V2 with 10-day target (fwd_alpha_10 rank)
  V8  V2 with 20-day target (fwd_alpha_20 rank)
  V9  ridge (alpha=1000) on CS-ranked features -> CS-rank 5d target
  V10 blend: mean daily rank of V2 and V7
  V11 EMA(3) smoothing (per ticker, past scores only) of the best of V1..V10 by the rule's IC criterion

SELECTION RULE (validation only): candidate must have valid IC >= baseline_IC + 0.002 AND valid
top-decile 5d alpha >= baseline's AND lag-5 score rank-autocorr >= baseline's - 0.02. Choose the max
valid IC among eligible; if none eligible keep the baseline (production pred_ens).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import config, model, transform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRanker, XGBRegressor

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
log = logging.getLogger("model_v2")
XGB = dict(
    max_depth=3,
    learning_rate=0.08,
    n_estimators=300,
    min_child_weight=200,
    subsample=0.8,
    colsample_bytree=0.7,
    tree_method="hist",
    n_jobs=-1,
    random_state=0,
)
DATE_LEVEL = [
    "mkt_ret_1",
    "mkt_ret_5",
    "breadth",
    "dispersion",
    "hmm_p_calm",
    "hmm_p_stressed",
    "usdthb_ret_1",
    "usdthb_ret_5",
    "usdthb_vol_20",
    "eem_ret_1",
    "eem_ret_5",
    "eem_vol_z",
    "spx_ret_5",
    "dxy_ret_5",
    "oil_ret_5",
    "gold_ret_5",
    "credit_spread",
    "credit_spread_chg_5",
    "vix",
    "vix_slope",
    "vix9d_ratio",
    "vvix",
    "skew",
    "vix_chg_5",
    "us10y_chg_5",
    "tbill3m",
]
ALREADY_CS = ["cs_ret_5", "cs_ret_20", "cs_resid_mom_20", "cs_vol_20", "cs_dvol"]
RULE = dict(ic_margin=0.002, ac_tol=0.02)


# ------------------------------------------------------------------ data
def load():
    df = transform.add_dummies(transform.load())
    sets = transform.feature_sets(df)
    feats = sets["NUMERIC"] + sets["DUMMIES"]
    # longer-horizon alphas, built exactly like transform (buy next open, sell close t+h, beta-hedged)
    pr = pd.read_parquet(config.RAW_DIR / "prices.parquet")
    cnt = pr.groupby("Ticker").size()
    pr = pr[pr.Ticker.isin(cnt[cnt >= 250].index)]
    W = lambda f: pr.pivot_table(index="Date", columns="Ticker", values=f).astype("float64")  # noqa: E731
    O, C = W("Open"), W("Close")  # noqa: E741
    liq = (
        df.pivot_table(index="Date", columns="Ticker", values="liquid", aggfunc="first")
        .reindex(C.index)
        .fillna(False)
        .astype(bool)
    )
    beta = df.pivot_table(index="Date", columns="Ticker", values="beta_mkt").reindex(
        index=C.index, columns=C.columns
    )
    for h in (10, 20):
        fr = C.shift(-h) / O.shift(-1) - 1
        fr = fr.where(fr.abs() < 1.0)
        fm = fr.where(liq).mean(axis=1)
        a = (
            (fr - beta.mul(fm, axis=0))
            .stack(future_stack=True)
            .rename(f"fwd_alpha_{h}")
            .reset_index()
        )
        df = df.merge(a, on=["Date", "Ticker"], how="left")
    # per-date cross-sectional rank features (stock-level numerics), liquid rows only
    stock = [
        f for f in feats if f in sets["NUMERIC"] and f not in DATE_LEVEL and f not in ALREADY_CS
    ]
    liqdf = df.loc[df.liquid, ["Date"] + stock]
    ranks = liqdf.groupby("Date")[stock].rank(pct=True).astype("float32").add_prefix("r_")
    df = df.join(ranks)
    rank_feats = [f for f in feats if f not in stock] + list(ranks.columns)
    return df, feats, rank_feats


def lab_rows(df, h_col="fwd_alpha_5", tradable_only=False):
    m = df["liquid"] & df[h_col].notna()
    return df[m & df["tradable"]] if tradable_only else df[m]


def embargo(d, boundary, h):
    days = np.sort(d.loc[d["Date"] < boundary, "Date"].unique())
    return d[d["Date"] < days[-h]] if len(days) > h else d


def train_window(df, start_excl_boundary, h, col, tradable_only=False):
    d = lab_rows(df, col, tradable_only)
    return embargo(d[d["Date"] < start_excl_boundary], start_excl_boundary, h)


# ------------------------------------------------------------------ targets
def cs_rank(d, col):
    return d.groupby("Date")[col].rank(pct=True) - 0.5


def winsor(s, lo=0.01, hi=0.99):
    return s.clip(s.quantile(lo), s.quantile(hi))


def make_target(kind, d):
    if kind == "alpha_w":
        return winsor(d["fwd_alpha_5"])
    if kind == "rank5":
        return cs_rank(d, "fwd_alpha_5")
    if kind == "rank10":
        return cs_rank(d, "fwd_alpha_10")
    if kind == "rank20":
        return cs_rank(d, "fwd_alpha_20")
    if kind == "volscaled":
        z = winsor(d["fwd_alpha_5"] / (d["vol_60"] * np.sqrt(5)))
        return z - z.groupby(d["Date"]).transform("mean")
    raise ValueError(kind)


# variant -> (target col needed for horizon/embargo, h, features key, tradable_only, learner, target kind)
SPEC = {
    "B0": dict(col="fwd_alpha_5", h=5, fs="raw", tr=False, learner="clf", tgt=None),
    "V1": dict(col="fwd_alpha_5", h=5, fs="raw", tr=False, learner="reg", tgt="alpha_w"),
    "V2": dict(col="fwd_alpha_5", h=5, fs="raw", tr=False, learner="reg", tgt="rank5"),
    "V3": dict(col="fwd_alpha_5", h=5, fs="raw", tr=False, learner="reg", tgt="volscaled"),
    "V4": dict(col="fwd_alpha_5", h=5, fs="raw", tr=False, learner="rank", tgt=None),
    "V5": dict(col="fwd_alpha_5", h=5, fs="rank", tr=False, learner="reg", tgt="rank5"),
    "V6": dict(col="fwd_alpha_5", h=5, fs="raw", tr=True, learner="reg", tgt="rank5"),
    "V7": dict(col="fwd_alpha_10", h=10, fs="raw", tr=False, learner="reg", tgt="rank10"),
    "V8": dict(col="fwd_alpha_20", h=20, fs="raw", tr=False, learner="reg", tgt="rank20"),
    "V9": dict(col="fwd_alpha_5", h=5, fs="rank", tr=False, learner="ridge", tgt="rank5"),
}


def fit_predict(name, df, feats, rank_feats, train_end, pred_rows):
    s = SPEC[name]
    tr = train_window(df, train_end, s["h"], s["col"], s["tr"])
    if s["tgt"] == "volscaled":
        tr = tr[tr["vol_60"].notna() & (tr["vol_60"] > 0)]
    F = feats if s["fs"] == "raw" else rank_feats
    if s["learner"] == "clf":
        m = XGBClassifier(eval_metric="auc", **XGB).fit(tr[F], tr[transform.TARGET].astype(int))
        return m.predict_proba(pred_rows[F])[:, 1]
    if s["learner"] == "reg":
        m = XGBRegressor(**XGB).fit(tr[F], make_target(s["tgt"], tr))
        return m.predict(pred_rows[F])
    if s["learner"] == "rank":
        tr = tr.sort_values("Date")
        q = tr.groupby("Date")["fwd_alpha_5"].rank(pct=True)
        y = np.minimum((q * 5).astype(int), 4)
        grp = tr.groupby("Date").size().to_numpy()
        m = XGBRanker(objective="rank:pairwise", **XGB).fit(tr[F], y, group=grp)
        return m.predict(pred_rows[F])
    if s["learner"] == "ridge":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1000.0))
        m.fit(tr[F], make_target(s["tgt"], tr))
        return m.predict(pred_rows[F])
    raise ValueError


def to_rank(df_keys, score):
    return pd.Series(score, index=df_keys.index).groupby(df_keys["Date"]).rank(pct=True)


def ema3(keys, rscore):
    """EMA(span 3) per ticker over past+current scores (rank scores), then re-ranked by date."""
    x = pd.DataFrame({"Date": keys["Date"], "Ticker": keys["Ticker"], "s": rscore}).sort_values(
        ["Ticker", "Date"]
    )
    x["e"] = x.groupby("Ticker")["s"].transform(lambda v: v.ewm(span=3, adjust=False).mean())
    x = x.sort_index()
    return x["e"].groupby(keys["Date"]).rank(pct=True)


# ------------------------------------------------------------------ metrics
def metrics(d, score, extra_ic=("fwd_alpha_10",)):
    """d: rows with Date, Ticker, tradable, fwd_alpha_5, is_alpha_5 and `score`."""
    ic = model.daily_ic(d, score)
    t = d[d["tradable"]].copy()
    t["rk"] = t.groupby("Date")[score].rank(pct=True)
    wide = t.pivot(index="Date", columns="Ticker", values="rk")
    ac1 = wide.corrwith(wide.shift(1), axis=1, method="spearman").mean()
    ac5 = wide.corrwith(wide.shift(5), axis=1, method="spearman").mean()
    top = wide >= 0.9
    overlap5 = (
        (top & top.shift(5, fill_value=False)).sum(axis=1) / top.sum(axis=1).replace(0, np.nan)
    ).mean()
    out = {
        "AUC": roc_auc_score(d[transform.TARGET].astype(int), d[score]) if score != "x" else np.nan,
        "IC": ic.mean(),
        "IC_t": ic.mean() / ic.std() * np.sqrt(ic.count()),
        "top_decile_alpha_%": 100 * model.top_decile_alpha(d, score),
        "rank_ac_lag1": ac1,
        "rank_ac_lag5": ac5,
        "top_decile_5d_overlap": overlap5,
    }
    for c in extra_ic:
        if c in d:
            tt = d[d["tradable"]].dropna(subset=[c])
            out[f"IC_{c}"] = (
                tt.groupby("Date").apply(lambda g: g[score].corr(g[c], method="spearman")).mean()
            )  # noqa: B023
    return out


def by_year(d, score):
    ic = model.daily_ic(d, score)
    return ic.groupby(ic.index.year).mean()


# ------------------------------------------------------------------ phases
def valid_phase(df, feats, rank_feats):
    base = pd.read_parquet(model.PREDICTIONS)
    keys_all = base[base.split == "valid"][["Date", "Ticker"]]
    extra = ["Date", "Ticker", "tradable", "fwd_alpha_5", "fwd_alpha_10", "is_alpha_5"]
    V = keys_all.merge(df, on=["Date", "Ticker"], how="left")
    scores = {}
    bsc = base[base.split == "valid"].merge(V[extra], on=["Date", "Ticker"])
    rows = {"baseline_ens (production)": metrics(bsc, "pred_ens")}
    ic_by_year = {"baseline_ens (production)": by_year(bsc, "pred_ens")}
    V = V.reset_index(drop=True)
    for name in SPEC:
        t0 = time.time()
        raw = fit_predict(name, df, feats, rank_feats, config.VALID_START, V)
        scores[name] = to_rank(V, raw)
        log.info("%s done in %.0fs", name, time.time() - t0)
    scores["V10"] = (scores["V2"] + scores["V7"]) / 2
    cand = [k for k in scores if k not in ("B0",)]
    # V11: smooth the best of V1..V10 by valid IC (selection of the base among trials is also on validation only)
    ics = {k: model.daily_ic(V.assign(x=scores[k]), "x").mean() for k in cand}
    best_pre = max(ics, key=ics.get)
    scores["V11"] = ema3(V, scores[best_pre])
    for k, s in scores.items():
        d = V.assign(x=s)
        rows[k] = metrics(d, "x") if k != "B0" else metrics(d.assign(x=s), "x")
        ic_by_year[k] = by_year(d, "x")
    tab = pd.DataFrame(rows).T
    tab["V11_base"] = ""
    tab.loc["V11", "V11_base"] = best_pre
    tab.to_csv(OUT / "valid_comparison.csv")
    pd.DataFrame(ic_by_year).T.to_csv(OUT / "valid_ic_by_year.csv")
    b = tab.loc["baseline_ens (production)"]
    elig = [
        k
        for k in scores
        if k != "B0"
        and tab.loc[k, "IC"] >= b["IC"] + RULE["ic_margin"]
        and tab.loc[k, "top_decile_alpha_%"] >= b["top_decile_alpha_%"]
        and tab.loc[k, "rank_ac_lag5"] >= b["rank_ac_lag5"] - RULE["ac_tol"]
    ]
    chosen = max(elig, key=lambda k: tab.loc[k, "IC"]) if elig else "baseline"
    json.dump(
        {
            "chosen": chosen,
            "eligible": elig,
            "v11_base": best_pre,
            "rule": RULE,
            "n_trials": len(SPEC) + 2,
        },
        open(OUT / "chosen.json", "w"),
        indent=1,
    )  # noqa: SIM115
    pd.set_option("display.width", 250)
    print(tab.drop(columns="V11_base").round(4).to_string())
    print("eligible:", elig, "-> chosen:", chosen)


def scorer(name, df, feats, rank_feats, train_end, rows):
    """Score for any variant name (incl. composites) as per-date rank on `rows`."""
    if name == "V10":
        return (
            scorer("V2", df, feats, rank_feats, train_end, rows)
            + scorer("V7", df, feats, rank_feats, train_end, rows)
        ) / 2
    return to_rank(rows, fit_predict(name, df, feats, rank_feats, train_end, rows))


def test_phase(df, feats, rank_feats):
    ch = json.load(open(OUT / "chosen.json"))  # noqa: SIM115
    chosen = ch["chosen"]
    base = pd.read_parquet(model.PREDICTIONS)
    extra = ["Date", "Ticker", "tradable", "fwd_alpha_5", "fwd_alpha_10", "is_alpha_5"]
    smooth = chosen == "V11"
    core = ch["v11_base"] if smooth else chosen
    if chosen == "baseline":
        log.info("baseline kept - writing baseline predictions as predictions_v2")
        base.to_parquet(OUT / "predictions_v2.parquet", index=False)
        new = base.copy()
    else:
        parts = []
        v = (
            base[base.split == "valid"][["Date", "Ticker", "split"]]
            .merge(df, on=["Date", "Ticker"], how="left")
            .reset_index(drop=True)
        )
        v["pred_v2"] = scorer(core, df, feats, rank_feats, config.VALID_START, v)
        parts.append(v[["Date", "Ticker", "split", "pred_v2"]])
        te = base[base.split == "test"][["Date", "Ticker", "split"]].merge(
            df, on=["Date", "Ticker"], how="left"
        )
        for yr in sorted(te.Date.dt.year.unique()):
            sub = te[te.Date.dt.year == yr].reset_index(drop=True)
            sub["pred_v2"] = scorer(core, df, feats, rank_feats, f"{yr}-01-01", sub)
            parts.append(sub[["Date", "Ticker", "split", "pred_v2"]])
            log.info("walk-forward %d done", yr)
        new = base.drop(columns=["pred_ens"]).merge(
            pd.concat(parts), on=["Date", "Ticker", "split"]
        )
        if smooth:
            new = new.sort_values(["Date", "Ticker"]).reset_index(drop=True)
            new["pred_v2"] = ema3(new, new["pred_v2"])
        new["pred_ens"] = new["pred_v2"]
        new.to_parquet(OUT / "predictions_v2.parquet", index=False)
    full = new.merge(df[extra], on=["Date", "Ticker"])
    old = base.merge(df[extra], on=["Date", "Ticker"])
    res, yr = {}, {}
    for sp in ("valid", "test"):
        for lbl, d in (("baseline", old), (f"v2:{chosen}", full)):
            x = d[d.split == sp]
            res[(sp, lbl)] = metrics(x, "pred_ens")
            yr[(sp, lbl)] = by_year(x, "pred_ens")
    pd.DataFrame(res).T.to_csv(OUT / "test_comparison.csv")
    pd.DataFrame(yr).T.to_csv(OUT / "ic_by_year_baseline_vs_v2.csv")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(res).T.round(4).to_string())
    print(pd.DataFrame(yr).T.round(4).to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["valid", "test", "both"], default="both")
    a = ap.parse_args()
    df, feats, rank_feats = load()
    if a.phase in ("valid", "both"):
        valid_phase(df, feats, rank_feats)
    if a.phase in ("test", "both"):
        test_phase(df, feats, rank_feats)
