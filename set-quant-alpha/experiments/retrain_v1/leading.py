"""Task 3b - which drift measures LEAD IC drops?  + fresh-vs-stale paired age curve.

Monthly panel 2013-2026 (A-exp = existing annual walk-forward IC from strategy_v3; 2016+ also the policy grid).
Predictors are measured at month-end m using information available then; targets are the IC of months m+1..m+3.
    uv run set-quant-alpha/experiments/retrain_v1/leading.py        (after fit_grid.py, drift.py, policies.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import DECISION, FITS, INFO, OUT, ic_by_day, load_frame
from scipy.stats import spearmanr

df, feats, best = load_frame()
real = df[["Date", "Ticker", "tradable", "fwd_alpha_5"]]

# --------------------------------------------------------------- A) paired fresh-vs-stale age curve (expanding window)
parts = [pd.read_parquet(f) for f in sorted(FITS.glob("exp_*.parquet"))]
x = pd.concat(parts, ignore_index=True)
x["age"] = (x.Date.dt.year - x.fit_month.dt.year) * 12 + x.Date.dt.month - x.fit_month.dt.month
ageic = {}
for k in range(12):
    s = x[x.age == k].copy()
    s["score"] = (
        s.groupby("Date")["p_logit"].rank(pct=True) + s.groupby("Date")["p_xgb"].rank(pct=True)
    ) / 2
    ageic[k] = ic_by_day(s.merge(real, on=["Date", "Ticker"]), "score")
A = pd.DataFrame(ageic)
A = A[A[0].notna()]  # dates where a fresh (age-0) model exists
rows = {}
for per, (a, b) in {"2016-21": DECISION, "2022-26": INFO}.items():
    S = A[a:b].dropna(how="all")
    for k in range(12):
        d = (S[k] - S[0]).dropna()
        ds = d.iloc[::5]
        rows[(per, k)] = {
            "IC": S[k].mean(),
            "n_days": S[k].count(),
            "dIC_vs_age0_same_dates": d.mean(),
            "t": ds.mean() / ds.std() * np.sqrt(len(ds)) if len(ds) > 2 else np.nan,
        }
AG = pd.DataFrame(rows).T
AG.index.names = ["period", "age_months"]
AG.to_csv(OUT / "staleness_paired_by_age.csv")
print(
    "Paired staleness (expanding window; IC of model of age k minus IC of age-0 model on the SAME dates)"
)
print(AG.round(4).to_string())
# slope of dIC on age
for per, (a, b) in {"2016-21": DECISION, "2022-26": INFO}.items():
    S = A[a:b]
    long = pd.concat(
        [(S[k] - S[0]).rename("d").to_frame().assign(age=k) for k in range(12)]
    ).dropna()
    sl = np.polyfit(long.age, long.d, 1)[0]
    print(f"{per}: IC loss per month of model age = {sl:.5f}  (12 months => {12 * sl:.4f})")
A.to_parquet(OUT / "ic_by_model_age_daily.parquet")

# --------------------------------------------------------------- B) monthly panel
ext = pd.read_parquet(
    Path(__file__).resolve().parents[1] / "strategy_v3/output/predictions_ext.parquet"
)
ic_a = ic_by_day(ext.merge(real, on=["Date", "Ticker"]), "pred_ens").rename("A")
mp = lambda s: s.groupby(s.index.to_period("M").astype(str)).mean()
P = pd.read_parquet(OUT / "policy_daily_ic.parquet")
dm = pd.read_csv(OUT / "drift_monthly.csv", index_col=0)
psi = pd.read_csv(OUT / "drift_feature_psi_monthly.csv", index_col=0)
# date-level (macro / regime) vs stock-level features
liq = df[df.liquid]
within = liq.groupby("Date")[list(psi.columns)].std().mean()
macro = [c for c in psi.columns if within[c] < 1e-9]
stock = [c for c in psi.columns if c not in macro]
print(f"\nfeatures: {len(stock)} stock-level, {len(macro)} date-level (macro/regime): {macro}")
dm["psi_stock_mean"] = psi[stock].mean(axis=1)
dm["psi_stock_max"] = psi[stock].max(axis=1)
dm["psi_macro_mean"] = psi[macro].mean(axis=1)
dm["n_stock_psi_gt_0.10"] = (psi[stock] > 0.10).sum(axis=1)
# raw score drift of the annual champion (xgb / logit probabilities on tradable names)
sc = x[(x.fit_month.dt.month == 1) & (x.age < 12)].merge(
    real[real.tradable][["Date", "Ticker"]], on=["Date", "Ticker"]
)
sc["m"] = sc.Date.dt.to_period("M").astype(str)
g = sc.groupby("m")
dm["score_xgb_std"] = g["p_xgb"].std()
dm["score_xgb_mean"] = g["p_xgb"].mean()
dm["score_logit_std"] = g["p_logit"].std()
dm["score_logit_mean"] = g["p_logit"].mean()
dm["score_xgb_p95"] = g["p_xgb"].quantile(0.95)


# score PSI vs previous 12 months
def _psi(r, u):
    e = np.unique(np.quantile(r, np.linspace(0, 1, 11)[1:-1]))
    e = np.r_[-np.inf, e, np.inf]
    a = np.clip(np.histogram(r, e)[0] / len(r), 1e-4, None)
    b = np.clip(np.histogram(u, e)[0] / len(u), 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())


per_ = sorted(sc.m.unique())
sp = {}
for i, m in enumerate(per_):
    if i >= 12:
        ref = sc[sc.m.isin(per_[i - 12 : i])]
        cur = sc[sc.m == m]
        sp[m] = _psi(ref.p_xgb.to_numpy(), cur.p_xgb.to_numpy())
dm["score_xgb_psi_vs_prev12m"] = pd.Series(sp)
dm["IC_A"] = mp(ic_a)
dm["IC_Mexp"] = mp(P["M-exp"])
dm["IC_Aexp_grid"] = mp(P["A-exp"])
# trailing IC with the 5-day label lag (what a live monitor can know on the last trading day of month m)
lagged = ic_a.shift(5)
dm["trail_ic_60d"] = (
    lagged.rolling(60, min_periods=40)
    .mean()
    .groupby(lagged.index.to_period("M").astype(str))
    .last()
)
dm["trail_ic_250d"] = (
    lagged.rolling(250, min_periods=150)
    .mean()
    .groupby(lagged.index.to_period("M").astype(str))
    .last()
)
IC = dm["IC_A"]
dm["IC_next1"] = IC.shift(-1)
dm["IC_next3"] = (IC.shift(-1) + IC.shift(-2) + IC.shift(-3)) / 3
dm["dIC_next3"] = dm["IC_next3"] - dm["trail_ic_250d"]
dm["dpsi_stock_3m"] = dm["psi_stock_mean"] - dm["psi_stock_mean"].shift(3)
dm.to_csv(OUT / "monthly_panel.csv")

preds = [
    "psi_mean",
    "psi_stock_mean",
    "psi_stock_max",
    "psi_macro_mean",
    "n_stock_psi_gt_0.10",
    "dpsi_stock_3m",
    "base_rate",
    "alpha_xs_std",
    "mkt_dispersion",
    "mkt_vol_20",
    "vix_mean",
    "score_xgb_std",
    "score_logit_std",
    "score_xgb_mean",
    "score_xgb_p95",
    "score_xgb_psi_vs_prev12m",
    "trail_ic_60d",
    "trail_ic_250d",
]
out = []
for p in preds:
    d = dm[
        list(dict.fromkeys([p, "IC_A", "IC_next1", "IC_next3", "dIC_next3", "trail_ic_60d"]))
    ].dropna()
    r = {"predictor": p, "n": len(d)}
    for t in ["IC_A", "IC_next1", "IC_next3", "dIC_next3"]:
        r["rho_" + t], r["p_" + t] = spearmanr(d[p], d[t])
    # incremental information over the trailing 60d IC for IC_next3
    if p != "trail_ic_60d":
        X = sm.add_constant(d[[p, "trail_ic_60d"]].apply(lambda s: (s - s.mean()) / s.std()))
        f = sm.OLS(d["IC_next3"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        r["incr_t_IC_next3_given_trailIC"] = f.tvalues[p]
    # robustness: HAC t of single regression, sign stability across halves, within-year (calendar-age) demeaned version
    z = (d[p] - d[p].mean()) / d[p].std()
    r["hac_t_IC_next3"] = (
        sm.OLS(d["IC_next3"], sm.add_constant(z))
        .fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        .tvalues.iloc[1]
    )
    h1, h2 = d[d.index < "2020-01"], d[d.index >= "2020-01"]
    r["rho_next3_2013-19"] = spearmanr(h1[p], h1["IC_next3"])[0]
    r["rho_next3_2020-26"] = spearmanr(h2[p], h2["IC_next3"])[0]
    yy = d.index.str[:4]
    r["rho_next3_yeardemeaned"] = spearmanr(
        d[p] - d[p].groupby(yy).transform("mean"),
        d["IC_next3"] - d["IC_next3"].groupby(yy).transform("mean"),
    )[0]
    out.append(r)
L = pd.DataFrame(out).set_index("predictor")
L.to_csv(OUT / "lead_lag_summary.csv")
print(
    "\nSpearman: predictor at month-end m vs IC of month m (coincident), m+1, m+3 avg, and IC change vs trailing year"
)
pd.set_option("display.width", 250)
print(
    L[
        [
            "n",
            "rho_IC_A",
            "rho_IC_next1",
            "rho_IC_next3",
            "hac_t_IC_next3",
            "rho_next3_2013-19",
            "rho_next3_2020-26",
            "rho_next3_yeardemeaned",
            "rho_dIC_next3",
            "p_dIC_next3",
            "incr_t_IC_next3_given_trailIC",
        ]
    ]
    .round(3)
    .to_string()
)
# cross-correlation profile for the main candidates
prof = {}
for p in [
    "psi_stock_mean",
    "psi_macro_mean",
    "base_rate",
    "alpha_xs_std",
    "mkt_dispersion",
    "score_xgb_std",
    "trail_ic_60d",
]:
    prof[p] = {k: dm[p].corr(IC.shift(-k), method="spearman") for k in range(-3, 7)}
CC = pd.DataFrame(prof).T
CC.columns = [f"IC(m{k:+d})" for k in CC.columns]
CC.to_csv(OUT / "lead_lag_profile.csv")
print("\nCross-correlation of predictor(m) with IC(m+k); k>0 = predictor LEADS")
print(CC.round(2).to_string())

# --------------------------------------------------------------- C) does retraining help more when drift is high?
if dm["IC_Mexp"].notna().any():
    d = dm.dropna(subset=["IC_Mexp", "IC_Aexp_grid"]).copy()
    d["gain_M_vs_A"] = d["IC_Mexp"] - d["IC_Aexp_grid"]
    print(
        "\nMonthly gain of monthly-refit over annual-refit (M-exp - A-exp), mean %.4f, sd %.4f, n=%d"
        % (d.gain_M_vs_A.mean(), d.gain_M_vs_A.std(), len(d))
    )
    for p in ["psi_stock_mean", "psi_macro_mean", "alpha_xs_std", "score_xgb_psi_vs_prev12m"]:
        dd = d[[p, "gain_M_vs_A"]].dropna()
        rho, pv = spearmanr(dd[p], dd["gain_M_vs_A"])
        hi = dd[dd[p] >= dd[p].median()]["gain_M_vs_A"].mean()
        lo = dd[dd[p] < dd[p].median()]["gain_M_vs_A"].mean()
        print(f"  {p:28s} rho={rho:+.2f} (p={pv:.2f})  gain when high={hi:+.4f}  low={lo:+.4f}")
    d["age"] = pd.PeriodIndex(d.index, freq="M").month - 1
    print("  gain by months since the Jan refit (age):")
    print(d.groupby("age")["gain_M_vs_A"].mean().round(4).to_string())
