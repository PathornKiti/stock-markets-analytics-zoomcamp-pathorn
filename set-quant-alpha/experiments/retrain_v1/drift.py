"""Task 3a - drift measures (label-free ones are usable as live retrain triggers).

* feature PSI: month (or year) of liquid rows vs the training window of the annual champion (all liquid rows
  from 2010 up to the previous 1 January), 10 reference-decile bins; summarised as mean / max PSI and the number of
  features with PSI > 0.25 (numeric features only; the one-hot dummies are excluded).
* label base rate: share of is_alpha_5 = 1 and cross-sectional std of fwd_alpha_5 (labels mature 5 days late).
* market-state context: dispersion, breadth, EW market 20d vol.
Outputs: output/drift_monthly.csv, drift_feature_psi_by_year.csv, drift_top_features_by_year.csv
    uv run set-quant-alpha/experiments/retrain_v1/drift.py
"""

import numpy as np
import pandas as pd
from common import OUT, load_frame
from setalpha import model

df, feats, best = load_frame()
sets_num = [f for f in feats if df[f].dtype != "int8"]
liq = df[df.liquid].copy()
X = liq[sets_num].astype("float32")
dates = liq["Date"]
EPS = 1e-4


def psi_vs(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    out = {}
    for c in ref.columns:
        r = ref[c].dropna().to_numpy()
        u = cur[c].dropna().to_numpy()
        if len(r) < 1000 or len(u) < 200:
            out[c] = np.nan
            continue
        e = np.unique(np.quantile(r, np.linspace(0, 1, 11)[1:-1]))
        e = np.r_[-np.inf, e, np.inf]
        pr = np.histogram(r, e)[0] / len(r)
        pu = np.histogram(u, e)[0] / len(u)
        pr = np.clip(pr, EPS, None)
        pu = np.clip(pu, EPS, None)
        out[c] = float(((pu - pr) * np.log(pu / pr)).sum())
    return pd.Series(out)


months = pd.period_range("2013-01", "2026-10", freq="M")
rows, psi_m = [], {}
for p in months:
    ref = X[dates < pd.Timestamp(year=p.year, month=1, day=1)]
    cur = X[(dates >= p.start_time) & (dates <= p.end_time)]
    psi = psi_vs(ref, cur)
    psi_m[str(p)] = psi
    rows.append(
        {
            "month": str(p),
            "psi_mean": psi.mean(),
            "psi_max": psi.max(),
            "n_psi_gt_0.25": int((psi > 0.25).sum()),
            "n_psi_gt_0.10": int((psi > 0.10).sum()),
            "n_rows": len(cur),
        }
    )
M = pd.DataFrame(rows).set_index("month")
# also a rolling 3-month window (more rows => less sampling noise in PSI)
lab = liq[liq[model.TARGET].notna()]
g = lab.groupby(lab.Date.dt.to_period("M").astype(str))
M["base_rate"] = g[model.TARGET].mean()
M["alpha_xs_std"] = g["fwd_alpha_5"].std()
M["alpha_mean_liquid"] = g["fwd_alpha_5"].mean()
daily = liq.groupby("Date")[["dispersion", "breadth", "mkt_ret_1", "vix"]].first()
dm = daily.groupby(daily.index.to_period("M").astype(str)).mean()
M = M.join(
    dm.rename(
        columns={
            "dispersion": "mkt_dispersion",
            "breadth": "mkt_breadth",
            "mkt_ret_1": "mkt_ret_1d_mean",
            "vix": "vix_mean",
        }
    )
)
M["mkt_vol_20"] = daily["mkt_ret_1"].groupby(daily.index.to_period("M").astype(str)).std()
M.to_csv(OUT / "drift_monthly.csv")
pd.DataFrame(psi_m).T.to_csv(OUT / "drift_feature_psi_monthly.csv")

# yearly PSI vs the training window before 1 Jan of that year
yrows = {}
for y in range(2013, 2027):
    ref = X[dates < pd.Timestamp(year=y, month=1, day=1)]
    cur = X[dates.dt.year == y]
    yrows[y] = psi_vs(ref, cur)
Y = pd.DataFrame(yrows)
Y.to_csv(OUT / "drift_feature_psi_by_year.csv")
top = {
    y: ", ".join(f"{k}({v:.2f})" for k, v in Y[y].sort_values(ascending=False).head(6).items())
    for y in Y
}
pd.Series(top).to_csv(OUT / "drift_top_features_by_year.csv", header=["top6_psi_features"])
print("Top-6 drifting features by year (PSI vs train window before 1 Jan)")
for y, s in top.items():
    print(y, s)
print("\nPSI summary by year (mean over features / n features > 0.25 / > 0.10)")
print(
    pd.DataFrame({"mean": Y.mean(), "gt0.25": (Y > 0.25).sum(), "gt0.10": (Y > 0.10).sum()})
    .round(3)
    .T.to_string()
)
# features that drift persistently
print("\nFeatures with mean PSI over 2016-2026 highest:")
print(Y.loc[:, 2016:].mean(axis=1).sort_values(ascending=False).head(12).round(3).to_string())
print("\nLabel base rate / xs std by year")
by = lab.groupby(lab.Date.dt.year).agg(
    base_rate=(model.TARGET, "mean"), alpha_xs_std=("fwd_alpha_5", "std")
)
print(by.round(4).T.to_string())
by.to_csv(OUT / "label_base_rate_by_year.csv")
