"""Task 1 - staleness curve of the EXISTING annual-refit walk-forward ensemble (strategy_v3/predictions_ext.parquet).

IC / top-decile alpha by months since the January refit, pooled by era, with t-stats from non-overlapping days
(every 5th).  Calendar seasonality is confounded with age here (annual refit => age == month of year); the
fresh-vs-stale paired comparison in policies.py (same dates, model age 0 vs k) removes that confound.
    uv run set-quant-alpha/experiments/retrain_v1/staleness.py
"""

import numpy as np
import pandas as pd
from common import OUT, ic_by_day, load_frame, top_decile

df, feats, best = load_frame()
p = pd.read_parquet(
    "set-quant-alpha/experiments/strategy_v3/output/predictions_ext.parquet"
    if False
    else __import__("pathlib").Path(__file__).resolve().parents[1]
    / "strategy_v3/output/predictions_ext.parquet"
)
d = p.merge(df[["Date", "Ticker", "tradable", "fwd_alpha_5"]], on=["Date", "Ticker"])
ic = ic_by_day(d, "pred_ens")
td = top_decile(d, "pred_ens")
t = pd.DataFrame({"ic": ic, "td": td})
t["year"] = t.index.year
t["age"] = t.index.month - 1  # months since the January refit


def agg(g):
    x = g["ic"].dropna()
    xs = x.iloc[::5]
    return pd.Series(
        {
            "IC": x.mean(),
            "t_nonoverlap": xs.mean() / xs.std() * np.sqrt(len(xs)),
            "top_decile_alpha_%": 100 * g["td"].mean(),
            "days": len(x),
        }
    )


eras = {"2013-19": t[t.year <= 2019], "2020-26": t[t.year >= 2020], "2013-26": t}
rows = {k: v.groupby("age").apply(agg) for k, v in eras.items()}
res = pd.concat(rows, axis=1)
res.to_csv(OUT / "staleness_annual_by_age.csv")
print(res.round(4).to_string())
# half-year buckets
t["half"] = np.where(t.age < 6, "H1 (age 0-5m)", "H2 (age 6-11m)")
h = t.groupby(["half", t.year >= 2020]).apply(agg)
print(h.round(4))
# IC by quarter of age, per year (H2 - H1 difference)
y = t.groupby(["year", "half"])["ic"].mean().unstack()
y["H2-H1"] = y.iloc[:, 1] - y.iloc[:, 0]
y.to_csv(OUT / "staleness_annual_h1_h2_by_year.csv")
print(y.round(4))
print(
    "mean H2-H1: all %.4f, 2013-19 %.4f, 2020-26 %.4f"
    % (y["H2-H1"].mean(), y.loc[:2019, "H2-H1"].mean(), y.loc[2020:, "H2-H1"].mean())
)
# regression of daily IC on age (months) with year fixed effects, SE via year-block bootstrap
tt = t.dropna(subset=["ic"]).copy()
tt["ic_dm"] = tt["ic"] - tt.groupby("year")["ic"].transform("mean")
tt["age_dm"] = tt["age"] - tt.groupby("year")["age"].transform("mean")
slope = (tt.ic_dm * tt.age_dm).sum() / (tt.age_dm**2).sum()
rng = np.random.default_rng(0)
yrs = tt.year.unique()
bs = []
for _ in range(500):
    s = rng.choice(yrs, len(yrs))
    b = pd.concat([tt[tt.year == y_] for y_ in s])
    bs.append((b.ic_dm * b.age_dm).sum() / (b.age_dm**2).sum())
print(
    "within-year IC slope per month of age: %.5f  (year-bootstrap 90%% CI %.5f..%.5f)"
    % (slope, *np.percentile(bs, [5, 95]))
)
pd.Series(
    {"slope_per_month": slope, "lo90": np.percentile(bs, 5), "hi90": np.percentile(bs, 95)}
).to_csv(OUT / "staleness_slope.csv")
