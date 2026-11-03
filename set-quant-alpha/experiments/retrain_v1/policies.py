"""Task 2 - retraining policy comparison at signal level (logit + XGB rank-average, production hyperparameters).

Policies (8, fixed before looking at results; decision on 2016-2021 only, 2022-2026 reported for information):
  A-exp   annual    expanding   (= research walk-forward, refit every January)
  Q-exp   quarterly expanding   (Jan/Apr/Jul/Oct)
  M-exp   monthly   expanding   (~ production: weekly expanding refit)
  A-r5    annual    rolling 5y
  Q-r5    quarterly rolling 5y
  M-r5    monthly   rolling 5y
  M-r3    monthly   rolling 3y
  M-expW  monthly   expanding, sample weight half-life 2y
Decision rule (pre-registered): a policy beats the incumbent M-exp/A-exp only if its 2016-21 mean IC is higher AND the
paired non-overlapping t-stat of the daily-IC difference is > 2; otherwise the cheapest cadence wins (ties -> simpler).
    uv run set-quant-alpha/experiments/retrain_v1/policies.py
"""

import numpy as np
import pandas as pd
from common import DECISION, FITS, INFO, OUT, ic_by_day, load_frame, top_decile

POLICIES = {  # name: (window, cadence months)
    "A-exp": ("exp", 12),
    "Q-exp": ("exp", 3),
    "M-exp": ("exp", 1),
    "A-r5": ("r5", 12),
    "Q-r5": ("r5", 3),
    "M-r5": ("r5", 1),
    "M-r3": ("r3", 1),
    "M-expW": ("expW", 1),
}
df, feats, best = load_frame()
real = df[["Date", "Ticker", "tradable", "fwd_alpha_5"]]


def load_window(w):
    parts = [pd.read_parquet(f) for f in sorted(FITS.glob(f"{w}_*.parquet"))]
    x = pd.concat(parts, ignore_index=True)
    x["age"] = (x.Date.dt.year - x.fit_month.dt.year) * 12 + x.Date.dt.month - x.fit_month.dt.month
    return x


def ensemble(x):
    r = (
        x.groupby("Date")["p_logit"].rank(pct=True) + x.groupby("Date")["p_xgb"].rank(pct=True)
    ) / 2
    return r


def select(x, c):
    """rows of the policy refitting every c months, Jan-aligned."""
    on_grid = (x.fit_month.dt.month - 1) % c == 0
    return x[on_grid & (x.age < c)]


def persistence(s, refit_dates):
    """s: DataFrame Date,Ticker,tradable,score. lag-1 rank autocorr and top-decile turnover (tradable only)."""
    s = s[s.tradable]
    w = s.assign(r=s.groupby("Date")["score"].rank(pct=True)).pivot(
        index="Date", columns="Ticker", values="r"
    )
    ac = w.corrwith(w.shift(1), axis=1)
    top = w >= 0.9
    prev = top.shift(1, fill_value=False)
    to = 1 - (top & prev).sum(axis=1) / top.sum(axis=1).clip(lower=1)
    ac = ac.iloc[1:]
    to = to.iloc[1:]
    first = ac.index.isin(refit_dates)
    return {
        "rank_autocorr_1d": ac.mean(),
        "autocorr_on_refit_day": ac[first].mean() if first.any() else np.nan,
        "autocorr_other_days": ac[~first].mean(),
        "top_decile_turnover_1d": to.mean(),
    }


series, rows, yearly, pers = {}, [], {}, {}
data = {}
for name, (w, c) in POLICIES.items():
    if w not in data:
        data[w] = load_window(w)
    x = select(data[w], c)
    assert not x.duplicated(["Date", "Ticker"]).any()
    x = x.assign(score=ensemble(x)).merge(real, on=["Date", "Ticker"])
    ic = ic_by_day(x, "score")
    td = top_decile(x, "score")
    series[name] = pd.DataFrame({"ic": ic, "td": td})
    # first trading day of each refit month
    refit_first = x.loc[x.age == 0].groupby("Date").size().index
    refit_first = pd.DatetimeIndex(
        sorted(
            x.groupby(x.Date.dt.to_period("M")).Date.min()[lambda s: s.dt.month.sub(1).mod(c).eq(0)]
        )
    )
    pers[name] = persistence(x, refit_first)
    print(name, "done", flush=True)

ICs = pd.DataFrame({k: v.ic for k, v in series.items()})
TDs = pd.DataFrame({k: v.td for k, v in series.items()})
ICs.to_parquet(OUT / "policy_daily_ic.parquet")
TDs.to_parquet(OUT / "policy_daily_td.parquet")
yearly = ICs.groupby(ICs.index.year).mean()
yearly.loc["2016-21"] = ICs[DECISION[0] : DECISION[1]].mean()
yearly.loc["2022-26"] = ICs[INFO[0] : INFO[1]].mean()
yearly.T.to_csv(OUT / "policy_ic_by_year.csv")
print("\nIC by year\n", yearly.T.round(4).to_string())
print("\nTop-decile 5d alpha % by year\n")
tdy = 100 * TDs.groupby(TDs.index.year).mean()
tdy.loc["2016-21"] = 100 * TDs[DECISION[0] : DECISION[1]].mean()
tdy.loc["2022-26"] = 100 * TDs[INFO[0] : INFO[1]].mean()
tdy.T.to_csv(OUT / "policy_td_by_year.csv")
print(tdy.T.round(3).to_string())


def tstat(a):
    a = a.dropna().iloc[::5]
    return a.mean() / a.std() * np.sqrt(len(a)) if len(a) > 2 else np.nan


summ = {}
for per, (a, b) in {"2016-21 (decision)": DECISION, "2022-26 (info)": INFO}.items():
    I = ICs[a:b]
    for n in ICs:
        d_base = I[n] - I["A-exp"]
        d_m = I[n] - I["M-exp"]
        s = I[n].dropna()
        summ[(per, n)] = {
            "IC": s.mean(),
            "ICIR_daily": s.mean() / s.std(),
            "ICIR_annualised_5d": s.mean() / s.std() * np.sqrt(50),
            "top_decile_alpha_%": 100 * TDs[a:b][n].mean(),
            "dIC_vs_A-exp": d_base.mean(),
            "t_vs_A-exp": tstat(d_base),
            "dIC_vs_M-exp": d_m.mean(),
            "t_vs_M-exp": tstat(d_m),
            **{k: v for k, v in pers[n].items()},
        }
S = pd.DataFrame(summ).T
S.to_csv(OUT / "policy_summary.csv")
print("\n", S.round(4).to_string())
