"""Task 3c - (i) leave-one-year-out upper bound (LOOK-AHEAD, diagnostic only), (ii) rolling-IC monitor false alarms,
(iii) stock-level vs macro feature drift by year.
    uv run set-quant-alpha/experiments/retrain_v1/diagnostics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from common import OUT, ic_by_day, labelled_frame
from setalpha import model

lab, feats, best = labelled_frame()
# ---- (i) LOYO: train on ALL other years (past AND future, 5-day embargo each side), predict the held-out year.
res = {}
for y in range(2013, 2027):
    s, e = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")
    days = np.sort(lab.Date.unique())
    before = days[days < s]
    after = days[days >= e]
    cut_lo = (
        before[-5] if len(before) >= 5 else s
    )  # drop last 5 days before the year (targets cross the boundary)
    tr = lab[(lab.Date < cut_lo) | (lab.Date >= e)]
    te = lab[(lab.Date >= s) & (lab.Date < e)].copy()
    sc = []
    for name in ["logit", "xgb"]:
        m = model.make_model(name, **best[name]).fit(tr[feats], tr[model.TARGET].astype(int))
        te["p_" + name] = m.predict_proba(te[feats])[:, 1]
        sc.append(te.groupby("Date")["p_" + name].rank(pct=True))
    te["score"] = sum(sc) / 2
    res[y] = ic_by_day(te, "score").mean()
    print(y, round(res[y], 4), flush=True)
loyo = pd.Series(res, name="LOYO_IC_lookahead")
ext = pd.read_csv(
    Path(__file__).resolve().parents[1] / "strategy_v3/output/ic_by_year_2013_2026.csv", index_col=0
)["mean"]
cmp = pd.concat([ext.rename("annual_walkforward_IC"), loyo], axis=1)
cmp["gap"] = cmp.LOYO_IC_lookahead - cmp.annual_walkforward_IC
cmp.to_csv(OUT / "loyo_vs_walkforward.csv")
print(cmp.round(4).to_string())
print(
    "mean 2013-19: walk-forward %.4f  LOYO %.4f | mean 2020-26: walk-forward %.4f  LOYO %.4f"
    % (
        cmp.loc[:2019, "annual_walkforward_IC"].mean(),
        cmp.loc[:2019, "LOYO_IC_lookahead"].mean(),
        cmp.loc[2020:, "annual_walkforward_IC"].mean(),
        cmp.loc[2020:, "LOYO_IC_lookahead"].mean(),
    )
)

# ---- (ii) rolling-IC monitor: false alarms and what a flag implies for the next 60 days
ic = ic_by_day(
    pd.read_parquet(
        Path(__file__).resolve().parents[1] / "strategy_v3/output/predictions_ext.parquet"
    ).merge(
        pd.read_parquet(
            Path(__file__).resolve().parents[2] / "data/processed/dataset.parquet"
            if False
            else OUT / "none.parquet"
        )
        if False
        else lab[["Date", "Ticker", "tradable", "fwd_alpha_5"]],
        on=["Date", "Ticker"],
    ),
    "pred_ens",
)
lag = ic.shift(5)  # an IC is only known 5 trading days after the signal date
rows = []
for W in (60, 120, 250):
    roll = lag.rolling(W, min_periods=int(W * 0.7)).mean()
    fwd = (
        ic[::-1].rolling(60, min_periods=40).mean()[::-1].shift(-1)
    )  # mean IC over the NEXT 60 days
    se = ic.std() * np.sqrt(5 / W)  # SE of a W-day mean with 5-day overlapping targets
    for thr in (0.0, 0.02, 0.03):
        f = roll < thr
        d = pd.DataFrame({"roll": roll, "fwd": fwd})["2016":].dropna()
        fl = d.roll < thr
        rows.append(
            {
                "window": W,
                "threshold": thr,
                "SE_of_window_mean": se,
                "share_of_days_flagged": fl.mean(),
                "next60d_IC_if_flagged": d.fwd[fl].mean(),
                "next60d_IC_if_not": d.fwd[~fl].mean(),
                "next60d_IC_below_0_if_flagged": (d.fwd[fl] < 0).mean(),
                "next60d_IC_below_0_if_not": (d.fwd[~fl] < 0).mean(),
            }
        )
MON = pd.DataFrame(rows)
MON.to_csv(OUT / "monitor_false_alarms.csv", index=False)
print("\nRolling-IC monitor (2016-2026, A-exp ensemble; IC lagged 5d)")
print(MON.round(3).to_string(index=False))
print(
    "daily IC sd = %.3f; non-overlapping-equivalent SE of a 60d mean = %.3f"
    % (ic.std(), ic.std() * np.sqrt(5 / 60))
)

# ---- (iii) stock-level vs macro feature drift
psi = pd.read_csv(OUT / "drift_feature_psi_by_year.csv", index_col=0)
macro = [
    "us10y_chg_5",
    "tbill3m",
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
]
st = psi.drop(index=[m for m in macro if m in psi.index])
S = pd.DataFrame(
    {
        "stock_mean_PSI": st.mean(),
        "stock_n_gt_0.10": (st > 0.10).sum(),
        "stock_n_gt_0.25": (st > 0.25).sum(),
        "macro_mean_PSI": psi.loc[psi.index.isin(macro)].mean(),
    }
)
print("\nStock-level features (39) vs macro/regime features (26): PSI vs training window by year")
print(S.round(3).T.to_string())
S.T.to_csv(OUT / "drift_stock_vs_macro_by_year.csv")
print("\nTop-5 STOCK-level drifters by year")
for y in st.columns:
    print(
        y, ", ".join(f"{k}({v:.2f})" for k, v in st[y].sort_values(ascending=False).head(5).items())
    )
