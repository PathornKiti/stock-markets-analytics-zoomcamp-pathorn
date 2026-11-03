"""Diagnostics for REPORT.md: exact-lot check of the TDEX-core overlay, the hindsight-best config's in-sample rank at
each walk-forward decision and its per-year returns vs the WF picks. Run after `integrated_v1.py wf`."""

import sys
from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integrated_v1 as m  # noqa: E402

P, G = joblib.load(m.PANEL), joblib.load(m.GRID)
S, _ = m.build_scores(P, G["ics"])
res, cfgs, dates = G["res"], m.grid(0.0), P["dates"]
bench_r = P["bench"]["Close"].reindex(dates).ffill().pct_change()
picks, tabs, sched = m.walk_forward(P, S, res, cfgs)
oos = slice("2016-01-01", None)

# 1. exact lots with a 50% TDEX core
ex = m.exact_sim(P, S, [(mk, replace(c, core=0.5)) for mk, c in sched])
fs = m.fast_sim(P, S, [(mk, replace(c, core=0.5)) for mk, c in sched])
r = ex["ret"].loc[oos]
print(
    f"50% core: exact CAGR {m.cagr(r):.2f} vs fast {m.cagr(fs['net'].loc[oos]):.2f}; corr {r.corr(fs['net'].loc[oos]):.4f}; "
    f"exact CAPM t {m.capm_t(r, bench_r.reindex(r.index))[1]:.2f}"
)

# 2. hindsight-best config: IS rank at each WF decision and per-year returns
hb = "C2|N10|R5|B|T"
rows = []
for Y in m.WF_YEARS:
    t = tabs[Y]
    el = t[t["eligible"].astype(bool)]["score"].astype(float).sort_values(ascending=False)
    yr = lambda k: 100 * ((1 + m.net_at(res[k].loc[str(Y)], m.FEE)).prod() - 1)  # noqa: E731
    beta = lambda k: m.evaluate.capm(m.net_at(res[k].loc[: f"{Y - 1}-12-31"], m.FEE), bench_r)[
        "beta"
    ]  # noqa: E731
    rows.append(
        {
            "year": Y,
            "WF pick": picks[Y].key,
            "pick IS t": el.iloc[0],
            "pick IS beta": beta(picks[Y].key),
            "hindsight cfg IS rank": int(list(el.index).index(hb)) + 1,
            "hindsight cfg IS t": el[hb],
            "hindsight cfg IS beta": beta(hb),
            "pick OOS %": yr(picks[Y].key),
            "hindsight cfg OOS %": yr(hb),
            "TDEX %": 100 * ((1 + bench_r.loc[str(Y)].dropna()).prod() - 1),
        }
    )
d = pd.DataFrame(rows).set_index("year")
d.to_csv(m.OUT / "hindsight_vs_wf_by_year.csv")
print(d.round(2).to_string())
