"""Fit logit+XGB (production hyperparameters) at the start of every month 2016-01..2026-10 for four window types,
predict the next 12 months (age 0..11). Each fit is cached to output/fits/<window>_<YYYY-MM>.parquet.

Window types: exp (expanding from 2010), r5 (rolling 5y), r3 (rolling 3y), expW (expanding, sample weights with a
2-year half-life).  No look-ahead: train = labelled liquid rows with Date < month start, minus model._embargo.
    uv run set-quant-alpha/experiments/retrain_v1/fit_grid.py [window ...]
"""

import sys
import time

import pandas as pd
from common import FITS, labelled_frame
from setalpha import model

WINDOWS = {"exp": None, "r5": 5, "r3": 3, "expW": None}
HALF_LIFE_Y = 2.0
MONTHS = pd.date_range("2016-01-01", "2026-10-01", freq="MS")


def main(which):
    lab, feats, best = labelled_frame()
    members = ["logit", "xgb"]
    for w in which:
        for m in MONTHS:
            f = FITS / f"{w}_{m:%Y-%m}.parquet"
            if f.exists():
                continue
            t0 = time.time()
            yrs = WINDOWS[w]
            lo = m - pd.DateOffset(years=yrs) if yrs else pd.Timestamp("2000-01-01")
            tr = model._embargo(lab[(lab["Date"] < m) & (lab["Date"] >= lo)], str(m.date()))
            sw = None
            if w == "expW":
                age_y = (tr["Date"].max() - tr["Date"]).dt.days.to_numpy() / 365.25
                sw = 0.5 ** (age_y / HALF_LIFE_Y)
            te = lab[(lab["Date"] >= m) & (lab["Date"] < m + pd.DateOffset(months=12))]
            part = te[["Date", "Ticker"]].copy()
            y = tr[model.TARGET].astype(int)
            for name in members:
                mdl = model.make_model(name, **best[name])
                kw = {}
                if sw is not None:
                    kw = (
                        {"sample_weight": sw}
                        if name == "xgb"
                        else {"logisticregression__sample_weight": sw}
                    )
                mdl.fit(tr[feats], y, **kw)
                part[f"p_{name}"] = mdl.predict_proba(te[feats])[:, 1]
            part["window"] = w
            part["fit_month"] = m
            part["train_end"] = tr["Date"].max()
            part["n_train"] = len(tr)
            part.to_parquet(f)
            print(
                f"{w} {m:%Y-%m} n_train={len(tr):,} pred={len(te):,} {time.time() - t0:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1:] or list(WINDOWS))
