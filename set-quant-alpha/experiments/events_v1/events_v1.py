"""events_v1 - do earnings events explain the worst misses, and is post-earnings drift an orthogonal signal?

    uv run set-quant-alpha/experiments/events_v1/events_v1.py

Diagnostics only (no strategy is selected here). Periods: 2013-19 and 2020-21 decide; 2022-26 is shown for
information because every earlier experiment has already used it. Scores are the walk-forward ensemble
(strategy_v3/output/predictions_ext.parquet), tradable top-100 names.

Questions
A. Do top-decile picks with an earnings release inside the 5-day holding window have worse / fatter-tailed alpha?
   (uses the look-ahead flag diag_earn_in_hold - diagnostic only)
B. Can the point-in-time flag exp_earn_next5 (last year's date + 364 days) anticipate those windows?
C. Post-earnings drift: do the earnings reaction (last_ear) and EPS surprise predict 5-day alpha, and are they
   orthogonal to the model score (Fama-MacBeth with both)?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import events, ingest, signal_eval, transform

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
PREDS = HERE.parent / "strategy_v3" / "output" / "predictions_ext.parquet"
PERIODS = {
    "2013-19": ("2013-01-01", "2019-12-31"),
    "2020-21": ("2020-01-01", "2021-12-31"),
    "2022-26 (info)": ("2022-01-01", "2026-12-31"),
}


def load() -> pd.DataFrame:
    df = transform.load()
    raw = ingest.ingest_all("file")
    feat = events.earnings_features(df, raw["earnings"], raw["prices"])
    p = pd.read_parquet(PREDS, columns=["Date", "Ticker", "pred_ens"])
    cols = [
        "Date",
        "Ticker",
        "tradable",
        "fwd_alpha_5",
        "cs_dvol",
        "resid_rev_5",
        "resid_mom_120",
        "idio_vol",
        "beta_mkt",
    ]
    d = p.merge(df[cols], on=["Date", "Ticker"]).merge(feat, on=["Date", "Ticker"], how="left")
    d = d[d["tradable"]].dropna(subset=["fwd_alpha_5", "pred_ens"])
    d["covered"] = d["days_since_earn"].notna()
    d["top"] = d.groupby("Date")["pred_ens"].rank(pct=True) >= 0.9
    return d


def _period(d, lo, hi):
    return d[(d["Date"] >= lo) & (d["Date"] <= hi)]


def tail_table(d: pd.DataFrame, flag: str) -> pd.DataFrame:
    rows = {}
    for name, (lo, hi) in PERIODS.items():
        s = _period(d, lo, hi)
        s = s[s["top"] & s["covered"]]
        for v, g in s.groupby(s[flag].fillna(0) > 0):
            a = 100 * g["fwd_alpha_5"]
            rows[(name, f"{flag}={int(v)}")] = {
                "picks": len(g),
                "share %": 100 * len(g) / len(s),
                "mean 5d alpha %": a.mean(),
                "std %": a.std(),
                "P(alpha < -10%) %": 100 * (a < -10).mean(),
                "5th pct %": a.quantile(0.05),
            }
    return pd.DataFrame(rows).T


def flag_quality(d: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, (lo, hi) in PERIODS.items():
        s = _period(d, lo, hi)
        s = s[s["covered"]]
        act, exp = s["diag_earn_in_hold"] > 0, s["exp_earn_next5"] > 0
        rows[name] = {
            "base rate %": 100 * act.mean(),
            "precision %": 100 * (act & exp).sum() / max(exp.sum(), 1),
            "recall %": 100 * (act & exp).sum() / max(act.sum(), 1),
        }
    return pd.DataFrame(rows).T


def drift_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, (lo, hi) in PERIODS.items():
        s = _period(d, lo, hi)
        for feat in ["last_ear", "last_surprise"]:
            g = s.dropna(subset=[feat])
            g = g[g.groupby("Date")[feat].transform("count") >= 10]
            if g.empty:
                continue
            ic = signal_eval.daily_ic(g, feat)
            summ = signal_eval.ic_summary(ic)
            corr = signal_eval.daily_ic(
                g, feat, "pred_ens"
            ).mean()  # rank correlation with the model score
            rows[(name, feat)] = {
                "names/day": g.groupby("Date").size().median(),
                "IC": summ["IC mean"],
                "IC t (non-overlap)": summ["t (non-overlapping)"],
                "corr with score": corr,
            }
    return pd.DataFrame(rows).T


def fama_macbeth(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """fwd_alpha_5 on cross-sectional ranks of the score and a drift feature; non-overlapping t-stats."""
    rows = {}
    for name, (lo, hi) in PERIODS.items():
        s = _period(d, lo, hi).dropna(subset=cols)
        coefs = []
        for _, g in s.groupby("Date"):
            if len(g) < 20:
                continue
            X = g[cols].rank(pct=True) - 0.5
            X.insert(0, "c", 1.0)
            b, *_ = np.linalg.lstsq(X.to_numpy(), g["fwd_alpha_5"].to_numpy(), rcond=None)
            coefs.append(b[1:])
        c = pd.DataFrame(coefs, columns=cols).iloc[::5]
        if len(c) < 10:
            continue
        for k in cols:
            rows[(name, k)] = {
                "slope %/5d (top-bottom)": 100 * c[k].mean(),
                "t (non-overlap)": c[k].mean() / c[k].std() * np.sqrt(len(c)),
                "dates": len(c),
            }
    return pd.DataFrame(rows).T


def to_md(t: pd.DataFrame) -> str:
    t = t.round(3)
    idx = [" / ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in t.index]
    lines = ["| | " + " | ".join(map(str, t.columns)) + " |", "|---" * (len(t.columns) + 1) + "|"]
    lines += [
        f"| {i} | " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for i, row in zip(idx, t.to_numpy(), strict=True)
    ]
    return "\n".join(lines)


def main() -> None:
    d = load()
    cov = (
        d.groupby(d["Date"].dt.year)["covered"]
        .mean()
        .mul(100)
        .rename("coverage % of tradable rows")
    )
    tables = {
        "coverage": cov.to_frame(),
        "A_tail_actual_earnings_in_hold (look-ahead, diagnostic)": tail_table(
            d, "diag_earn_in_hold"
        ),
        "B_flag_quality": flag_quality(d),
        "B_tail_expected_flag (point-in-time)": tail_table(d, "exp_earn_next5"),
        "C_drift_ic": drift_table(d),
        "C_fama_macbeth_score_plus_ear": fama_macbeth(d, ["pred_ens", "last_ear"]),
        "C_fama_macbeth_score_plus_surprise": fama_macbeth(d, ["pred_ens", "last_surprise"]),
    }
    md = []
    for k, t in tables.items():
        t.to_csv(OUT / f"{k.split(' ')[0]}.csv")
        md.append(f"## {k}\n\n{to_md(t)}\n")
    (OUT / "tables.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
