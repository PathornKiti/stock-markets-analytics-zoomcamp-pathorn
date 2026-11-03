"""adaptive_v1 - adaptive signal weights, wider books, earnings-surprise overlay (protocol: output/protocol_frozen.txt)

    uv run python set-quant-alpha/experiments/adaptive_v1/adaptive_v1.py ml3        # ML refit on the last 3 years (~3 min)
    uv run python set-quant-alpha/experiments/adaptive_v1/adaptive_v1.py discover   # simulations, 2013-21 decision
    uv run python set-quant-alpha/experiments/adaptive_v1/adaptive_v1.py check      # 2022-26 consistency (burned period)

Needs survivorship_v1, fundamentals_v1 and lseg_signals_v1 outputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import config, validity

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
EXP = HERE.parent
sys.path.insert(0, str(EXP / "lseg_signals_v1"))
sys.path.insert(0, str(EXP / "integrated_v1"))
import integrated_v1 as iv  # noqa: E402
import lseg_signals_v1 as ls  # noqa: E402

fv, sv = ls.fv, ls.sv
POOL = [
    "pred_ens",
    "pred_v8",
    "rev1",
    "rev3",
    "recup3",
    "bp",
    "ep",
    "roe",
    "ptup",
    "ptrev3",
    "sue",
    "surp",
    "smartdiv",
    "disp",
    "fdy",
    "dpsrev3",
    "gpa",
    "accr",
    "iss",
]
DISCOVERY, LATE = ("2013-01-01", "2021-12-31"), ("2022-01-01", "2026-12-31")
DECISION = OUT / "discovery_decision.txt"
STRATS = {
    "A": 10,
    "S1a": 10,
    "S1b": 10,
    "S2a": 20,
    "S2b": 30,
    "S3a": 10,
    "S3b": 10,
}  # name -> book size


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ============================================================================ S1b: ML on the last 3 years (L data dir)
def stage_ml3() -> None:
    root = sv.ROOT / "L"
    if os.environ.get("SETALPHA_DATA_DIR") != str(root):
        env = {**os.environ, "SETALPHA_DATA_DIR": str(root), "SETALPHA_PROD_DATA_DIR": str(sv.PROD)}
        sys.exit(subprocess.call([sys.executable, __file__, "ml3"], env=env))
    from setalpha import model, report

    sys.path.insert(0, str(EXP / "model_v2"))
    import model_v2 as mv

    t0 = time.time()
    df, feats, _ = mv.load()
    best = json.loads((config.MODELS_DIR / "best_params.json").read_text())
    members = report.load_meta()["ensemble"]
    lab = model.labelled(df)
    rows_all = df[df["liquid"] & (df["Date"].dt.year >= sv.FIRST_YEAR)]
    parts, fits = [], {}
    for yr in range(sv.FIRST_YEAR, sv.LAST_YEAR + 1):
        fit_year = 2020 if yr in (2020, 2021) else yr
        start = f"{fit_year}-01-01"
        if fit_year not in fits:
            tr = lab[(lab["Date"] < start) & (lab["Date"] >= f"{fit_year - 3}-01-01")]
            train = model._embargo(tr, start)
            fits[fit_year] = {n: model.fit(n, train, feats, **best[n]) for n in members}
        rows = rows_all[rows_all["Date"].dt.year == yr]
        part = rows[["Date", "Ticker"]].copy()
        for n in members:
            part[f"pred_{n}"] = model.predict(fits[fit_year][n], rows, feats)
        parts.append(part)
        log(f"[ML3] {yr}: {len(rows):,} rows, {time.time() - t0:.0f}s")
    out = model.add_ensemble(pd.concat(parts, ignore_index=True), members).rename(
        columns={"pred_ens": "pred_ml3"}
    )
    out[["Date", "Ticker", "pred_ml3"]].to_parquet(OUT / "scores_ML3.parquet", index=False)


# ============================================================================ scores
def _fresh(d: pd.DataFrame) -> pd.Series:
    """True for the 10 trading days starting the day after an earnings report (point-in-time, as lseg_signals_v1)."""
    px = pd.read_parquet(sv.LSEG / "prices.parquet", columns=["Ticker", "Date"]).drop_duplicates()
    days = pd.DatetimeIndex(sorted(px.Date.unique()))
    q = pd.read_parquet(ls.X / "epsq.parquet")
    q["report_date"] = pd.to_datetime(q.report_date)
    q = q.dropna(subset=["report_date"]).drop_duplicates(["ric", "report_date"])
    w = ls._events(q.assign(eff=q.report_date, value=1.0), days, 9)
    for old, new in json.load(open(sv.OUT / "added_rics.json"))["renamed"].items():
        if old in w.columns:
            w[new] = w[new].combine_first(w[old]) if new in w.columns else w[old]
    f = w.stack().reset_index()
    f.columns = ["Date", "Ticker", "fresh"]
    return (
        d[["Date", "Ticker"]]
        .merge(f, on=["Date", "Ticker"], how="left")["fresh"]
        .fillna(0)
        .astype(bool)
        .to_numpy()
    )


def daily_ic(d: pd.DataFrame, x: str, y: str = "fwd_alpha_5") -> pd.Series:
    s = d[["Date", x, y]].dropna()
    g = s.groupby("Date")
    s = s.assign(a=g[x].rank(), b=g[y].rank())
    n = s.groupby("Date").size()
    c = s.groupby("Date")[["a", "b"]].corr().xs("a", level=1)["b"]
    return c[n >= fv.MIN_NAMES]


def icw_weights(d: pd.DataFrame) -> pd.DataFrame:
    """Monthly weights from trailing 504-day IC ending 6 trading days before each month's first trading day."""
    ics = pd.DataFrame({k: daily_ic(d, f"r_{k}") for k in POOL}).sort_index()
    dates = pd.DatetimeIndex(sorted(d.Date.unique()))
    firsts = dates[np.r_[True, dates.month[1:] != dates.month[:-1]]]
    rows = {}
    for t in firsts:
        pos = dates.get_loc(t)
        if pos < 6:
            continue
        hist = ics[ics.index <= dates[pos - 6]].tail(504)
        if len(hist) < 250:
            continue
        w = hist.mean().clip(lower=0)
        if w.sum() > 0:
            rows[t] = w / w.sum()
    W = pd.DataFrame(rows).T.reindex(columns=POOL)
    return W.reindex(dates).ffill()


def build_scores() -> tuple[dict, dict, pd.DataFrame]:
    d = pd.read_parquet(EXP / "lseg_signals_v1" / "output" / "frame.parquet")
    d = d.merge(pd.read_parquet(OUT / "scores_ML3.parquet"), on=["Date", "Ticker"], how="left")
    d["r_ml3"] = d.groupby("Date")["pred_ml3"].rank(pct=True)
    d["fresh_surp"] = np.where(_fresh(d), d["r_surp"], np.nan)
    W_icw = icw_weights(d)
    W_icw.to_csv(OUT / "icw_weights.csv")
    wd = W_icw.reindex(d.Date).to_numpy()
    R = np.nan_to_num(d[[f"r_{k}" for k in POOL]].to_numpy() - 0.5)
    d["icw"] = np.where(np.isnan(wd).all(axis=1), np.nan, np.nansum(np.nan_to_num(wd) * R, axis=1))
    panel, base, _ = sv.panel_for("L")
    dates, tickers = panel["dates"], panel["tickers"]
    P = (
        lambda c: d.pivot_table(index="Date", columns="Ticker", values=c)
        .reindex(  # noqa: E731
            index=dates, columns=tickers
        )
        .to_numpy()
    )
    trad = ~np.isnan(base["C2"])
    z = lambda a: np.nan_to_num(a - 0.5)  # noqa: E731
    rml, rv8 = P("r_pred_ens"), P("r_pred_v8")
    icw = P("icw")
    icw = np.where(np.isnan(icw), base["C2"], icw)  # < 250 IC days -> C2
    scores = {
        "A": base["C2"],
        "S2a": base["C2"],
        "S2b": base["C2"],
        "S1a": np.where(trad, icw, np.nan),
        "S1b": np.where(trad, (z(P("r_ml3")) + z(rv8)) / 2, np.nan),
        "S3a": np.where(trad, (z(rml) + z(rv8) + z(P("r_surp"))) / 3, np.nan),
        "S3b": np.where(trad, (z(rml) + z(rv8) + z(P("fresh_surp"))) / 3, np.nan),
    }
    return panel, scores, d


def run_all(panel, scores) -> dict[int, pd.DataFrame]:
    """Net daily returns per strategy for each of the 5 weekly rebalance phases."""
    orig = iv.decision_days
    out = {}
    try:
        for k in range(5):
            iv.decision_days = lambda dates, reb, k=k: (_pos(dates) + k) % reb == 0
            out[k] = pd.DataFrame(
                {
                    s: iv.fast_sim(
                        panel, scores, iv.Cfg(s, n, 5, True, False), fee=config.FEE_PER_SIDE
                    )["net"]
                    for s, n in STRATS.items()
                }
            )
            log(f"phase {k} done")
    finally:
        iv.decision_days = orig
    return out


def _pos(dates: pd.DatetimeIndex) -> np.ndarray:
    y = dates.year.to_numpy()
    pos = np.zeros(len(dates), dtype=int)
    for yr in np.unique(y):
        m = y == yr
        pos[m] = np.arange(m.sum())
    return pos


def table(ph: dict[int, pd.DataFrame], bench: pd.Series, period) -> pd.DataFrame:
    a, b = period
    p0 = ph[0].loc[a:b]
    rows = {}
    for s in STRATS:
        r = sv.stats(p0[s], bench)
        cagrs = [iv.cagr(ph[k][s].loc[a:b].dropna()) for k in ph]
        r["phase-avg CAGR %"], r["path noise (sd) %"] = (
            float(np.mean(cagrs)),
            float(np.std(cagrs, ddof=1)),
        )
        if s != "A":
            bt = validity.sharpe_diff_test(p0[s].dropna(), p0["A"].dropna())
            r["Sharpe diff vs A"], r["CI lo"], r["CI hi"] = (
                bt["diff"],
                bt["CI 2.5%"],
                bt["CI 97.5%"],
            )
        rows[s] = r
    t = pd.DataFrame(rows).T
    tb = bench.loc[a:b].dropna()
    t.loc["TDEX", ["CAGR %", "Sharpe"]] = [
        100 * ((1 + tb).prod() ** (252 / len(tb)) - 1),
        tb.mean() / tb.std() * np.sqrt(252),
    ]
    return t


def ml3_ic(d: pd.DataFrame, period) -> dict:
    p = d[(d.Date >= period[0]) & (d.Date <= period[1])]
    a, b = daily_ic(p, "r_pred_ens"), daily_ic(p, "r_ml3")
    diff = (b - a).dropna()
    return {
        "IC ML": a.mean(),
        "IC ML3": b.mean(),
        "diff": diff.mean(),
        "diff NW t": validity.nw_tstat(diff, lags=5),
    }


def stage_discover() -> None:
    panel, scores, d = build_scores()
    ph = run_all(panel, scores)
    pd.concat(ph, names=["phase"]).to_parquet(OUT / "daily_phases.parquet")
    d[["Date", "Ticker", "r_pred_ens", "r_ml3", "fwd_alpha_5"]].to_parquet(
        OUT / "ml3_frame.parquet", index=False
    )
    bench = panel["bench"]["Close"].pct_change().reindex(panel["dates"])
    t = table(ph, bench, DISCOVERY)
    ic = ml3_ic(d, DISCOVERY)
    A = t.loc["A"]
    cand = {}
    for s in ("S1a", "S1b", "S3a", "S3b"):
        r = t.loc[s]
        cand[s] = bool(
            r["Sharpe"] > A["Sharpe"]
            and r["CAGR %"] > A["CAGR %"]
            and r["phase-avg CAGR %"] > A["phase-avg CAGR %"]
        )
    for s in ("S2a", "S2b"):
        r = t.loc[s]
        cand[s] = bool(
            r["Sharpe"] >= A["Sharpe"] - 0.05 and r["path noise (sd) %"] < A["path noise (sd) %"]
        )
    DECISION.write_text(
        f"discovery decision written {pd.Timestamp.now(tz='UTC').isoformat()}\n"
        f"candidates (2013-21 rules): {json.dumps(cand)}\nML3 vs ML IC 2013-21: {json.dumps(ic, default=float)}\n"
    )
    w = pd.read_csv(OUT / "icw_weights.csv", index_col=0, parse_dates=True)
    wy = w.groupby(w.index.year).mean().round(2)
    with open(OUT / "tables.md", "w") as f:
        f.write(
            "# adaptive_v1 tables\n\n## 2013-21 (decision period), net of 0.168%/side, phase 0\n\n"
            + sv.md(t, 2)
        )
        f.write("\n\n## S1a average monthly weights by year\n\n" + sv.md(wy, 2))
        f.write("\n\n## Decision\n\n```\n" + DECISION.read_text() + "```\n")
    print(open(OUT / "tables.md").read())


def stage_check() -> None:
    if not DECISION.exists():
        sys.exit("run `discover` first")
    daily = pd.read_parquet(OUT / "daily_phases.parquet")
    ph = {k: daily.xs(k, level="phase") for k in range(5)}
    panel, _, _ = sv.panel_for("L")
    bench = panel["bench"]["Close"].pct_change().reindex(panel["dates"])
    t = table(ph, bench, LATE)
    full = table(ph, bench, ("2013-01-01", "2026-12-31"))
    ic = ml3_ic(pd.read_parquet(OUT / "ml3_frame.parquet"), LATE)
    cand = json.loads(DECISION.read_text().split("rules): ")[1].split("\n")[0])
    cons = {
        s: bool(
            c
            and t.loc[s, "Sharpe"] >= t.loc["A", "Sharpe"]
            and t.loc[s, "CAGR %"] > t.loc["TDEX", "CAGR %"]
        )
        for s, c in cand.items()
    }
    with open(OUT / "tables.md", "a") as f:
        f.write("\n## 2022-26 (burned: consistency check only)\n\n" + sv.md(t, 2))
        f.write("\n\n## 2013-26\n\n" + sv.md(full, 2))
        f.write(f"\n\nML3 vs ML IC 2022-26: {json.dumps(ic, default=float)}\n")
        f.write("\n## Consistent candidates\n\n```\n" + json.dumps(cons) + "\n```\n")
    print(open(OUT / "tables.md").read().split("## 2022-26")[1])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"ml3": stage_ml3, "discover": stage_discover, "check": stage_check}[sys.argv[1]]()
