"""lseg_signals_v1 - do the other LSEG datasets carry alpha beyond S5's score? (protocol: output/protocol_frozen.txt)

    uv run --group lseg python set-quant-alpha/scripts/fetch_lseg.py extra             # LSEG pull, ~15 min
    uv run python set-quant-alpha/experiments/lseg_signals_v1/lseg_signals_v1.py features   # point-in-time daily signals
    uv run python set-quant-alpha/experiments/lseg_signals_v1/lseg_signals_v1.py discover   # 2013-21 decision
    uv run python set-quant-alpha/experiments/lseg_signals_v1/lseg_signals_v1.py holdout    # 2022-26, once

Needs survivorship_v1 and fundamentals_v1 outputs (universe L, scores, frame.parquet).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import config, validity

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
EXP = HERE.parent
sys.path.insert(0, str(EXP / "fundamentals_v1"))
import fundamentals_v1 as fv  # noqa: E402

sv = fv.sv
X = sv.LSEG / "extra"
SIGN = {
    "ptup": 1,
    "ptrev3": 1,
    "sue": 1,
    "surp": 1,
    "smartdiv": 1,
    "disp": -1,
    "fdy": 1,
    "dpsrev3": 1,
    "gpa": 1,
    "accr": -1,
    "iss": -1,
    "esg": 1,
    "ca": 1,
}
BLEND = [
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
SIGNALS = list(SIGN)
DISCOVERY, HOLDOUT = fv.DISCOVERY, fv.HOLDOUT
DECISION = OUT / "discovery_decision.txt"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ============================================================================ 1. point-in-time daily signals
def _monthly(
    name: str, col: str, days: pd.DatetimeIndex, limit: int = 70, shift_days: int = 0
) -> pd.DataFrame:
    d = pd.read_parquet(X / f"{name}.parquet")
    d["date"] = pd.to_datetime(d["date"]) + pd.Timedelta(days=shift_days)
    d[col] = pd.to_numeric(d[col], errors="coerce").astype("float64")
    w = d.dropna(subset=[col]).pivot_table(index="date", columns="ric", values=col, aggfunc="last")
    w = w.reindex(w.index.union(days)).ffill(limit=limit).reindex(days)
    return w.shift(1)  # usable from the next trading day


def _events(ev: pd.DataFrame, days: pd.DatetimeIndex, limit: int) -> pd.DataFrame:
    """ev: ric, eff (calendar timestamp after which the value is public), value -> daily panel, effective next day."""
    ev = ev.dropna(subset=["eff", "value"]).copy()
    pos = days.searchsorted(pd.to_datetime(ev["eff"]).dt.normalize() + pd.Timedelta(days=1))
    ev = ev[pos < len(days)].assign(Date=days[pos[pos < len(days)]])
    w = ev.sort_values("eff").pivot_table(
        index="Date", columns="ric", values="value", aggfunc="last"
    )
    return w.reindex(days).ffill(limit=limit)


def _sue(days) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.read_parquet(X / "epsq.parquet")
    q["period_end"], q["report_date"] = pd.to_datetime(q.period_end), pd.to_datetime(q.report_date)
    q["eps_act"] = pd.to_numeric(q.eps_act, errors="coerce").astype("float64")
    q = q.dropna(subset=["period_end", "report_date", "eps_act"]).sort_values("report_date")
    q = q.drop_duplicates(["ric", "period_end"]).sort_values(["ric", "period_end"])
    prev = q[["ric", "period_end", "eps_act"]].rename(
        columns={"period_end": "pe_prev", "eps_act": "eps_prev"}
    )
    q["pe_prev"] = q["period_end"] - pd.Timedelta(days=365)
    q = pd.merge_asof(
        q.sort_values("pe_prev"),
        prev.sort_values("pe_prev"),
        on="pe_prev",
        by="ric",
        tolerance=pd.Timedelta(days=20),
        direction="nearest",
    ).sort_values(["ric", "period_end"])
    q["diff"] = q["eps_act"] - q["eps_prev"]
    sd = q.groupby("ric")["diff"].transform(lambda s: s.shift(1).rolling(8, min_periods=4).std())
    q["sue"] = (q["diff"] / sd.where(sd > 0)).clip(-10, 10)
    sue = _events(q.rename(columns={"report_date": "eff", "sue": "value"}), days, 126)
    s = pd.read_parquet(X / "surpq.parquet").rename(columns={"date": "period_end"})
    s["period_end"] = pd.to_datetime(s.period_end)
    s["surprise"] = pd.to_numeric(s.surprise, errors="coerce").astype("float64")
    s = s.drop_duplicates(["ric", "period_end"]).merge(
        q[["ric", "period_end", "report_date"]], on=["ric", "period_end"]
    )
    surp = _events(s.assign(eff=s.report_date, value=s.surprise.clip(-100, 100)), days, 63)
    return sue, surp


def _fin(days) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_parquet(X / "finq.parquet").rename(columns={"date": "period_end"})
    f["period_end"] = pd.to_datetime(f.period_end)
    f["announced"] = pd.to_datetime(f.announced, errors="coerce").dt.tz_localize(None)
    for c in ["gp", "ni", "cfo", "ta"]:
        f[c] = pd.to_numeric(f[c], errors="coerce").astype("float64")
    f = (
        f.dropna(subset=["period_end"])
        .drop_duplicates(["ric", "period_end"])
        .sort_values(["ric", "period_end"])
    )
    g = f.groupby("ric")
    ok = (f["period_end"] - g["period_end"].shift(3)) <= pd.Timedelta(days=370)
    ttm = {
        c: g[c].transform(lambda s: s.rolling(4, min_periods=4).sum()).where(ok)
        for c in ["gp", "ni", "cfo"]
    }
    ta = f["ta"].where(f["ta"] > 0)
    f["eff"] = f["announced"].fillna(f["period_end"] + pd.Timedelta(days=60))
    gpa = _events(f.assign(value=ttm["gp"] / ta), days, 300)
    accr = _events(f.assign(value=(ttm["ni"] - ttm["cfo"]) / ta), days, 300)
    return gpa, accr


def stage_features() -> None:
    px = pd.read_parquet(sv.LSEG / "prices.parquet").drop_duplicates(["Ticker", "Date"])
    close = px.assign(Close=px.Close.astype("float64")).pivot_table(
        index="Date", columns="Ticker", values="Close"
    )
    days = close.index
    pt, ptn = _monthly("pt", "pt", days), _monthly("pt", "pt_n", days)
    smart, mean = _monthly("smart", "smart", days), _monthly("smart", "mean", days)
    sdv = _monthly("epssd", "eps_sd", days)
    dps = _monthly("dps", "dps", days)
    sh = _monthly("shares", "shares", days).shift(21)
    esg = _monthly("esg", "esg", days, limit=300, shift_days=180)
    ca = _monthly("ca", "ca_country", days)
    sue, surp = _sue(days)
    gpa, accr = _fin(days)
    cols = close.columns
    al = lambda x: x.reindex(columns=cols)  # noqa: E731
    pt, ptn, smart, mean, sdv, dps, sh, esg, ca, sue, surp, gpa, accr = map(
        al, (pt, ptn, smart, mean, sdv, dps, sh, esg, ca, sue, surp, gpa, accr)
    )
    amean = mean.abs().where(mean.abs() > 0)
    feats = {
        "ptup": (pt / close - 1).where(ptn >= 2).clip(-0.9, 3),
        "ptrev3": np.log(pt.where(pt > 0) / pt.shift(63).where(pt.shift(63) > 0)),
        "sue": sue,
        "surp": surp,
        "smartdiv": ((smart - mean) / amean).clip(-1, 1),
        "disp": (sdv / amean).clip(0, 5),
        "fdy": (dps / close).where(dps >= 0),
        "dpsrev3": (
            (dps - dps.shift(63)) / dps.shift(63).abs().where(dps.shift(63).abs() > 0)
        ).clip(-1, 1),
        "gpa": gpa,
        "accr": accr,
        "iss": np.log(sh.where(sh > 0) / sh.shift(252).where(sh.shift(252) > 0)),
        "esg": esg,
        "ca": ca,
    }
    renamed = json.load(open(sv.OUT / "added_rics.json"))["renamed"]
    for k, v in feats.items():
        v = v * SIGN[k]  # sign-adjusted: higher = prior says better
        for old, new in renamed.items():
            if old in v.columns:
                v[new] = v[new].combine_first(v[old]) if new in v.columns else v[old]
        feats[k] = v
    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long.index.names = ["Date", "Ticker"]
    long = long.dropna(how="all").reset_index()
    long = long[long.Date >= "2010-01-01"]
    long.to_parquet(OUT / "signals.parquet", index=False)
    fr = pd.read_parquet(
        EXP / "fundamentals_v1" / "output" / "frame.parquet", columns=["Date", "Ticker"]
    )
    cov = fr.merge(long, on=["Date", "Ticker"], how="left")
    log(f"features: {len(long):,} stock-days; coverage inside the tradable top-100 by year:")
    print(
        cov.groupby(cov.Date.dt.year)[SIGNALS]
        .apply(lambda d: d.notna().mean())
        .round(2)
        .to_string()
    )


# ============================================================================ 2. evaluation
def frame() -> pd.DataFrame:
    d = pd.read_parquet(EXP / "fundamentals_v1" / "output" / "frame.parquet")
    d = d.merge(pd.read_parquet(OUT / "signals.parquet"), on=["Date", "Ticker"], how="left")
    g = d.groupby("Date")
    for c in SIGNALS:
        d[f"r_{c}"] = g[c].rank(pct=True)
    d["LC_all"] = d[[f"r_{c}" for c in BLEND]].fillna(0.5).mean(axis=1)
    d["r_LC_all"] = d.groupby("Date")["LC_all"].rank(pct=True)
    return d


def add_sel(d: pd.DataFrame, sel: list[str]) -> pd.DataFrame:
    if sel:
        d["LC_sel"] = d[[f"r_{c}" for c in sel]].fillna(0.5).mean(axis=1)
        d["r_LC_sel"] = d.groupby("Date")["LC_sel"].rank(pct=True)
    return d


def stage1(d: pd.DataFrame, period, cols) -> pd.DataFrame:
    p = d[(d.Date >= period[0]) & (d.Date <= period[1])]
    rows = []
    for c in cols:
        for h in (5, 20):
            y = f"fwd_alpha_{h}"
            ic, fm = fv._ic(p, c, y), fv._fm(p, c, y)
            rows.append(
                {
                    "signal": c,
                    "h": h,
                    "names/day": p.groupby("Date")[c].count().mean(),
                    "IC": ic.mean(),
                    "IC t": validity.nw_tstat(ic, lags=h),
                    "FM slope": fm.mean(),
                    "FM t": validity.nw_tstat(fm, lags=h),
                }
            )
    return pd.DataFrame(rows)


def _passes(t: pd.DataFrame, col: str, c: str) -> pd.Series:
    """Per-horizon pass of a t-stat gate in the prior direction (two-sided for esg)."""
    return t[col].abs() >= 2 if c == "esg" else t[col] >= 2


def strategies(d: pd.DataFrame, use_sel: bool) -> dict[str, pd.DataFrame]:
    sys.path.insert(0, str(EXP / "integrated_v1"))
    import integrated_v1 as iv

    panel, base, _ = sv.panel_for("L")
    dates, tickers = panel["dates"], panel["tickers"]
    W = (
        lambda c: d.pivot_table(index="Date", columns="Ticker", values=c)
        .reindex(  # noqa: E731
            index=dates, columns=tickers
        )
        .to_numpy()
    )
    trad = ~np.isnan(base["C2"])
    z = lambda a: np.nan_to_num(a - 0.5)  # noqa: E731
    rml, rv8, rfc, rla = W("r_pred_ens"), W("r_pred_v8"), W("r_FC"), W("r_LC_all")
    scores = {
        "A": base["C2"],
        "F": np.where(trad, (z(rml) + z(rv8) + z(rla)) / 3, np.nan),
        "G": np.where(trad, (z(rml) + z(rv8) + z(rfc) + z(rla)) / 4, np.nan),
    }
    if use_sel:
        scores["E"] = np.where(trad, (z(rml) + z(rv8) + z(W("r_LC_sel"))) / 3, np.nan)
    res = {
        k: iv.fast_sim(panel, scores, iv.Cfg(k, 10, 5, True, False), fee=config.FEE_PER_SIDE)
        for k in scores
    }
    res["_bench"] = panel["bench"]["Close"].pct_change().reindex(dates).to_frame("ret")
    return res


def stage_discover() -> None:
    d = frame()
    s1 = stage1(d, DISCOVERY, SIGNALS + ["LC_all"])
    sel = sorted(
        set(s1.loc[s1.signal.isin(BLEND) & (s1["IC t"] >= 2) & (s1["FM t"] >= 2), "signal"])
    )
    d = add_sel(d, sel)
    comp = stage1(d, DISCOVERY, ["LC_sel"]) if sel else pd.DataFrame()
    d.to_parquet(OUT / "frame.parquet", index=False)
    res = strategies(d, bool(sel))
    pd.DataFrame({k: v["net"] for k, v in res.items() if k != "_bench"}).to_parquet(
        OUT / "strategies_daily.parquet"
    )
    st = fv.strat_table(res, DISCOVERY)
    cand = [k for k in ("E", "F", "G") if k in st.index]
    pick = st.loc[cand, "alpha t"].astype(float).idxmax()
    added = comp if pick == "E" else s1[s1.signal == "LC_all"]
    s_1 = float(added.query("h == 5")["FM t"].iloc[0]) >= 2
    s_2 = (
        st.loc[pick, "CAGR %"] > st.loc["A", "CAGR %"]
        and st.loc[pick, "Sharpe"] > st.loc["A", "Sharpe"]
    )
    g12 = {
        c: [
            int(h)
            for h in (5, 20)
            if _passes(s1[(s1.signal == c) & (s1.h == h)], "IC t", c).iloc[0]
            and _passes(s1[(s1.signal == c) & (s1.h == h)], "FM t", c).iloc[0]
        ]
        for c in SIGNALS
    }
    DECISION.write_text(
        f"discovery decision written {pd.Timestamp.now(tz='UTC').isoformat()}\n"
        f"signals passing G1+G2 (horizons): {json.dumps({k: v for k, v in g12.items() if v})}\n"
        f"LC_sel members: {sel}\n"
        f"pick (max net CAPM alpha t, 2013-21, among {cand}): {pick}\n"
        f"S1 information gate: {bool(s_1)}   S2 beats A on 2013-21: {bool(s_2)}\n"
    )
    with open(OUT / "tables.md", "w") as f:
        f.write(
            "# lseg_signals_v1 tables\n\n## Stage 1: signals (sign-adjusted), discovery 2013-21\n\n"
        )
        f.write(sv.md(pd.concat([s1, comp]), 4, index=False))
        f.write(
            "\n\n## Stage 3: strategies, discovery 2013-21 (net, 0.168%/side)\n\n" + sv.md(st, 2)
        )
        f.write("\n\n## Decision\n\n```\n" + DECISION.read_text() + "```\n")
    print(open(OUT / "tables.md").read())


def stage_holdout() -> None:
    if not DECISION.exists():
        sys.exit("run `discover` first")
    if (OUT / "holdout.json").exists():
        sys.exit("holdout already computed once; see output/tables.md")
    d = pd.read_parquet(OUT / "frame.parquet")
    dec = DECISION.read_text()
    pick = dec.split("2013-21, among ")[1].split("): ")[1].split("\n")[0].strip()
    g12 = json.loads(dec.split("(horizons): ")[1].split("\n")[0])
    s_1, s_2 = "S1 information gate: True" in dec, "S2 beats A on 2013-21: True" in dec
    cols = SIGNALS + ["LC_all"] + (["LC_sel"] if "LC_sel" in d else [])
    s1 = stage1(d, HOLDOUT, cols)
    passed = {}
    for c, hs in g12.items():
        ok = [h for h in hs if _passes(s1[(s1.signal == c) & (s1.h == h)], "IC t", c).iloc[0]]
        if ok:
            passed[c] = ok
    daily = pd.read_parquet(OUT / "strategies_daily.parquet")
    res = {k: pd.DataFrame({"net": daily[k], "turn": 0.0}) for k in daily}
    panel, _, _ = sv.panel_for("L")
    res["_bench"] = panel["bench"]["Close"].pct_change().reindex(panel["dates"]).to_frame("ret")
    st = fv.strat_table(res, HOLDOUT).drop(columns="turnover x/yr")
    full = fv.strat_table(res, ("2013-01-01", "2026-12-31")).drop(columns="turnover x/yr")
    s_3 = (
        st.loc[pick, "Sharpe"] >= st.loc["A", "Sharpe"]
        and st.loc[pick, "CAGR %"] > st.loc["TDEX", "CAGR %"]
    )
    boot = validity.sharpe_diff_test(daily[pick].loc["2022":], daily["A"].loc["2022":])
    verdict = {
        "signals passing G1+G2+G3": passed,
        "pick": pick,
        "S1": s_1,
        "S2": s_2,
        "S3": bool(s_3),
        "strategy pass": bool(s_1 and s_2 and s_3),
        "holdout Sharpe diff vs A": boot["diff"],
        "95% CI": [boot["CI 2.5%"], boot["CI 97.5%"]],
    }
    json.dump(verdict, open(OUT / "holdout.json", "w"), indent=1, default=float)
    with open(OUT / "tables.md", "a") as f:
        f.write(
            "\n## Holdout 2022-26 (computed once)\n\n### Stage 1\n\n" + sv.md(s1, 4, index=False)
        )
        f.write("\n\n### Strategies 2022-26\n\n" + sv.md(st, 2))
        f.write("\n\n### Strategies 2013-26\n\n" + sv.md(full, 2))
        f.write(
            "\n\n### Verdict\n\n```\n" + json.dumps(verdict, indent=1, default=float) + "\n```\n"
        )
    print(open(OUT / "tables.md").read().split("## Holdout")[1])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"features": stage_features, "discover": stage_discover, "holdout": stage_holdout}[
        sys.argv[1]
    ]()
