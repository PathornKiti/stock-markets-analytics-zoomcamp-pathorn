"""fundamentals_v1 - do analyst-consensus and fundamental data improve the ML model and S5? (protocol: output/protocol_frozen.txt)

    uv run --group lseg python set-quant-alpha/scripts/fetch_lseg.py fundamentals     # LSEG monthly as-of history
    uv run python set-quant-alpha/experiments/fundamentals_v1/fundamentals_v1.py features   # point-in-time daily features
    uv run python set-quant-alpha/experiments/fundamentals_v1/fundamentals_v1.py scores     # ML_F walk-forward (~2 min)
    uv run python set-quant-alpha/experiments/fundamentals_v1/fundamentals_v1.py discover   # 2013-21 decision
    uv run python set-quant-alpha/experiments/fundamentals_v1/fundamentals_v1.py holdout    # 2022-26, once

Universe and baseline come from survivorship_v1 (universe L, delisted stocks included); run that first.
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
sys.path.insert(0, str(EXP / "survivorship_v1"))
import survivorship_v1 as sv  # noqa: E402

LSEG = sv.LSEG
SIGNALS = ["rev1", "rev3", "recup3", "bp", "ep", "roe"]
EXTRA = ["log_mcap", "log_nest", "has_ibes"]
DISCOVERY, HOLDOUT = ("2013-01-01", "2021-12-31"), ("2022-01-01", "2026-12-31")
DECISION = OUT / "discovery_decision.txt"
MIN_NAMES = 10


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ============================================================================ 1. point-in-time features
def _asof(
    long: pd.DataFrame,
    field: str,
    col: str,
    days: pd.DatetimeIndex,
    limit: int,
    shift_days: int = 0,
) -> pd.DataFrame:
    d = long[long.field == field].copy()
    d["date"] = pd.to_datetime(d["date"]) + pd.Timedelta(days=shift_days)
    d[col] = (
        pd.to_numeric(d[col], errors="coerce") if col != "period_end" else pd.to_datetime(d[col])
    )
    w = d.pivot_table(index="date", columns="ric", values=col, aggfunc="last")
    idx = w.index.union(days)
    w = w.reindex(idx).ffill(limit=limit).reindex(days)
    return w.shift(1)  # usable from the next trading day


def stage_features() -> None:
    f = pd.read_parquet(LSEG / "fundamentals.parquet")
    f = f[f["value"].notna()]
    px = pd.read_parquet(LSEG / "prices.parquet").drop_duplicates(["Ticker", "Date"])
    close = px.assign(Close=px.Close.astype("float64")).pivot_table(
        index="Date", columns="Ticker", values="Close"
    )
    days = close.index
    eps = _asof(f, "eps", "value", days, 70)
    pe_dt = pd.to_datetime(f.loc[f.field == "eps", "period_end"], errors="coerce")
    pend = f[f.field == "eps"].assign(
        value=(pe_dt - pd.Timestamp("1970-01-01")).dt.days.astype("float64")
    )
    pend = _asof(pend, "eps", "value", days, 70)
    nest = _asof(f, "nest", "value", days, 70)
    rec = _asof(f, "rec", "value", days, 70)
    pb = _asof(f, "pb", "value", days, 70)
    pe = _asof(f, "pe", "value", days, 70)
    roe = _asof(f, "roe", "value", days, 300, shift_days=90)  # fiscal period end + 90 days
    mcap = _asof(f, "mcap", "value", days, 70)
    cols = close.columns.union(eps.columns)
    eps, pend, nest, rec, pb, pe, roe, mcap, close = (
        x.reindex(columns=cols) for x in (eps, pend, nest, rec, pb, pe, roe, mcap, close)
    )

    def rev(k: int) -> pd.DataFrame:
        r = (eps - eps.shift(k)) / eps.shift(k).abs()
        r = r.clip(-1, 1).where(pend.eq(pend.shift(k)) & (nest >= 2))
        return r

    pb_l, pe_l, c_l = pb.shift(63), pe.shift(63), close.shift(63)
    feats = {
        "rev1": rev(21),
        "rev3": rev(63),
        "recup3": -(rec - rec.shift(63)),
        "bp": ((c_l / pb_l) / close).where(pb_l > 0),
        "ep": ((c_l / pe_l) / close).where(pe_l > 0),
        "roe": roe,
        "log_mcap": np.log(mcap.where(mcap > 0)),
        "log_nest": np.log1p(nest),
        "has_ibes": (nest.fillna(0) > 0).astype(float),
    }
    # renamed tickers: the Yahoo name carries the old RIC's history (e.g. Yahoo BANPU.BK = LSEG BPP.BK^G26)
    renamed = json.load(open(sv.OUT / "added_rics.json"))["renamed"]
    for k, v in feats.items():
        for old, new in renamed.items():
            if old in v.columns:
                v[new] = v[new].combine_first(v[old]) if new in v.columns else v[old]
        feats[k] = v
    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long.index.names = ["Date", "Ticker"]
    long = long.dropna(how="all", subset=SIGNALS + ["log_mcap", "log_nest"]).reset_index()
    long = long[long.Date >= "2010-01-01"]
    long.to_parquet(OUT / "fund_features.parquet", index=False)
    cov = long.groupby(long.Date.dt.year)[SIGNALS + EXTRA].apply(
        lambda d: d.notna().mean() if len(d) else d
    )
    log(f"features: {len(long):,} stock-days, {long.Ticker.nunique()} tickers")
    print(cov.round(2).to_string())


# ============================================================================ 2. ML_F walk-forward (in the L data dir)
def stage_scores() -> None:
    root = sv.ROOT / "L"
    if os.environ.get("SETALPHA_DATA_DIR") != str(root):
        env = {**os.environ, "SETALPHA_DATA_DIR": str(root), "SETALPHA_PROD_DATA_DIR": str(sv.PROD)}
        sys.exit(subprocess.call([sys.executable, __file__, "scores"], env=env))
    from setalpha import model, report

    sys.path.insert(0, str(EXP / "model_v2"))
    import model_v2 as mv

    t0 = time.time()
    df, feats, _ = mv.load()
    fund = pd.read_parquet(OUT / "fund_features.parquet")
    df = df.merge(fund, on=["Date", "Ticker"], how="left")
    new = SIGNALS + ["log_mcap", "log_nest"]
    liq = df["liquid"]
    for c in new:  # per-date pct rank among liquid names, missing = 0.5
        df[f"f_{c}"] = (
            df.loc[liq]
            .groupby("Date")[c]
            .rank(pct=True)
            .reindex(df.index)
            .fillna(0.5)
            .astype("float32")
        )
    df["f_has_ibes"] = df["has_ibes"].fillna(0).astype("float32")
    feats_f = feats + [f"f_{c}" for c in new] + ["f_has_ibes"]
    best = json.loads((config.MODELS_DIR / "best_params.json").read_text())
    members = report.load_meta()["ensemble"]
    lab = model.labelled(df)
    rows_all = df[liq & (df["Date"].dt.year >= sv.FIRST_YEAR)]
    parts, fits = [], {}
    for yr in range(sv.FIRST_YEAR, sv.LAST_YEAR + 1):
        fit_year = 2020 if yr in (2020, 2021) else yr
        start = f"{fit_year}-01-01"
        if fit_year not in fits:
            train = model._embargo(lab[lab["Date"] < start], start)
            fits[fit_year] = {n: model.fit(n, train, feats_f, **best[n]) for n in members}
        rows = rows_all[rows_all["Date"].dt.year == yr]
        part = rows[["Date", "Ticker"]].copy()
        for n in members:
            part[f"pred_{n}"] = model.predict(fits[fit_year][n], rows, feats_f)
        parts.append(part)
        log(f"[ML_F] {yr}: {len(rows):,} rows, {time.time() - t0:.0f}s")
    out = model.add_ensemble(pd.concat(parts, ignore_index=True), members).rename(
        columns={"pred_ens": "pred_ens_F"}
    )
    keep = [
        "Date",
        "Ticker",
        "tradable",
        "liquid",
        "fwd_alpha_5",
        "fwd_alpha_20",
        "dvol60",
        "beta_mkt",
    ]
    out = out[["Date", "Ticker", "pred_ens_F"]].merge(df[keep], on=["Date", "Ticker"], how="left")
    out.to_parquet(OUT / "scores_F.parquet", index=False)
    imp = pd.Series(fits[2026]["xgb"].feature_importances_, index=feats_f).sort_values(
        ascending=False
    )
    imp.to_csv(OUT / "xgb_importance_2026.csv")
    log(
        f"[ML_F] saved; fundamental features in xgb top-25: {[c for c in imp.head(25).index if c.startswith('f_')]}"
    )


# ============================================================================ 3. evaluation
def frame() -> pd.DataFrame:
    s = pd.read_parquet(sv.OUT / "scores_L.parquet")[["Date", "Ticker", "pred_ens", "pred_v8"]]
    f = pd.read_parquet(OUT / "scores_F.parquet")
    fund = pd.read_parquet(OUT / "fund_features.parquet")
    d = f.merge(s, on=["Date", "Ticker"], how="left").merge(fund, on=["Date", "Ticker"], how="left")
    px = pd.read_parquet(sv.ROOT / "L" / "raw" / "prices.parquet")
    C = px.assign(Close=px.Close.astype("float64")).pivot_table(
        index="Date", columns="Ticker", values="Close"
    )
    mom = (C.shift(21) / C.shift(252) - 1).stack(future_stack=True).rename("mom121").reset_index()
    d = d.merge(mom, on=["Date", "Ticker"], how="left")
    d["logsize"] = np.log(d["dvol60"])
    d = d[d["tradable"].fillna(False).astype(bool)].copy()
    g = d.groupby("Date")
    for c in ["pred_ens", "pred_ens_F", "pred_v8"] + SIGNALS:
        d[f"r_{c}"] = g[c].rank(pct=True)
    d["FC"] = d[[f"r_{c}" for c in SIGNALS]].fillna(0.5).mean(axis=1)
    d["r_FC"] = d.groupby("Date")["FC"].rank(pct=True)
    d["C2"] = (d["r_pred_ens"].fillna(0.5) + d["r_pred_v8"].fillna(0.5)) / 2
    return d


def _ic(d: pd.DataFrame, x: str, y: str) -> pd.Series:
    def one(g):
        g = g[[x, y]].dropna()
        return g[x].corr(g[y], method="spearman") if len(g) >= MIN_NAMES else np.nan

    return d.groupby("Date").apply(one).dropna()


def _fm(d: pd.DataFrame, x: str, y: str) -> pd.Series:
    ctl = ["C2", "logsize", "beta_mkt", "mom121"]

    def one(g):
        g = g[[x, y] + ctl].dropna()
        if len(g) < 15:
            return np.nan
        R = g.rank(pct=True)
        Z = (R - R.mean()) / R.std()
        A = np.column_stack([np.ones(len(Z)), Z[[x] + ctl].to_numpy()])
        return np.linalg.lstsq(A, Z[y].to_numpy(), rcond=None)[0][1]

    return d.groupby("Date").apply(one).dropna()


def stage1(d: pd.DataFrame, period) -> pd.DataFrame:
    p = d[(d.Date >= period[0]) & (d.Date <= period[1])]
    rows = []
    for c in SIGNALS + ["FC"]:
        for h in (5, 20):
            y = f"fwd_alpha_{h}"
            ic, fm = _ic(p, c, y), _fm(p, c, y)
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


def ml_compare(d: pd.DataFrame, period) -> dict:
    p = d[(d.Date >= period[0]) & (d.Date <= period[1])]
    a, b = _ic(p, "pred_ens", "fwd_alpha_5"), _ic(p, "pred_ens_F", "fwd_alpha_5")
    p2 = p.assign(c2f=(p.r_pred_ens_F.fillna(0.5) + p.r_pred_v8.fillna(0.5)) / 2)
    c2, c2f = _ic(p2, "C2", "fwd_alpha_5"), _ic(p2, "c2f", "fwd_alpha_5")
    diff = (b - a).dropna()
    return {
        "IC ml": a.mean(),
        "IC ml_F": b.mean(),
        "diff": diff.mean(),
        "diff NW t": validity.nw_tstat(diff, lags=5),
        "IC C2": c2.mean(),
        "IC C2F": c2f.mean(),
        "ml_F > ml in years": f"{int(((b - a).groupby(b.index.year).mean() > 0).sum())}/{b.index.year.nunique()}",
    }


def strategies(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
    rml, rmlf, rv8, rfc = W("r_pred_ens"), W("r_pred_ens_F"), W("r_pred_v8"), W("r_FC")
    z = lambda a: np.nan_to_num(a - 0.5)  # noqa: E731
    scores = {
        "A": base["C2"],
        "B": np.where(trad, (z(rmlf) + z(rv8)) / 2, np.nan),
        "C": np.where(trad, (z(rml) + z(rv8) + z(rfc)) / 3, np.nan),
    }
    cfgs = {
        "A": iv.Cfg("A", 10, 5, True, False),
        "B": iv.Cfg("B", 10, 5, True, False),
        "C": iv.Cfg("C", 10, 5, True, False),
        "D": iv.Cfg("B", 10, 10, True, False),
    }
    res = {k: iv.fast_sim(panel, scores, c, fee=config.FEE_PER_SIDE) for k, c in cfgs.items()}
    res["_bench"] = panel["bench"]["Close"].pct_change().reindex(dates).to_frame("ret")
    return res


def strat_table(res: dict, period) -> pd.DataFrame:
    b = res["_bench"]["ret"]
    rows = {
        k: sv.stats(v["net"].loc[period[0] : period[1]], b) for k, v in res.items() if k != "_bench"
    }
    for k in rows:
        rows[k]["turnover x/yr"] = res[k]["turn"].loc[period[0] : period[1]].mean() * 252
    t = pd.DataFrame(rows).T
    tb = b.loc[period[0] : period[1]].dropna()
    t.loc["TDEX", ["CAGR %", "Sharpe"]] = [
        100 * ((1 + tb).prod() ** (252 / len(tb)) - 1),
        tb.mean() / tb.std() * np.sqrt(252),
    ]
    return t


def stage_discover() -> None:
    d = frame()
    d.to_parquet(OUT / "frame.parquet", index=False)
    s1 = stage1(d, DISCOVERY)
    ml = ml_compare(d, DISCOVERY)
    res = strategies(d)
    pd.DataFrame({k: v["net"] for k, v in res.items() if k != "_bench"}).to_parquet(
        OUT / "strategies_daily.parquet"
    )
    st = strat_table(res, DISCOVERY)
    pick = st.loc[["B", "C", "D"], "alpha t"].astype(float).idxmax()
    g1 = (
        (ml["diff NW t"] >= 2 and ml["diff"] > 0)
        if pick in ("B", "D")
        else float(s1.query("signal == 'FC' and h == 5")["FM t"].iloc[0]) >= 2
    )
    g2 = (
        st.loc[pick, "CAGR %"] > st.loc["A", "CAGR %"]
        and st.loc[pick, "Sharpe"] > st.loc["A", "Sharpe"]
    )
    DECISION.write_text(
        f"discovery decision written {pd.Timestamp.now(tz='UTC').isoformat()}\n"
        f"pick (max net CAPM alpha t, 2013-21, among B/C/D): {pick}\n"
        f"G1 information gate: {bool(g1)}   G2 beats A on 2013-21: {bool(g2)}\n"
        f"ML compare 2013-21: {json.dumps(ml, default=float)}\n"
    )
    with open(OUT / "tables.md", "w") as f:
        f.write(
            "# fundamentals_v1 tables\n\n## Stage 1: signals, discovery 2013-21\n\n"
            + sv.md(s1, 4, index=False)
        )
        f.write(
            "\n\n## Stage 2: ML vs ML_F, discovery 2013-21\n\n"
            + sv.md(pd.Series(ml).to_frame("value"), 4)
        )
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
    pick = dec.split("among B/C/D): ")[1].split("\n")[0].strip()
    g1 = "G1 information gate: True" in dec
    g2 = "G2 beats A on 2013-21: True" in dec
    s1 = stage1(d, HOLDOUT)
    ml = ml_compare(d, HOLDOUT)
    daily = pd.read_parquet(OUT / "strategies_daily.parquet")
    res = {k: pd.DataFrame({"net": daily[k], "turn": 0.0}) for k in daily}
    panel, _, _ = sv.panel_for("L")
    res["_bench"] = panel["bench"]["Close"].pct_change().reindex(panel["dates"]).to_frame("ret")
    st = strat_table(res, HOLDOUT).drop(columns="turnover x/yr")
    full = strat_table(res, ("2013-01-01", "2026-12-31")).drop(columns="turnover x/yr")
    g3 = (
        st.loc[pick, "Sharpe"] >= st.loc["A", "Sharpe"]
        and st.loc[pick, "CAGR %"] > st.loc["TDEX", "CAGR %"]
    )
    boot = validity.sharpe_diff_test(daily[pick].loc["2022":], daily["A"].loc["2022":])
    verdict = {
        "pick": pick,
        "G1": g1,
        "G2": g2,
        "G3": bool(g3),
        "pass all": bool(g1 and g2 and g3),
        "holdout Sharpe diff vs A": boot["diff"],
        "95% CI": [boot["CI 2.5%"], boot["CI 97.5%"]],
    }
    json.dump(verdict, open(OUT / "holdout.json", "w"), indent=1, default=float)
    with open(OUT / "tables.md", "a") as f:
        f.write(
            "\n## Holdout 2022-26 (computed once)\n\n### Stage 1\n\n" + sv.md(s1, 4, index=False)
        )
        f.write("\n\n### ML vs ML_F\n\n" + sv.md(pd.Series(ml).to_frame("value"), 4))
        f.write("\n\n### Strategies 2022-26\n\n" + sv.md(st, 2))
        f.write("\n\n### Strategies 2013-26\n\n" + sv.md(full, 2))
        f.write(
            "\n\n### Verdict\n\n```\n" + json.dumps(verdict, indent=1, default=float) + "\n```\n"
        )
    print(open(OUT / "tables.md").read().split("## Holdout")[1])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {
        "features": stage_features,
        "scores": stage_scores,
        "discover": stage_discover,
        "holdout": stage_holdout,
    }[sys.argv[1]]()
