"""survivorship_v1 - how much does survivorship bias flatter the S5 rule? (protocol: output/protocol_frozen.txt)

    uv run --group lseg python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py data     # build Y and L raw stores
    uv run python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py scores Y              # ~20-30 min each
    uv run python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py scores L
    uv run python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py sim                   # tables

Y = production Yahoo prices (today's listings). L = Y + LSEG prices of every RIC ever in SET100 that Y lacks
(delisted ones included). Each variant gets its own data dir (SETALPHA_DATA_DIR) under data/processed/survivorship/,
so production data and models are never touched. LSEG inputs come from scripts/fetch_lseg.py.
"""

from __future__ import annotations

import json
import os
import shutil
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
PROD = Path(
    os.environ.get("SETALPHA_PROD_DATA_DIR", config.DATA_DIR)
)  # production store (the child sees a variant dir)
LSEG = PROD / "raw" / "lseg"
ROOT = PROD / "processed" / "survivorship"  # gitignored (data/processed/*)
VARIANTS = ("Y", "L")
FIRST_YEAR, LAST_YEAR = 2013, 2026
TRBC_TO_YAHOO = {
    "Energy": "Energy",
    "Financials": "Financial Services",
    "Basic Materials": "Basic Materials",
    "Industrials": "Industrials",
    "Consumer Cyclicals": "Consumer Cyclical",
    "Consumer Non-Cyclicals": "Consumer Defensive",
    "Healthcare": "Healthcare",
    "Technology": "Technology",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Telecommunications Services": "Communication Services",
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def md(t: pd.DataFrame, digits: int = 2, index: bool = True) -> str:
    sys.path.insert(0, str(EXP / "strategy_v2"))
    import sv2

    return sv2.to_md(t if index else t.set_index(t.columns[0]), digits)


# ============================================================================ 1. data
def lseg_adjusted(l: pd.DataFrame) -> pd.DataFrame:
    """LSEG split-adjusted OHLCV -> Yahoo-style dividend-adjusted OHLC (from daily total return), split-adj volume."""
    parts = []
    for t, g in l.drop_duplicates(["Ticker", "Date"]).groupby("Ticker"):
        g = g.sort_values("Date").set_index("Date")
        g = g.astype(
            {c: "float64" for c in ["Open", "High", "Low", "Close", "Volume", "TotalReturn"]}
        )
        g = g[g["Close"].notna() & (g["Close"] > 0)]
        if len(g) < 2:
            continue
        r = g["TotalReturn"].astype(float)
        r = r.where(r.notna(), g["Close"].pct_change())
        r.iloc[0] = 0.0
        adj = (1 + r.fillna(0)).cumprod()
        adj = adj / adj.iloc[-1] * g["Close"].iloc[-1]  # anchored at the last close, like Yahoo
        f = adj / g["Close"]
        parts.append(
            pd.DataFrame(
                {
                    "Date": g.index,
                    "Open": g["Open"] * f,
                    "High": g["High"] * f,
                    "Low": g["Low"] * f,
                    "Close": adj,
                    "Volume": g["Volume"].astype(float),
                    "Ticker": t,
                }
            ).reset_index(drop=True)
        )
    return pd.concat(parts, ignore_index=True)


def renamed_duplicates(add: pd.DataFrame, y: pd.DataFrame) -> dict[str, str]:
    """Missing RICs whose daily returns match a Yahoo ticker (corr > 0.95, >= 250 shared days): ticker renames."""
    yr = y.pivot_table(index="Date", columns="Ticker", values="Close").pct_change(fill_method=None)
    dup = {}
    for t, g in add.groupby("Ticker"):
        r = g.set_index("Date")["Close"].pct_change().dropna()
        sub = yr.reindex(r.index)
        n = sub.notna().sum()
        c = sub.loc[:, n >= 250].corrwith(r)
        if len(c) and c.max() > 0.95:
            dup[t] = c.idxmax()
    return dup


def trbc_sectors(rics: list[str]) -> pd.Series:
    path = LSEG / "trbc.parquet"
    if not path.exists():
        from setalpha import lseg

        with lseg.session() as ld:
            s = ld.get_data(rics, ["TR.TRBCEconomicSector"])
        s.columns = ["Ticker", "trbc"]
        s.to_parquet(path, index=False)
    s = pd.read_parquet(path).set_index("Ticker")["trbc"]
    return s.map(TRBC_TO_YAHOO).fillna("")


def stage_data() -> None:
    raw = PROD / "raw"
    y = pd.read_parquet(raw / "prices.parquet")
    l = pd.read_parquet(LSEG / "prices.parquet")
    members = pd.read_parquet(LSEG / "index_members.parquet")
    m100 = members[members["index"] == "SET100"]
    have = set(y.Ticker)
    missing = sorted(set(m100.ric) - have)
    add = lseg_adjusted(l[l.Ticker.isin(missing)])
    dup = renamed_duplicates(add, y)
    add = add[~add.Ticker.isin(dup)]
    log(
        f"SET100-ever RICs {m100.ric.nunique()}, missing from Yahoo {len(missing)}, renamed duplicates {len(dup)}: {dup}"
    )
    log(
        f"added to L: {add.Ticker.nunique()} RICs, {len(add):,} rows; no LSEG data: "
        f"{sorted(set(missing) - set(add.Ticker) - set(dup))}"
    )

    # validation: LSEG total return vs Yahoo for names in both
    both = sorted(set(l.Ticker) & have)
    rows = []
    for t in both:
        a = lseg_adjusted(l[l.Ticker == t]).set_index("Date")["Close"].pct_change()
        b = y[y.Ticker == t].set_index("Date")["Close"].pct_change()
        j = pd.concat([a, b], axis=1, keys=["lseg", "yahoo"], sort=True).dropna()
        j = j[j.yahoo.abs() < 0.45]
        if len(j) > 250:
            rows.append(
                {
                    "Ticker": t,
                    "days": len(j),
                    "corr": j.lseg.corr(j.yahoo),
                    "cum_diff_%": 100 * ((1 + j.lseg).prod() / (1 + j.yahoo).prod() - 1),
                }
            )
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "lseg_vs_yahoo.csv", index=False)
    log(
        f"LSEG vs Yahoo on {len(val)} shared names: median daily corr {val['corr'].median():.3f}, "
        f"median |cum diff| {val['cum_diff_%'].abs().median():.1f}%"
    )

    sec = trbc_sectors(sorted(add.Ticker.unique()))
    for v in VARIANTS:
        d = ROOT / v / "raw"
        d.mkdir(parents=True, exist_ok=True)
        for f in raw.glob("*.parquet"):
            if f.name not in ("prices.parquet", "universe.parquet", "sectors.parquet"):
                shutil.copy2(f, d / f.name)
        uni = pd.read_parquet(raw / "universe.parquet")
        sectors = pd.read_parquet(raw / "sectors.parquet")
        prices = y
        if v == "L":
            extra = sorted(add.Ticker.unique())
            uni = pd.concat(
                [uni, pd.DataFrame({"Ticker": extra, "name": extra, "marketCap": np.nan})],
                ignore_index=True,
            )
            sectors = pd.concat(
                [
                    sectors,
                    pd.DataFrame(
                        {"Ticker": extra, "sector": [sec.get(t, "") for t in extra], "industry": ""}
                    ),
                ],
                ignore_index=True,
            )
            prices = pd.concat([y, add.astype({"Ticker": y["Ticker"].dtype})], ignore_index=True)
        prices.to_parquet(d / "prices.parquet", index=False)
        uni.to_parquet(d / "universe.parquet", index=False)
        sectors.to_parquet(d / "sectors.parquet", index=False)
        log(f"{v}: {prices.Ticker.nunique()} tickers, {len(prices):,} price rows -> {d}")
    json.dump(
        {"missing": missing, "renamed": dup, "added": sorted(add.Ticker.unique())},
        open(OUT / "added_rics.json", "w"),
        indent=1,
    )


# ============================================================================ 2. scores (one process per variant)
def stage_scores(v: str) -> None:
    if os.environ.get("SETALPHA_DATA_DIR") != str(
        ROOT / v
    ):  # re-exec with the variant's isolated data dir
        env = {
            **os.environ,
            "SETALPHA_DATA_DIR": str(ROOT / v),
            "SETALPHA_PROD_DATA_DIR": str(PROD),
        }
        sys.exit(subprocess.call([sys.executable, __file__, "scores", v], env=env))
    from setalpha import ingest, model, report, transform

    sys.path.insert(0, str(EXP / "model_v2"))
    import model_v2 as mv

    assert ROOT / v == config.DATA_DIR, config.DATA_DIR
    t0 = time.time()
    if not (config.PROCESSED_DIR / "dataset.parquet").exists():
        df = transform.build_dataset(ingest.ingest_all("file"))
        transform.save(df)
        log(f"[{v}] dataset {df.shape}, {time.time() - t0:.0f}s")
        del df
    df, feats, rank_feats = mv.load()  # dataset + dummies + 10/20-day alphas + rank features
    best = json.loads((config.MODELS_DIR / "best_params.json").read_text())
    members = report.load_meta()["ensemble"]
    lab = model.labelled(df)
    cols = list(dict.fromkeys(feats + rank_feats))  # the two lists share the dummy columns
    rows_all = df[df["liquid"] & (df["Date"].dt.year >= FIRST_YEAR)][["Date", "Ticker"] + cols]
    parts = []
    fits: dict[str, dict] = {}
    for yr in range(FIRST_YEAR, LAST_YEAR + 1):
        fit_year = 2020 if yr in (2020, 2021) else yr  # production: 2020-21 scored by the <2020 fit
        start = f"{fit_year}-01-01"
        if fit_year not in fits:
            train = model._embargo(lab[lab["Date"] < start], start)
            fits[fit_year] = {n: model.fit(n, train, feats, **best[n]) for n in members}
        rows = rows_all[rows_all["Date"].dt.year == yr].reset_index(drop=True)
        part = rows[["Date", "Ticker"]].copy()
        for n in members:
            part[f"pred_{n}"] = model.predict(fits[fit_year][n], rows, feats)
        part["pred_v8"] = mv.scorer("V8", df, feats, rank_feats, f"{yr}-01-01", rows).to_numpy()
        parts.append(part)
        log(f"[{v}] {yr}: {len(rows):,} rows scored, {time.time() - t0:.0f}s")
    out = model.add_ensemble(pd.concat(parts, ignore_index=True), members)
    out = out.merge(
        df[["Date", "Ticker", "tradable", "fwd_alpha_5"]], on=["Date", "Ticker"], how="left"
    )
    out.to_parquet(OUT / f"scores_{v}.parquet", index=False)
    log(f"[{v}] scores saved, {time.time() - t0:.0f}s")


# ============================================================================ 3. simulation and tables
def panel_for(v: str) -> tuple[dict, dict, pd.DataFrame]:
    sys.path.insert(0, str(EXP / "integrated_v1"))
    import integrated_v1 as iv  # noqa: F401  (imported for fast_sim / Cfg)

    s = pd.read_parquet(OUT / f"scores_{v}.parquet")
    s = s[s["tradable"].fillna(False).astype(bool)].copy()
    g = s.groupby("Date")
    s["r_ml"], s["r_v8"] = g["pred_ens"].rank(pct=True), g["pred_v8"].rank(pct=True)
    prices = pd.read_parquet(ROOT / v / "raw" / "prices.parquet")
    num = ["Open", "High", "Low", "Close", "Volume"]
    prices[num] = prices[num].astype("float64")  # LSEG rows arrive as nullable Float64
    dates_all = pd.DatetimeIndex(sorted(prices["Date"].unique()), name="Date")
    dates = dates_all[(dates_all >= s["Date"].min()) & (dates_all <= s["Date"].max())]
    tickers = sorted(s["Ticker"].unique())
    W = lambda c: prices.pivot_table(index="Date", columns="Ticker", values=c).reindex(  # noqa: E731
        index=dates_all, columns=pd.Index(tickers, name="Ticker")
    )
    O, C, L = W("Open"), W("Close"), W("Low")
    # delisting exit: after a stock's last price, its open is its last close (a held position is sold there)
    last = C.apply(lambda col: col.last_valid_index())
    for t, d in last.items():
        if d is not None and d < dates_all[-1]:
            after = O.index > d
            O.loc[after, t] = C.at[d, t]
    wide = (
        lambda col: s.pivot_table(index="Date", columns="Ticker", values=col)
        .reindex(  # noqa: E731
            index=dates, columns=tickers
        )
        .to_numpy()
    )
    r_ml, r_v8 = wide("r_ml"), wide("r_v8")
    trad = ~np.isnan(r_ml)
    c2 = np.where(trad, (np.nan_to_num(r_ml - 0.5) + np.nan_to_num(r_v8 - 0.5)) / 2, np.nan)
    mk = pd.read_parquet(ROOT / v / "raw" / "market.parquet")
    bench = mk[mk.Ticker == config.BENCHMARK].set_index("Date").sort_index()
    panel = {
        "dates": dates,
        "tickers": tickers,
        "O": O.reindex(dates).to_numpy(),
        "C": C.reindex(dates).to_numpy(),
        "L": L.reindex(dates).to_numpy(),
        "C_prev0": C.shift(1).reindex(dates).to_numpy()[0],
        "bench": bench[["Open", "Close"]],
    }
    # EW tradable universe, close-to-close, next day (diagnostic)
    ret = C.pct_change(fill_method=None).reindex(dates)
    ew = pd.Series(
        np.nanmean(np.where(np.roll(trad, 1, axis=0), ret.to_numpy(), np.nan), axis=1), index=dates
    )
    return panel, {"C2": c2}, pd.DataFrame({"ew": ew.iloc[1:]})


def stats(r: pd.Series, b: pd.Series) -> dict:
    sys.path.insert(0, str(EXP / "integrated_v1"))
    import integrated_v1 as iv

    r = r.dropna()
    a, t = iv.capm_t(r, b.reindex(r.index))
    return {
        "CAGR %": iv.cagr(r),
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
        "MaxDD %": iv.maxdd(r),
        "CAPM alpha %": a,
        "alpha t": t,
        "IR vs TDEX": iv.ir(r, b),
    }


def daily_ic(v: str) -> pd.Series:
    s = pd.read_parquet(OUT / f"scores_{v}.parquet")
    s = s[s["tradable"].fillna(False).astype(bool) & s["fwd_alpha_5"].notna()].copy()
    g = s.groupby("Date")
    s["c2"] = (g["pred_ens"].rank(pct=True) + g["pred_v8"].rank(pct=True)) / 2
    return s.groupby("Date").apply(lambda d: d["c2"].corr(d["fwd_alpha_5"], method="spearman"))


def index_bias() -> pd.DataFrame:
    """EW return of all SET100 members vs members present in Yahoo, both from LSEG total return (one source)."""
    m = pd.read_parquet(LSEG / "index_members.parquet")
    m = m[m["index"] == "SET100"]
    l = pd.read_parquet(LSEG / "prices.parquet").drop_duplicates(["Ticker", "Date"])
    tr = l.pivot_table(index="Date", columns="Ticker", values="TotalReturn")
    y = set(pd.read_parquet(PROD / "raw" / "prices.parquet", columns=["Ticker"]).Ticker)
    y |= set(
        json.load(open(OUT / "added_rics.json"))["renamed"]
    )  # renamed tickers Yahoo has under a new name
    mem = (
        m.assign(v=1)
        .pivot_table(index="Date", columns="ric", values="v")
        .reindex(tr.index)
        .shift(1)
    )  # held from t-1
    mem = mem.reindex(columns=tr.columns)
    all_ = tr.where(mem == 1).mean(axis=1)
    surv = tr.where((mem == 1) & tr.columns.isin(list(y))[None, :]).mean(axis=1)
    d = pd.DataFrame({"all": all_, "in_yahoo": surv}).dropna()
    yr = d.groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    yr["bias_pts"] = yr["in_yahoo"] - yr["all"]
    cov = m.assign(year=m.Date.dt.year, iny=m.ric.isin(y)).groupby("year")["iny"].mean()
    yr["missing_member_days_%"] = 100 * (1 - cov)
    return yr


def stage_sim() -> None:
    sys.path.insert(0, str(EXP / "integrated_v1"))
    import integrated_v1 as iv

    s5 = iv.Cfg("C2", 10, 5, True, False)
    res, ew = {}, {}
    bench = None
    for v in VARIANTS:
        panel, scores, e = panel_for(v)
        res[v] = iv.fast_sim(panel, scores, s5, fee=config.FEE_PER_SIDE)
        ew[v] = e["ew"]
        b = panel["bench"]["Close"]
        bench = b.pct_change().reindex(panel["dates"]) if bench is None else bench
        log(f"{v}: simulated {len(res[v])} days")
    pd.DataFrame(
        {f"{v}_{k}": res[v][k] for v in VARIANTS for k in ("gross", "net", "turn")}
    ).to_parquet(OUT / "daily.parquet")

    periods = {
        "2013-26": ("2013", "2026"),
        "2016-26": ("2016", "2026"),
        "2013-21": ("2013", "2021"),
        "2022-26": ("2022", "2026"),
    }
    rows = []
    for p, (a, b_) in periods.items():
        for v in VARIANTS:
            rows.append(
                {"period": p, "series": f"S5 rule, {v}", **stats(res[v]["net"].loc[a:b_], bench)}
            )
            rows.append(
                {
                    "period": p,
                    "series": f"EW tradable top-100, {v}",
                    **stats(ew[v].loc[a:b_], bench),
                }
            )
        rows.append(
            {
                "period": p,
                "series": "TDEX",
                **stats(bench.loc[a:b_], bench),
                "CAPM alpha %": np.nan,
                "alpha t": np.nan,
                "IR vs TDEX": np.nan,
            }
        )
    main = pd.DataFrame(rows)
    main.to_csv(OUT / "main.csv", index=False)

    per_year = pd.DataFrame(
        {
            f"S5 {v}": res[v]["net"].groupby(res[v].index.year).apply(lambda x: (1 + x).prod() - 1)
            * 100
            for v in VARIANTS
        }
    )
    per_year["S5 L - Y"] = per_year["S5 L"] - per_year["S5 Y"]
    for v in VARIANTS:
        per_year[f"EW {v}"] = (
            ew[v].groupby(ew[v].index.year).apply(lambda x: (1 + x).prod() - 1) * 100
        )
    per_year["EW L - Y"] = per_year["EW L"] - per_year["EW Y"]
    per_year.to_csv(OUT / "per_year.csv")

    ic = {v: daily_ic(v) for v in VARIANTS}
    ic_t = pd.DataFrame({v: ic[v].groupby(ic[v].index.year).mean() for v in VARIANTS})
    ic_t.loc["all"] = [ic[v].mean() for v in VARIANTS]
    ic_t.to_csv(OUT / "ic_by_year.csv")

    ib = index_bias()
    ib.to_csv(OUT / "index_bias.csv")

    # replication: Y run vs the existing integrated_v1 S5-rule fast sim
    ref = __import__("joblib").load(EXP / "integrated_v1" / "output" / "grid.joblib")["res"][s5.key]
    ref_net = iv.net_at(ref, config.FEE_PER_SIDE)
    j = pd.concat([res["Y"]["net"], ref_net], axis=1, keys=["new", "ref"]).dropna()
    rep = {
        "key": s5.key,
        "days": len(j),
        "corr daily": float(j.new.corr(j.ref)),
        "CAGR new %": iv.cagr(j.new),
        "CAGR integrated_v1 %": iv.cagr(j.ref),
        "note": "new = scores on all liquid rows; integrated_v1 = scores on rows with a known 5-day target",
    }

    # is L - Y distinguishable from noise? paired daily differences and a paired Sharpe bootstrap
    sig = {}
    for p, (a, b_) in periods.items():
        dl, dy = res["L"]["net"].loc[a:b_], res["Y"]["net"].loc[a:b_]
        diff = (dl - dy).dropna()
        bt = validity.sharpe_diff_test(dl, dy)
        sig[p] = {
            "L - Y % a year": 252 * 100 * diff.mean(),
            "NW t": validity.nw_tstat(diff, lags=5),
            "Sharpe L - Y": bt["diff"],
            "95% CI low": bt["CI 2.5%"],
            "95% CI high": bt["CI 97.5%"],
        }
    sig = pd.DataFrame(sig).T
    sig.to_csv(OUT / "l_minus_y_test.csv")
    json.dump(rep, open(OUT / "replication.json", "w"), indent=1, default=float)

    with open(OUT / "tables.md", "w") as f:
        f.write("# survivorship_v1 tables\n\n## S5 rule and universe, net of 0.168%/side\n\n")
        f.write(md(main, 2, index=False) + "\n\n## Per year (%)\n\n")
        f.write(md(per_year, 1) + "\n\n## C2 daily rank IC vs 5-day alpha\n\n")
        f.write(md(ic_t, 4) + "\n\n## SET100 index-level bias (LSEG total return, EW)\n\n")
        f.write(md(ib, 1) + "\n\n## Replication of the existing S5-rule sim (Y)\n\n")
        f.write(json.dumps(rep, indent=1, default=float) + "\n\n## Is L - Y more than noise?\n\n")
        f.write(md(sig, 2) + "\n")
    print(open(OUT / "tables.md").read())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1]
    if stage == "data":
        stage_data()
    elif stage == "scores":
        stage_scores(sys.argv[2])
    elif stage == "sim":
        stage_sim()
