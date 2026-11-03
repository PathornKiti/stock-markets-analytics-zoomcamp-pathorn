"""foreign_v1 - does foreign-board demand predict Thai stock alpha? (protocol: output/protocol_frozen.txt)

    uv run python set-quant-alpha/experiments/foreign_v1/foreign_v1.py discover   # 2013-2021: all decisions
    uv run python set-quant-alpha/experiments/foreign_v1/foreign_v1.py holdout    # 2022-26, once, after discover

Inputs: data/raw/lseg/{foreign_board,prices}.parquet (scripts/fetch_lseg.py), the integrated_v1 panel (universe,
forward alphas, ML/V8 ranks, styles) and the alpha_portfolio_v1 extras (HMM state). Nothing is written outside
output/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from setalpha import config, validity

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
EXP = HERE.parent
LSEG = config.RAW_DIR / "lseg"
SIGNALS = {"prem": 0, "dprem20": +1, "fshare20": +1}  # prior sign (0 = two-sided)
HORIZONS = (5, 20)
DISCOVERY = ("2013-01-01", "2021-12-31")
HOLDOUT = ("2022-01-01", "2026-12-31")
CARRY, MIN_NAMES, MIN_NAMES_FM = 20, 10, 15
DECISION = OUT / "discovery_decision.txt"


def md(t: pd.DataFrame, digits: int = 2, index: bool = True) -> str:
    sys.path.insert(0, str(EXP / "strategy_v2"))
    import sv2

    return sv2.to_md(t if index else t.set_index(t.columns[0]), digits)


def load() -> dict:
    panel = joblib.load(EXP / "integrated_v1" / "output" / "panel.joblib")
    extra = joblib.load(EXP / "alpha_portfolio_v1" / "output" / "extra.joblib")
    dates, tickers = panel["dates"], panel["tickers"]
    fb = pd.read_parquet(LSEG / "foreign_board.parquet")
    lp = pd.read_parquet(LSEG / "prices.parquet").drop_duplicates(["Ticker", "Date"])
    W = (
        lambda d, c: d.assign(**{c: d[c].astype("float64")})
        .pivot_table(  # noqa: E731  (LSEG nullable dtypes)
            index="Date", columns="Ticker", values=c
        )
        .reindex(columns=tickers)
    )
    f_close, f_vol, f_val = W(fb, "f_close"), W(fb, "f_volume"), W(fb, "f_value")
    l_close, l_vol = W(lp, "Close"), W(lp, "Volume")
    idx = f_close.index.union(l_close.index).union(dates)
    f_close, f_vol, f_val, l_close, l_vol = (
        x.reindex(idx) for x in (f_close, f_vol, f_val, l_close, l_vol)
    )
    traded = f_vol.fillna(0) > 0
    f_val = f_val.where(f_val.notna(), f_vol * f_close).where(traded, 0.0)
    prem = (f_close / l_close - 1).where(traded)
    prem = prem.where(prem.abs() < 2.0)  # bad prints
    prem = prem.ffill(limit=CARRY)
    dprem = prem - prem.shift(20)
    l_val = (l_close * l_vol).where(l_close.notna())
    fshare = f_val.rolling(20, min_periods=10).sum() / l_val.rolling(20, min_periods=10).sum()
    fshare = fshare.where(
        f_close.notna().rolling(250, min_periods=1).max() > 0
    )  # only stocks with a foreign line
    sig = {
        k: v.reindex(dates).to_numpy(dtype="float64")
        for k, v in {"prem": prem, "dprem20": dprem, "fshare20": fshare}.items()
    }
    trad = panel["tradable"]
    r = {k: np.where(trad, v - 0.5, np.nan) for k, v in panel["sig"].items()}
    c2 = np.where(trad, (np.nan_to_num(r["ml"]) + np.nan_to_num(r["v8"])) / 2, np.nan)
    return {
        "dates": dates,
        "tickers": tickers,
        "trad": trad,
        "sig": sig,
        "fwd": panel["fwd"],
        "c2": c2,
        "aux": panel["aux"],
        "hmm": extra["hmm"],
        "has_line": fb.Ticker.nunique(),
    }


def _rank(a: np.ndarray) -> np.ndarray:
    return pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()


def daily_ic(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xr, yr = _rank(np.where(mask, x, np.nan)), _rank(np.where(mask, y, np.nan))
    ok = ~np.isnan(xr) & ~np.isnan(yr)
    out = np.full(len(x), np.nan)
    for t in np.where(ok.sum(1) >= MIN_NAMES)[0]:
        out[t] = np.corrcoef(xr[t, ok[t]], yr[t, ok[t]])[0, 1]
    return out


def fm_slope(d: dict, s: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cols = [s, d["c2"], d["aux"]["logsize"], d["aux"]["beta_mkt"], d["aux"]["mom121"]]
    X = [_rank(np.where(mask, c, np.nan)) for c in cols]
    yr = _rank(np.where(mask, y, np.nan))
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        ok = ~np.isnan(yr[t]) & np.all([~np.isnan(x[t]) for x in X], axis=0)
        if ok.sum() < MIN_NAMES_FM:
            continue
        A = np.column_stack(
            [np.ones(ok.sum())] + [(x[t, ok] - x[t, ok].mean()) / x[t, ok].std() for x in X]
        )
        b = np.linalg.lstsq(A, (yr[t, ok] - yr[t, ok].mean()) / yr[t, ok].std(), rcond=None)[0]
        out[t] = b[1]
    return out


def q_spread(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xr = _rank(np.where(mask & ~np.isnan(y), x, np.nan))
    hi, lo = xr > 0.8, xr <= 0.2
    with np.errstate(invalid="ignore"):
        s = np.nanmean(np.where(hi, y, np.nan), axis=1) - np.nanmean(
            np.where(lo, y, np.nan), axis=1
        )
    s[(~np.isnan(xr)).sum(1) < MIN_NAMES] = np.nan
    return s


def evaluate(d: dict, period: tuple[str, str]) -> tuple[pd.DataFrame, dict]:
    sel = (d["dates"] >= period[0]) & (d["dates"] <= period[1])
    rows, series = [], {}
    for k in SIGNALS:
        s = d["sig"][k]
        mask = d["trad"] & ~np.isnan(s)
        for h in HORIZONS:
            y = d["fwd"][h]
            ic = pd.Series(daily_ic(s, y, mask), index=d["dates"])[sel].dropna()
            fm = pd.Series(fm_slope(d, s, y, mask), index=d["dates"])[sel].dropna()
            qs = pd.Series(q_spread(s, y, mask), index=d["dates"])[sel].dropna()
            series[(k, h)] = ic
            rows.append(
                {
                    "signal": k,
                    "h": h,
                    "days": len(ic),
                    "names/day": mask[sel].sum(1).mean(),
                    "IC": ic.mean(),
                    "IC t (NW)": validity.nw_tstat(ic, lags=h),
                    "FM slope": fm.mean(),
                    "FM t (NW)": validity.nw_tstat(fm, lags=h),
                    "Q5-Q1 % per h": 100 * qs.mean(),
                    "Q t (NW)": validity.nw_tstat(qs, lags=h),
                }
            )
    return pd.DataFrame(rows), series


def coverage(d: dict) -> pd.DataFrame:
    yrs = d["dates"].year
    out = {
        k: pd.Series((d["trad"] & ~np.isnan(v)).sum(1), index=d["dates"]).groupby(yrs).mean()
        for k, v in d["sig"].items()
    }
    out["tradable"] = pd.Series(d["trad"].sum(1), index=d["dates"]).groupby(yrs).mean()
    return pd.DataFrame(out)


def gates(disc: pd.DataFrame) -> pd.DataFrame:
    g = disc[["signal", "h", "IC", "IC t (NW)", "FM t (NW)"]].copy()
    prior = g["signal"].map(SIGNALS)
    sign = np.where(prior == 0, np.sign(g["IC"]), prior)
    g["direction"] = sign
    g["G1"] = g["IC t (NW)"] * sign >= 2
    g["G2"] = g["FM t (NW)"] * sign >= 2
    return g


def stage_discover() -> None:
    d = load()
    cov = coverage(d)
    disc, series = evaluate(d, DISCOVERY)
    by_year = pd.DataFrame(
        {f"{k} h{h}": s.groupby(s.index.year).mean() for (k, h), s in series.items()}
    )
    hmm = pd.Series(d["hmm"].to_numpy(), index=d["dates"])
    by_regime = pd.DataFrame(
        {f"{k} h{h}": s.groupby(hmm.reindex(s.index)).mean() for (k, h), s in series.items()}
    )
    g = gates(disc)
    cov.to_csv(OUT / "coverage.csv")
    disc.to_csv(OUT / "discovery.csv", index=False)
    by_year.to_csv(OUT / "discovery_ic_by_year.csv")
    g.to_csv(OUT / "discovery_gates.csv", index=False)
    passed = g[g.G1 & g.G2]
    DECISION.write_text(
        f"discovery decision written {pd.Timestamp.now(tz='UTC').isoformat()}\n"
        f"stocks with a foreign-board line: {d['has_line']}\n"
        + g.to_string(index=False)
        + "\n\n"
        + (
            "go to holdout for: "
            + ", ".join(
                f"{r.signal} h{r.h} (sign {int(r.direction):+d})" for r in passed.itertuples()
            )
            if len(passed)
            else "no signal passed G1+G2: holdout is still computed once and reported, but nothing can pass"
        )
        + "\n"
    )
    with open(OUT / "tables.md", "w") as f:
        f.write(
            "# foreign_v1 tables\n\n## Coverage: tradable names per day with the signal\n\n"
            + md(cov, 1)
        )
        f.write("\n\n## Discovery 2013-2021\n\n" + md(disc, 4, index=False))
        f.write("\n\n## Discovery IC by year\n\n" + md(by_year, 3))
        f.write("\n\n## Discovery IC by HMM regime\n\n" + md(by_regime, 3))
        f.write("\n\n## Gates\n\n" + md(g, index=False) + "\n")
    print(open(OUT / "tables.md").read())
    print(DECISION.read_text())


def stage_holdout() -> None:
    if not DECISION.exists():
        sys.exit(
            "run `discover` first: the holdout is read only after the discovery decision is written"
        )
    if (OUT / "holdout.csv").exists():
        sys.exit("holdout already computed once; see output/holdout.csv")
    d = load()
    g = pd.read_csv(OUT / "discovery_gates.csv")
    hold, series = evaluate(d, HOLDOUT)
    hold = hold.merge(g[["signal", "h", "direction", "G1", "G2"]], on=["signal", "h"])
    hold["G3"] = hold["IC t (NW)"] * hold["direction"] >= 1.5
    hold["pass all"] = hold.G1 & hold.G2 & hold.G3
    hold.to_csv(OUT / "holdout.csv", index=False)
    by_year = pd.DataFrame(
        {f"{k} h{h}": s.groupby(s.index.year).mean() for (k, h), s in series.items()}
    )
    with open(OUT / "tables.md", "a") as f:
        f.write("\n## Holdout 2022-2026 (computed once)\n\n" + md(hold, 4, index=False))
        f.write("\n\n## Holdout IC by year\n\n" + md(by_year, 3) + "\n")
    print(md(hold, 4, index=False))
    json.dump(
        {"passed": hold.loc[hold["pass all"], ["signal", "h"]].to_dict("records")},
        open(OUT / "summary.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"discover": stage_discover, "holdout": stage_holdout}[sys.argv[1]]()
