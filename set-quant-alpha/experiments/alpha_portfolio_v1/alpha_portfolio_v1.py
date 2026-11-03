"""alpha_portfolio_v1 - factor diagnostics -> correlation control -> portfolio construction -> robustness (2026-11-02).

    uv run python set-quant-alpha/experiments/alpha_portfolio_v1/alpha_portfolio_v1.py diag    # stage 1 (~1 min)
    uv run python set-quant-alpha/experiments/alpha_portfolio_v1/alpha_portfolio_v1.py corr    # stage 2 (~1 min)
    uv run python set-quant-alpha/experiments/alpha_portfolio_v1/alpha_portfolio_v1.py build   # stage 3 (~10 min)
    uv run python set-quant-alpha/experiments/alpha_portfolio_v1/alpha_portfolio_v1.py robust  # stage 4 (~10 min)

The protocol (hypotheses, grid, selection objective, gates) was frozen in output/protocol_frozen.txt before any of
this code ran. Inputs are integrated_v1's cached panel (8 walk-forward signals, forward alphas, prices, benchmark and
factor data); nothing is re-trained. Every choice is walk-forward: signal weights for year Y use ICs whose forward
windows end before Y; the composite and the portfolio construction for Y are chosen on 2013 .. Y-1 only.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from setalpha import config, factors, portfolio, risk, validity
from setalpha import signal_eval as se
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "integrated_v1"))
import integrated_v1 as iv  # noqa: E402

EXTRA = OUT / "extra.joblib"
SCORES = OUT / "scores.joblib"
GRID = OUT / "grid.joblib"
FEE, TFEE = config.FEE_PER_SIDE, iv.TFEE
DAYS = 252
SIGNALS = iv.SIGNALS
FIRST_YEAR, WF_YEARS = 2013, iv.WF_YEARS
COMPOSITES = ["C1", "C2", "K1", "K2", "K3", "K4", "K5", "K6"]
H_ALPHA, IC_PRIOR = (
    20,
    0.05,
)  # Grinold alpha horizon (days) and the a-priori IC before any history (2013)
CAND, CAP, SECTOR_CAP, BAND, MIN_W, COV_WIN = 40, 0.10, 0.30, 0.5, 0.005, 120
VT = 0.15  # ablation A1 ex-ante vol target
OOS = slice(pd.Timestamp("2016-01-01"), None)
STYLES = ["size", "beta", "momentum", "volatility", "illiquidity"]
OWN = {
    "lowbeta": "beta",
    "lowvol": "volatility",
}  # a signal's own characteristic is not used as its control
PERIODS = {"2013-15": ("2013", "2015"), "2016-21": ("2016", "2021"), "2022-26": ("2022", "2026")}
TRIALS_BEFORE = validity.N_TRIALS - validity.TRIALS.get("alpha_portfolio_v1", 0)  # 679


def log(msg: str, t0: float | None = None) -> None:
    print(msg + (f"  [{time.time() - t0:.0f}s]" if t0 else ""), flush=True)


# ============================================================================ data
def build_extra(panel: dict) -> dict:
    """Point-in-time extras on the panel grid: idio / 60d vol, Amihud, traded value, HMM regime, sector, returns."""
    dates, tickers = panel["dates"], panel["tickers"]
    cols = ["Date", "Ticker", "idio_vol", "vol_60", "amihud_20", "dvol60", "hmm_state"]
    d = pd.read_parquet(
        config.PROCESSED_DIR / "dataset.parquet",
        columns=cols,
        filters=[("Date", ">=", pd.Timestamp("2012-06-01"))],
    )
    d = d[d["Ticker"].isin(tickers)]
    wide = {
        c: d.pivot_table(index="Date", columns="Ticker", values=c)
        .reindex(index=dates, columns=tickers)
        .to_numpy()
        for c in ("idio_vol", "vol_60", "amihud_20", "dvol60")
    }
    hmm = d.groupby("Date")["hmm_state"].first().reindex(dates)
    sec = (
        pd.read_parquet(config.RAW_DIR / "sectors.parquet")
        .drop_duplicates("Ticker")
        .set_index("Ticker")["sector"]
        .reindex(tickers)
        .fillna("")
        .replace("", "Unknown")
    )
    px = pd.read_parquet(
        config.RAW_DIR / "prices.parquet",
        columns=["Date", "Ticker", "Close"],
        filters=[("Date", ">=", pd.Timestamp("2012-06-01"))],
    )
    C = (
        px[px["Ticker"].isin(tickers)]
        .pivot_table(index="Date", columns="Ticker", values="Close")
        .reindex(columns=tickers)
    )
    R = C.pct_change(fill_method=None)
    R = R.where(R.abs() < 0.45)
    row = R.index.get_indexer(dates)
    assert (row >= 0).all(), "panel dates missing from prices"
    return {**wide, "hmm": hmm, "sector": sec, "R": R.to_numpy(), "R_dates": R.index, "R_row": row}


def load() -> tuple[dict, dict]:
    panel = joblib.load(iv.PANEL)
    if not EXTRA.exists():
        joblib.dump(build_extra(panel), EXTRA, compress=3)
    return panel, joblib.load(EXTRA)


def zrank(a: np.ndarray, mask: np.ndarray, fill: bool = True) -> np.ndarray:
    """Per-row rank -> centred, unit-variance score inside `mask`; missing values inside the mask -> 0 if `fill`."""
    r = pd.DataFrame(np.where(mask, a, np.nan)).rank(axis=1, pct=True).to_numpy() - 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        z = r / np.nanstd(r, axis=1, keepdims=True)
    if fill:
        z = np.where(mask & np.isnan(z), 0.0, z)
    return np.where(mask, z, np.nan)


def zscore_rows(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - np.nanmean(a, axis=1, keepdims=True)) / np.nanstd(a, axis=1, keepdims=True)


def styles(panel: dict, ex: dict) -> dict[str, np.ndarray]:
    trad = panel["tradable"]
    raw = {
        "size": panel["aux"]["logsize"],
        "beta": panel["aux"]["beta_mkt"],
        "momentum": panel["aux"]["mom121"],
        "volatility": ex["vol_60"],
        "illiquidity": np.log(ex["amihud_20"]),
    }
    return {k: np.where(trad, v, np.nan) for k, v in raw.items()}


def first_row(dates: pd.DatetimeIndex, Y: int) -> int:
    return int(dates.searchsorted(pd.Timestamp(f"{Y}-01-01")))


def nw(x, lags: int = 5) -> float:
    x = pd.Series(x).dropna()
    return validity.nw_tstat(x, lags) if len(x) > 10 else np.nan


# ============================================================================ stage 1: diagnostics
def deciles(s: np.ndarray, y: np.ndarray, q: int = 10) -> np.ndarray:
    """T x q mean of y by per-row decile of s (NaN-safe)."""
    m = ~np.isnan(s) & ~np.isnan(y)
    r = pd.DataFrame(np.where(m, s, np.nan)).rank(axis=1, pct=True, method="first").to_numpy()
    dec = np.ceil(r * q)
    out = np.full((len(s), q), np.nan)
    for d in range(1, q + 1):
        k = dec == d
        cnt = k.sum(1)
        out[:, d - 1] = np.where(cnt > 0, np.where(k, y, 0.0).sum(1) / np.maximum(cnt, 1), np.nan)
    return out


def top_set(s: np.ndarray, top: float = 0.9) -> np.ndarray:
    r = pd.DataFrame(s).rank(axis=1, pct=True).to_numpy()
    return r > top


def top_spread(s: np.ndarray, y: np.ndarray, dd: np.ndarray, fee: float = FEE) -> pd.DataFrame:
    """Per decision day: top-decile mean forward alpha - universe mean, top-decile turnover vs the previous decision
    day, and the spread net of 2 x fee x turnover (a long-only top-decile book's round trip)."""
    top = top_set(s)
    valid = ~np.isnan(s) & ~np.isnan(y)
    idx = np.flatnonzero(dd)
    rows, prev = [], None
    for t in idx:
        k = top[t] & valid[t]
        if k.sum() < 3 or valid[t].sum() < 20:
            prev = top[t]
            continue
        spread = y[t, k].mean() - y[t, valid[t]].mean()
        tau = 1.0 - (top[t] & prev).sum() / max(top[t].sum(), 1) if prev is not None else np.nan
        rows.append((t, spread, tau))
        prev = top[t]
    d = pd.DataFrame(rows, columns=["t", "spread", "tau"]).set_index("t")
    d["net"] = d["spread"] - 2 * fee * d["tau"]
    return d


def fm_one(
    sig: np.ndarray, y: np.ndarray, ctl: dict[str, np.ndarray], trad: np.ndarray, sec: np.ndarray
) -> dict:
    """Daily Fama-MacBeth slopes of y (% 5d alpha) on the z-scored signal: raw, + styles, + styles + sector FE;
    and the cross-sectional R^2 of the signal on the styles (+ sector)."""
    names = list(ctl)
    Zc = np.stack([ctl[k] for k in names], axis=-1)  # T x K x m (already rank-z, missing 0)
    out = {
        "raw": [],
        "styles": [],
        "styles+sector": [],
        "R2 styles": [],
        "R2 styles+sector": [],
        "neutral IC": [],
    }
    for t in range(len(sig)):
        m = trad[t] & ~np.isnan(sig[t]) & ~np.isnan(y[t])
        n = m.sum()
        if n < 30:
            continue
        r = pd.Series(sig[t, m]).rank(pct=True).to_numpy() - 0.5
        zs = r / r.std()
        yy = 100 * y[t, m]
        X = np.column_stack([np.ones(n), Zc[t, m]])
        out["raw"].append(np.linalg.lstsq(np.column_stack([np.ones(n), zs]), yy, rcond=None)[0][1])
        out["styles"].append(np.linalg.lstsq(np.column_stack([X, zs]), yy, rcond=None)[0][-1])
        b = np.linalg.lstsq(X, zs, rcond=None)[0]
        res = zs - X @ b
        out["R2 styles"].append(1 - res.var() / zs.var())
        out["neutral IC"].append(pd.Series(res).corr(pd.Series(yy), method="spearman"))
        g = sec[t, m]
        cnt = np.bincount(g, minlength=g.max() + 1)

        def dm(v):
            v = v.reshape(n, -1)
            mu = np.stack(
                [np.bincount(g, weights=v[:, j], minlength=len(cnt)) for j in range(v.shape[1])], 1
            )
            return v - mu[g] / np.maximum(cnt[g], 1)[:, None]

        Xs, ys, zss = dm(Zc[t, m]), dm(yy)[:, 0], dm(zs)[:, 0]
        out["styles+sector"].append(
            np.linalg.lstsq(np.column_stack([Xs, zss]), ys, rcond=None)[0][-1]
        )
        b2 = np.linalg.lstsq(Xs, zss, rcond=None)[0]
        out["R2 styles+sector"].append(1 - (zss - Xs @ b2).var() / zs.var())
    return {k: np.asarray(v) for k, v in out.items()}


def stage_diag() -> None:
    t0 = time.time()
    panel, ex = load()
    dates, trad = panel["dates"], panel["tradable"]
    sig, fwd = panel["sig"], panel["fwd"]
    hmm = ex["hmm"].to_numpy()
    sec_codes = pd.factorize(ex["sector"])[0]
    sec = np.broadcast_to(sec_codes, trad.shape)
    st = styles(panel, ex)
    stz = {k: zrank(v, trad) for k, v in st.items()}
    log("data loaded", t0)

    # ---- IC at 5 / 10 / 20 days, decay, by year, by regime
    ic = {
        h: pd.DataFrame({k: se.rank_ic_panel(sig[k], fwd[h]) for k in SIGNALS}, index=dates)
        for h in (5, 10, 20)
    }
    ref = joblib.load(iv.GRID)["ics"][5]
    log(
        f"check vs integrated_v1 ICs: max |diff| = {np.nanmax(np.abs(ic[5].to_numpy() - ref.to_numpy())):.2e}",
        t0,
    )
    marg = {"6-10": fwd[10] - fwd[5], "11-20": fwd[20] - fwd[10]}
    icm = {
        w: pd.DataFrame({k: se.rank_ic_panel(sig[k], y) for k in SIGNALS}, index=dates)
        for w, y in marg.items()
    }
    summ, decay = {}, {}
    for k in SIGNALS:
        for h in (5, 10, 20):
            s = se.ic_summary(ic[h][k], h)
            summ[(k, h)] = {
                "IC mean": s["IC mean"],
                "ICIR (daily)": s["ICIR (daily obs)"],
                "ICIR annualised": s["ICIR annualised"],
                "t NW": s["t (Newey-West)"],
                "t non-overlap": s["t (non-overlapping)"],
                "IC>0 %": s["IC > 0 % of days"],
            }
        decay[k] = {
            "IC 1-5": ic[5][k].mean(),
            "IC 1-10": ic[10][k].mean(),
            "IC 1-20": ic[20][k].mean(),
            "IC 6-10 (marginal)": icm["6-10"][k].mean(),
            "t 6-10": nw(icm["6-10"][k], 5),
            "IC 11-20 (marginal)": icm["11-20"][k].mean(),
            "t 11-20": nw(icm["11-20"][k], 10),
            "IC20 / IC5": ic[20][k].mean() / ic[5][k].mean(),
            "IC20/sqrt4 / IC5 (1 = IC per bet constant)": ic[20][k].mean() / 2 / ic[5][k].mean(),
        }
    summ = pd.DataFrame(summ).T
    summ.index.names = ["signal", "h"]
    decay = pd.DataFrame(decay).T
    yr = ic[5].groupby(dates.year).mean().T
    yr_t = ic[5].groupby(dates.year).apply(lambda g: g.apply(lambda c: nw(c, 5))).T
    per = pd.DataFrame({p: ic[5].loc[a:b].mean() for p, (a, b) in PERIODS.items()})
    per_t = pd.DataFrame(
        {p + " t": ic[5].loc[a:b].apply(lambda c: nw(c, 5)) for p, (a, b) in PERIODS.items()}
    )
    reg = {}
    for s_ in ("calm", "normal", "stressed"):
        g = ic[5].loc[hmm == s_]
        reg[(s_, "IC")] = g.mean()
        reg[(s_, "t NW")] = g.apply(lambda c: nw(c, 5))
        reg[(s_, "days")] = pd.Series(len(g), index=SIGNALS)
    reg = pd.DataFrame(reg)
    log("IC tables", t0)

    # ---- deciles, persistence, breakeven cost
    dec_rows, mono = {}, {}
    for k in SIGNALS:
        D = deciles(sig[k], fwd[5]) * 100
        spread = D[:, -1] - D[:, 0]
        means = np.nanmean(D, 0)
        dec_rows[k] = {
            **{f"D{i + 1}": means[i] for i in range(10)},
            "D10-D1": np.nanmean(spread),
            "D10-D1 t NW": nw(spread, 5),
            "monotonicity (Spearman)": float(
                pd.Series(range(10)).corr(pd.Series(means), method="spearman")
            ),
        }
    dec_t = pd.DataFrame(dec_rows).T
    pers = {}
    for k in SIGNALS:
        row = {}
        top = top_set(sig[k])
        for L in (1, 5, 20):
            row[f"rank autocorr lag {L}"] = np.nanmean(se.rank_ic_panel(sig[k][L:], sig[k][:-L]))
        for L in (5, 20):
            stay = (top[L:] & top[:-L]).sum(1) / np.maximum(top[L:].sum(1), 1)
            row[f"top-decile turnover lag {L} %"] = 100 * (1 - stay[top[L:].sum(1) > 0]).mean()
        pers[k] = row
    pers = pd.DataFrame(pers).T
    be = {}
    for k in SIGNALS:
        for lab, reb, h in (("weekly", 5, 5), ("monthly", 21, 20)):
            d = top_spread(sig[k], fwd[h], iv.decision_days(dates, reb))
            sp, tau = d["spread"].mean(), d["tau"].mean()
            be[(k, lab)] = {
                "top-decile spread % per period": 100 * sp,
                "one-way turnover per period %": 100 * tau,
                "breakeven cost %/side": 100 * sp / (2 * tau),
                "breakeven turnover at 0.168% (x one-way per period)": sp / (2 * FEE),
                "net spread % per period": 100 * d["net"].mean(),
                "net spread t": d["net"].mean() / d["net"].std() * np.sqrt(len(d)),
                "net spread % p.a.": 100 * d["net"].mean() * DAYS / reb,
            }
    be = pd.DataFrame(be).T
    log("deciles / persistence / breakeven", t0)

    # ---- Fama-MacBeth and style exposures
    fm, expo = {}, {}
    for k in SIGNALS:
        ctl = {c: stz[c] for c in STYLES if c != OWN.get(k)}
        r = fm_one(sig[k], fwd[5], ctl, trad, sec)
        fm[k] = {
            "slope raw": r["raw"].mean(),
            "t raw": nw(r["raw"]),
            "slope +styles": r["styles"].mean(),
            "t +styles": nw(r["styles"]),
            "slope +styles+sector": r["styles+sector"].mean(),
            "t +styles+sector": nw(r["styles+sector"]),
            "share of raw slope kept": r["styles"].mean() / r["raw"].mean(),
            "neutral IC": np.nanmean(r["neutral IC"]),
            "neutral IC t": nw(r["neutral IC"]),
            "R2 on styles": r["R2 styles"].mean(),
            "R2 on styles+sector": r["R2 styles+sector"].mean(),
            "controls": ",".join(ctl),
        }
        expo[k] = {c: np.nanmean(se.rank_ic_panel(sig[k], st[c])) for c in STYLES}
        log(f"  FM {k}", t0)
    fm = pd.DataFrame(fm).T
    expo = pd.DataFrame(expo).T.add_prefix("rho ")
    flags = pd.DataFrame(
        {
            "raw t >= 2": fm["t raw"].astype(float) >= 2,
            "|t +styles| < 2": fm["t +styles"].astype(float).abs() < 2,
            "R2 on styles+sector > 0.5": fm["R2 on styles+sector"].astype(float) > 0.5,
        }
    )
    flags["style beta (H1 flag)"] = (flags.iloc[:, 0] & flags.iloc[:, 1]) | flags.iloc[:, 2]
    flags["sign flips with controls"] = np.sign(fm["slope raw"].astype(float)) != np.sign(
        fm["slope +styles"].astype(float)
    )

    for name, t in {
        "diag_ic_summary": summ,
        "diag_ic_decay": decay,
        "diag_ic_by_year": yr,
        "diag_ic_by_year_t": yr_t,
        "diag_ic_by_period": pd.concat([per, per_t], axis=1),
        "diag_ic_by_regime": reg,
        "diag_deciles": dec_t,
        "diag_persistence": pers,
        "diag_breakeven": be,
        "diag_fama_macbeth": fm,
        "diag_exposures": expo,
        "diag_flags": flags,
    }.items():
        t.to_csv(OUT / f"{name}.csv")
    joblib.dump({"ic": ic}, OUT / "diag_ic.joblib", compress=3)
    hs = summ.xs(5, level="h")[["IC mean", "ICIR annualised", "t NW", "IC>0 %"]].join(
        summ.xs(20, level="h")[["IC mean", "t NW"]], rsuffix=" h20"
    )
    txt = [
        "# alpha_portfolio_v1 - stage 1 factor diagnostics (2013-01 .. 2026-10, tradable top-100)",
        "",
        "## IC summary (h = 5; h = 20 alongside)",
        iv.md(hs, 3),
        "",
        "## IC by period (h = 5)",
        iv.md(pd.concat([per, per_t], axis=1), 3),
        "",
        "## IC by year (h = 5)",
        iv.md(yr, 3),
        "",
        "## IC by walk-forward HMM regime (h = 5)",
        iv.md(reg, 3),
        "",
        "## IC decay",
        iv.md(decay, 3),
        "",
        "## Deciles (mean 5d alpha %, per 5 days)",
        iv.md(dec_t, 3),
        "",
        "## Persistence",
        iv.md(pers, 3),
        "",
        "## Breakeven cost (top-decile long-only book)",
        iv.md(be, 3),
        "",
        "## Fama-MacBeth (slope = % 5d alpha per 1 sd; NW t)",
        iv.md(fm, 3),
        "",
        "## Style exposures (mean daily Spearman)",
        iv.md(expo, 3),
        "",
        "## Flags",
        iv.md(flags.astype(str)),
    ]
    (OUT / "diag.md").write_text("\n".join(txt))
    print("\n".join(txt))
    log("stage diag done", t0)


# ============================================================================ stage 2: correlation control
def centred_ranks(panel: dict) -> dict[str, np.ndarray]:
    trad = panel["tradable"]
    return {
        k: np.where(trad, np.nan_to_num(v - 0.5, nan=0.0), np.nan) for k, v in panel["sig"].items()
    }


def past_rows(dates: pd.DatetimeIndex, Y: int, h: int) -> slice:
    """Rows whose h-day forward window ends before year Y (expanding window from the panel start)."""
    return slice(0, max(first_row(dates, Y) - h, 0))


def scheme_weights(ic5: pd.DataFrame, Y: int, dates, scheme: str) -> tuple[pd.Series, dict]:
    """Signal weights of a combination scheme for year Y from past daily ICs only."""
    hist = ic5.iloc[past_rows(dates, Y, 5)].dropna(how="all")
    k = ic5.columns
    eq = pd.Series(1.0 / len(k), index=k)
    if len(hist) < 120 or scheme == "K1":
        return eq, {}
    mu, sd = hist.mean(), hist.std()
    if scheme in ("K2", "K4"):
        w = mu.clip(lower=0).fillna(0.0)
        return (w / w.sum() if w.sum() > 0 else eq), {}
    if scheme == "K3":
        w = (mu / sd).clip(lower=0).fillna(0.0)
        w = w / w.sum() if w.sum() > 0 else eq
        return 0.5 * w + 0.5 * eq, {}
    if scheme == "K5":
        h = hist.fillna(hist.mean())
        cov = pd.DataFrame(LedoitWolf().fit(h.to_numpy()).covariance_, index=k, columns=k)
        return (se.mv_signal_weights(mu, cov) if (mu > 0).any() else eq), {}
    if scheme == "K6":
        lab = se.cluster_signals(hist.corr(), cut=0.5)
        cl_ic = mu.groupby(lab).mean()
        keep = cl_ic[cl_ic > 0].index
        if len(keep) == 0:
            return eq, {"clusters": lab.to_dict()}
        w = pd.Series(0.0, index=k)
        for c in keep:
            mem = lab[lab == c].index
            w[mem] = 1.0 / len(keep) / len(mem)
        return w, {"clusters": lab.to_dict()}
    raise ValueError(scheme)


def lowdin_panel(R: dict[str, np.ndarray], trad: np.ndarray) -> dict[str, np.ndarray]:
    X = np.stack([R[k] for k in SIGNALS], axis=-1)
    O = np.full_like(X, np.nan)
    for t in range(len(X)):
        m = trad[t]
        if m.sum() >= 20:
            O[t, m] = se.lowdin(X[t, m])
    return {k: O[..., j] for j, k in enumerate(SIGNALS)}


def ls_returns(s: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Daily top-tercile minus bottom-tercile EW close-to-close return of the next day (diagnostic)."""
    Cf = pd.DataFrame(C).ffill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        rn = np.r_[Cf[1:] / Cf[:-1] - 1, np.full((1, C.shape[1]), np.nan)]
    rn = np.where(np.isfinite(rn) & (np.abs(rn) < 0.45), rn, np.nan)
    r = pd.DataFrame(s).rank(axis=1, pct=True).to_numpy()
    top, bot = (r > 2 / 3) & ~np.isnan(rn), (r <= 1 / 3) & ~np.isnan(rn)
    return (np.where(top, rn, 0).sum(1) / np.maximum(top.sum(1), 1)) - (
        np.where(bot, rn, 0).sum(1) / np.maximum(bot.sum(1), 1)
    )


def stage_corr() -> None:
    t0 = time.time()
    panel, ex = load()
    dates, trad, fwd = panel["dates"], panel["tradable"], panel["fwd"]
    years = dates.year.to_numpy()
    ic = joblib.load(OUT / "diag_ic.joblib")["ic"]
    ic5 = ic[5]
    R = centred_ranks(panel)

    # ---- correlation structure of the 8 signals (descriptive, full period and pre-OOS 2013-15)
    ic_corr = ic5.corr()
    ic_corr_pre = ic5.loc[:"2015"].corr()
    ls = pd.DataFrame({k: ls_returns(panel["sig"][k], panel["C"]) for k in SIGNALS}, index=dates)
    ls_corr = ls.corr()
    cs = pd.DataFrame(np.eye(len(SIGNALS)), index=SIGNALS, columns=SIGNALS)
    for a, b in itertools.combinations(SIGNALS, 2):
        cs.loc[a, b] = cs.loc[b, a] = np.nanmean(se.rank_ic_panel(panel["sig"][a], panel["sig"][b]))
    enb = pd.DataFrame(
        {
            "IC series": se.effective_bets(ic_corr),
            "long-short returns": se.effective_bets(ls_corr),
            "cross-sectional scores": se.effective_bets(cs),
        }
    ).T
    clus_full = pd.DataFrame(
        {
            "IC corr clusters (cut 0.5)": se.cluster_signals(ic_corr, 0.5),
            "LS corr clusters (cut 0.5)": se.cluster_signals(ls_corr, 0.5),
            "score corr clusters (cut 0.5)": se.cluster_signals(cs, 0.5),
        }
    )
    log("correlation structure", t0)

    # ---- walk-forward scheme weights
    O = lowdin_panel(R, trad)
    ic5_orth = pd.DataFrame({k: se.rank_ic_panel(O[k], fwd[5]) for k in SIGNALS}, index=dates)
    W, meta = {}, {}
    for sch in ("K1", "K2", "K3", "K4", "K5", "K6"):
        rows = {}
        for Y in range(FIRST_YEAR, 2027):
            w, m = scheme_weights(ic5_orth if sch == "K4" else ic5, Y, dates, sch)
            rows[Y] = w
            if m:
                meta[(sch, Y)] = m
        W[sch] = pd.DataFrame(rows).T
    log("scheme weights", t0)

    comp = {"C1": R["ml"], "C2": (R["ml"] + R["v8"]) / 2}
    for sch, w in W.items():
        base = O if sch == "K4" else R
        s = np.zeros_like(R["ml"])
        for k in SIGNALS:
            s += w[k].reindex(years).to_numpy()[:, None] * np.nan_to_num(base[k], nan=0.0)
        comp[sch] = np.where(trad, s, np.nan)

    # ---- composite diagnostics and the walk-forward composite (WFC)
    dd5 = iv.decision_days(dates, 5)
    cic5 = pd.DataFrame({c: se.rank_ic_panel(comp[c], fwd[5]) for c in COMPOSITES}, index=dates)
    sp = {c: top_spread(comp[c], fwd[5], dd5) for c in COMPOSITES}
    choice, sel_rows = {FIRST_YEAR: "K1"}, {}
    for Y in range(FIRST_YEAR + 1, 2027):
        lim = first_row(dates, Y) - 5
        stat = {}
        for c in COMPOSITES:
            n = sp[c]["net"][sp[c].index < lim].dropna()
            stat[c] = n.mean() / n.std() * np.sqrt(len(n)) if len(n) > 20 else np.nan
        st = pd.Series(stat)
        choice[Y] = st.idxmax()
        sel_rows[Y] = {**{f"t {c}": v for c, v in stat.items()}, "pick": choice[Y]}
    wfc = np.full_like(comp["C1"], np.nan)
    for Y, c in choice.items():
        m = years == Y
        wfc[m] = comp[c][m]
    comp["WFC"] = wfc
    cic5["WFC"] = se.rank_ic_panel(wfc, fwd[5])
    sp["WFC"] = top_spread(wfc, fwd[5], dd5)
    cic20 = pd.DataFrame(
        {c: se.rank_ic_panel(comp[c], fwd[20]) for c in [*COMPOSITES, "WFC"]}, index=dates
    )
    ic20_by_year = pd.DataFrame(
        {
            c: {
                Y: (
                    cic20[c].iloc[past_rows(dates, Y, 20)].mean()
                    if first_row(dates, Y) - 20 >= 120
                    else IC_PRIOR
                )
                for Y in range(FIRST_YEAR, 2027)
            }
            for c in cic20.columns
        }
    )
    log("composites + WFC", t0)

    ctab = {}
    for c in [*COMPOSITES, "WFC"]:
        for p, (a, b) in {
            "2016-26": ("2016", "2026"),
            "2022-26": ("2022", "2026"),
            "2013-15": ("2013", "2015"),
        }.items():
            x = cic5[c].loc[a:b]
            d = sp[c][
                (dates[sp[c].index] >= pd.Timestamp(a))
                & (dates[sp[c].index] <= pd.Timestamp(f"{b}-12-31"))
            ]
            ctab[(c, p)] = {
                "IC5": x.mean(),
                "IC5 t NW": nw(x, 5),
                "ICIR ann.": x.mean() / x.std() * np.sqrt(DAYS / 5),
                "IC20": cic20[c].loc[a:b].mean(),
                "top-decile spread %/wk": 100 * d["spread"].mean(),
                "turnover %/wk": 100 * d["tau"].mean(),
                "net spread %/wk": 100 * d["net"].mean(),
                "net spread t": d["net"].mean() / d["net"].std() * np.sqrt(len(d)),
            }
    ctab = pd.DataFrame(ctab).T
    # ENB and diversification ratio of each scheme, year by year on the past IC covariance
    dr = {}
    for Y in range(2016, 2027):
        hist = ic5.iloc[past_rows(dates, Y, 5)].dropna()
        cov = hist.cov()
        row = {"ENB entropy (past IC corr)": se.effective_bets(hist.corr())["ENB entropy"]}
        for sch in W:
            if sch != "K4":
                row[f"DR {sch}"] = se.diversification_ratio(W[sch].loc[Y], cov)
        row["DR C2"] = se.diversification_ratio(pd.Series({"ml": 0.5, "v8": 0.5}), cov)
        dr[Y] = row
    dr = pd.DataFrame(dr).T
    wtab = pd.concat(
        {k: v.loc[[2016, 2019, 2022, 2026]] for k, v in W.items() if k != "K1"}, axis=0
    )
    clus_wf = pd.DataFrame(
        {Y: meta[("K6", Y)]["clusters"] for Y in range(2014, 2027) if ("K6", Y) in meta}
    ).T

    joblib.dump(
        {
            "comp": comp,
            "choice": choice,
            "ic20_by_year": ic20_by_year,
            "weights": W,
            "cic5": cic5,
            "cic20": cic20,
        },
        SCORES,
        compress=3,
    )
    for name, t in {
        "corr_ic": ic_corr,
        "corr_ic_2013_15": ic_corr_pre,
        "corr_ls_returns": ls_corr,
        "corr_cs_scores": cs,
        "corr_enb": enb,
        "corr_clusters_full": clus_full,
        "corr_clusters_wf": clus_wf,
        "corr_composites": ctab,
        "corr_wfc_selection": pd.DataFrame(sel_rows).T,
        "corr_diversification": dr,
        "corr_ic20_by_year": ic20_by_year,
        "corr_weights": wtab,
    }.items():
        t.to_csv(OUT / f"{name}.csv")
    txt = [
        "# alpha_portfolio_v1 - stage 2 correlation control",
        "",
        "## Daily IC (h=5) correlation, 2013-26",
        iv.md(ic_corr, 2),
        "",
        "## Daily long-short (tercile) return correlation",
        iv.md(ls_corr, 2),
        "",
        "## Mean cross-sectional score correlation",
        iv.md(cs, 2),
        "",
        "## Effective number of bets (8 signals)",
        iv.md(enb, 2),
        "",
        "## Clusters (full period, descriptive)",
        iv.md(clus_full),
        "",
        "## Clusters used by K6, walk-forward",
        iv.md(clus_wf),
        "",
        "## Scheme weights (walk-forward; selected years)",
        iv.md(wtab, 2),
        "",
        "## Composites (weights out of sample from 2016)",
        iv.md(ctab, 3),
        "",
        "## WFC selection (t of weekly net top-decile spread, 2013..Y-1)",
        iv.md(pd.DataFrame(sel_rows).T, 2),
        "",
        "## Diversification ratio of the weights on the past IC covariance",
        iv.md(dr, 2),
        "",
        "## IC20 used for the Grinold alpha (past only)",
        iv.md(ic20_by_year, 3),
    ]
    (OUT / "corr.md").write_text("\n".join(txt))
    print("\n".join(txt))
    log("stage corr done", t0)


# ============================================================================ stage 3: construction
@dataclass(frozen=True)
class PC:
    """One portfolio-construction config (all long-only, fully invested in stocks, weekly unless reb says)."""

    score: str  # C2 | WFC (| C3 for the S5-record replication)
    kind: str  # EW | TILT | IV | HYS | MVO
    n: int = 10
    bufm: float = 3.0  # hold buffer = min(bufm x N, 80) ranks (EW / TILT / IV)
    lam: float = 0.0  # MVO risk aversion
    neutral: bool = False  # MVO size / beta bands
    cap: float = CAP  # MVO name cap
    hurdle: float = 1.0  # cost multiplier in HYS / MVO decisions
    costterm: bool = True  # MVO cost term on (A2 ablation: off)
    reb: int = 5
    vt: float = 0.0  # ex-ante vol target (A1 ablation)

    @property
    def key(self) -> str:
        k = f"{self.score}|{self.kind}"
        k += (
            f"|L{self.lam:g}|{'NEU' if self.neutral else 'free'}"
            if self.kind == "MVO"
            else f"|N{self.n}"
        )
        if self.kind in ("EW", "TILT", "IV") and self.bufm != 3.0:
            k += f"|B{self.bufm:g}N"
        if self.kind in ("HYS", "MVO") and self.hurdle != 1.0:
            k += f"|H{self.hurdle:g}"
        if self.kind == "MVO" and self.cap != CAP:
            k += f"|cap{self.cap:g}"
        if not self.costterm:
            k += "|nocost"
        if self.reb != 5:
            k += f"|R{self.reb}"
        if self.vt:
            k += f"|VT{self.vt:g}"
        return k

    @property
    def buffer(self) -> int:
        return min(int(round(self.bufm * self.n)), 80)


def grid() -> list[PC]:
    out = []
    for s in ("C2", "WFC"):
        for kind in ("EW", "TILT", "IV", "HYS"):
            out += [PC(s, kind, n) for n in (10, 20)]
        out += [PC(s, "MVO", lam=lam, neutral=neu) for lam in (3.0, 30.0) for neu in (False, True)]
    return out


class Ctx:
    """Everything a target-weight rule needs, on the panel grid (T dates x K tickers)."""

    def __init__(self):
        panel, ex = load()
        S = joblib.load(SCORES)
        self.panel, self.ex = panel, ex
        self.dates, self.trad = panel["dates"], panel["tradable"]
        years = self.dates.year.to_numpy()
        R = centred_ranks(panel)
        self.scores = {
            "C2": S["comp"]["C2"].astype(float),
            "WFC": S["comp"]["WFC"].astype(float),
            "C3": (R["lowbeta"] + R["lowvol"] + R["resmom"] + R["hi52"]) / 4,
        }
        idio = ex["idio_vol"].copy()
        idio = np.where(
            np.isnan(idio),
            np.nanmedian(np.where(self.trad, idio, np.nan), axis=1, keepdims=True),
            idio,
        )
        self.alpha = {}
        for k in ("C2", "WFC"):
            ic20 = S["ic20_by_year"][k].reindex(years).to_numpy()[:, None]
            self.alpha[k] = ic20 * idio * np.sqrt(H_ALPHA) * zscore_rows(self.scores[k])
        vol = ex["vol_60"]
        self.vol = np.where(
            np.isnan(vol),
            np.nanmedian(np.where(self.trad, vol, np.nan), axis=1, keepdims=True),
            vol,
        )
        self.zsize = zscore_rows(np.where(self.trad, panel["aux"]["logsize"], np.nan))
        self.zbeta = zscore_rows(np.where(self.trad, panel["aux"]["beta_mkt"], np.nan))
        self.dvol = np.where(self.trad, ex["dvol60"], np.nan)
        self.sector = pd.factorize(ex["sector"])[0]
        self.R, self.R_row = ex["R"], ex["R_row"]
        self.fee = FEE

    def cov(self, t: int, names: np.ndarray) -> np.ndarray:
        r = self.R[max(self.R_row[t] - COV_WIN + 1, 0) : self.R_row[t] + 1][:, names]
        return LedoitWolf().fit(np.nan_to_num(r, nan=0.0)).covariance_


_CTX: Ctx | None = None


def get_ctx() -> Ctx:
    global _CTX
    if _CTX is None:
        _CTX = Ctx()
    return _CTX


def book_names(s: np.ndarray, held: np.ndarray, n: int, buffer: int) -> list[int]:
    """integrated_v1.pick's name rule: keep held names ranked inside the buffer, fill with the best others."""
    elig = np.flatnonzero(~np.isnan(s))
    if len(elig) == 0:
        return []
    order = elig[np.argsort(-s[elig], kind="stable")]
    keep = [i for i in order[:buffer] if held[i]][:n]
    ks = set(keep)
    names = keep + [i for i in order if i not in ks][: n - len(keep)]
    return sorted(names, key=lambda i: -s[i])


def targets(ctx: Ctx, c: PC, t: int, w: np.ndarray, fee: float, info: dict) -> np.ndarray:
    """Target weights decided at the close of day t from the current (drifted) weights w."""
    s = ctx.scores[c.score][t]
    held = w > 1e-12
    tgt = np.zeros_like(w)
    if c.kind in ("HYS", "MVO") and np.isnan(s).all():
        info["no_scores_kept_book"] = info.get("no_scores_kept_book", 0) + 1
        return w.copy()  # no scored names that day (2021-12-24..30 embargo gap): no trade
    if c.kind in ("EW", "TILT", "IV"):
        names = book_names(s, held, c.n, c.buffer)
        if names:
            if c.kind == "EW":
                tgt[names] = 1.0 / c.n
            elif c.kind == "TILT":
                lin = (
                    1.5 - np.arange(len(names)) / (len(names) - 1) if len(names) > 1 else np.ones(1)
                )
                tgt[names] = lin / lin.sum() * len(names) / c.n
            else:
                inv = 1.0 / ctx.vol[t, names]
                tgt[names] = inv / inv.sum() * len(names) / c.n
    elif c.kind == "HYS":
        a = ctx.alpha[c.score][t]
        elig = ~np.isnan(s)
        book = sorted(np.flatnonzero(held & elig), key=lambda i: a[i])  # worst first
        while len(book) > c.n:
            book.pop(0)
        cands = sorted(np.flatnonzero(elig & ~held), key=lambda i: -a[i])  # best first
        while len(book) < c.n and cands:
            book.append(cands.pop(0))
        book = sorted(book, key=lambda i: a[i])
        hurdle = 2 * fee * c.hurdle
        while cands and book and a[cands[0]] - a[book[0]] > hurdle:
            book[0] = cands.pop(0)
            book = sorted(book, key=lambda i: a[i])
            info["swaps"] = info.get("swaps", 0) + 1
        tgt[book] = 1.0 / c.n
    elif c.kind == "MVO":
        a = ctx.alpha[c.score][t]
        elig = np.flatnonzero(~np.isnan(s))
        order = elig[np.argsort(-s[elig], kind="stable")]
        cand = set(order[:CAND]) | set(np.flatnonzero(held & ~np.isnan(s)))
        if c.neutral:
            dv = ctx.dvol[t, elig]
            cand |= set(elig[np.argsort(-np.nan_to_num(dv, nan=0.0))[:20]])
        cand = np.array(sorted(cand))
        S = ctx.cov(t, cand) * H_ALPHA
        kw = {}
        if c.neutral:
            E = np.column_stack(
                [np.nan_to_num(ctx.zsize[t, cand]), np.nan_to_num(ctx.zbeta[t, cand])]
            )
            dv = np.nan_to_num(ctx.dvol[t, elig], nan=0.0)
            tg = (
                np.array([np.nansum(dv * ctx.zsize[t, elig]), np.nansum(dv * ctx.zbeta[t, elig])])
                / dv.sum()
            )
            kw = {"exposures": E, "targets": tg, "band": BAND}
        args = dict(
            fee=fee * c.hurdle if c.costterm else 0.0,
            risk_aversion=c.lam,
            max_weight=c.cap,
            groups=ctx.sector[cand],
            group_cap=SECTOR_CAP,
        )
        wn, ok = portfolio.cost_aware_mvo(np.nan_to_num(a[cand]), S, w[cand], **args, **kw)
        info["solves"] = info.get("solves", 0) + 1
        if not ok and kw:
            info["band_dropped"] = info.get("band_dropped", 0) + 1
            wn, ok = portfolio.cost_aware_mvo(np.nan_to_num(a[cand]), S, w[cand], **args)
        if not ok:
            info["failed"] = info.get("failed", 0) + 1
            wn = w[cand].copy()  # no trade on a failed solve
        wn[wn < MIN_W] = 0.0
        if wn.sum() > 0:
            wn = wn / wn.sum()
        tgt[cand] = wn
    if c.vt and tgt.sum() > 0:
        nm = np.flatnonzero(tgt > 0)
        S = ctx.cov(t, nm)
        vol = float(np.sqrt(tgt[nm] @ S @ tgt[nm] * DAYS))
        if vol > c.vt:
            tgt *= c.vt / vol
    return tgt


def schedule_array(T: int, sched) -> np.ndarray:
    cfg_at = np.empty(T, dtype=object)
    for mask, c in sched:
        cfg_at[mask] = c
    return cfg_at


def next_cfg(cfg_at, dates, t: int, c: PC) -> PC:
    nxt = cfg_at[t + 1]
    return nxt if (nxt is not None and nxt != c and dates[t + 1].year != dates[t].year) else c


def sim(ctx: Ctx, sched, fee: float = FEE) -> tuple[pd.DataFrame, dict]:
    """Proportional-cost book (integrated_v1.fast_sim accounting, target weights from `targets`). `fee` is both the
    cost charged (net column) and the cost the cost-aware rules plan with."""
    p = ctx.panel
    O, C = p["O"], p["C"]
    T, K = C.shape
    Cf = pd.DataFrame(C).ffill().to_numpy()
    Of = np.where(np.isnan(O), Cf, O)
    prevC = pd.DataFrame(np.r_[[p["C_prev0"]], Cf[:-1]]).ffill().to_numpy()
    if isinstance(sched, PC):
        sched = [(np.ones(T, bool), sched)]
    cfg_at = schedule_array(T, sched)
    dd, info = {}, {}
    w = np.zeros(K)
    pending = None
    gross, turn, nn = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        c = cfg_at[t]
        if c is None:
            continue
        on = np.where(prevC[t] > 0, Of[t] / prevC[t], 1.0)
        on = np.where(np.isfinite(on), on, 1.0)
        intra = np.where(Of[t] > 0, Cf[t] / Of[t], 1.0)
        intra = np.where(np.isfinite(intra), intra, 1.0)
        v = w * on
        cash = 1.0 - w.sum()
        tv = 0.0
        if pending is not None:
            eq = v.sum() + cash
            cur = v / eq
            tgt = pending.copy()
            stale = np.isnan(O[t])
            tgt[stale] = cur[stale]
            small = (np.abs(tgt - cur) < iv.MIN_TRADE) & (cur > 0) & (tgt > 0)
            tgt[small] = cur[small]
            tv = np.abs(tgt - cur).sum()
            v, cash = tgt * eq, eq - (tgt * eq).sum()
            pending = None
        v1 = v * intra
        eq1 = v1.sum() + cash
        gross[t], turn[t] = eq1 - 1.0, tv
        w = v1 / eq1
        nn[t] = (w > 0).sum()
        if c.reb not in dd:
            dd[c.reb] = iv.decision_days(ctx.dates, c.reb)
        if dd[c.reb][t] and t < T - 1:
            cc = next_cfg(cfg_at, ctx.dates, t, c)
            pending = targets(ctx, cc, t, w, fee, info)
    out = pd.DataFrame({"gross": gross, "turn": turn, "names": nn}, index=ctx.dates)
    first = next(t for t in range(T) if cfg_at[t] is not None)
    out = out.iloc[first:]
    out["net"] = (1 + out["gross"]) * (1 - fee * out["turn"]) - 1
    return out, info


def run_cfg(c: PC, fee: float = FEE) -> tuple[str, pd.DataFrame, dict]:
    t0 = time.time()
    r, info = sim(get_ctx(), c, fee)
    info["seconds"] = time.time() - t0
    return c.key, r, info


def run_many(cfgs: list[PC], fee: float = FEE, n_jobs: int = 8) -> dict:
    from joblib import Parallel, delayed

    out = Parallel(n_jobs=n_jobs, verbose=0)(delayed(run_cfg)(c, fee) for c in cfgs)
    return {k: (r, info) for k, r, info in out}


def wf(res: dict, cfgs: list[PC], objective: str = "alpha_t", fee: float = FEE):
    """integrated_v1's walk-forward selection (max net CAPM alpha t on 2013..Y-1, turnover <= 40x/yr)."""
    panel = get_ctx().panel
    r = {k: (v[0] if isinstance(v, tuple) else v)[["gross", "turn"]] for k, v in res.items()}
    return iv.walk_forward(panel, None, r, cfgs, objective=objective, fee=fee)


def s5_record_schedule(ctx: Ctx) -> list:
    """The deployed S5 record (integrated_v1 walk-forward picks) as PC configs in this harness."""
    picks = pd.read_csv(iv.OUT / "wf_per_year.csv", index_col=0)["pick"]
    sched = []
    for Y, key in picks.items():
        comp, n, reb, b, wt = key.split("|")
        c = PC(
            comp,
            "TILT" if wt == "T" else "EW",
            int(n[1:]),
            3.0 if b == "B" else 1.0,
            reb=int(reb[1:]),
        )
        sched.append((np.asarray(ctx.dates.year == int(Y)), c))
    return sched


def stage_build() -> None:
    t0 = time.time()
    ctx = get_ctx()
    log("context ready", t0)
    # ---- harness check 1: the S5 rule (C2 EW N10, buffer 30, weekly) reproduces integrated_v1's fast simulator
    G = joblib.load(iv.GRID)["res"]
    r, _ = sim(ctx, PC("C2", "EW", 10))
    ref = G["C2|N10|R5|B|E"]
    d1 = (
        float(np.abs(r["gross"] - ref["gross"]).max()),
        float(np.abs(r["turn"] - ref["turn"]).max()),
    )
    # ---- harness check 2: the S5 record (integrated_v1 walk-forward schedule) reproduces wf_daily.parquet
    rec, _ = sim(ctx, s5_record_schedule(ctx))
    wfd = pd.read_parquet(iv.OUT / "wf_daily.parquet")
    d2 = float(np.abs(rec["net"] - wfd["wf_fast_net"].reindex(rec.index)).max())
    log(
        f"harness checks: S5 rule max|d gross|={d1[0]:.1e} max|d turn|={d1[1]:.1e}; S5 record max|d net|={d2:.1e}",
        t0,
    )

    cfgs = grid()
    res = run_many(cfgs)
    log(f"grid: {len(res)} configs", t0)
    for k, (_, info) in res.items():
        if "solves" in info:
            log(
                f"  {k}: solves {info['solves']}, bands dropped {info.get('band_dropped', 0)}, failed "
                f"{info.get('failed', 0)}, {info['seconds']:.0f}s"
            )
    picks, tabs, sched = wf(res, cfgs)
    ap, ap_info = sim(ctx, sched)
    log("WF: " + "; ".join(f"{Y}:{picks[Y].key}" for Y in WF_YEARS), t0)

    # ---- pre-declared ablations
    abl = {}
    abl["A1 WF + 15% vol target"] = sim(ctx, [(m, replace(c, vt=VT)) for m, c in sched])[0]
    a2 = run_many([PC("WFC", "MVO", lam=lam, costterm=False) for lam in (3.0, 30.0)], n_jobs=2)
    for k, (r_, _) in a2.items():
        abl[f"A2 {k}"] = r_
    p_ir, _, s_ir = wf(res, cfgs, objective="ir")
    abl["A3 WF, IR objective"] = sim(ctx, s_ir)[0]
    c2only = [c for c in cfgs if c.score == "C2"]
    p_c2, _, s_c2 = wf(res, c2only)
    abl["A4 WF, score = C2 only"] = sim(ctx, s_c2)[0]
    alt = pd.DataFrame(
        {
            "AP (primary)": {Y: picks[Y].key for Y in WF_YEARS},
            "A3 IR objective": {Y: p_ir[Y].key for Y in WF_YEARS},
            "A4 C2 only": {Y: p_c2[Y].key for Y in WF_YEARS},
        }
    )
    sel = pd.concat(
        {
            Y: tabs[Y].assign(rank=tabs[Y]["score"].astype(float).rank(ascending=False))
            for Y in WF_YEARS
        }
    )
    joblib.dump(
        {
            "res": {k: v[0].astype("float32") for k, v in res.items()},
            "info": {k: v[1] for k, v in res.items()},
            "picks": picks,
            "ap": ap,
            "ap_info": ap_info,
            "abl": abl,
            "alt": alt,
            "s5_record_fast": rec,
            "checks": {"S5 rule": d1, "S5 record": d2},
        },
        GRID,
        compress=3,
    )
    sel.to_csv(OUT / "build_selection_tables.csv")
    alt.to_csv(OUT / "build_wf_picks.csv")
    print(alt.to_string())
    log("stage build done", t0)


# ============================================================================ stage 4: robustness
def exact(
    ctx: Ctx, sched, fee: float = FEE, capital: float = config.INITIAL_CAPITAL, lot: int = 100
):
    """integrated_v1.exact_sim (THB 1M, 100-share lots, sells first, next-open fills) with targets from `targets`.
    Returns (daily frame: equity, traded, ret), trade blotter, end-of-day weights (dates x tickers)."""
    p, dates = ctx.panel, ctx.dates
    O, C = p["O"], p["C"]
    T, K = C.shape
    Cf = pd.DataFrame(C).ffill().to_numpy()
    if isinstance(sched, PC):
        sched = [(np.ones(T, bool), sched)]
    cfg_at = schedule_array(T, sched)
    dd, info = {}, {}
    cash, sh = capital, np.zeros(K)
    pending = None
    eq, traded = np.full(T, np.nan), np.zeros(T)
    W = np.zeros((T, K), dtype="float32")
    trades = []
    for t in range(T):
        c = cfg_at[t]
        if c is None:
            continue
        if pending is not None:
            tgt = pending
            po = O[t]
            mark = np.where(np.isfinite(po), po, Cf[t])
            value = cash + np.nansum(sh * mark)
            budget = value * 0.998
            want = np.where(
                np.isfinite(po) & (po > 0),
                np.floor(budget * tgt / np.where(po > 0, po, 1) / lot) * lot,
                sh,
            )
            dq = want - sh
            notional = np.abs(dq) * np.nan_to_num(po)
            small = (notional < iv.MIN_TRADE * value) & (want > 0) & (sh > 0)
            dq[small] = 0
            dq[~np.isfinite(po)] = 0
            for side in (-1, 1):
                idx = np.where(np.sign(dq) == side)[0]
                if side == -1:
                    cash += (-dq[idx] * po[idx]).sum() - (np.abs(dq[idx]) * po[idx]).sum() * fee
                    traded[t] += (np.abs(dq[idx]) * po[idx]).sum()
                else:
                    cost = dq[idx] * po[idx] * (1 + fee)
                    if cost.sum() > cash and cost.sum() > 0:
                        f = max(cash, 0) / cost.sum()
                        dq[idx] = np.floor(dq[idx] * f / lot) * lot
                        cost = dq[idx] * po[idx] * (1 + fee)
                    cash -= cost.sum()
                    traded[t] += (dq[idx] * po[idx]).sum()
                sh[idx] += dq[idx]
                trades += [
                    (dates[t], ctx.panel["tickers"][i], dq[i], po[i]) for i in idx if dq[i] != 0
                ]
            pending = None
        value = cash + np.nansum(sh * Cf[t])
        eq[t] = value
        W[t] = sh * Cf[t] / value
        if c.reb not in dd:
            dd[c.reb] = iv.decision_days(dates, c.reb)
        if dd[c.reb][t] and t < T - 1:
            cc = next_cfg(cfg_at, dates, t, c)
            pending = targets(ctx, cc, t, sh * Cf[t] / value, fee, info)
    out = pd.DataFrame({"equity": eq, "traded": traded}, index=dates).dropna(subset=["equity"])
    out["ret"] = out["equity"].pct_change()
    out.iloc[0, out.columns.get_loc("ret")] = out["equity"].iloc[0] / capital - 1
    out["turn"] = out["traded"] / out["equity"].shift(1).fillna(capital)
    tr = pd.DataFrame(trades, columns=["Date", "Ticker", "qty", "price"])
    Wd = pd.DataFrame(W, index=dates, columns=ctx.panel["tickers"]).loc[out.index]
    return out, tr, Wd


def neighbours(c: PC) -> list[PC]:
    """Pre-declared one-at-a-time perturbations (only those that apply to the design)."""
    out = []
    if c.kind != "MVO":
        out += [replace(c, n=c.n - 5), replace(c, n=c.n + 5)]
    if c.kind in ("EW", "TILT", "IV"):
        out += [replace(c, bufm=2.0), replace(c, bufm=4.0)]
    if c.kind == "MVO":
        out += [
            replace(c, lam=c.lam / 2),
            replace(c, lam=c.lam * 2),
            replace(c, cap=0.075),
            replace(c, cap=0.15),
        ]
    if c.kind in ("HYS", "MVO"):
        out += [replace(c, hurdle=0.5), replace(c, hurdle=2.0)]
    return out + [replace(c, reb=10)]


def prior_trial_srs() -> list[float]:
    """Daily Sharpe ratios of the 679 trials before this experiment (integrated_v1's list + retrain_v1 at 0)."""
    s = iv.v3.prior_trial_srs()
    s = s + [0.0] * max(0, 89 - len(s))
    G = joblib.load(iv.GRID)["res"]
    for c in iv.grid(0.0) + iv.grid(0.08):
        r = iv.net_at(G[c.key], FEE).loc[OOS]
        s.append(float(r.mean() / r.std()))
    s += [0.0] * (6 + validity.TRIALS["retrain_v1"])
    assert len(s) == TRIALS_BEFORE, (len(s), TRIALS_BEFORE)
    return s


def sr_daily(r: pd.Series) -> float:
    r = r.loc[OOS].dropna()
    return float(r.mean() / r.std())


def period_stats(r: pd.Series) -> dict:
    r = r.dropna()
    return {
        "CAGR %": iv.cagr(r),
        "Sharpe": r.mean() / r.std() * np.sqrt(DAYS),
        "max DD %": iv.maxdd(r),
    }


def stage_robust() -> None:
    t0 = time.time()
    ctx = get_ctx()
    panel, dates = ctx.panel, ctx.dates
    B = joblib.load(GRID)
    res, picks, ap_fast, abl = B["res"], B["picks"], B["ap"], B["abl"]
    cfgs = grid()
    bench_r = panel["bench"]["Close"].reindex(dates).ffill().pct_change()
    sched = [(np.asarray(dates.year == Y), picks[Y]) for Y in WF_YEARS]
    s5_sched = s5_record_schedule(ctx)

    # ---- neighbourhood of the 2026 pick (fixed designs)
    nb_cfgs = neighbours(picks[2026])
    nb = run_many([picks[2026], *nb_cfgs], n_jobs=6)
    log(f"neighbours of {picks[2026].key}: {[c.key for c in nb_cfgs]}", t0)

    # ---- exact lots: AP and the S5 record (net and gross)
    ex_ap, tr_ap, w_ap = exact(ctx, sched)
    exg_ap, _, _ = exact(ctx, sched, fee=0.0)
    ex_s5, tr_s5, w_s5 = exact(ctx, s5_sched)
    exg_s5, _, _ = exact(ctx, s5_sched, fee=0.0)
    wfd = pd.read_parquet(iv.OUT / "wf_daily.parquet")
    chk = float(np.abs(ex_s5["ret"].loc[OOS] - wfd["wf_exact_net"].loc[OOS]).max())
    log(
        f"exact sims done; S5 record exact replication max|d ret| = {chk:.1e}; fast/exact corr AP "
        f"{ap_fast['net'].loc[OOS].corr(ex_ap['ret'].loc[OOS]):.4f}",
        t0,
    )

    # ---- trial ledger, DSR inputs
    n_sig = 7  # K1..K6 + WFC (signal level, SR counted as 0)
    mine = {c.key: sr_daily(res[c.key]["net"]) for c in cfgs}
    mine["AP walk-forward"] = sr_daily(ap_fast["net"])
    mine.update({k: sr_daily(v["net"]) for k, v in abl.items()})
    mine.update(
        {f"neighbour {k}": sr_daily(v[0]["net"]) for k, v in nb.items() if k != picks[2026].key}
    )
    if (OUT / "check_c3.joblib").exists():  # deviation D3: 12 C3 configs + their WF (counted)
        C3 = joblib.load(OUT / "check_c3.joblib")
        mine.update({f"D3 {k}": sr_daily(v["net"]) for k, v in C3["res"].items()})
        mine["D3 WF incl. C3"] = sr_daily(C3["wf"]["net"])
    n_mine = len(mine) + n_sig
    n_total = TRIALS_BEFORE + n_mine
    trial_srs = prior_trial_srs() + list(mine.values()) + [0.0] * n_sig
    log(
        f"trials: {n_mine} in this experiment, {n_total} in total (validity.N_TRIALS now {validity.N_TRIALS})",
        t0,
    )

    # ---- headline table
    rows = {}
    rows["AP walk-forward, exact lots"] = iv.full_stats(
        ex_ap["ret"].loc[OOS], exg_ap["ret"].loc[OOS], ex_ap["turn"].loc[OOS], panel, trial_srs
    )
    rows["AP walk-forward, fast"] = iv.full_stats(
        ap_fast["net"].loc[OOS],
        ap_fast["gross"].loc[OOS],
        ap_fast["turn"].loc[OOS],
        panel,
        trial_srs,
    )
    rows["S5 deployed record, exact lots"] = iv.full_stats(
        ex_s5["ret"].loc[OOS], exg_s5["ret"].loc[OOS], ex_s5["turn"].loc[OOS], panel, trial_srs
    )
    s5r = res["C2|EW|N10"]
    rows["S5 rule fixed (C2 EW N10), fast"] = iv.full_stats(
        s5r["net"].loc[OOS], s5r["gross"].loc[OOS], s5r["turn"].loc[OOS], panel, trial_srs
    )
    for k, v in abl.items():
        rows[k] = iv.full_stats(
            v["net"].loc[OOS], v["gross"].loc[OOS], v["turn"].loc[OOS], panel, trial_srs
        )
    rows["TDEX buy & hold"] = iv.full_stats(bench_r.loc[OOS].dropna(), None, None, panel)
    for k in (
        "CAPM alpha %",
        "CAPM alpha t",
        "TH-Carhart alpha %",
        "TH-Carhart alpha t",
        "IR vs TDEX",
        "TH-Carhart (net factors) alpha %",
        "TH-Carhart (net factors) alpha t",
        "TH b_SMB",
        "TH b_WML",
    ):
        rows["TDEX buy & hold"][k] = np.nan
    main = pd.DataFrame(rows).T
    ap_r, s5_r = ex_ap["ret"].loc[OOS], ex_s5["ret"].loc[OOS]
    log("headline stats", t0)

    # ---- per year, sub-periods, regimes
    yr = []
    for Y in WF_YEARS:
        a, s_, b = ex_ap["ret"].loc[str(Y)], ex_s5["ret"].loc[str(Y)], bench_r.loc[str(Y)].dropna()
        yr.append(
            {
                "year": Y,
                "AP pick": picks[Y].key,
                "S5 pick": dict(
                    (int(y), k)
                    for y, k in pd.read_csv(iv.OUT / "wf_per_year.csv", index_col=0)["pick"].items()
                )[Y],
                "AP net %": 100 * ((1 + a).prod() - 1),
                "S5 net %": 100 * ((1 + s_).prod() - 1),
                "TDEX %": 100 * ((1 + b).prod() - 1),
                "AP - S5 pp": 100 * ((1 + a).prod() - (1 + s_).prod()),
                "AP turnover x": float(ex_ap["turn"].loc[str(Y)].sum()),
                "AP cost drag pp": 100 * ((1 + exg_ap["ret"].loc[str(Y)]).prod() - (1 + a).prod()),
                "S5 turnover x": float(ex_s5["turn"].loc[str(Y)].sum()),
                "S5 cost drag pp": 100 * ((1 + exg_s5["ret"].loc[str(Y)]).prod() - (1 + s_).prod()),
            }
        )
    yrt = pd.DataFrame(yr).set_index("year")
    sub = {}
    for per, (a, b) in {
        "2016-19": ("2016", "2019"),
        "2020-21": ("2020", "2021"),
        "2016-21": ("2016", "2021"),
        "2022-26": ("2022", "2026"),
    }.items():
        for name, r in (("AP", ex_ap["ret"]), ("S5", ex_s5["ret"]), ("TDEX", bench_r)):
            sub[(per, name)] = period_stats(r.loc[a:b])
        sub[(per, "AP - S5")] = {
            "CAGR %": sub[(per, "AP")]["CAGR %"] - sub[(per, "S5")]["CAGR %"],
            "Sharpe": sub[(per, "AP")]["Sharpe"] - sub[(per, "S5")]["Sharpe"],
            "max DD %": np.nan,
        }
    subt = pd.DataFrame(sub).T
    hmm = ctx.ex["hmm"].reindex(ap_r.index)
    reg = {}
    for st_ in ("calm", "normal", "stressed"):
        m = (hmm == st_).to_numpy()
        d = (ap_r - s5_r)[m]
        reg[st_] = {
            "days": int(m.sum()),
            "AP % p.a.": 100 * DAYS * ap_r[m].mean(),
            "S5 % p.a.": 100 * DAYS * s5_r[m].mean(),
            "TDEX % p.a.": 100 * DAYS * bench_r.reindex(ap_r.index)[m].mean(),
            "AP - S5 % p.a.": 100 * DAYS * d.mean(),
            "AP - S5 t (NW)": nw(d, 5),
        }
    regt = pd.DataFrame(reg).T

    # ---- cost sensitivity (same schedule; (a) same trades re-priced, (b) re-run with the cost in the decisions)
    cs = {}
    for f in (0.0008, 0.0010, 0.00125, 0.0015, FEE, 0.0020, 0.0025):
        a_same = iv.net_at(ap_fast, f).loc[OOS]
        a_rerun = sim(ctx, sched, fee=f)[0]["net"].loc[OOS]
        s_same = iv.net_at(B["s5_record_fast"], f).loc[OOS]
        b = bench_r.reindex(a_same.index)
        cs[f"{100 * f:.3f}%"] = {
            "AP CAGR % (same trades)": iv.cagr(a_same),
            "AP CAGR % (re-run)": iv.cagr(a_rerun),
            "S5 record CAGR %": iv.cagr(s_same),
            "TDEX CAGR %": iv.cagr(b),
            "AP - S5 pp": iv.cagr(a_same) - iv.cagr(s_same),
            "AP CAPM alpha t": iv.capm_t(a_same, b)[1],
            "AP IR vs TDEX": iv.ir(a_same, b),
            "AP TH-Carhart t": factors.thai_local(a_same, panel["local"])["alpha t"],
        }
    cst = pd.DataFrame(cs).T
    log("periods / regimes / costs", t0)

    # ---- bootstrap, PBO, DSR / HLZ
    boot = {
        "AP vs S5 record (block 21)": validity.sharpe_diff_test(ap_r, s5_r, 21, 5000, 0),
        "AP vs TDEX (block 21)": validity.sharpe_diff_test(
            ap_r, bench_r.reindex(ap_r.index), 21, 5000, 0
        ),
        "S5 record vs TDEX (block 21)": validity.sharpe_diff_test(
            s5_r, bench_r.reindex(s5_r.index), 21, 5000, 0
        ),
        "AP vs S5 record (block 5)": validity.sharpe_diff_test(ap_r, s5_r, 5, 5000, 0),
        "AP vs S5 record (block 63)": validity.sharpe_diff_test(ap_r, s5_r, 63, 5000, 0),
    }
    boott = pd.DataFrame(boot).T
    Rg = pd.DataFrame({c.key: res[c.key]["net"].astype(float) for c in cfgs}).dropna()
    pbo = {
        "Sharpe, 2013-26": iv.cscv(Rg, bench_r, metric="sharpe"),
        "CAPM alpha t, 2013-26": iv.cscv(Rg, bench_r, metric="alpha_t"),
        "Sharpe, 2016-26": iv.cscv(Rg.loc[OOS], bench_r, metric="sharpe"),
    }
    pbot = pd.DataFrame(pbo).T
    years = len(ap_r) / DAYS
    hl = {
        "AP Sharpe t": validity.haircut(validity.sharpe_tstat(ap_r), years, n_trials=n_total),
        "AP CAPM alpha t": validity.haircut(
            main.loc["AP walk-forward, exact lots", "CAPM alpha t"], years, n_trials=n_total
        ),
        "S5 Sharpe t": validity.haircut(validity.sharpe_tstat(s5_r), years, n_trials=n_total),
    }
    hlt = pd.DataFrame(hl).T
    log("bootstrap / PBO / HLZ", t0)

    # ---- neighbourhood surface, grid surface
    s5_fast_cagr = iv.cagr(B["s5_record_fast"]["net"].loc[OOS])
    nbt = {}
    for k, (r, _) in nb.items():
        n_ = r["net"].loc[OOS]
        b = bench_r.reindex(n_.index)
        nbt[k] = {
            "role": "pick (2026)" if k == picks[2026].key else "neighbour",
            "net CAGR 16-26 %": iv.cagr(n_),
            "Sharpe": n_.mean() / n_.std() * np.sqrt(DAYS),
            "CAPM alpha t": iv.capm_t(n_, b)[1],
            "turnover 2-way": float(r["turn"].loc[OOS].sum() / (len(n_) / DAYS)),
            "beats S5 record (fast)": iv.cagr(n_) > s5_fast_cagr,
        }
    nbt = pd.DataFrame(nbt).T
    gs = {}
    for c in cfgs:
        r = res[c.key]
        n_ = r["net"].loc[OOS].astype(float)
        b = bench_r.reindex(n_.index)
        gs[c.key] = {
            "score": c.score,
            "kind": c.kind,
            "net CAGR 16-26 %": iv.cagr(n_),
            "gross CAGR 16-26 %": iv.cagr(r["gross"].loc[OOS].astype(float)),
            "Sharpe": n_.mean() / n_.std() * np.sqrt(DAYS),
            "max DD %": iv.maxdd(n_),
            "CAPM alpha t": iv.capm_t(n_, b)[1],
            "IR vs TDEX": iv.ir(n_, b),
            "turnover 2-way": float(r["turn"].loc[OOS].sum() / (len(n_) / DAYS)),
            "avg names": float(r["names"].loc[OOS].mean()),
            "times picked": sum(picks[Y].key == c.key for Y in WF_YEARS),
        }
    gst = pd.DataFrame(gs).T
    by_kind = gst.groupby(["kind"])[
        ["net CAGR 16-26 %", "gross CAGR 16-26 %", "Sharpe", "CAPM alpha t", "turnover 2-way"]
    ].median()
    by_score = gst.groupby(["score"])[
        ["net CAGR 16-26 %", "gross CAGR 16-26 %", "Sharpe", "CAPM alpha t", "turnover 2-way"]
    ].median()

    # ---- capacity and decomposition
    adv = pd.DataFrame(ctx.ex["dvol60"], index=dates, columns=panel["tickers"])
    vol = pd.DataFrame(ctx.ex["vol_60"], index=dates, columns=panel["tickers"])
    cap, caps = {}, {}
    for name, ex_, tr_, w_ in (("AP", ex_ap, tr_ap, w_ap), ("S5", ex_s5, tr_s5, w_s5)):
        e = ex_.loc[OOS]
        tl = risk.trade_liquidity(tr_[tr_["Date"] >= OOS.start], adv, vol)
        curve = risk.capacity_curve(e["ret"], e["equity"], tl, w_.loc[OOS], adv)
        cap[name] = curve
        caps[name] = {
            "AUM THB m at 1% p.a. impact drag": risk.capacity_at(curve, "impact drag % p.a.", 1.0),
            "AUM THB m at 2% p.a. impact drag": risk.capacity_at(curve, "impact drag % p.a.", 2.0),
            "median trade % ADV at THB 100m": float(curve.loc[100.0, "median trade % ADV"]),
            "net CAGR % at THB 100m": float(curve.loc[100.0, "net CAGR %"]),
            "net CAGR % at THB 1bn": float(curve.loc[1000.0, "net CAGR %"]),
        }
    capt = pd.concat(cap, axis=0)
    capst = pd.DataFrame(caps).T
    fac = panel["local"][["MKT", "SMB", "WML"]]
    dec = pd.DataFrame(
        {
            "AP": risk.return_decomposition(ap_r, exg_ap["ret"].loc[OOS], fac),
            "S5": risk.return_decomposition(s5_r, exg_s5["ret"].loc[OOS], fac),
        }
    )
    log("capacity / decomposition", t0)

    # ---- gates (pre-registered)
    A, S = main.loc["AP walk-forward, exact lots"], main.loc["S5 deployed record, exact lots"]
    g = {}
    g["G1 beats S5 (net CAGR and Sharpe)"] = (
        A["CAGR %"] > S["CAGR %"] and A["Sharpe"] > S["Sharpe"],
        f"CAGR {A['CAGR %']:.2f} vs {S['CAGR %']:.2f}; Sharpe {A['Sharpe']:.2f} vs {S['Sharpe']:.2f}",
    )
    g["G2 beats TDEX (CAGR and IR > 0)"] = (
        A["CAGR %"] > A["TDEX CAGR %"] and A["IR vs TDEX"] > 0,
        f"CAGR {A['CAGR %']:.2f} vs {A['TDEX CAGR %']:.2f}; IR {A['IR vs TDEX']:.2f}",
    )
    d1, d2 = subt.loc[("2016-21", "AP - S5"), "CAGR %"], subt.loc[("2022-26", "AP - S5"), "CAGR %"]
    g["G3 time-stable (AP - S5 > 0 in 2016-21 and 2022-26)"] = (
        d1 > 0 and d2 > 0,
        f"{d1:+.2f} / {d2:+.2f} pp",
    )
    nbn = nbt[nbt["role"] == "neighbour"]
    share = float(nbn["beats S5 record (fast)"].astype(bool).mean())
    g["G4 neighbour-stable (>= 2/3 beat S5)"] = (
        share >= 2 / 3,
        f"{int(nbn['beats S5 record (fast)'].astype(bool).sum())} of {len(nbn)}",
    )
    pb = float(pbot.loc["Sharpe, 2013-26", "PBO"])
    g["G5 PBO < 0.5 (Sharpe, 2013-26)"] = (pb < 0.5, f"{pb:.3f}")
    pv = boot["AP vs S5 record (block 21)"]["p one-sided (H0 diff <= 0)"]
    g["G6 Sharpe(AP) > Sharpe(S5) significant (p < 0.05)"] = (
        pv < 0.05,
        f"p = {pv:.3f}; diff {boot['AP vs S5 record (block 21)']['diff']:.2f}",
    )
    g["G7 DSR > 0.95"] = (A["DSR"] > 0.95, f"{A['DSR']:.2f} at {n_total} trials")
    gates = pd.DataFrame(
        {"pass": {k: bool(v[0]) for k, v in g.items()}, "value": {k: v[1] for k, v in g.items()}}
    )
    improves = all(gates["pass"].iloc[:5])
    capital = all(gates["pass"])
    verdict = ("IMPROVES (G1-G5 pass)" if improves else "DOES NOT IMPROVE (a G1-G5 gate fails)") + (
        "; significant / capital-ready" if capital else "; not significant / not capital-ready"
    )
    log(f"VERDICT: {verdict}", t0)

    # ---- save
    for name, t in {
        "robust_main": main,
        "robust_per_year": yrt,
        "robust_subperiods": subt,
        "robust_regimes": regt,
        "robust_cost_sensitivity": cst,
        "robust_bootstrap": boott,
        "robust_pbo": pbot,
        "robust_hlz": hlt,
        "robust_neighbours": nbt,
        "robust_grid_2016_2026": gst,
        "robust_capacity_curve": capt,
        "robust_capacity": capst,
        "robust_decomposition": dec,
        "robust_gates": gates,
    }.items():
        t.to_csv(OUT / f"{name}.csv")
    pd.DataFrame(
        {
            "ap_exact_net": ex_ap["ret"],
            "ap_exact_gross": exg_ap["ret"],
            "ap_turn": ex_ap["turn"],
            "s5_exact_net": ex_s5["ret"],
            "s5_exact_gross": exg_s5["ret"],
            "ap_fast_net": ap_fast["net"],
            "tdex": bench_r,
        }
    ).to_parquet(OUT / "robust_daily.parquet")
    summ = {
        "verdict": verdict,
        "n_trials_this_experiment": n_mine,
        "n_trials_total": n_total,
        "trial_list": list(mine) + ["K1", "K2", "K3", "K4", "K5", "K6", "WFC"],
        "s5_exact_replication_max_abs_diff": chk,
        "harness_checks": B["checks"],
        "fast_exact_corr_ap": float(ap_fast["net"].loc[OOS].corr(ex_ap["ret"].loc[OOS])),
        "picks": {Y: picks[Y].key for Y in WF_YEARS},
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    KEY = [
        "CAGR %",
        "TDEX CAGR %",
        "gross CAGR %",
        "cost drag %",
        "turnover 2-way x/yr",
        "Sharpe",
        "max DD %",
        "IR vs TDEX",
        "TE %",
        "CAPM alpha %",
        "CAPM alpha t",
        "gross CAPM alpha t",
        "CAPM beta",
        "FF3-EM alpha %",
        "FF3-EM alpha t",
        "Carhart-EM alpha %",
        "Carhart-EM alpha t",
        "TH-Carhart alpha %",
        "TH-Carhart alpha t",
        "gross TH-Carhart alpha t",
        "TH-Carhart (net factors) alpha %",
        "TH-Carhart (net factors) alpha t",
        "TH b_SMB",
        "TH b_WML",
        "vs EW top-100 % p.a.",
        "PSR",
        "DSR",
        "MinTRL yrs",
        "years",
    ]
    txt = [
        "# alpha_portfolio_v1 - stage 4 robustness (auto-generated)",
        "",
        f"VERDICT: {verdict}",
        "",
        f"trials: {n_mine} here, {n_total} in total",
        "",
        "## Gates (pre-registered)",
        iv.md(gates),
        "",
        "## Walk-forward OOS 2016-01 -> 2026-10",
        iv.md(main[[k for k in KEY if k in main.columns]].T.astype(float)),
        "",
        "## Per year",
        iv.md(yrt),
        "",
        "## Sub-periods",
        iv.md(subt),
        "",
        "## HMM regimes (exact, net)",
        iv.md(regt),
        "",
        "## Cost sensitivity",
        iv.md(cst),
        "",
        "## Stationary bootstrap of the Sharpe difference",
        iv.md(boott, 3),
        "",
        "## PBO (CSCV, 16 blocks, 24 configs)",
        iv.md(pbot.astype(float)),
        "",
        "## HLZ haircut",
        iv.md(hlt, 3),
        "",
        f"## Neighbourhood of the 2026 pick (S5 record fast CAGR {s5_fast_cagr:.2f}%)",
        iv.md(nbt),
        "",
        "## Grid 2016-26 (diagnostic; selection used 2013..Y-1 only)",
        iv.md(gst),
        "",
        "### medians by construction",
        iv.md(by_kind),
        "",
        "### medians by score",
        iv.md(by_score),
        "",
        "## Capacity",
        iv.md(capst),
        "",
        iv.md(capt),
        "",
        "## Return decomposition (% p.a., local factors)",
        iv.md(dec),
        "",
        f"summary: {json.dumps(summ, default=str)}",
    ]
    (OUT / "tables.md").write_text("\n".join(txt))
    print("\n".join(txt))
    log("stage robust done", t0)


# ============================================================================ post-hoc check (deviation D3; counted, not a gate)
def stage_c3check() -> None:
    """Was AP's 2016-20 edge only the exclusion of C3 (the literature book S5's record held in 2016-20)?
    Re-run the same walk-forward over the 24 configs + the 12 constructions on C3 (13 extra trials)."""
    t0 = time.time()
    ctx = get_ctx()
    ctx.alpha["C3"] = ctx.alpha["C2"] * 0.0  # placeholder, replaced below with C3's own past IC20
    S = joblib.load(SCORES)
    years = ctx.dates.year.to_numpy()
    fwd20 = ctx.panel["fwd"][20]
    ic20 = pd.Series(se.rank_ic_panel(ctx.scores["C3"], fwd20), index=ctx.dates)
    by_y = pd.Series(
        {
            Y: (
                ic20.iloc[past_rows(ctx.dates, Y, 20)].mean()
                if first_row(ctx.dates, Y) - 20 >= 120
                else IC_PRIOR
            )
            for Y in range(FIRST_YEAR, 2027)
        }
    )
    idio = ctx.ex["idio_vol"]
    idio = np.where(
        np.isnan(idio), np.nanmedian(np.where(ctx.trad, idio, np.nan), axis=1, keepdims=True), idio
    )
    ctx.alpha["C3"] = (
        by_y.reindex(years).to_numpy()[:, None]
        * idio
        * np.sqrt(H_ALPHA)
        * zscore_rows(ctx.scores["C3"])
    )
    _ = S
    c3 = [replace(c, score="C3") for c in grid() if c.score == "C2"]
    out = {}
    for c in c3:  # serial: the C3 alpha lives in this process's context only
        r, info = sim(ctx, c)
        out[c.key] = (r, info)
    log(f"C3 configs simulated: {len(out)}", t0)
    B = joblib.load(GRID)
    res = {k: (v, {}) for k, v in B["res"].items()} | out
    cfgs = grid() + c3
    picks, _, sched = wf(res, cfgs)
    r, _ = sim(ctx, sched)
    bench_r = ctx.panel["bench"]["Close"].reindex(ctx.dates).ffill().pct_change()
    rows = {}
    for name, x in (
        ("AP (24 configs)", B["ap"]["net"]),
        ("AP + C3 constructions (36 configs)", r["net"]),
        ("S5 deployed record (fast)", B["s5_record_fast"]["net"]),
    ):
        for per, sl in (
            ("2016-26", OOS),
            ("2016-21", slice("2016", "2021")),
            ("2022-26", slice("2022", None)),
        ):
            n_ = x.loc[sl]
            rows[(name, per)] = {
                "net CAGR %": iv.cagr(n_),
                "Sharpe": n_.mean() / n_.std() * np.sqrt(DAYS),
                "CAPM alpha t": iv.capm_t(n_, bench_r.reindex(n_.index))[1],
            }
    t = pd.DataFrame(rows).T
    pk = pd.DataFrame(
        {
            "AP": {Y: B["picks"][Y].key for Y in WF_YEARS},
            "AP + C3": {Y: picks[Y].key for Y in WF_YEARS},
        }
    )
    gs = pd.DataFrame(
        {
            k: {
                "net CAGR 16-26 %": iv.cagr(v[0]["net"].loc[OOS]),
                "CAPM alpha t 16-26": iv.capm_t(
                    v[0]["net"].loc[OOS], bench_r.reindex(v[0].loc[OOS].index)
                )[1],
            }
            for k, v in out.items()
        }
    ).T
    joblib.dump(
        {
            "res": {k: v[0][["gross", "turn", "net"]].astype("float32") for k, v in out.items()},
            "wf": r[["gross", "turn", "net"]],
        },
        OUT / "check_c3.joblib",
        compress=3,
    )
    t.to_csv(OUT / "check_c3_wf.csv")
    pk.to_csv(OUT / "check_c3_picks.csv")
    gs.to_csv(OUT / "check_c3_grid.csv")
    txt = [
        "# Post-hoc check D3: is AP's edge the exclusion of C3? (13 extra trials, not a gate)",
        "",
        iv.md(t),
        "",
        iv.md(pk),
        "",
        iv.md(gs),
    ]
    (OUT / "check_c3.md").write_text("\n".join(txt))
    print("\n".join(txt))
    log("c3check done", t0)


if __name__ == "__main__":
    {
        "diag": stage_diag,
        "corr": stage_corr,
        "build": stage_build,
        "robust": stage_robust,
        "c3check": stage_c3check,
    }[sys.argv[1]]()
