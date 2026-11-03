"""strategy_v3 - fix the four leaks between the ML signal and factor-adjusted alpha.

    uv run set-quant-alpha/experiments/strategy_v3/v3.py preds     # extended walk-forward 2013-2019 (~5 min)
    uv run set-quant-alpha/experiments/strategy_v3/v3.py select    # run every design on 2013-2021, freeze the pick
    uv run set-quant-alpha/experiments/strategy_v3/v3.py test      # one-shot test 2022-01 -> 2026-10

Leak -> fix
1. Costs (top-decile alpha 0.27%/5d < 0.30% round trip)  -> costs inside the optimiser: a trade is made only
   when its expected alpha beats its cost (a principled no-trade band instead of a fixed rank buffer).
2. Construction adds factor tilts (equal-weight top-10 is small-cap vs a megacap SET50)
   -> TDEX core + stock tilts, with the book's local MKT / SMB / WML loadings held at TDEX's (1, 0, 0).
3. Too few bets (10 names, low transfer coefficient) -> many small Grinold-sized positions, risk-penalised.
4. Test too short (4.8 years) -> walk-forward predictions back to 2013; designs are selected on 2013-2021
   (9 years) instead of 2020-2021, and 2022-26 stays a one-shot test.

Pre-registered (written before any simulation): designs in DESIGNS, the selection rule in `select`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from setalpha import config, model, portfolio, report, transform

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "strategy_v2"))
import sv2  # noqa: E402  (reused read-only: simulator for E1/E2, attribution, stats)

PREDS_EXT = OUT / "predictions_ext.parquet"
FROZEN = OUT / "picks_frozen_before_test.txt"
SEL_END = "2021-12-31"
CAPITAL = 20e6  # THB; large enough that 100-share lots do not block 1-3% tilts
EXT_YEARS = range(2013, 2020)
LOAD_WIN, LOAD_MIN = 250, 200  # factor-loading regression window / minimum observations
WMAX = 0.05  # max weight per stock
EPS = 0.05  # band on active MKT / SMB / WML loading
TRIALS_BEFORE = 60  # 37 cost_aware + 6 pipeline + 17 strategy_v2


@dataclass(frozen=True)
class Design:
    name: str
    te: float = 0.04  # ex-ante tracking-error target used to calibrate risk aversion (cost-free)
    neutral: bool = True  # hold MKT/SMB/WML loadings at TDEX's
    tdex: bool = True  # TDEX allowed in the book
    kappa: float = 1.0  # weight of trading costs in the objective (0 = cost-unaware)
    pickable: bool = True
    family: str = "F"


DESIGNS = [
    Design("F1 enhanced index: TDEX + neutral tilts, TE 4%, cost-aware"),
    Design("F2 enhanced index: TDEX + neutral tilts, TE 8%, cost-aware", te=0.08),
    Design("F3 all-stock factor-matched (no TDEX), cost-aware", te=0.08, tdex=False),
    Design(
        "A1 ablation: F2 without factor constraints",
        te=0.08,
        neutral=False,
        pickable=False,
        family="A",
    ),
    Design("A2 ablation: F2 cost-unaware", te=0.08, kappa=0.0, pickable=False, family="A"),
]
BASELINES = [
    sv2.Design("E1 equal-weight top-10", (sv2.Sleeve(),), "E", pickable=False),
    sv2.Design("E2 top-10 + buffer 30 + 8% stop", (sv2.E2S,), "E", pickable=True),
]


# ============================================================================ 4. longer out-of-sample
def make_ext_predictions() -> pd.DataFrame:
    """Walk-forward logit + XGB (production hyperparameters), refit each January 2013-2019."""
    df = transform.add_dummies(transform.load())
    sets = transform.feature_sets(df)
    feats = sets["NUMERIC"] + sets["DUMMIES"]
    best = json.loads((config.MODELS_DIR / "best_params.json").read_text())
    members = report.load_meta()["ensemble"]
    lab = model.labelled(df)
    parts = []
    for yr in EXT_YEARS:
        start = f"{yr}-01-01"
        train = model._embargo(lab[lab["Date"] < start], start)
        test = lab[lab["Date"].dt.year == yr]
        part = test[["Date", "Ticker"]].copy()
        for name in members:
            part[f"pred_{name}"] = model.predict(
                model.fit(name, train, feats, **best[name]), test, feats
            )
        parts.append(part)
        print(f"  {yr}: trained on {len(train):,} rows, predicted {len(test):,}", flush=True)
    ext = model.add_ensemble(pd.concat(parts, ignore_index=True), members)
    prod = pd.read_parquet(model.PREDICTIONS)
    cols = ["Date", "Ticker"] + [f"pred_{m}" for m in members] + ["pred_ens"]
    out = pd.concat([ext[cols], prod[cols]], ignore_index=True)
    out["split"] = np.where(out["Date"] <= SEL_END, "sel", "test")
    out.to_parquet(PREDS_EXT, index=False)

    # IC by year over the whole 2013-2026 out-of-sample history
    d = out.merge(df[["Date", "Ticker", "tradable", "fwd_alpha_5"]], on=["Date", "Ticker"])
    ic = model.daily_ic(d, "pred_ens")
    by = ic.groupby(ic.index.year).agg(["mean", "count"])
    by["t"] = ic.groupby(ic.index.year).apply(
        lambda x: x.iloc[::5].mean() / x.iloc[::5].std() * np.sqrt(len(x.iloc[::5]))
    )
    by.to_csv(OUT / "ic_by_year_2013_2026.csv")
    print(by.round(4).to_string())
    return out


# ============================================================================ 1-3. optimiser
def loadings(
    rets: pd.DataFrame, fac: pd.DataFrame, day: pd.Timestamp, names: list[str]
) -> pd.DataFrame:
    """Rolling OLS of daily returns on local MKT/SMB/WML up to `day`; adds residual daily vol."""
    f = fac.loc[:day].tail(LOAD_WIN)
    r = rets.reindex(index=f.index, columns=names)
    ok = r.notna().sum() >= LOAD_MIN
    r = r.loc[:, ok].fillna(0.0)
    X = np.column_stack([np.ones(len(f)), f[["MKT", "SMB", "WML"]].to_numpy()])
    b, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    resid = r.to_numpy() - X @ b
    return pd.DataFrame(
        {"b_MKT": b[1], "b_SMB": b[2], "b_WML": b[3], "s_eps": resid.std(axis=0, ddof=4)},
        index=r.columns,
    )


def solve(alpha, s2, B, w0, t0, lam, des: Design, fee=config.FEE_PER_SIDE, tfee=sv2.TDEX_FEE):
    """max a'w - lam/2 sum s2 w^2 - kappa * cost(|dw|)  s.t. fully invested, 0<=w<=WMAX, loadings ~ TDEX's.

    Variables x = [u (buys), v (sells), uT, vT]; w = w0 + u - v, wT = t0 + uT - vT.
    """
    n = len(alpha)
    k = des.kappa

    def split(x):
        return x[:n], x[n : 2 * n], x[2 * n], x[2 * n + 1]

    def obj(x):
        u, v, ut, vt = split(x)
        w = w0 + u - v
        return -(alpha @ w - 0.5 * lam * (s2 * w * w).sum()) + k * (
            fee * (u + v).sum() + tfee * (ut + vt)
        )

    def grad(x):
        u, v, ut, vt = split(x)
        gw = -(alpha - lam * s2 * (w0 + u - v))
        return np.concatenate([gw + k * fee, -gw + k * fee, [k * tfee, k * tfee]])

    A_w = np.hstack([np.eye(n), -np.eye(n), np.zeros((n, 2))])  # w = w0 + A_w x
    a_t = np.r_[np.zeros(2 * n), 1.0, -1.0]  # wT = t0 + a_t x
    cons = [
        {
            "type": "eq",
            "fun": lambda x: (w0 + A_w @ x).sum() + t0 + a_t @ x - 1.0,
            "jac": lambda x: A_w.sum(0) + a_t,
        },
        {"type": "ineq", "fun": lambda x: w0 + A_w @ x, "jac": lambda x: A_w},
        {"type": "ineq", "fun": lambda x: WMAX - (w0 + A_w @ x), "jac": lambda x: -A_w},
        {
            "type": "ineq",
            "fun": lambda x: np.atleast_1d(t0 + a_t @ x),
            "jac": lambda x: a_t[None, :],
        },
    ]
    if des.neutral:
        target = np.array([1.0, 0.0, 0.0])
        Bt = B.T  # 3 x n
        tdex_b = np.array([1.0, 0.0, 0.0])

        def expo(x):
            return Bt @ (w0 + A_w @ x) + tdex_b * (t0 + a_t @ x) - target

        jac_e = Bt @ A_w + np.outer(tdex_b, a_t)
        cons += [
            {"type": "ineq", "fun": lambda x: EPS - expo(x), "jac": lambda x: -jac_e},
            {"type": "ineq", "fun": lambda x: EPS + expo(x), "jac": lambda x: jac_e},
        ]
    tb = (0.0, 1.0) if des.tdex else (0.0, 0.0)
    bounds = (
        [(0.0, WMAX)] * n
        + [(0.0, max(float(w), 0.0)) for w in w0]
        + [tb, (0.0, max(t0, 0.0)) if des.tdex else (0.0, 0.0)]
    )
    res = minimize(
        obj,
        np.zeros(2 * n + 2),
        jac=grad,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    u, v, ut, vt = split(res.x)
    w = np.clip(w0 + u - v, 0.0, WMAX)
    w[w < 1e-4] = 0.0
    return w, max(t0 + ut - vt, 0.0), bool(res.success)


def problem(day_rows: pd.DataFrame, held: pd.Series, rets, fac, day, ic):
    """Alpha, risk and loadings for the candidates: names held + names with positive alpha."""
    tr = day_rows[day_rows["tradable"].fillna(False).astype(bool)].dropna(
        subset=["pred_ens", "idio_vol"]
    )
    tr = tr.set_index("Ticker")
    a = portfolio.alpha_from_scores(tr["pred_ens"], tr["idio_vol"], ic)  # 5-day expected alpha
    cand = sorted(set(a.index[a > 0]) | (set(held.index[held > 0]) & set(a.index)))
    L = loadings(rets, fac, day, cand)
    cand = list(L.index)
    s2 = (L["s_eps"] ** 2 * config.HORIZON).to_numpy()  # 5-day residual variance
    return cand, a.reindex(cand).to_numpy(), s2, L[["b_MKT", "b_SMB", "b_WML"]].to_numpy()


def calibrate_lambda(des: Design, f, rets, fac, ic, n_days: int = 15) -> float:
    """Risk aversion giving median ex-ante TE = des.te for the cost-free, from-TDEX optimum (no returns used)."""
    days = sorted(f["Date"].unique())
    pick = [days[i] for i in np.linspace(260, len(days) - 1, n_days).astype(int)]
    free = Design("cal", neutral=des.neutral, tdex=des.tdex, kappa=0.0)
    probs = []
    for d in pick:
        cand, a, s2, B = problem(
            f[f["Date"] == d], pd.Series(dtype=float), rets, fac, pd.Timestamp(d), ic
        )
        probs.append((a, s2, B))

    def te(lam):
        out = []
        for a, s2, B in probs:
            w, _, _ = solve(a, s2, B, np.zeros(len(a)), 1.0 if des.tdex else 0.0, lam, free)
            out.append(np.sqrt((s2 * w * w).sum() * 52))
        return float(np.median(out))

    lo, hi = 1.0, 2000.0
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if te(mid) > des.te else (lo, mid)
    return float(np.sqrt(lo * hi))


@dataclass
class Run:
    equity: pd.Series
    trades: list = field(default_factory=list)
    stock_w: dict = field(default_factory=dict)
    tdex_w: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)
    expo: list = field(default_factory=list)
    fails: int = 0


def simulate_opt(
    des: Design,
    lam: float,
    f,
    px,
    bench,
    rets,
    fac,
    ic,
    end=None,
    fee=config.FEE_PER_SIDE,
    tfee=sv2.TDEX_FEE,
    replay: dict | None = None,
) -> Run:
    """Weekly decision at the close, execution at the next open, 100-share lots, costs per side."""
    O, C = px["Open"], px["Close"].ffill()  # noqa: E741
    bO, bC = bench["Open"].reindex(C.index), bench["Close"].reindex(C.index).ffill()
    dates = [
        d
        for d in C.index
        if f["Date"].min() <= d <= (pd.Timestamp(end) if end else f["Date"].max())
    ]
    by_day = {d: g for d, g in f.groupby("Date")}
    cash, sh, tu = CAPITAL, {}, 0.0
    run, pending = Run(pd.Series(dtype=float)), None
    eq = {}
    for i, d in enumerate(dates):
        if pending is not None:  # execute at today's open
            tw, ttdex = pending
            po = {
                t: (O.at[d, t] if t in O.columns and pd.notna(O.at[d, t]) else np.nan)
                for t in set(sh) | set(tw.index)
            }
            pb = bO.at[d] if pd.notna(bO.at[d]) else bC.at[d]
            value = (
                cash
                + sum(q * (po[t] if pd.notna(po[t]) else C.at[d, t]) for t, q in sh.items())
                + tu * pb
            )
            budget = value * 0.997  # leave room for costs
            orders = []
            for t in set(sh) | set(tw.index):
                if pd.isna(po[t]):
                    continue
                tgt = int(budget * tw.get(t, 0.0) / po[t] / sv2.LOT) * sv2.LOT
                dq = tgt - sh.get(t, 0)
                if dq:
                    orders.append((dq, t))
            for dq, t in sorted(orders):  # sells first
                cash -= dq * po[t] + abs(dq * po[t]) * fee
                sh[t] = sh.get(t, 0) + dq
                run.trades.append((d, t, dq, po[t], "rebalance"))
            dtu = budget * ttdex / pb - tu
            if abs(dtu * pb) > 1e-6 * value:
                cash -= dtu * pb + abs(dtu * pb) * tfee
                tu += dtu
                run.trades.append((d, config.BENCHMARK, dtu, pb, "tdex"))
            sh = {t: q for t, q in sh.items() if q}
            pending = None
        value = (
            cash
            + sum(q * C.at[d, t] for t, q in sh.items() if pd.notna(C.at[d, t]))
            + tu * bC.at[d]
        )
        eq[d] = value
        stock_val = pd.Series({t: q * C.at[d, t] for t, q in sh.items()}, dtype=float).fillna(0.0)
        run.stock_w[d] = stock_val.sum() / value
        run.tdex_w[d] = tu * bC.at[d] / value
        if i % config.HORIZON == 0 and d in by_day and i < len(dates) - 1:  # decision at the close
            if replay is not None:
                pending = replay.get(d)
                continue
            held = stock_val / value
            cand, a, s2, B = problem(by_day[d], held, rets, fac, d, ic)
            w0 = held.reindex(cand).fillna(0.0).to_numpy()
            t0 = float(run.tdex_w[d])
            w, wt, ok = solve(a, s2, B, w0, t0, lam, des)
            run.fails += not ok
            tw = pd.Series(w, index=cand)
            tw = tw[tw > 0]
            pending = (tw, wt)
            run.targets[d] = pending
            e = B.T @ w + np.array([1.0, 0.0, 0.0]) * wt
            run.expo.append((d, *e, float(np.sqrt((s2 * w * w).sum() * 52)), int((w > 0).sum())))
    run.equity = pd.Series(eq)
    return run


def as_sim(name: str, run: Run) -> sv2.SimResult:
    tr = pd.DataFrame(run.trades, columns=["Date", "Ticker", "qty", "price", "reason"])
    return sv2.SimResult(
        name,
        run.equity.rename(name),
        tr,
        pd.Series(run.stock_w),
        pd.Series(run.tdex_w),
        pd.DataFrame(),
    )


# ============================================================================ stages
def setup(split: str):
    inp = sv2.load_inputs(PREDS_EXT, v8_path=None)
    f = sv2.make_frame(inp, split)
    px = sv2.price_panel(inp, f)
    rets = (
        inp["prices"]
        .pivot_table(index="Date", columns="Ticker", values="Close")
        .pct_change(fill_method=None)
    )
    rets = rets.where(rets.abs() < 0.45)
    return inp, f, px, rets, inp["local"]


def run_all(
    split: str, lams: dict | None = None, end: str | None = None
) -> tuple[pd.DataFrame, dict, dict]:
    inp, f, px, rets, fac = setup(split)
    ic = inp["ic"]
    rows, daily, lam_used = {}, {}, {}
    ew = sv2.universe_ew(px["Close"], f)
    for des in BASELINES:
        net = sv2.simulate(des, f, px, inp["bench"], capital=CAPITAL)
        gross = sv2.simulate(des, f, px, inp["bench"], fee=0.0, tdex_fee=0.0, capital=CAPITAL)
        if end:
            net.equity, gross.equity = net.equity.loc[:end], gross.equity.loc[:end]
        b = sv2.bench_returns(inp["bench"], net.equity.index)
        rows[des.name] = {
            "family": des.family,
            "pickable": des.pickable,
            **sv2.stats(net, gross, inp, b, ew),
        }
        daily[des.name] = net.returns
        print(f"  {des.name}: CAGR {rows[des.name]['CAGR %']:.2f}", flush=True)
    for des in DESIGNS:
        lam = lams[des.name] if lams else calibrate_lambda(des, f, rets, fac, ic)
        lam_used[des.name] = lam
        net = simulate_opt(des, lam, f, px, inp["bench"], rets, fac, ic, end=end)
        gross = simulate_opt(
            des,
            lam,
            f,
            px,
            inp["bench"],
            rets,
            fac,
            ic,
            end=end,
            fee=0.0,
            tfee=0.0,
            replay=net.targets,
        )
        b = sv2.bench_returns(inp["bench"], net.equity.index)
        ex = pd.DataFrame(net.expo, columns=["Date", "MKT", "SMB", "WML", "exante_TE", "names"])
        rows[des.name] = {
            "family": des.family,
            "pickable": des.pickable,
            **sv2.stats(as_sim(des.name, net), as_sim(des.name, gross), inp, b, ew),
            "lambda": lam,
            "ex-ante TE %": 100 * ex["exante_TE"].median(),
            "names held": ex["names"].median(),
            "ex-ante b_SMB": ex["SMB"].mean(),
            "ex-ante b_WML": ex["WML"].mean(),
            "solver fails": net.fails,
        }
        daily[des.name] = net.equity.pct_change().dropna()
        print(
            f"  {des.name}: lambda {lam:.1f}, CAGR {rows[des.name]['CAGR %']:.2f}, "
            f"TH alpha t {rows[des.name].get('TH alpha t', np.nan):.2f}",
            flush=True,
        )
    b = sv2.bench_returns(inp["bench"], next(iter(daily.values())).index).dropna()
    rows["TDEX buy & hold"] = {
        "family": "bench",
        "pickable": False,
        "CAGR %": 100 * ((1 + b).prod() ** (252 / len(b)) - 1),
        "Sharpe": b.mean() / b.std() * np.sqrt(252),
        **sv2.attribution(b, inp),
    }
    return pd.DataFrame(rows).T, daily, lam_used


COLS = [
    "family",
    "pickable",
    "CAGR %",
    "gross CAGR %",
    "cost drag %",
    "Sharpe",
    "max DD %",
    "TE %",
    "IR vs TDEX",
    "turnover 2-way x/yr",
    "CAPM alpha %",
    "CAPM alpha t",
    "CAPM beta",
    "FF3 alpha %",
    "FF3 alpha t",
    "C4 alpha %",
    "C4 alpha t",
    "TH alpha %",
    "TH alpha t",
    "TH b_MKT",
    "TH b_SMB",
    "TH b_WML",
    "TDEX weight %",
    "names held",
    "ex-ante TE %",
    "lambda",
    "solver fails",
    "DSR",
]


def add_dsr(t: pd.DataFrame, daily: dict, prior_srs: list[float]) -> pd.DataFrame:
    srs = prior_srs + [float(r.mean() / r.std()) for r in daily.values()]
    t["DSR"] = [sv2.deflated(daily[k], srs) if k in daily else np.nan for k in t.index]
    return t


def prior_trial_srs() -> list[float]:
    v = pd.read_csv(
        HERE.parent / "strategy_v2" / "output" / "validation_all_with_dsr.csv", index_col=0
    )
    srs = v["sr_daily"].dropna().tolist() if "sr_daily" in v else []
    return srs + [0.0] * max(0, TRIALS_BEFORE - len(srs))  # unknown earlier trials counted at SR 0


def stage_select() -> None:
    t, daily, lams = run_all("sel", end=SEL_END)
    t = add_dsr(t, daily, prior_trial_srs())
    t[[c for c in COLS if c in t.columns]].to_csv(OUT / "select_2013_2021.csv")
    elig = t[t["pickable"] == True]  # noqa: E712
    pick = elig["TH alpha t"].astype(float).idxmax()
    FROZEN.write_text(
        f"frozen {datetime.now(UTC).isoformat(timespec='seconds')} (before any strategy_v3 test simulation)\n"
        "rule = highest 2013-2021 local-Thai (MKT/SMB/WML) alpha t among pickable designs (E2, F1, F2, F3)\n"
        f"PICK = {pick}  (TH alpha {float(t.at[pick, 'TH alpha %']):.2f}%/yr, t {float(t.at[pick, 'TH alpha t']):.2f})\n"
        f"lambdas = {json.dumps(lams)}\n"
        f"trials for DSR = {TRIALS_BEFORE} earlier + {len(DESIGNS) + len(BASELINES)} here\n"
    )
    print(FROZEN.read_text())
    print(sv2.to_md(t[[c for c in COLS[2:] if c in t.columns]].astype(float)))


def stage_test() -> None:
    assert FROZEN.exists(), "run `select` first"
    lams = json.loads(FROZEN.read_text().split("lambdas = ")[1].splitlines()[0])
    t, daily, _ = run_all("test", lams=lams)
    t = add_dsr(t, daily, prior_trial_srs())
    t[[c for c in COLS if c in t.columns]].to_csv(OUT / "test_2022_2026.csv")
    pd.DataFrame(daily).to_parquet(OUT / "test_daily_returns.parquet")
    print(FROZEN.read_text())
    print(sv2.to_md(t[[c for c in COLS[2:] if c in t.columns]].astype(float)))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "select"
    {"preds": make_ext_predictions, "select": stage_select, "test": stage_test}[stage]()
