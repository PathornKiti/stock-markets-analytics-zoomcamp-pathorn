"""strategy_v4 - widen the tradable universe from the top-100 to the top-200 liquid names.

    uv run set-quant-alpha/experiments/strategy_v4/v4.py select    # 2013-2021, freezes the pick
    uv run set-quant-alpha/experiments/strategy_v4/v4.py test      # one-shot 2022-01 -> 2026-10

Motivation (validation 2020-21): daily IC is 0.047 in liquidity ranks 1-50, 0.059 in 51-100 and 0.087 in 101-200.
Smaller names cost more to trade, so every design here uses one liquidity-aware cost model:

    cost per side = 0.11% (commission + VAT) + Y * sigma_daily * sqrt(trade value / ADV),   Y = 0.5

(square-root market impact; at THB 20M this is about the old flat 0.15% for top-50 names). Positions are capped at
min(5% of the book, 20% of the name's 60-day median daily traded value). Execution is at the next open (call auction),
100-share lots. Stop-losses are not used: strategy_v2 found the 8% stop is not robust.

Pre-registered before any simulation: DESIGNS, Y, the caps, lambda = 1 (as v3 F2), and the selection rule.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from setalpha import config, portfolio

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "strategy_v3"))
sys.path.insert(0, str(HERE.parent / "strategy_v2"))
import sv2  # noqa: E402  (read-only reuse: stats, attribution, frames)
import v3  # noqa: E402  (read-only reuse: extended predictions, factor loadings)

FROZEN = OUT / "picks_frozen_before_test.txt"
COMMISSION = 0.0011  # per side, commission + VAT
IMPACT_Y = 0.5  # square-root impact coefficient (sensitivity: 1.0)
ADV_CAP = 0.20  # max position as a fraction of 60-day median daily traded value
WMAX = 0.05
LAM = 1.0
TRIALS_BEFORE = 67  # 60 earlier + 7 in strategy_v3


@dataclass(frozen=True)
class Design:
    name: str
    universe: int  # max liquidity rank
    kind: str  # "rule" (equal-weight top-10 + buffer) | "opt" (TDEX + neutral tilts, cost-aware)
    buffer: int = 0
    pickable: bool = True


DESIGNS = [
    Design("R100 top-10 EW + buffer 30, ranks 1-100", 100, "rule", buffer=30),
    Design("R200 top-10 EW + buffer 60, ranks 1-200", 200, "rule", buffer=60),
    Design("N100 TDEX + neutral cost-aware tilts, ranks 1-100", 100, "opt"),
    Design("N200 TDEX + neutral cost-aware tilts, ranks 1-200", 200, "opt"),
]


# ============================================================================ inputs per decision
def names_info(day_rows: pd.DataFrame, rets: pd.DataFrame, day, names) -> pd.DataFrame:
    """sigma (60-day daily vol) and ADV for impact costs and caps."""
    r = rets.loc[:day].tail(60).reindex(columns=names)
    adv = day_rows.set_index("Ticker")["dvol60"].reindex(names)
    return pd.DataFrame(
        {"sigma": r.std().fillna(0.03).clip(0.005, 0.10), "adv": adv.fillna(config.MIN_DVOL)}
    )


def universe(day_rows: pd.DataFrame, U: int) -> pd.DataFrame:
    return (
        day_rows[(day_rows["lrank"] <= U)]
        .dropna(subset=["pred_ens", "idio_vol"])
        .set_index("Ticker")
    )


def rule_targets(des: Design, day_rows, held: pd.Series, n: int = 10) -> pd.Series:
    u = universe(day_rows, des.universe)
    rank = u["pred_ens"].rank(ascending=False, method="first")
    keep = [t for t in held.index[held > 0] if t in rank.index and rank[t] <= n + des.buffer]
    keep = sorted(keep, key=lambda t: rank[t])[:n]
    fill = [t for t in rank.sort_values().index if t not in keep][: n - len(keep)]
    names = keep + fill
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


def solve(alpha, s2, B, w0, t0, imp, wmax):
    """max a'w - LAM/2 sum s2 w^2 - sum(c|dw| + imp |dw|^1.5) - tdex fee; fully invested, loadings ~ TDEX's."""
    n = len(alpha)
    tfee = sv2.TDEX_FEE

    def split(x):
        return x[:n], x[n : 2 * n], x[2 * n], x[2 * n + 1]

    def obj(x):
        u, v, ut, vt = split(x)
        w = w0 + u - v
        cost = COMMISSION * (u + v).sum() + (imp * (u**1.5 + v**1.5)).sum() + tfee * (ut + vt)
        return -(alpha @ w - 0.5 * LAM * (s2 * w * w).sum()) + cost

    def grad(x):
        u, v, ut, vt = split(x)
        gw = -(alpha - LAM * s2 * (w0 + u - v))
        return np.concatenate(
            [
                gw + COMMISSION + 1.5 * imp * np.sqrt(u),
                -gw + COMMISSION + 1.5 * imp * np.sqrt(v),
                [tfee, tfee],
            ]
        )

    A_w = np.hstack([np.eye(n), -np.eye(n), np.zeros((n, 2))])
    a_t = np.r_[np.zeros(2 * n), 1.0, -1.0]
    tb = np.array([1.0, 0.0, 0.0])
    jac_e = B.T @ A_w + np.outer(tb, a_t)

    def expo(x):
        return B.T @ (w0 + A_w @ x) + tb * (t0 + a_t @ x) - tb

    cons = [
        {
            "type": "eq",
            "fun": lambda x: (w0 + A_w @ x).sum() + t0 + a_t @ x - 1.0,
            "jac": lambda x: A_w.sum(0) + a_t,
        },
        {"type": "ineq", "fun": lambda x: w0 + A_w @ x, "jac": lambda x: A_w},
        {"type": "ineq", "fun": lambda x: wmax - (w0 + A_w @ x), "jac": lambda x: -A_w},
        {
            "type": "ineq",
            "fun": lambda x: np.atleast_1d(t0 + a_t @ x),
            "jac": lambda x: a_t[None, :],
        },
        {"type": "ineq", "fun": lambda x: v3.EPS - expo(x), "jac": lambda x: -jac_e},
        {"type": "ineq", "fun": lambda x: v3.EPS + expo(x), "jac": lambda x: jac_e},
    ]
    bounds = (
        [(0.0, float(m)) for m in wmax]
        + [(0.0, max(float(w), 0.0)) for w in w0]
        + [(0.0, 1.0), (0.0, max(t0, 0.0))]
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
    w = np.clip(w0 + u - v, 0.0, wmax)
    w[w < 1e-4] = 0.0
    return w, max(t0 + ut - vt, 0.0), bool(res.success)


def opt_targets(
    des: Design, day_rows, held: pd.Series, t0: float, rets, fac, day, ic, equity, info
):
    u = universe(day_rows, des.universe)
    a = portfolio.alpha_from_scores(u["pred_ens"], u["idio_vol"], ic)
    cand = sorted(set(a.index[a > 0]) | (set(held.index[held > 0]) & set(a.index)))
    L = v3.loadings(rets, fac, day, cand)
    cand = list(L.index)
    s2 = (L["s_eps"] ** 2 * config.HORIZON).to_numpy()
    nf = info.loc[cand]
    imp = IMPACT_Y * nf["sigma"].to_numpy() * np.sqrt(equity / nf["adv"].to_numpy())
    wmax = np.minimum(WMAX, ADV_CAP * nf["adv"].to_numpy() / equity)
    w0 = held.reindex(cand).fillna(0.0).to_numpy()
    w, wt, ok = solve(
        a.reindex(cand).to_numpy(), s2, L[["b_MKT", "b_SMB", "b_WML"]].to_numpy(), w0, t0, imp, wmax
    )
    tw = pd.Series(w, index=cand)
    e = L[["b_MKT", "b_SMB", "b_WML"]].to_numpy().T @ w + np.array([1.0, 0.0, 0.0]) * wt
    return tw[tw > 0], wt, ok, e


# ============================================================================ simulator
def simulate(
    des: Design,
    f,
    px,
    bench,
    rets,
    fac,
    ic,
    end=None,
    y=IMPACT_Y,
    commission=COMMISSION,
    tfee=sv2.TDEX_FEE,
    replay: dict | None = None,
):
    O, C = px["Open"], px["Close"].ffill()  # noqa: E741
    bO, bC = bench["Open"].reindex(C.index), bench["Close"].reindex(C.index).ffill()
    last = pd.Timestamp(end) if end else f["Date"].max()
    dates = [d for d in C.index if f["Date"].min() <= d <= last]
    by_day = {d: g for d, g in f.groupby("Date")}
    cash, sh, tu, pending = v3.CAPITAL, {}, 0.0, None
    eq, trades, stock_w, tdex_w, targets, expo, fails = {}, [], {}, {}, {}, [], 0
    known: dict[str, tuple[float, float]] = {}  # last sigma / ADV per name (for exits)
    impact_paid = 0.0
    for i, d in enumerate(dates):
        if pending is not None:
            tw, ttdex = pending
            names = set(sh) | set(tw.index)
            po = {
                t: (O.at[d, t] if t in O.columns and pd.notna(O.at[d, t]) else np.nan)
                for t in names
            }
            pb = bO.at[d] if pd.notna(bO.at[d]) else bC.at[d]
            value = (
                cash
                + sum(q * (po[t] if pd.notna(po[t]) else C.at[d, t]) for t, q in sh.items())
                + tu * pb
            )
            budget = value * 0.995
            orders = []
            for t in names:
                if pd.isna(po[t]):
                    continue
                dq = int(budget * tw.get(t, 0.0) / po[t] / sv2.LOT) * sv2.LOT - sh.get(t, 0)
                if dq:
                    orders.append((dq, t))
            for dq, t in sorted(orders):  # sells first
                tv = abs(dq * po[t])
                sig, adv = known.get(t, (0.03, config.MIN_DVOL))
                imp = y * sig * np.sqrt(tv / adv)
                cash -= dq * po[t] + tv * (commission + imp)
                impact_paid += tv * imp
                sh[t] = sh.get(t, 0) + dq
                trades.append((d, t, dq, po[t], "rebalance"))
            dtu = budget * ttdex / pb - tu
            if abs(dtu * pb) > 1e-6 * value:
                cash -= dtu * pb + abs(dtu * pb) * tfee
                tu += dtu
                trades.append((d, config.BENCHMARK, dtu, pb, "tdex"))
            sh = {t: q for t, q in sh.items() if q}
            pending = None
        value = (
            cash
            + sum(q * C.at[d, t] for t, q in sh.items() if pd.notna(C.at[d, t]))
            + tu * bC.at[d]
        )
        eq[d] = value
        sv = pd.Series({t: q * C.at[d, t] for t, q in sh.items()}, dtype=float).fillna(0.0)
        stock_w[d], tdex_w[d] = sv.sum() / value, tu * bC.at[d] / value
        if i % config.HORIZON == 0 and d in by_day and i < len(dates) - 1:
            if replay is not None:
                pending = replay.get(d)
                continue
            rows = by_day[d]
            held = sv / value
            info = names_info(rows, rets, d, sorted(set(rows["Ticker"]) | set(held.index)))
            known.update({t: (r.sigma, r.adv) for t, r in info.iterrows()})
            if des.kind == "rule":
                tw, wt = rule_targets(des, rows, held), 0.0
            else:
                tw, wt, ok, e = opt_targets(
                    des, rows, held, float(tdex_w[d]), rets, fac, d, ic, value, info
                )
                fails += not ok
                expo.append((d, *e, len(tw)))
            pending = (tw, wt)
            targets[d] = pending
    run = v3.Run(pd.Series(eq), trades, stock_w, tdex_w, targets, expo, fails)
    return run, impact_paid


def run_all(
    split: str, end: str | None = None, designs=DESIGNS, y=IMPACT_Y
) -> tuple[pd.DataFrame, dict]:
    inp, f, px, rets, fac = v3.setup(split)
    ic = inp["ic"]
    ew = sv2.universe_ew(px["Close"], f)
    rows, daily = {}, {}
    for des in designs:
        net, impact = simulate(des, f, px, inp["bench"], rets, fac, ic, end=end, y=y)
        gross, _ = simulate(
            des,
            f,
            px,
            inp["bench"],
            rets,
            fac,
            ic,
            end=end,
            y=0.0,
            commission=0.0,
            tfee=0.0,
            replay=net.targets,
        )
        b = sv2.bench_returns(inp["bench"], net.equity.index)
        st = sv2.stats(v3.as_sim(des.name, net), v3.as_sim(des.name, gross), inp, b, ew)
        years = len(net.equity) / 252
        ex = (
            pd.DataFrame(net.expo, columns=["Date", "MKT", "SMB", "WML", "names"])
            if net.expo
            else pd.DataFrame()
        )
        held_rank = _avg_rank(net, f)
        rows[des.name] = {
            "pickable": des.pickable,
            **st,
            "impact cost %/yr": 100 * impact / net.equity.mean() / years,
            "avg liq rank held": held_rank,
            "solver fails": net.fails,
            "names held": ex["names"].median() if len(ex) else 10,
            "TDEX weight %": 100 * pd.Series(net.tdex_w).mean(),
        }
        daily[des.name] = net.equity.pct_change().dropna()
        print(
            f"  {des.name}: CAGR {st['CAGR %']:.2f}, TH alpha t {st.get('TH alpha t', np.nan):.2f}",
            flush=True,
        )
    b = sv2.bench_returns(inp["bench"], next(iter(daily.values())).index).dropna()
    rows["TDEX buy & hold"] = {
        "pickable": False,
        "CAGR %": 100 * ((1 + b).prod() ** (252 / len(b)) - 1),
        "Sharpe": b.mean() / b.std() * np.sqrt(252),
        **sv2.attribution(b, inp),
    }
    return pd.DataFrame(rows).T, daily


def _avg_rank(run, f) -> float:
    lr = f.set_index(["Date", "Ticker"])["lrank"]
    ranks = [lr.get((d, t)) for d, (tw, _) in run.targets.items() for t in tw.index]
    ranks = [r for r in ranks if r is not None and not np.isnan(r)]
    return float(np.mean(ranks)) if ranks else np.nan


COLS = [
    "pickable",
    "CAGR %",
    "gross CAGR %",
    "cost drag %",
    "impact cost %/yr",
    "Sharpe",
    "max DD %",
    "IR vs TDEX",
    "turnover 2-way x/yr",
    "CAPM alpha %",
    "CAPM alpha t",
    "FF3 alpha %",
    "FF3 alpha t",
    "C4 alpha %",
    "C4 alpha t",
    "TH alpha %",
    "TH alpha t",
    "TH b_MKT",
    "TH b_SMB",
    "TH b_WML",
    "avg liq rank held",
    "names held",
    "TDEX weight %",
    "solver fails",
    "DSR",
]


def stage_select() -> None:
    t, daily = run_all("sel", end=v3.SEL_END)
    t = v3.add_dsr(t, daily, v3.prior_trial_srs() + [0.0] * 7)
    t[[c for c in COLS if c in t.columns]].to_csv(OUT / "select_2013_2021.csv")
    pick = t[t["pickable"] == True]["TH alpha t"].astype(float).idxmax()  # noqa: E712
    FROZEN.write_text(
        f"frozen {datetime.now(UTC).isoformat(timespec='seconds')} (before any strategy_v4 test simulation)\n"
        "rule = highest 2013-2021 local-Thai (MKT/SMB/WML) alpha t among R100, R200, N100, N200\n"
        f"PICK = {pick}  (TH alpha {float(t.at[pick, 'TH alpha %']):.2f}%/yr, t {float(t.at[pick, 'TH alpha t']):.2f})\n"
        f"cost model = {COMMISSION:.4f}/side + {IMPACT_Y} * sigma * sqrt(trade/ADV); caps min({WMAX}, {ADV_CAP} x ADV)\n"
        f"trials for DSR = {TRIALS_BEFORE} earlier + {len(DESIGNS)} here\n"
    )
    print(FROZEN.read_text())
    print(sv2.to_md(t[[c for c in COLS[1:] if c in t.columns]].astype(float)))


def stage_test() -> None:
    assert FROZEN.exists(), "run `select` first"
    t, daily = run_all("test")
    t = v3.add_dsr(t, daily, v3.prior_trial_srs() + [0.0] * 7)
    t[[c for c in COLS if c in t.columns]].to_csv(OUT / "test_2022_2026.csv")
    pd.DataFrame(daily).to_parquet(OUT / "test_daily_returns.parquet")
    pick = FROZEN.read_text().split("PICK = ")[1].split("  (")[0]
    st, _ = run_all(
        "test", designs=[d for d in DESIGNS if d.name == pick], y=1.0
    )  # cost stress, not a trial
    st.to_csv(OUT / "test_pick_impact_Y1.csv")
    print(FROZEN.read_text())
    print(sv2.to_md(t[[c for c in COLS[1:] if c in t.columns]].astype(float)))
    print("\npick with impact Y = 1.0:")
    print(sv2.to_md(st[[c for c in COLS[1:] if c in st.columns]].astype(float)))


if __name__ == "__main__":
    {"select": stage_select, "test": stage_test}[sys.argv[1] if len(sys.argv) > 1 else "select"]()
