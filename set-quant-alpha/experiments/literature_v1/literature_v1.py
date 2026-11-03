"""literature_v1 - published market-beating strategies, applied to Thai stocks with the papers' own parameters.

    uv run set-quant-alpha/experiments/literature_v1/literature_v1.py select    # 2013-2021, freezes the pick
    uv run set-quant-alpha/experiments/literature_v1/literature_v1.py test      # one shot, 2022-01 -> 2026-10

Hypotheses (pre-registered 2026-11-02, before any simulation; parameters from the papers, not tuned):
  L1 low beta          Frazzini & Pedersen (2014); Thai low-risk anomaly (AAMJAF 2017, SET 2004-15)
                       score = -beta to the market (120-day beta_mkt, known at t)
  L2 low volatility    Blitz & van Vliet (2007); Blitz, Pang & van Vliet (2012, emerging markets)
                       score = -volatility of daily returns over the last 250 trading days
  L3 residual momentum Blitz, Huij & Martens (2011)
                       score = sum of daily residual returns over months t-12..t-1 / their std (skip the last month)
  L4 52-week high      George & Hwang (2004)
                       score = price / highest close of the last 250 trading days
  L5 multi-factor      average percentile rank of L2, L3 and L4 (a standard composite)
  L6 L5 + ML           average percentile rank of L5 and the walk-forward ML score (pred_ens)

Portfolio rules, the same for every design (Novy-Marx & Velikov 2016: low turnover + a buy/hold spread):
  universe = liquidity top-100 (tradable); long the top quintile (20 names), equal weight; rebalance every 21
  trading days; a held name stays while it ranks in the top 40; execution at the next open, 100-share lots,
  0.15% per side, THB 1M, profits reinvested.
Selection rule: the highest 2013-2021 worst-case net alpha t-stat across CAPM (TDEX), local-Thai Carhart and
Fama-French EM Carhart. The pick is frozen before the test run; 2022-26 is reported once for every design.
Caveat: we already knew 2022-26 was a megacap / low-risk-friendly market when choosing which papers to test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import config, events

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "strategy_v3"))
sys.path.insert(0, str(HERE.parent / "strategy_v2"))
import sv2  # noqa: E402  (read-only reuse: stats, attribution)
import v3  # noqa: E402  (read-only reuse: data setup on the extended 2013-2026 frame)

FROZEN = OUT / "picks_frozen_before_test.txt"
CAPITAL = config.INITIAL_CAPITAL
N_HOLD, BUFFER, EVERY, UNIVERSE = 20, 40, 21, 100
FEE = config.FEE_PER_SIDE


@dataclass(frozen=True)
class Design:
    name: str
    score: str
    pickable: bool = True


DESIGNS = [
    Design("L1 low beta", "s_lowbeta"),
    Design("L2 low volatility", "s_lowvol"),
    Design("L3 residual momentum", "s_resmom"),
    Design("L4 52-week high", "s_hi52"),
    Design("L5 multi-factor (L2+L3+L4)", "s_multi"),
    Design("L6 multi-factor + ML score", "s_multi_ml"),
]


def add_scores(f: pd.DataFrame, inp: dict) -> pd.DataFrame:
    """Literature scores from prices (all known at the close of t)."""
    px = inp["prices"]
    C = px.pivot_table(index="Date", columns="Ticker", values="Close")
    r = C.pct_change(fill_method=None)
    r = r.where(r.abs() < 0.45)
    vol = r.rolling(250, min_periods=200).std()
    resid = events.daily_alpha(inp["df"], px).reindex_like(C)
    win = resid.shift(21)  # skip the most recent month
    resmom = win.rolling(231, min_periods=180).sum() / win.rolling(231, min_periods=180).std()
    hi52 = C / C.rolling(250, min_periods=200).max()
    wide = pd.concat({"s_lowvol": -vol, "s_resmom": resmom, "s_hi52": hi52}, axis=1).stack(
        future_stack=True
    )
    wide.index.names = ["Date", "Ticker"]
    f = f.merge(wide.reset_index(), on=["Date", "Ticker"], how="left")
    f["s_lowbeta"] = -f["beta_mkt"]
    g = f.groupby("Date")
    rk = {c: g[c].rank(pct=True) for c in ["s_lowvol", "s_resmom", "s_hi52", "pred_ens"]}
    f["s_multi"] = (rk["s_lowvol"] + rk["s_resmom"] + rk["s_hi52"]) / 3
    f["s_multi_ml"] = (f.groupby("Date")["s_multi"].rank(pct=True) + rk["pred_ens"]) / 2
    return f


def targets(day: pd.DataFrame, score: str, held: list[str]) -> pd.Series:
    u = day[(day["lrank"] <= UNIVERSE)].dropna(subset=[score]).sort_values(score, ascending=False)
    ranked = list(u["Ticker"])
    keep = [t for t in ranked[:BUFFER] if t in held][:N_HOLD]
    names = keep + [t for t in ranked if t not in keep][: N_HOLD - len(keep)]
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


def simulate(
    des: Design,
    f: pd.DataFrame,
    px: dict,
    end: str | None = None,
    fee: float = FEE,
    replay: dict | None = None,
) -> v3.Run:
    O, C = px["Open"], px["Close"].ffill()  # noqa: E741
    last = pd.Timestamp(end) if end else f["Date"].max()
    dates = [d for d in C.index if f["Date"].min() <= d <= last]
    by_day = {d: g for d, g in f.groupby("Date")}
    cash, sh, pending, eq, trades, stock_w, tgt = CAPITAL, {}, None, {}, [], {}, {}
    for i, d in enumerate(dates):
        if pending is not None:
            value = cash + sum(
                q * (O.at[d, t] if pd.notna(O.at[d, t]) else C.at[d, t]) for t, q in sh.items()
            )
            orders = []
            for t in set(sh) | set(pending.index):
                p = O.at[d, t] if t in O.columns and pd.notna(O.at[d, t]) else np.nan
                if pd.isna(p) or p <= 0:
                    continue
                dq = int(pending.get(t, 0.0) * value * 0.998 / p / sv2.LOT) * sv2.LOT - sh.get(t, 0)
                if dq and (abs(dq * p) >= 0.01 * value or t not in pending.index):
                    orders.append((dq, t, p))
            for dq, t, p in sorted(orders):  # sells first
                cash -= dq * p + abs(dq * p) * fee
                sh[t] = sh.get(t, 0) + dq
                trades.append((d, t, dq, p, "rebalance"))
            sh = {t: q for t, q in sh.items() if q}
            pending = None
        value = cash + sum(q * C.at[d, t] for t, q in sh.items() if pd.notna(C.at[d, t]))
        eq[d] = value
        stock_w[d] = (value - cash) / value
        if i % EVERY == 0 and d in by_day and i < len(dates) - 1:
            pending = (
                replay.get(d) if replay is not None else targets(by_day[d], des.score, list(sh))
            )
            tgt[d] = pending
    return v3.Run(pd.Series(eq), trades, stock_w, {d: 0.0 for d in eq}, tgt, [], 0)


def run_all(split: str, end: str | None = None) -> tuple[pd.DataFrame, dict]:
    inp, f, px, _, _ = v3.setup(split)
    f = add_scores(f, inp)
    ew = sv2.universe_ew(px["Close"], f)
    rows, daily = {}, {}
    for des in DESIGNS:
        net = simulate(des, f, px, end)
        gross = simulate(des, f, px, end, fee=0.0, replay=net.targets)
        b = sv2.bench_returns(inp["bench"], net.equity.index)
        st = sv2.stats(v3.as_sim(des.name, net), v3.as_sim(des.name, gross), inp, b, ew)
        g_only = sv2.attribution(gross.equity.pct_change().dropna(), inp)
        st["gross TH alpha t"], st["gross C4 alpha t"] = (
            g_only.get("TH alpha t"),
            g_only.get("C4 alpha t"),
        )
        st["worst net alpha t"] = min(
            st.get("CAPM alpha t", np.nan),
            st.get("TH alpha t", np.nan),
            st.get("C4 alpha t", np.nan),
        )
        rows[des.name] = {"pickable": des.pickable, **st}
        daily[des.name] = net.equity.pct_change().dropna()
        print(
            f"  {des.name}: CAGR {st['CAGR %']:.2f} (TDEX {st['TDEX CAGR %']:.2f}), worst net alpha t {st['worst net alpha t']:.2f}",
            flush=True,
        )
    return pd.DataFrame(rows).T, daily


COLS = [
    "CAGR %",
    "TDEX CAGR %",
    "gross CAGR %",
    "cost drag %",
    "turnover 2-way x/yr",
    "Sharpe",
    "max DD %",
    "IR vs TDEX",
    "CAPM alpha %",
    "CAPM alpha t",
    "CAPM beta",
    "FF3 alpha %",
    "FF3 alpha t",
    "C4 alpha %",
    "C4 alpha t",
    "TH alpha %",
    "TH alpha t",
    "TH b_SMB",
    "TH b_WML",
    "gross TH alpha t",
    "gross C4 alpha t",
    "worst net alpha t",
    "DSR",
]


def stage_select() -> None:
    t, daily = run_all("sel", end=v3.SEL_END)
    t = v3.add_dsr(t, daily, v3.prior_trial_srs() + [0.0] * 11)
    t[[c for c in COLS if c in t.columns]].to_csv(OUT / "select_2013_2021.csv")
    pick = t["worst net alpha t"].astype(float).idxmax()
    FROZEN.write_text(
        f"frozen {datetime.now(UTC).isoformat(timespec='seconds')} (before any literature_v1 test simulation)\n"
        "rule = highest 2013-2021 worst-case net alpha t across CAPM, local-Thai Carhart and FF-EM Carhart\n"
        f"PICK = {pick}  (worst net alpha t {float(t.at[pick, 'worst net alpha t']):.2f})\n"
        f"trials = 71 earlier strategy designs + {len(DESIGNS)} here\n"
    )
    print(FROZEN.read_text())
    print(sv2.to_md(t[[c for c in COLS if c in t.columns]].astype(float)))


def stage_test() -> None:
    assert FROZEN.exists(), "run `select` first"
    t, daily = run_all("test")
    t = v3.add_dsr(t, daily, v3.prior_trial_srs() + [0.0] * 11)
    t[[c for c in COLS if c in t.columns]].to_csv(OUT / "test_2022_2026.csv")
    pd.DataFrame(daily).to_parquet(OUT / "test_daily_returns.parquet")
    print(FROZEN.read_text())
    print(sv2.to_md(t[[c for c in COLS if c in t.columns]].astype(float)))


if __name__ == "__main__":
    {"select": stage_select, "test": stage_test}[sys.argv[1] if len(sys.argv) > 1 else "select"]()
