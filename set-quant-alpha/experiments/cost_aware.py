"""Cost-aware / benchmark-relative portfolio construction experiments.

Goal: can the existing `pred_ens` signal beat the TDEX (SET50 ETF) benchmark after costs
with a better portfolio-construction / execution design?  No model is retrained.

Protocol (see REPORT.md):
  1. Every configuration in `grid()` is simulated on the VALIDATION split (2020-2021) only.
  2. Picks are made by rules on validation results only:
       A = highest validation information ratio vs TDEX (all pickable trials);
       B = highest validation "net signal IR" (strategy minus the equal-weight return of its
           own universe, costs included), among full-sleeve equal-weight configurations;
       C = pick A wrapped as core-satellite: sleeve = min(1, 4% / validation TE of A),
           the rest held in TDEX (an enhanced-index risk budget);
     A-C were coded before any result was seen. D (style-controlled: best validation IR among
     configs whose holdings match E2's idio vol and liquidity rank within 10%) was added after
     the first validation pass, before any test run.
  3. A-D (plus the published E2 baseline as reference) are run ONCE on the test split.
  4. DSR deflates by the number of validation trials (37), using the dispersion of the
     validation Sharpe ratios (absolute) and information ratios (active vs TDEX).

Run:  .venv/bin/python set-quant-alpha/experiments/cost_aware.py          (~30 seconds)
      add --valid-only to stop before the test split.
Outputs: set-quant-alpha/experiments/output/*.csv and *.md tables.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from setalpha import config, evaluate, ingest, model, report, strategy, transform

log = logging.getLogger("cost_aware")
OUT = Path(__file__).resolve().parent / "output"
LOT = strategy.LOT
TDEX_FEE = strategy.HEDGE_FEE  # 0.05 % per side on TDEX trades
TE_BUDGET = 0.04  # pick C: annualised tracking-error budget vs TDEX
DAYS = 252


# ============================================================================ config
@dataclass(frozen=True)
class CAConfig:
    name: str
    family: str = "F1"
    score: str = "pred_ens"  # or "pred_sm5" (5-day smoothed rank)
    universe: tuple[int, int] = (
        1,
        100,
    )  # liquidity-rank range (dvol60 rank among liquid stocks); (1,100) == `tradable`
    uni_slack: int = 0  # held names stay eligible while rank <= hi + slack (universe hysteresis)
    n: int = 10  # number of slots in the stock sleeve
    select: str = "topn"  # "topn" | "buffer" | "hurdle" | "dvol_tilt"
    buffer: int = 30  # buffer mode: keep a holding while it ranks within the top `buffer`
    hurdle: float = 1.0  # hurdle mode: trade only if alpha gain > hurdle x round-trip cost
    vol_scaled: bool = (
        True  # hurdle mode: Grinold alpha uses each name's idio vol (False: median vol)
    )
    pickable: bool = True  # False = diagnostic trial (counted, but excluded from the pick pools)
    tilt: float = 0.5  # dvol_tilt: weight ~ dvol60 * max(0, 1 + tilt * z)
    rebalance_every: int = config.HORIZON  # trading days between decisions
    sleeve: float = 1.0  # weight of the stock sleeve; 1 - sleeve is held in TDEX (core)
    equitize: bool = False  # park unused slots / lot residue in TDEX instead of cash
    stop_loss: float | None = 0.08
    min_trade_frac: float = 0.01  # skip trades smaller than 1% of equity (as in exact_sim)
    resize_band: float = 0.0  # skip resizing a kept name unless |w - target| > band * target


# ============================================================================ data
def load_inputs() -> dict:
    df = transform.load()
    preds = pd.read_parquet(model.PREDICTIONS)
    raw = ingest.ingest_all("file")
    meta = report.load_meta()
    liq = df.loc[df["liquid"], ["Date", "Ticker", "dvol60"]].copy()
    liq["lrank"] = liq.groupby("Date")["dvol60"].rank(ascending=False, method="first")
    preds = preds.sort_values(["Ticker", "Date"]).copy()
    # 5-day smoothed score: rolling mean of the daily percentile rank (past data only), re-ranked
    preds["pct"] = preds.groupby("Date")["pred_ens"].rank(pct=True)
    preds["sm"] = preds.groupby("Ticker")["pct"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    preds["pred_sm5"] = preds.groupby("Date")["sm"].rank(pct=True)
    bench = (
        raw["market"][raw["market"]["Ticker"] == config.BENCHMARK].set_index("Date").sort_index()
    )
    return {
        "df": df,
        "preds": preds,
        "liq": liq[["Date", "Ticker", "lrank"]],
        "prices": raw["prices"],
        "market": raw["market"],
        "bench": bench,
        "ic": meta["ic_valid"],
    }


def make_frame(inp: dict, split: str) -> pd.DataFrame:
    """Candidate rows for one split. Unlike strategy.sim_frame, rows are NOT filtered on a known
    forward return (that filter silently drops names about to be suspended: a small look-ahead)."""
    cols = ["Date", "Ticker", "tradable", "dvol60", "idio_vol", "beta_mkt"]
    p = inp["preds"][inp["preds"]["split"] == split][["Date", "Ticker", "pred_ens", "pred_sm5"]]
    f = p.merge(inp["df"][cols], on=["Date", "Ticker"], how="left").merge(
        inp["liq"], on=["Date", "Ticker"], how="left"
    )
    return f[f["lrank"].notna()].reset_index(drop=True)


def price_panel(inp: dict, f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    start, end = f["Date"].min() - pd.Timedelta(days=250), f["Date"].max() + pd.Timedelta(days=10)
    p = inp["prices"]
    return strategy.wide_prices(
        p[(p["Date"] >= start) & (p["Date"] <= end) & p["Ticker"].isin(f["Ticker"].unique())]
    )


# ============================================================================ decisions
def _eligible(cfg: CAConfig, day: pd.DataFrame, held: list[str]) -> pd.DataFrame:
    lo, hi = cfg.universe
    if cfg.universe == (1, 100):
        base = day["tradable"].fillna(False).astype(bool)
    else:
        base = (day["lrank"] >= lo) & (day["lrank"] <= hi)
    keep = (
        day["Ticker"].isin(held)
        & (day["lrank"] >= max(1, lo - cfg.uni_slack))
        & (day["lrank"] <= hi + cfg.uni_slack)
    )
    return day[base | keep]


def grinold_alpha(day: pd.DataFrame, score: str, ic: float, vol_scaled: bool = True) -> pd.Series:
    """5-day expected residual return: IC x sigma_5d x normal score of the cross-sectional rank."""
    z = norm.ppf((day[score].rank() - 0.5) / len(day))
    vol = day["idio_vol"].fillna(day["idio_vol"].median())
    if not vol_scaled:
        vol = pd.Series(vol.median(), index=vol.index)
    return pd.Series(
        ic * vol.to_numpy() * np.sqrt(config.HORIZON) * z, index=day["Ticker"].to_numpy()
    )


def decide(
    cfg: CAConfig, day: pd.DataFrame, held: list[str], ic: float, fee: float
) -> tuple[pd.Series, float]:
    """Return (stock target weights, TDEX target weight) for one decision date."""
    el = _eligible(cfg, day, held)
    if cfg.select == "dvol_tilt":
        z = pd.Series(
            norm.ppf((el[cfg.score].rank() - 0.5) / len(el)), index=el["Ticker"].to_numpy()
        )
        w = el.set_index("Ticker")["dvol60"] * np.maximum(0.0, 1 + cfg.tilt * z)
        w = (w / w.sum()).clip(upper=0.10)
        w = w / w.sum() * cfg.sleeve
        return w[w > 0.005], 1.0 - float(w[w > 0.005].sum())
    ranked = list(el.sort_values(cfg.score, ascending=False)["Ticker"])
    if cfg.select == "topn":
        names = list(el.nlargest(cfg.n, cfg.score)["Ticker"])  # same tie-breaking as exact_sim
    elif cfg.select == "buffer":  # identical rule to strategy.target_weights
        keep = [t for t in held if t in ranked[: cfg.buffer]][: cfg.n]
        names = keep + [t for t in ranked if t not in keep][: cfg.n - len(keep)]
    elif cfg.select == "hurdle":
        a = grinold_alpha(el, cfg.score, ic, cfg.vol_scaled)
        c = 2 * fee * cfg.hurdle  # round-trip cost scaled by the hurdle multiple
        book = [
            t for t in held if t in a.index and a[t] > -fee * cfg.hurdle
        ]  # sell only clear losers
        cands = [t for t in a.sort_values(ascending=False).index if t not in book]
        for t in cands:  # fill empty slots: entry must pay its round trip
            if len(book) >= cfg.n or a[t] <= c:
                break
            book.append(t)
        cands = [t for t in cands if t not in book]
        for t in cands:  # swaps: alpha gain must exceed the round trip
            worst = min(book, key=lambda x: a[x]) if book else None
            if worst is None or a[t] - a[worst] <= c:
                break
            book.remove(worst)
            book.append(t)
        names = book
    else:
        raise ValueError(cfg.select)
    slot = cfg.sleeve / cfg.n
    w = pd.Series(slot, index=names, dtype=float)
    if cfg.select != "hurdle" and len(names) and not cfg.equitize and cfg.sleeve == 1.0:
        w = pd.Series(
            1.0 / len(names), index=names
        )  # exact_sim convention: fully invest the chosen names
    tdex = (1.0 - float(w.sum())) if (cfg.equitize or cfg.sleeve < 1.0) else 0.0
    if not cfg.equitize and cfg.sleeve < 1.0:
        tdex = 1.0 - cfg.sleeve
    return w, tdex


# ============================================================================ simulator
@dataclass
class SimResult:
    name: str
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    tdex_weight: pd.Series
    holdings: pd.DataFrame = (
        None  # per decision: mean liquidity rank / idio vol / beta of the target names
    )

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def ca_sim(
    cfg: CAConfig,
    f: pd.DataFrame,
    px: dict,
    bench_px: pd.DataFrame,
    ic: float,
    fee: float = config.FEE_PER_SIDE,
    tdex_fee: float = TDEX_FEE,
    capital: float = config.INITIAL_CAPITAL,
    decision_fee: float = config.FEE_PER_SIDE,
) -> SimResult:
    """Day-by-day simulator. With select in {topn, buffer}, sleeve=1, no equitize, no band it reproduces
    strategy.exact_sim exactly (checked in `replication_check`). Additions: TDEX core / cash
    equitisation (100-unit lots, 0.05% per side), alpha-hurdle selection, resize band, universes."""
    O, L, C = px["Open"], px["Low"], px["Close"].ffill()  # noqa: E741
    bO = bench_px["Open"].reindex(C.index)
    bC = bench_px["Close"].reindex(C.index).ffill()
    dates = [d for d in C.index if d >= f["Date"].min()]
    by_day = {d: g for d, g in f.groupby("Date")}
    cash, shares, entry, tunits = capital, {}, {}, 0
    pending, trades, equity, expo, tw, hold = None, [], {}, {}, {}, []

    def trade(tkr, qty, price, day, reason, cost=fee):
        nonlocal cash
        cash -= qty * price + abs(qty * price) * cost
        trades.append((day, tkr, qty, price, reason))

    for i, d in enumerate(dates):
        if pending is not None:
            targets, tdex_w = pending
            b_open = bO.get(d, np.nan) if pd.notna(bO.get(d, np.nan)) else bC.get(d, np.nan)
            eq_open = (
                cash
                + sum(
                    q * (O.at[d, t] if pd.notna(O.at[d, t]) else C.at[d, t])
                    for t, q in shares.items()
                )
                + tunits * (b_open if pd.notna(b_open) else 0.0)
            )
            for t in list(shares):  # sells first
                if t not in targets.index and pd.notna(O.at[d, t]):
                    trade(t, -shares.pop(t), O.at[d, t], d, "rebalance-out")
                    entry.pop(t, None)
            if pd.notna(b_open) and b_open > 0:  # TDEX sells before stock buys
                want_u = int(tdex_w * eq_open / b_open / LOT) * LOT
                if (
                    want_u < tunits
                    and abs((want_u - tunits) * b_open) >= cfg.min_trade_frac * eq_open
                ):
                    trade(config.BENCHMARK, want_u - tunits, b_open, d, "tdex", tdex_fee)
                    tunits = want_u
            for t, w in targets.items():  # buys / resizes
                price = O.at[d, t] if t in O.columns else np.nan
                if pd.isna(price) or price <= 0:
                    continue
                want = int(w * eq_open / price / LOT) * LOT
                cur = shares.get(t, 0)
                diff = want - cur
                if (
                    cur
                    and cfg.resize_band > 0
                    and abs(cur * price - w * eq_open) <= cfg.resize_band * w * eq_open
                ):
                    continue
                if diff != 0 and abs(diff * price) >= cfg.min_trade_frac * eq_open:
                    trade(t, diff, price, d, "rebalance")
                    shares[t] = cur + diff
                    entry[t] = price if diff > 0 and t not in entry else entry.get(t, price)
                    if shares[t] == 0:
                        shares.pop(t), entry.pop(t, None)
            if pd.notna(b_open) and b_open > 0:  # TDEX buys last, with the cash left
                want_u = int(tdex_w * eq_open / b_open / LOT) * LOT
                spare = int(max(cash - 0.005 * eq_open, 0) / (b_open * (1 + tdex_fee)) / LOT) * LOT
                want_u = tunits + spare if cfg.equitize else min(want_u, tunits + spare)
                if (
                    want_u > tunits
                    and abs((want_u - tunits) * b_open) >= cfg.min_trade_frac * eq_open
                ):
                    trade(config.BENCHMARK, want_u - tunits, b_open, d, "tdex", tdex_fee)
                    tunits = want_u
            pending = None
        if cfg.stop_loss:
            for t in list(shares):
                stop_px = entry[t] * (1 - cfg.stop_loss)
                if pd.notna(L.at[d, t]) and L.at[d, t] <= stop_px:
                    fill = min(O.at[d, t], stop_px) if pd.notna(O.at[d, t]) else stop_px
                    trade(t, -shares.pop(t), fill, d, "stop-loss")
                    entry.pop(t)
        long_val = sum(q * C.at[d, t] for t, q in shares.items())
        tval = tunits * bC.get(d, 0.0)
        equity[d] = cash + long_val + tval
        expo[d] = long_val / equity[d] if equity[d] else 0.0
        tw[d] = tval / equity[d] if equity[d] else 0.0
        if i % cfg.rebalance_every == 0 and i < len(dates) - 1 and d in by_day:
            pending = decide(
                cfg, by_day[d], list(shares), ic, decision_fee
            )  # rule fixed at 0.15%; `fee` is what is charged
            h = by_day[d].set_index("Ticker").reindex(pending[0].index)
            hold.append((d, len(h), h["lrank"].mean(), h["idio_vol"].mean(), h["beta_mkt"].mean()))
    tr = pd.DataFrame(trades, columns=["Date", "Ticker", "qty", "price", "reason"])
    hd = pd.DataFrame(hold, columns=["Date", "names", "lrank", "idio_vol", "beta_mkt"])
    return SimResult(
        cfg.name, pd.Series(equity, name=cfg.name), tr, pd.Series(expo), pd.Series(tw), hd
    )


# ============================================================================ metrics
def bench_returns(bench: pd.DataFrame, idx: pd.Index) -> pd.Series:
    return bench["Close"].reindex(idx).ffill().pct_change()


def universe_ew(px_close: pd.DataFrame, f: pd.DataFrame, universe: tuple[int, int]) -> pd.Series:
    """Equal-weight daily return (no costs) of a liquidity-rank universe, previous-day membership."""
    r = px_close.pct_change(fill_method=None)
    lo, hi = universe
    m = (
        f["tradable"].fillna(False).astype(bool)
        if universe == (1, 100)
        else f["lrank"].between(lo, hi)
    )
    mem = (
        f[m]
        .assign(v=1.0)
        .pivot_table(index="Date", columns="Ticker", values="v")
        .reindex(index=r.index, columns=r.columns)
    )
    return r.where(mem.shift(1).notna()).mean(axis=1)


def stats(res: SimResult, bench_r: pd.Series, own_bench: pd.Series | None = None) -> dict:
    r = res.returns
    b = bench_r.reindex(r.index)
    s = evaluate.perf_stats(r, b)
    act = (r - b).dropna()
    years = len(r) / DAYS
    stock_tr = (
        res.trades[res.trades["Ticker"] != config.BENCHMARK] if len(res.trades) else res.trades
    )
    traded = float((stock_tr["qty"].abs() * stock_tr["price"]).sum()) if len(stock_tr) else 0.0
    wk = lambda x: (1 + x).resample("W-FRI").prod() - 1  # noqa: E731
    out = {
        "CAGR %": s["CAGR %"],
        "TDEX CAGR %": 100 * ((1 + b.dropna()).prod() ** (DAYS / len(b.dropna())) - 1),
        "vol %": s["vol %"],
        "Sharpe": s["Sharpe"],
        "max DD %": s["max DD %"],
        "TE %": 100 * act.std() * np.sqrt(DAYS),
        "TE weekly %": 100 * (wk(r) - wk(b)).std() * np.sqrt(52),
        "info ratio": s["info ratio"],
        "alpha % p.a.": s["alpha % p.a."],
        "alpha t": s["alpha t-stat"],
        "beta": s["beta"],
        "PSR": s["PSR (SR>0)"],
        "PSR active": evaluate.psr(act),
        "turnover x/yr (one-way)": traded / 2 / res.equity.mean() / years,
        "stock exposure %": 100 * res.exposure.mean(),
        "TDEX weight %": 100 * res.tdex_weight.mean(),
        "trades": len(stock_tr),
        "sr_daily": float(r.mean() / r.std()),
        "ir_daily": float(act.mean() / act.std()),
        "avg names": res.holdings["names"].mean(),
        "avg liq rank": res.holdings["lrank"].mean(),
        "avg idio vol %": 100 * res.holdings["idio_vol"].mean(),
    }
    out["active vs TDEX % p.a."] = 100 * act.mean() * DAYS
    if own_bench is not None:
        ew = own_bench.reindex(r.index)
        sig = (r - ew).dropna()
        out["signal IR (vs own-universe EW)"] = float(sig.mean() / sig.std() * np.sqrt(DAYS))
        out["= vs own-universe EW % p.a."] = 100 * sig.mean() * DAYS  # signal net of costs
        out["+ own-universe EW - TDEX % p.a."] = (
            100 * (ew - b).dropna().mean() * DAYS
        )  # size / style
    return out


# ============================================================================ grid (fixed before any run)
def grid() -> list[CAConfig]:
    E1 = CAConfig("F0 E1 replica: equal top-10, weekly", "F0", select="topn", stop_loss=None)
    E2 = CAConfig("F0 E2 replica: top-10 + buffer 30 + 8% stop", "F0", select="buffer", buffer=30)
    H = CAConfig("hurdle 1.0", "F1", select="hurdle", hurdle=1.0)
    cfgs = [E1, E2]
    # F1 execution / cost control on the tradable top-100
    cfgs += [replace(E2, name=f"F1 buffer {b}", family="F1", buffer=b) for b in (20, 40, 50)]
    cfgs += [replace(H, name=f"F1 hurdle {h}", hurdle=h) for h in (0.5, 1.0, 1.5, 2.0)]
    cfgs += [
        replace(E2, name="F1 buffer 30 + resize band 25%", family="F1", resize_band=0.25),
        replace(H, name="F1 hurdle 1.0 + resize band 25%", resize_band=0.25),
        replace(E2, name="F1 buffer 30, no stop", family="F1", stop_loss=None),
        replace(H, name="F1 hurdle 1.0, no stop", stop_loss=None),
        replace(E2, name="F1 buffer 30 + equitize (idle cash in TDEX)", family="F1", equitize=True),
        replace(H, name="F1 hurdle 1.0 + equitize", equitize=True),
        replace(E2, name="F1 buffer 30, rebalance every 10d", family="F1", rebalance_every=10),
        replace(H, name="F1 hurdle 1.0, rebalance every 10d", rebalance_every=10),
        replace(E2, name="F1 buffer 30, 5d-smoothed score", family="F1", score="pred_sm5"),
        replace(E2, name="F1 top-20 + buffer 50", family="F1", n=20, buffer=50),
        replace(H, name="F1 hurdle 1.0, 20 slots", n=20),
        replace(E2, name="F1 top-5 + buffer 15", family="F1", n=5, buffer=15),
        replace(H, name="F1 hurdle 1.0, 5 slots", n=5),
        replace(E2, name="F1 buffer 30 + universe slack 25", family="F1", uni_slack=25),
        # decomposition diagnostic added after the first validation pass: is the hurdle gain the
        # cost rule or the idio-vol scaling of the alpha? Counted as a trial, never pickable.
        replace(
            H, name="F1 hurdle 1.0, z-only alpha (diagnostic)", vol_scaled=False, pickable=False
        ),
    ]
    # F2 universe choice (validation: alpha concentrated outside the top-30 by liquidity)
    for u in [(1, 50), (31, 150), (51, 150), (1, 150)]:
        cfgs += [
            replace(
                E2, name=f"F2 ranks {u[0]}-{u[1]}: buffer 30", family="F2", universe=u, uni_slack=25
            ),
            replace(
                H, name=f"F2 ranks {u[0]}-{u[1]}: hurdle 1.0", family="F2", universe=u, uni_slack=25
            ),
        ]
    # F3 core-satellite (TDEX core). IR is ~scale-invariant, included to verify that.
    cfgs += [
        replace(
            E2,
            name="F3 50% TDEX core + 50% buffer-30 sleeve",
            family="F3",
            sleeve=0.5,
            equitize=True,
        ),
        replace(
            H,
            name="F3 50% TDEX core + 50% hurdle-1.0 sleeve",
            family="F3",
            sleeve=0.5,
            equitize=True,
        ),
    ]
    # F4 enhanced index: traded-value-weighted large caps tilted by the score
    cfgs += [
        CAConfig(
            "F4 dvol-weighted ranks 1-30, tilt 0.5",
            "F4",
            select="dvol_tilt",
            universe=(1, 30),
            uni_slack=10,
            tilt=0.5,
            stop_loss=None,
            resize_band=0.25,
            equitize=True,
        ),
        CAConfig(
            "F4 dvol-weighted ranks 1-50, tilt 1.0",
            "F4",
            select="dvol_tilt",
            universe=(1, 50),
            uni_slack=10,
            tilt=1.0,
            stop_loss=None,
            resize_band=0.25,
            equitize=True,
        ),
    ]
    return cfgs


# ============================================================================ checks
def replication_check(inp: dict) -> pd.DataFrame:
    """ca_sim must reproduce strategy.exact_sim on the original sim_frame (validation)."""
    f0 = strategy.sim_frame(
        inp["df"], inp["preds"].drop(columns=["pct", "sm", "pred_sm5"]), inp["market"], "valid"
    )
    f0["lrank"] = 1.0
    px = price_panel(inp, f0)
    rows = {}
    for ex, ca in [
        (strategy.ExactConfig("E1"), CAConfig("E1", select="topn", stop_loss=None)),
        (
            strategy.ExactConfig("E2", buffer=30, stop_loss=0.08),
            CAConfig("E2", select="buffer", buffer=30),
        ),
    ]:
        a = strategy.exact_sim(ex, f0, px, inp["bench"], inp["ic"]).equity
        b = ca_sim(ca, f0, px, inp["bench"], inp["ic"]).equity
        rows[ex.name] = {
            "exact_sim final": a.iloc[-1],
            "ca_sim final": b.iloc[-1],
            "max abs diff": float((a - b).abs().max()),
        }
    return pd.DataFrame(rows).T


# ============================================================================ main
def run_split(cfgs, inp, split, fee=config.FEE_PER_SIDE):
    f = make_frame(inp, split)
    px = price_panel(inp, f)
    res = {c.name: ca_sim(c, f, px, inp["bench"], inp["ic"], fee=fee) for c in cfgs}
    idx = next(iter(res.values())).equity.index
    b = bench_returns(inp["bench"], idx)
    ew = {u: universe_ew(px["Close"], f, u) for u in {c.universe for c in cfgs}}
    rows = {
        c.name: {
            "family": c.family,
            **stats(
                res[c.name],
                b,
                ew[c.universe] if c.sleeve == 1.0 and c.select != "dvol_tilt" else None,
            ),
        }
        for c in cfgs
    }
    return pd.DataFrame(rows).T, res, b, ew


def to_md(t: pd.DataFrame, digits: int = 2) -> str:
    """Minimal markdown table (no tabulate dependency)."""
    fmt = lambda x: f"{x:.{digits}f}" if isinstance(x, float | np.floating) else str(x)  # noqa: E731
    idx_name = " / ".join(str(n) for n in t.index.names if n) or ""
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    lines = ["| " + " | ".join([idx_name] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, row in t.iterrows():
        i = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        lines.append("| " + " | ".join([i] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def md(t: pd.DataFrame, cols: list[str], digits: int = 2) -> str:
    return to_md(t[cols], digits)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("setalpha").setLevel(logging.WARNING)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    inp = load_inputs()
    rep = replication_check(inp)
    log.info("replication check (validation):\n%s", rep.to_string())
    assert (rep["max abs diff"] < 1e-6).all(), "ca_sim does not reproduce exact_sim"

    # ---------------------------------------------------------------- 1) validation grid
    cfgs = grid()
    val, _, _, _ = run_split(cfgs, inp, "valid")
    n_trials = len(cfgs)
    log.info("validation: %d configurations in %.0fs", n_trials, time.time() - t0)

    # ---------------------------------------------------------------- 2) pre-declared picks
    by_name = {c.name: c for c in cfgs}
    pickable = pd.Series([by_name[k].pickable for k in val.index], index=val.index)
    pick_a = val.loc[pickable, "info ratio"].astype(float).idxmax()
    pool_b = val[(val["family"].isin(["F0", "F1", "F2"])) & pickable]
    pool_b = pool_b[
        [by_name[k].sleeve == 1.0 and by_name[k].select != "dvol_tilt" for k in pool_b.index]
    ]
    pick_b = pool_b["signal IR (vs own-universe EW)"].astype(float).idxmax()
    a_cfg = by_name[pick_a]
    s_c = float(min(1.0, TE_BUDGET / (float(val.loc[pick_a, "TE %"]) / 100) * a_cfg.sleeve))
    c_cfg = replace(
        a_cfg,
        name=f"C: pick A as {100 * s_c:.0f}% sleeve + TDEX core (4% TE budget)",
        family="C",
        sleeve=round(s_c, 2),
        equitize=True,
    )
    c_val, _, _, _ = run_split([c_cfg], inp, "valid")
    val = pd.concat([val, c_val])
    n_trials += 1
    val["pick"] = ""
    val.loc[pick_a, "pick"] += "A"
    val.loc[pick_b, "pick"] += "B"
    val.loc[c_cfg.name, "pick"] += "C"
    # D (defined after the first validation pass, BEFORE any test run): style-controlled pick.
    # Best validation IR among pickable full-sleeve configs whose holdings have the same idio vol and
    # liquidity rank as the E2 baseline (+-10%), i.e. no size / volatility tilt vs the published design.
    e2 = val.loc["F0 E2 replica: top-10 + buffer 30 + 8% stop"]
    pool_d = val[
        pickable.reindex(val.index).fillna(False).astype(bool)
        & val.index.map(
            lambda k: k in by_name and by_name[k].sleeve == 1.0 and by_name[k].select != "dvol_tilt"
        )
        & ((val["avg idio vol %"].astype(float) / float(e2["avg idio vol %"]) - 1).abs() <= 0.10)
        & ((val["avg liq rank"].astype(float) / float(e2["avg liq rank"]) - 1).abs() <= 0.10)
    ]
    pick_d = pool_d["info ratio"].astype(float).idxmax()
    val.loc[pick_d, "pick"] += "D"
    trial_sr = val["sr_daily"].astype(float).tolist()
    trial_ir = val["ir_daily"].astype(float).tolist()
    picks = {"A": a_cfg, "B": by_name[pick_b], "C": c_cfg, "D": by_name[pick_d]}
    log.info("picks: A=%s | B=%s | C=%s | D=%s", pick_a, pick_b, c_cfg.name, pick_d)

    if "--valid-only" in sys.argv:
        val.to_csv(OUT / "validation_all.csv")
        pd.set_option("display.width", 250, "display.max_columns", 30)
        log.info(
            "\n%s", val.drop(columns=["family"]).astype(float, errors="ignore").round(3).to_string()
        )
        return
    # ---------------------------------------------------------------- 3) test: run once
    ref = by_name["F0 E2 replica: top-10 + buffer 30 + 8% stop"]
    test_cfgs = list({c.name: c for c in [*picks.values(), ref]}.values())
    test, tres, tb, _ = run_split(test_cfgs, inp, "test")
    labels = {n: "".join(k for k, c in picks.items() if c.name == n) for n in test.index}
    test.insert(0, "pick", [labels[n] or "ref (published E2)" for n in test.index])
    bs = evaluate.perf_stats(tb.dropna())
    test.loc[
        "TDEX buy & hold (benchmark)", ["pick", "CAGR %", "vol %", "Sharpe", "max DD %", "PSR"]
    ] = ["bench", bs["CAGR %"], bs["vol %"], bs["Sharpe"], bs["max DD %"], bs["PSR (SR>0)"]]
    for name in [n for n in test.index if n in tres]:
        r = tres[name].returns
        test.loc[name, "DSR"] = evaluate.dsr(r, trial_sr)
        test.loc[name, "DSR active"] = evaluate.dsr((r - tb.reindex(r.index)).dropna(), trial_ir)
    # in-sample DSR on validation for the picks
    vf = make_frame(inp, "valid")
    vpx = price_panel(inp, vf)
    for cfg in picks.values():
        r = ca_sim(cfg, vf, vpx, inp["bench"], inp["ic"]).returns
        b = bench_returns(inp["bench"], r.index)
        val.loc[cfg.name, "DSR"] = evaluate.dsr(r, trial_sr)
        val.loc[cfg.name, "DSR active"] = evaluate.dsr((r - b.reindex(r.index)).dropna(), trial_ir)

    # ---------------------------------------------------------------- 4) cost sensitivity (picks only)
    sens = []
    for fee in (0.0, 0.0010, 0.0015, 0.0020):
        for split in ("valid", "test"):
            t, _, _, _ = run_split(test_cfgs, inp, split, fee=fee)
            for name in t.index:
                sens.append(
                    {
                        "split": split,
                        "fee/side %": 100 * fee,
                        "config": name,
                        "CAGR %": t.loc[name, "CAGR %"],
                        "TDEX CAGR %": t.loc[name, "TDEX CAGR %"],
                        "Sharpe": t.loc[name, "Sharpe"],
                        "info ratio": t.loc[name, "info ratio"],
                        "alpha % p.a.": t.loc[name, "alpha % p.a."],
                    }
                )
    sens = pd.DataFrame(sens)

    # ---------------------------------------------------------------- 5) write
    val.to_csv(OUT / "validation_all.csv")
    test.to_csv(OUT / "test_picks.csv")
    sens.to_csv(OUT / "cost_sensitivity.csv", index=False)
    rep.to_csv(OUT / "replication_check.csv")
    pd.DataFrame(
        [{"pick": k, **{kk: str(vv) for kk, vv in asdict(v).items()}} for k, v in picks.items()]
    ).to_csv(OUT / "picks.csv", index=False)
    vcols = [
        "pick",
        "family",
        "CAGR %",
        "TDEX CAGR %",
        "Sharpe",
        "max DD %",
        "TE %",
        "info ratio",
        "signal IR (vs own-universe EW)",
        "alpha % p.a.",
        "beta",
        "turnover x/yr (one-way)",
        "stock exposure %",
    ]
    tcols = [
        "pick",
        "CAGR %",
        "TDEX CAGR %",
        "vol %",
        "Sharpe",
        "max DD %",
        "TE %",
        "TE weekly %",
        "info ratio",
        "alpha % p.a.",
        "alpha t",
        "beta",
        "turnover x/yr (one-way)",
        "stock exposure %",
        "TDEX weight %",
        "PSR",
        "PSR active",
        "DSR",
        "DSR active",
    ]
    dcols = [
        "pick",
        "active vs TDEX % p.a.",
        "= vs own-universe EW % p.a.",
        "+ own-universe EW - TDEX % p.a.",
        "signal IR (vs own-universe EW)",
        "avg names",
        "avg liq rank",
        "avg idio vol %",
    ]
    piv = sens.pivot_table(
        index=["split", "config"], columns="fee/side %", values=["CAGR %", "info ratio"]
    ).round(2)
    text = (
        f"# Generated tables ({n_trials} validation trials)\n\n## Validation (2020-2021)\n\n"
        + md(val, vcols)
        + "\n\n## Validation DSR of picks\n\n"
        + md(val[val["pick"] != ""], ["pick", "PSR", "PSR active", "DSR", "DSR active"], 3)
        + "\n\n## Test (2022-01 to 2026-10), run once\n\n"
        + md(test, tcols)
        + "\n\n## Test decomposition of active return vs TDEX (arithmetic, % p.a.)\n\n"
        + md(test.drop(index="TDEX buy & hold (benchmark)"), dcols)
        + "\n\n## Validation decomposition (picks + E2)\n\n"
        + md(val[(val["pick"] != "") | val.index.str.startswith("F0 E2")], dcols[1:] + ["pick"])
        + "\n\n## Cost sensitivity\n\n"
        + to_md(piv)
        + "\n"
    )
    (OUT / "tables.md").write_text(text)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    log.info("\nVALIDATION\n%s", val[vcols].round(2).to_string())
    log.info("\nTEST (picks, run once)\n%s", test[tcols].round(3).to_string())
    log.info("\nCOST SENSITIVITY\n%s", piv.to_string())
    log.info("done in %.0fs; tables in %s", time.time() - t0, OUT)


if __name__ == "__main__":
    sys.exit(main())
