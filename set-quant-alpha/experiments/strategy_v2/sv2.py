"""strategy_v2 library: inputs, a sleeve-based exact simulator, design grid and attribution metrics.

The simulator generalises `strategy.exact_sim` / `experiments/cost_aware.ca_sim` (same accounting:
decisions after the close every 5 trading days, execution at the next open, sells first, 100-share
lots, 0.15 %/side on stocks, 0.05 %/side on TDEX, per-name stop on the daily low, profits reinvested).
For a single equal-weight sleeve with no core it reproduces `strategy.exact_sim` exactly
(`replication_check`). Additions:
  * several sleeves in one book (each sleeve keeps its own held list for its buffer rule);
  * a TDEX core / remainder (cash not given to stock sleeves goes to TDEX when `tdex_remainder`);
  * stop cooldown (a stopped name cannot be re-bought for `cooldown` trading days);
  * volatility-scaled stops (2 x 5-day sigma, clipped 5-15 %);
  * ex-ante volatility management of the stock sleeve (Ledoit-Wolf, 120d), remainder in TDEX;
  * score variants: EMA-smoothed rank, momentum blend, V8 (20-day-target) score.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from setalpha import (
    config,
    evaluate,
    factors,
    ingest,
    model,
    portfolio,
    report,
    strategy,
    transform,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
LOT = strategy.LOT
TDEX_FEE = strategy.HEDGE_FEE
DAYS = 252
V8_PATH = OUT / "predictions_v8.parquet"


# ============================================================================ inputs
def load_inputs(
    preds_path: str | Path = model.PREDICTIONS, v8_path: str | Path | None = V8_PATH
) -> dict:
    df = transform.load()
    raw = ingest.ingest_all("file")
    preds = pd.read_parquet(preds_path)[["Date", "Ticker", "split", "pred_ens"]].copy()
    preds = preds.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    # EMA-smoothed score: per-ticker EWM (halflife 2 trading days) of the daily percentile rank, past data only
    preds["pct"] = preds.groupby("Date")["pred_ens"].rank(pct=True)
    preds["pred_ema"] = preds.groupby("Ticker")["pct"].transform(
        lambda x: x.ewm(halflife=2, adjust=False).mean()
    )
    preds = preds.drop(columns="pct")
    if v8_path is not None and Path(v8_path).exists():
        v8 = pd.read_parquet(v8_path)[["Date", "Ticker", "pred_ens"]].rename(
            columns={"pred_ens": "pred_v8"}
        )
        preds = preds.merge(v8, on=["Date", "Ticker"], how="left")
    liq = df.loc[df["liquid"], ["Date", "Ticker", "dvol60"]].copy()
    liq["lrank"] = liq.groupby("Date")["dvol60"].rank(ascending=False, method="first")
    bench = (
        raw["market"][raw["market"]["Ticker"] == config.BENCHMARK].set_index("Date").sort_index()
    )
    usdthb = (
        raw["market"][raw["market"]["Ticker"] == "THB=X"].set_index("Date").sort_index()["Close"]
    )
    local = factors.local_factors(df, raw["prices"], bench["Close"])
    return {
        "df": df,
        "preds": preds,
        "liq": liq[["Date", "Ticker", "lrank"]],
        "prices": raw["prices"],
        "market": raw["market"],
        "bench": bench,
        "ff": raw.get("fama_french"),
        "usdthb": usdthb,
        "local": local,
        "ic": report.load_meta()["ic_valid"],
        "preds_path": str(preds_path),
    }


def make_frame(inp: dict, split: str) -> pd.DataFrame:
    """All liquid candidate rows of one split (rows are NOT filtered on a known forward return)."""
    cols = [
        "Date",
        "Ticker",
        "tradable",
        "dvol60",
        "idio_vol",
        "beta_mkt",
        "vol_60",
        "resid_mom_120",
        "hmm_p_stressed",
        "vix_slope",
    ]
    score_cols = [c for c in ["pred_ens", "pred_ema", "pred_v8"] if c in inp["preds"].columns]
    p = inp["preds"][inp["preds"]["split"] == split][["Date", "Ticker"] + score_cols]
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


# ============================================================================ designs
@dataclass(frozen=True)
class Sleeve:
    frac: float = 1.0  # share of equity for this sleeve
    score: str = (
        "pred_ens"  # column, or "momblend" (0.75 rank(pred_ens) + 0.25 rank(resid_mom_120))
    )
    universe: tuple[int, int] = (1, 100)  # liquidity ranks; (1, 100) == `tradable`
    uni_slack: int = 0
    n: int = 10
    buffer: int | None = None  # None = plain top-n each decision
    stop: float | str | None = None  # 0.08 | "vol" | None
    cooldown: int = 0  # trading days a stopped name is barred from re-entry
    weighting: str = "equal"  # "equal" | "sqrt_dvol" (cap 20 %)
    every: int = 1  # full re-rank every k decisions; in between only refill empty slots


@dataclass(frozen=True)
class Design:
    name: str
    sleeves: tuple[Sleeve, ...]
    family: str = "D"
    pickable: bool = True
    composable: str | None = None  # tag used by the pre-registered combination rule
    vol_manage: float | None = None  # ex-ante vol target of the stock sleeves (remainder in TDEX)
    tdex_remainder: bool = False  # equity not given to stock sleeves is held in TDEX
    scale: float = 1.0  # multiplies every sleeve fraction (core-satellite wrapper)
    note: str = ""


E2S = Sleeve(buffer=30, stop=config.STOP_LOSS)


def grid(has_v8: bool) -> list[Design]:
    """Pre-registered design list (frozen before the first validation run)."""
    d = [
        # ---- E1 -> E2 ablation lattice (diagnostic; counted as trials, not pickable except E2 itself)
        Design("AB0 E1: top-10 weekly", (Sleeve(),), "AB", pickable=False),
        Design("AB1 E1 + 8% stop", (Sleeve(stop=0.08),), "AB", pickable=False),
        Design("AB2 E1 + 30-rank buffer", (Sleeve(buffer=30),), "AB", pickable=False),
        Design("AB3 E2: buffer 30 + 8% stop", (E2S,), "AB", pickable=True, note="incumbent"),
        # ---- single modifications of E2
        Design("D01 E2 + stop cooldown 10d", (replace(E2S, cooldown=10),), composable="cooldown"),
        Design(
            "D02 E2 broad: top-20 + buffer 50",
            (replace(E2S, n=20, buffer=50),),
            composable="breadth",
        ),
        Design(
            "D03 E2 + EMA(hl 2d) score", (replace(E2S, score="pred_ema"),), composable="score:ema"
        ),
        Design("D04 E2 + 25% residual-momentum blend score", (replace(E2S, score="momblend"),)),
        Design(
            "D05 50% E2 + 50% residual-momentum sleeve (4-weekly)",
            (replace(E2S, frac=0.5), Sleeve(frac=0.5, score="resid_mom_120", buffer=30, every=4)),
        ),
        Design(
            "D06 E2 + sqrt(traded value) weights",
            (replace(E2S, weighting="sqrt_dvol"),),
            composable="sqrtdvol",
        ),
        Design(
            "D07 ranks 31-150 + E2 rules + cooldown",
            (replace(E2S, universe=(31, 150), uni_slack=25, cooldown=10),),
        ),
        Design(
            "D08 70% TDEX + 30% best-5 of liquidity ranks 1-30",
            (Sleeve(frac=0.3, universe=(1, 30), uni_slack=5, n=5, buffer=10),),
            tdex_remainder=True,
        ),
        Design(
            "D09 E2 vol-managed to 15% ex-ante, remainder TDEX",
            (E2S,),
            vol_manage=config.TARGET_VOL,
            tdex_remainder=True,
        ),
        Design(
            "D10 E2 with vol-scaled stop (2 x 5d sigma) + cooldown",
            (replace(E2S, stop="vol", cooldown=10),),
            composable="volstop",
        ),
    ]
    if has_v8:
        d.append(
            Design(
                "D12 E2 rules on V8 (20d-target) score",
                (replace(E2S, score="pred_v8"),),
                composable="score:v8",
            )
        )
    return d


def combine(base: Design, mods: list[str], name: str) -> Design:
    """Apply composable modifications to the E2 sleeve (pre-registered combination rule)."""
    s = base.sleeves[0]
    for m in mods:
        if m == "cooldown":
            s = replace(s, cooldown=10)
        elif m == "breadth":
            s = replace(s, n=20, buffer=50)
        elif m == "score:ema":
            s = replace(s, score="pred_ema")
        elif m == "score:v8":
            s = replace(s, score="pred_v8")
        elif m == "sqrtdvol":
            s = replace(s, weighting="sqrt_dvol")
        elif m == "volstop":
            s = replace(s, stop="vol", cooldown=10)
    return Design(name, (s,), "D", note="combo of: " + ", ".join(mods))


def core_satellite(d: Design, s: float) -> Design:
    return replace(
        d,
        name=f"P-CS: pick as {100 * s:.0f}% sleeve + TDEX core (4% TE budget)",
        family="P-CS",
        scale=round(s, 3),
        tdex_remainder=True,
        pickable=False,
    )


# ============================================================================ decisions
def _eligible(sl: Sleeve, day: pd.DataFrame, held: list[str]) -> pd.DataFrame:
    lo, hi = sl.universe
    base = (
        day["tradable"].fillna(False).astype(bool)
        if sl.universe == (1, 100)
        else day["lrank"].between(lo, hi)
    )
    keep = (
        day["Ticker"].isin(held)
        & (day["lrank"] >= max(1, lo - sl.uni_slack))
        & (day["lrank"] <= hi + sl.uni_slack)
    )
    return day[base | keep]


def _score(sl: Sleeve, el: pd.DataFrame) -> pd.Series:
    if sl.score == "momblend":
        mom = el["resid_mom_120"].fillna(el["resid_mom_120"].median())
        return 0.75 * el["pred_ens"].rank(pct=True) + 0.25 * mom.rank(pct=True)
    return el[sl.score]


def sleeve_names(
    sl: Sleeve, day: pd.DataFrame, held: list[str], barred: set[str], full: bool
) -> list[str]:
    el = _eligible(sl, day, held).copy()
    el["_s"] = _score(sl, el)
    el = el[el["_s"].notna()]
    if not full:  # off-week: keep current names, refill empty slots
        ranked = [t for t in el.sort_values("_s", ascending=False)["Ticker"] if t not in barred]
        keep = [t for t in held if t in set(el["Ticker"])][: sl.n]
        return keep + [t for t in ranked if t not in keep][: sl.n - len(keep)]
    if sl.buffer is None:
        if not barred:
            return list(el.nlargest(sl.n, "_s")["Ticker"])  # same tie-breaking as exact_sim
        return list(el[~el["Ticker"].isin(barred)].nlargest(sl.n, "_s")["Ticker"])
    ranked = list(
        el.sort_values("_s", ascending=False)["Ticker"]
    )  # identical rule to strategy.target_weights
    keep = [t for t in held if t in ranked[: sl.buffer]][: sl.n]
    return keep + [t for t in ranked if t not in keep and t not in barred][: sl.n - len(keep)]


def sleeve_weights(sl: Sleeve, names: list[str], day: pd.DataFrame) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    if sl.weighting == "sqrt_dvol":
        dv = day.set_index("Ticker").reindex(names)["dvol60"]
        w = np.sqrt(dv.fillna(dv.median()))
        w = w / w.sum()
        for _ in range(10):  # iterative 20 % cap
            over = w > 0.2
            if not over.any():
                break
            w[over] = 0.2
            w[~over] = w[~over] / w[~over].sum() * (1 - 0.2 * over.sum())
        return w * sl.frac
    return pd.Series(sl.frac / len(names), index=names)


# ============================================================================ simulator
@dataclass
class SimResult:
    name: str
    equity: pd.Series
    trades: pd.DataFrame
    exposure: pd.Series
    tdex_weight: pd.Series
    holdings: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()


def simulate(
    des: Design,
    f: pd.DataFrame,
    px: dict,
    bench_px: pd.DataFrame,
    fee: float = config.FEE_PER_SIDE,
    tdex_fee: float = TDEX_FEE,
    capital: float = config.INITIAL_CAPITAL,
    min_trade_frac: float = 0.01,
) -> SimResult:
    O, L, C = px["Open"], px["Low"], px["Close"].ffill()  # noqa: E741
    min_trade_frac = min_trade_frac * min(
        1.0, des.scale
    )  # 1% of the stock sleeve's capital (scaled wrappers)
    rets = px["Close"].pct_change(fill_method=None)
    bO = bench_px["Open"].reindex(C.index)
    bC = bench_px["Close"].reindex(C.index).ffill()
    dates = [d for d in C.index if d >= f["Date"].min()]
    by_day = {d: g for d, g in f.groupby("Date")}
    K = len(des.sleeves)
    cash, shares, entry, stop_frac, tunits = capital, {}, {}, {}, 0
    sleeve_held: list[list[str]] = [[] for _ in range(K)]
    stopped_at: dict[str, int] = {}
    stop_names: set[str] = set()
    pending, trades, equity, expo, tw, hold = None, [], {}, {}, {}, []
    n_dec = 0

    def trade(tkr, qty, price, day, reason, cost=fee):
        nonlocal cash
        cash -= qty * price + abs(qty * price) * cost
        trades.append((day, tkr, qty, price, reason))

    for i, d in enumerate(dates):
        if pending is not None:
            targets, tdex_w, names_by_sleeve, stop_set, new_stop = pending
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
                    stop_frac.pop(t, None)
            if pd.notna(b_open) and b_open > 0:  # TDEX sells before stock buys
                want_u = int(tdex_w * eq_open / b_open / LOT) * LOT
                if want_u < tunits and abs((want_u - tunits) * b_open) >= min_trade_frac * eq_open:
                    trade(config.BENCHMARK, want_u - tunits, b_open, d, "tdex", tdex_fee)
                    tunits = want_u
            for t, w in targets.items():  # buys / resizes
                price = O.at[d, t] if t in O.columns else np.nan
                if pd.isna(price) or price <= 0:
                    continue
                want = int(w * eq_open / price / LOT) * LOT
                cur = shares.get(t, 0)
                diff = want - cur
                if diff != 0 and abs(diff * price) >= min_trade_frac * eq_open:
                    trade(t, diff, price, d, "rebalance")
                    shares[t] = cur + diff
                    if diff > 0 and t not in entry:
                        entry[t] = price
                        stop_frac[t] = new_stop.get(t)
                    else:
                        entry[t] = entry.get(t, price)
                    if shares[t] == 0:
                        shares.pop(t), entry.pop(t, None), stop_frac.pop(t, None)
            if pd.notna(b_open) and b_open > 0 and tdex_w > 0:  # TDEX buys last, with the cash left
                want_u = int(tdex_w * eq_open / b_open / LOT) * LOT
                spare = int(max(cash - 0.005 * eq_open, 0) / (b_open * (1 + tdex_fee)) / LOT) * LOT
                want_u = min(want_u, tunits + spare)
                if want_u > tunits and abs((want_u - tunits) * b_open) >= min_trade_frac * eq_open:
                    trade(config.BENCHMARK, want_u - tunits, b_open, d, "tdex", tdex_fee)
                    tunits = want_u
            sleeve_held = [[t for t in nm if t in shares] for nm in names_by_sleeve]
            stop_names = stop_set
            pending = None

        # stop-loss on the daily low (only names owned by a stop-enabled sleeve)
        for t in list(shares):
            sf = stop_frac.get(t)
            if t not in stop_names or not sf:
                continue
            stop_px = entry[t] * (1 - sf)
            if pd.notna(L.at[d, t]) and L.at[d, t] <= stop_px:
                fill = min(O.at[d, t], stop_px) if pd.notna(O.at[d, t]) else stop_px
                trade(t, -shares.pop(t), fill, d, "stop-loss")
                entry.pop(t)
                stop_frac.pop(t, None)
                stopped_at[t] = i

        long_val = sum(q * C.at[d, t] for t, q in shares.items())
        tval = tunits * bC.get(d, 0.0)
        equity[d] = cash + long_val + tval
        expo[d] = long_val / equity[d] if equity[d] else 0.0
        tw[d] = tval / equity[d] if equity[d] else 0.0

        if i % config.HORIZON == 0 and i < len(dates) - 1 and d in by_day:
            day = by_day[d]
            owned = set().union(*[set(h) for h in sleeve_held]) if K > 1 else set()
            targets = pd.Series(dtype=float)
            names_by_sleeve, stop_set, new_stop = [], set(), {}
            vol_d = day.set_index("Ticker")["vol_60"] if "vol_60" in day.columns else None
            for k, sl in enumerate(des.sleeves):
                if K == 1:
                    held = list(shares)  # exact_sim convention
                else:
                    orphans = [t for t in shares if t not in owned] if k == 0 else []
                    held = [t for t in shares if t in sleeve_held[k] or t in orphans]
                barred = (
                    {t for t, j in stopped_at.items() if i - j <= sl.cooldown}
                    if sl.cooldown
                    else set()
                )
                full = n_dec % sl.every == 0
                names = sleeve_names(sl, day, held, barred, full)
                w = sleeve_weights(sl, names, day) * des.scale
                targets = targets.add(w, fill_value=0.0)
                names_by_sleeve.append(names)
                if sl.stop:
                    stop_set |= set(names)
                    for t in names:
                        if sl.stop == "vol":
                            v = vol_d.get(t, np.nan) if vol_d is not None else np.nan
                            v = v if pd.notna(v) else 0.021
                            new_stop[t] = float(
                                np.clip(2 * v * np.sqrt(config.HORIZON), 0.05, 0.15)
                            )
                        else:
                            new_stop[t] = float(sl.stop)
            if des.vol_manage and len(targets):
                window = rets.loc[:d, [t for t in targets.index if t in rets.columns]].tail(120)
                cov = portfolio.ledoit_wolf_cov(window)
                w = targets.reindex(cov.index).dropna()
                if len(w) and w.sum() > 0:
                    vol = portfolio.portfolio_vol(w / w.sum() * targets.sum(), cov)
                    targets = targets * min(1.0, des.vol_manage / vol) if vol > 0 else targets
            tdex_w = max(0.0, 1.0 - float(targets.sum())) if des.tdex_remainder else 0.0
            pending = (targets, tdex_w, names_by_sleeve, stop_set, new_stop)
            n_dec += 1
            h = day.set_index("Ticker").reindex(targets.index)
            hold.append(
                (
                    d,
                    len(h),
                    float((h["lrank"] * targets).sum() / targets.sum())
                    if targets.sum()
                    else np.nan,
                    h["idio_vol"].mean(),
                    h["beta_mkt"].mean(),
                    float(targets.sum()),
                )
            )
    tr = pd.DataFrame(trades, columns=["Date", "Ticker", "qty", "price", "reason"])
    hd = pd.DataFrame(hold, columns=["Date", "names", "lrank_w", "idio_vol", "beta_mkt", "stock_w"])
    return SimResult(
        des.name, pd.Series(equity, name=des.name), tr, pd.Series(expo), pd.Series(tw), hd
    )


def replication_check(inp: dict) -> pd.DataFrame:
    """simulate() must reproduce strategy.exact_sim (E1, E2) on the pipeline's sim_frame (validation)."""
    preds = inp["preds"][["Date", "Ticker", "split", "pred_ens"]]
    f0 = strategy.sim_frame(inp["df"], preds, inp["market"], "valid")
    f0["lrank"] = 1.0
    px = price_panel(inp, f0)
    rows = {}
    for ex, des in [
        (strategy.ExactConfig("E1"), Design("E1", (Sleeve(),))),
        (strategy.ExactConfig("E2", buffer=30, stop_loss=0.08), Design("E2", (E2S,))),
    ]:
        a = strategy.exact_sim(ex, f0, px, inp["bench"], inp["ic"]).equity
        b = simulate(des, f0, px, inp["bench"]).equity
        rows[ex.name] = {
            "exact_sim final": a.iloc[-1],
            "sv2 final": b.iloc[-1],
            "max abs diff": float((a - b).abs().max()),
        }
    return pd.DataFrame(rows).T


# ============================================================================ metrics
def bench_returns(bench: pd.DataFrame, idx: pd.Index) -> pd.Series:
    return bench["Close"].reindex(idx).ffill().pct_change()


def universe_ew(px_close: pd.DataFrame, f: pd.DataFrame) -> pd.Series:
    r = px_close.pct_change(fill_method=None)
    m = f[f["tradable"].fillna(False).astype(bool)].assign(v=1.0)
    mem = m.pivot_table(index="Date", columns="Ticker", values="v").reindex(
        index=r.index, columns=r.columns
    )
    return r.where(mem.shift(1).notna()).mean(axis=1)


def ff3_em(daily_thb: pd.Series, ff: pd.DataFrame, usdthb: pd.Series) -> dict:
    f = ff.set_index("Date")[["Mkt-RF", "SMB", "HML", "RF"]] / 100
    r = factors.to_monthly_usd(daily_thb, usdthb)
    f = f.reindex(r.index)
    return factors.regress(
        r - f["RF"], f[["Mkt-RF", "SMB", "HML"]], periods_per_year=12, hac_lags=3
    )


def attribution(r: pd.Series, inp: dict) -> dict:
    out = {}
    loc = factors.thai_local(r, inp["local"])
    out.update(
        {
            "TH alpha %": loc["alpha % p.a."],
            "TH alpha t": loc["alpha t"],
            "TH b_MKT": loc["b_MKT"],
            "TH b_SMB": loc["b_SMB"],
            "TH t_SMB": loc["t_SMB"],
            "TH b_WML": loc["b_WML"],
            "TH t_WML": loc["t_WML"],
            "TH R2": loc["R2"],
        }
    )
    if inp["ff"] is not None:
        try:
            f3 = ff3_em(r, inp["ff"], inp["usdthb"])
            c4 = factors.carhart_em(r, inp["ff"], inp["usdthb"])
            out.update(
                {
                    "FF3 alpha %": f3["alpha % p.a."],
                    "FF3 alpha t": f3["alpha t"],
                    "FF3 b_Mkt": f3["b_Mkt-RF"],
                    "FF3 b_SMB": f3["b_SMB"],
                    "FF3 b_HML": f3["b_HML"],
                    "C4 alpha %": c4["alpha % p.a."],
                    "C4 alpha t": c4["alpha t"],
                    "C4 b_Mkt": c4["b_Mkt-RF"],
                    "C4 b_SMB": c4["b_SMB"],
                    "C4 b_HML": c4["b_HML"],
                    "C4 b_WML": c4["b_WML"],
                    "C4 t_WML": c4["t_WML"],
                    "FF obs (months)": c4["obs"],
                }
            )
        except Exception:  # noqa: BLE001
            pass
    return out


def stats(
    res: SimResult, gross: SimResult | None, inp: dict, bench_r: pd.Series, ew: pd.Series
) -> dict:
    r = res.returns
    b = bench_r.reindex(r.index)
    s = evaluate.perf_stats(r, b)
    years = len(r) / DAYS
    st = res.trades[res.trades["Ticker"] != config.BENCHMARK] if len(res.trades) else res.trades
    traded = float((st["qty"].abs() * st["price"]).sum()) if len(st) else 0.0
    two_way = traded / res.equity.mean() / years
    bench_cagr = 100 * ((1 + b.dropna()).prod() ** (DAYS / len(b.dropna())) - 1)
    out = {
        "CAGR %": s["CAGR %"],
        "TDEX CAGR %": bench_cagr,
        "vol %": s["vol %"],
        "Sharpe": s["Sharpe"],
        "max DD %": s["max DD %"],
        "TE %": s["tracking error %"],
        "IR vs TDEX": s["info ratio"],
        "CAPM alpha %": s["alpha % p.a."],
        "CAPM alpha t": s["alpha t-stat"],
        "CAPM beta": s["beta"],
        "PSR": s["PSR (SR>0)"],
        "turnover 1-way x/yr": two_way / 2,
        "turnover 2-way x/yr": two_way,
        "stock trades": len(st),
        "stop-outs": int((st["reason"] == "stop-loss").sum()) if len(st) else 0,
        "stock exposure %": 100 * res.exposure.mean(),
        "TDEX weight %": 100 * res.tdex_weight.mean(),
        "avg liq rank (w)": res.holdings["lrank_w"].mean() if len(res.holdings) else np.nan,
        "sr_daily": float(r.mean() / r.std()),
        "ir_daily": float((r - b).dropna().mean() / (r - b).dropna().std()),
    }
    sig = (r - ew.reindex(r.index)).dropna()
    out["vs EW top-100 % p.a."] = 100 * sig.mean() * DAYS
    out.update(attribution(r, inp))
    if gross is not None:
        g = gross.returns
        gc = 100 * ((1 + g).prod() ** (DAYS / len(g)) - 1)
        out["gross CAGR %"] = gc
        out["cost drag %"] = gc - out["CAGR %"]
        out["break-even vs TDEX %/side"] = (gc - bench_cagr) / two_way if two_way else np.nan
        g_th = factors.thai_local(g, inp["local"])["alpha % p.a."]
        out["gross TH alpha %"] = g_th
        out["break-even TH alpha %/side"] = g_th / two_way if two_way else np.nan
    return out


def to_md(t: pd.DataFrame, digits: int = 2) -> str:
    fmt = lambda x: f"{x:.{digits}f}" if isinstance(x, float | np.floating) else str(x)  # noqa: E731
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    lines = ["| " + " | ".join([""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, row in t.iterrows():
        i = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        lines.append("| " + " | ".join([i] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def deflated(r: pd.Series, trial_srs: list[float]) -> float:
    return evaluate.dsr(r, trial_srs)
