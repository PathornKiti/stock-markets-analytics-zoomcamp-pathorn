"""Risk, liquidity / capacity and performance-reporting views used by risk teams and allocators.

* Expected Shortfall 97.5% (Basel FRTB): mean loss beyond the 2.5% quantile, 1-day and 10-day.
* Historical stress windows: strategy vs benchmark over named market episodes.
* Ex-ante vs realised tracking error and the bias statistic (std of active returns standardised by
  their ex-ante forecast; ~1 = well calibrated risk model, > 1 = risk under-forecast).
* Risk decomposition (Euler): share of variance from each local Thai factor vs stock-specific.
* Return decomposition: gross -> costs -> net = market beta + style (SMB, WML) + specific alpha.
* Liquidity and capacity: % of ADV per trade, days to liquidate, and net CAGR vs AUM under a
  square-root market-impact model  cost = fee + k x sigma_daily x sqrt(trade / ADV)  (k = IMPACT_K,
  an assumption; 0.5-1 is the usual range in Almgren et al. 2005 / Kissell's I-star family).
* GIPS-style reporting tables: monthly and calendar-year returns, rolling 12-month alpha and IR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from setalpha import config, portfolio

DAYS = 252
IMPACT_K = 1.0  # square-root impact coefficient (assumption, see docstring)
AUM_GRID = [1e5, 1e6, 1e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, 2.5e9, 5e9]  # THB
STRESS_WINDOWS = {  # fixed calendar windows of named episodes (not fitted)
    "COVID crash (2020-02-14 to 03-23)": ("2020-02-14", "2020-03-23"),
    "COVID rebound (2020-03-24 to 06-30)": ("2020-03-24", "2020-06-30"),
    "2022 H1 global rate shock": ("2022-01-01", "2022-06-30"),
    "2023 H2 Thai sell-off (Aug-Nov)": ("2023-08-01", "2023-11-30"),
    "2025 H1 sell-off incl. US tariffs": ("2025-01-01", "2025-06-30"),
    "Apr-2025 tariff week (04-02 to 04-09)": ("2025-04-02", "2025-04-09"),
    "2025-26 megacap rally (Jul-25 on)": ("2025-07-01", "2026-12-31"),
}


# ----------------------------------------------------------------------------- tail risk
def expected_shortfall(r: pd.Series, level: float = 0.975, horizon: int = 1) -> float:
    """Historical ES in % (positive = loss) of overlapping `horizon`-day compounded returns."""
    r = r.dropna()
    x = r if horizon == 1 else ((1 + r).rolling(horizon).apply(np.prod, raw=True) - 1).dropna()
    q = np.quantile(x, 1 - level)
    return float(-100 * x[x <= q].mean())


def stress_table(returns: pd.DataFrame, bench: str, windows: dict = STRESS_WINDOWS) -> pd.DataFrame:
    """Compounded return (%) of each column over each window, plus active vs `bench` and in-window max DD."""
    rows = {}
    for name, (a, b) in windows.items():
        w = returns.loc[a:b].dropna(how="all")
        if len(w) < 3:
            continue
        tot = 100 * ((1 + w.fillna(0)).prod() - 1)
        curve = (1 + w.fillna(0)).cumprod()
        dd = 100 * (curve / curve.cummax() - 1).min()
        row = {"from": w.index[0].date(), "to": w.index[-1].date(), "days": len(w)}
        for c in returns.columns:
            row[f"{c} %"] = tot[c]
            if c != bench:
                row[f"{c} active %"] = tot[c] - tot[bench]
                row[f"{c} max DD %"] = dd[c]
        rows[name] = row
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- tracking error
def ex_ante_te(
    w: pd.DataFrame,
    stock_rets: pd.DataFrame,
    bench_ret: pd.Series,
    active: pd.Series,
    decision_days: pd.Index,
    window: int = 120,
) -> pd.DataFrame:
    """Ex-ante daily TE vs the benchmark at each rebalance (Ledoit-Wolf on the trailing `window` days),
    and the realised active return (`active` = strategy - benchmark, daily) until the next rebalance.

    Active weights = [stock weights, -1 on the benchmark]; cash has zero risk. Only data up to the
    rebalance date enter the forecast.
    """
    rows = []
    days = [d for d in decision_days if d in w.index]
    for d, nxt in zip(days[:-1], days[1:], strict=False):
        pos = w.index.get_loc(d)
        if pos + 1 >= len(w):
            break
        held = w.iloc[pos + 1]  # book after the next-open execution
        held = held[held > 1e-6]
        if held.empty:
            continue
        hist = (
            stock_rets.loc[:d, held.index]
            .tail(window)
            .assign(_bench=bench_ret.loc[:d].tail(window))
        )
        cov = portfolio.ledoit_wolf_cov(hist)
        a = pd.Series(held, index=cov.index).fillna(0.0)
        a["_bench"] = -1.0
        te = float(np.sqrt(a.to_numpy() @ cov.to_numpy() @ a.to_numpy()))
        realised = active.loc[d:nxt].iloc[1:]
        rows.append(
            {
                "Date": d,
                "ex-ante TE (daily)": te,
                "active return": float(realised.sum()),
                "days": len(realised),
            }
        )
    return pd.DataFrame(rows).set_index("Date")


def te_summary(te: pd.DataFrame, active_daily: pd.Series) -> dict:
    z = te["active return"] / (te["ex-ante TE (daily)"] * np.sqrt(te["days"]))
    return {
        "ex-ante TE % (mean)": 100 * te["ex-ante TE (daily)"].mean() * np.sqrt(DAYS),
        "realised TE %": 100 * active_daily.std() * np.sqrt(DAYS),
        "realised / ex-ante": active_daily.std() / te["ex-ante TE (daily)"].mean(),
        "bias statistic (std z)": float(z.std()),
        "rebalances": len(te),
    }


# ----------------------------------------------------------------------------- decompositions
def _ols(y: pd.Series, X: pd.DataFrame):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    return sm.OLS(d["y"], sm.add_constant(d[X.columns])).fit(), d


def risk_decomposition(r: pd.Series, factors: pd.DataFrame) -> dict:
    """Euler variance shares (%): b_k cov(f_k, r) / var(r) per factor; specific = var(e) / var(r)."""
    fit, d = _ols(r, factors)
    var = d["y"].var()
    out = {f"{f} %": 100 * fit.params[f] * d[f].cov(d["y"]) / var for f in factors.columns}
    out["factor total %"] = sum(out.values())
    out["specific %"] = 100 * fit.resid.var() / var
    return out


def return_decomposition(net: pd.Series, gross: pd.Series, factors: pd.DataFrame) -> dict:
    """Arithmetic % p.a.: gross -> costs -> net, and net = beta + style + specific (local Thai factors)."""
    fit, d = _ols(net, factors)
    contrib = {f: 100 * DAYS * fit.params[f] * d[f].mean() for f in factors.columns}
    style = sum(v for f, v in contrib.items() if f != "MKT")
    g_fit, _ = _ols(gross, factors)
    return {
        "gross return": 100 * DAYS * gross.reindex(d.index).mean(),
        "costs": -100 * DAYS * (gross.reindex(d.index) - d["y"]).mean(),
        "net return": 100 * DAYS * d["y"].mean(),
        "= market beta (MKT)": contrib.get("MKT", 0.0),
        "+ style (SMB + WML)": style,
        **{f"   of which {f}": v for f, v in contrib.items() if f != "MKT"},
        "+ specific alpha (net)": 100 * DAYS * fit.params["const"],
        "specific alpha (gross)": 100 * DAYS * g_fit.params["const"],
    }


# ----------------------------------------------------------------------------- liquidity / capacity
def trade_liquidity(trades: pd.DataFrame, adv: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Each trade with its value, the previous day's ADV (60d median traded value) and daily vol."""
    t = trades.copy()
    t["value"] = (t["qty"] * t["price"]).abs()
    adv_l, vol_l = adv.shift(1).stack(), vol.shift(1).stack()  # known before the open
    key = pd.MultiIndex.from_arrays([t["Date"], t["Ticker"]])
    t["adv"] = adv_l.reindex(key).to_numpy()
    t["sigma"] = vol_l.reindex(key).to_numpy()
    return t


def capacity_curve(
    net: pd.Series,
    equity: pd.Series,
    tl: pd.DataFrame,
    w: pd.DataFrame,
    adv: pd.DataFrame,
    aum: list[float] = AUM_GRID,
    k: float = IMPACT_K,
    capital: float = config.INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Net CAGR vs AUM: the simulated book scaled by AUM / capital, minus square-root impact.

    First-order approximation: the same trades scaled up (no change in lots or in the stop / buffer
    decisions); impact is charged on top of the per-side fee (config.FEE_PER_SIDE) already inside `net`.
    """
    tl = tl.dropna(subset=["adv", "sigma"])
    eq_prev = equity.shift(1).fillna(capital)
    hold_val = w.mul(equity, axis=0)
    adv_al = adv.shift(1).reindex(index=w.index, columns=w.columns)
    years = len(net) / DAYS

    def cagr(r: pd.Series) -> float:
        return 100 * ((1 + r).prod() ** (1 / years) - 1)

    rows = {}
    for a in aum:
        s = a / capital
        part = tl["value"] * s / tl["adv"]
        unit = (
            (tl["value"] * s * tl["sigma"] * np.sqrt(part)).groupby(tl["Date"]).sum()
        )  # THB per unit of k
        drag = (unit / (eq_prev * s)).reindex(net.index).fillna(0.0)
        dtl = (
            (hold_val * s / adv_al).where(hold_val > 0).max(axis=1)
        )  # days at 100% ADV, slowest name
        rows[a] = {
            "AUM (THB m)": a / 1e6,
            "net CAGR %": cagr(net - k * drag),
            f"net CAGR % (k={k / 2:g})": cagr(net - k / 2 * drag),
            "impact drag % p.a.": 100 * DAYS * k * drag.mean(),
            "median trade % ADV": 100 * float(part.median()),
            "95th pct trade % ADV": 100 * float(part.quantile(0.95)),
            "trades > 10% ADV %": 100 * float((part > 0.10).mean()),
            "days to liquidate @10% ADV (avg)": float((dtl / 0.10).mean()),
            "days to liquidate @20% ADV (avg)": float((dtl / 0.20).mean()),
            "days to liquidate @20% ADV (95th pct)": float((dtl / 0.20).quantile(0.95)),
        }
    return pd.DataFrame(rows).T.set_index("AUM (THB m)")


def capacity_at(curve: pd.DataFrame, column: str, level: float) -> float:
    """AUM (THB m) at which the rising `column` (e.g. impact drag) first reaches `level`; log-linear
    interpolation on the AUM grid, nan if never reached."""
    c = curve[column]
    above = np.flatnonzero(c.to_numpy() >= level)
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return float(c.index[0])
    x0, x1, y0, y1 = np.log(c.index[i - 1]), np.log(c.index[i]), c.iloc[i - 1], c.iloc[i]
    return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))


# ----------------------------------------------------------------------------- reporting tables
def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Year x month compounded returns (%), with the calendar-year total."""
    m = (1 + r.dropna()).resample("ME").prod() - 1
    t = (
        (100 * m)
        .to_frame("r")
        .assign(y=m.index.year, m=m.index.strftime("%b"))
        .pivot(index="y", columns="m", values="r")
    )
    t = t[
        [
            c
            for c in pd.date_range("2000-01-01", periods=12, freq="ME").strftime("%b")
            if c in t.columns
        ]
    ]
    t["Year"] = 100 * ((1 + r.dropna()).groupby(r.dropna().index.year).prod() - 1)
    return t.rename_axis(None).rename_axis(None, axis=1)


def calendar_years(returns: pd.DataFrame, bench: str) -> pd.DataFrame:
    y = 100 * ((1 + returns.fillna(0)).groupby(returns.index.year).prod() - 1)
    for c in returns.columns:
        if c != bench:
            y[f"{c} active"] = y[c] - y[bench]
    return y


def rolling_alpha_ir(r: pd.Series, bench: pd.Series, window: int = DAYS) -> pd.DataFrame:
    """Rolling 12-month CAPM alpha (% p.a.), beta and information ratio vs the benchmark."""
    d = pd.concat([r, bench], axis=1, keys=["p", "b"]).dropna()
    cov = d["p"].rolling(window).cov(d["b"])
    beta = cov / d["b"].rolling(window).var()
    alpha = d["p"].rolling(window).mean() - beta * d["b"].rolling(window).mean()
    act = d["p"] - d["b"]
    ir = act.rolling(window).mean() / act.rolling(window).std() * np.sqrt(DAYS)
    return pd.DataFrame({"alpha % p.a.": 100 * DAYS * alpha, "beta": beta, "IR": ir}).dropna()
