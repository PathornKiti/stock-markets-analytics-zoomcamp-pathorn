"""Presentation layer of the HTML dashboard (`report.build_dashboard` writes what `render` returns).

Six tabs, in the order a user needs them: Today (orders for the next open, signal health, paper trading),
Performance, Factors, Signal, Trust & risk, Diagnostics.

Design rules (see README section 9 for the screenshot):
* colour by role, never decoration: the selected strategy is the one blue series, TDEX is the neutral reference
  grey, the equal-weight universe is orange, every other strategy is a recessive light grey;
* the categorical slots come from a palette validated for colour-blind separation in light and dark mode;
* one y-axis per chart (the old dual-axis charts are split into panels); a table twin exists for every chart;
* light and dark themes are both designed: charts are built from colour tokens that the page resolves at render
  time, so a theme switch re-draws them with the other mode's steps.
Charts are Plotly (loaded from its CDN); everything else is plain HTML/CSS in this one file.
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd

from setalpha import config

PLOTLY_JS = "https://cdn.plot.ly/plotly-3.7.0.min.js"
BENCH = f"{config.BENCHMARK} buy & hold"
EW = "EW tradable top-100 (no costs)"


# ----------------------------------------------------------------------------- small helpers
def short(name: str) -> str:
    if name == BENCH:
        return "TDEX"
    if name == EW:
        return "EW top-100"
    if name[:1] == "E" and name[1:2].isdigit():
        return name.split()[0] + (", " + name.split(", ", 1)[1] if ", " in name else "")
    return name


def _day(d) -> str:
    return pd.Timestamp(d).strftime("%-d %b %Y")


def _esc(x) -> str:
    return html.escape(str(x))


def _num(v, digits: int = 2, sign: bool = False, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if isinstance(v, int | np.integer) and not isinstance(v, bool):
        return f"{v:,}"
    if not isinstance(v, float | np.floating):
        return _esc(v)
    if math.isinf(v):
        return "never" if v > 0 else "−∞"
    s = f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"
    return s.replace("-", "−") + ("%" if pct else "")


def _digits(col: str) -> int:
    c = col.lower()
    if (
        re.search(r"\b(ic|icir|auc|tc|pbo|psr|dsr|r2|score)\b", c)
        or "corr" in c
        or "precision@" in c
    ):
        return 3
    if "thb" in c or c in ("trades", "picks", "days", "obs", "rebalances"):
        return 0
    return 2


def table(
    df: pd.DataFrame,
    highlight: dict[str, str] | None = None,
    tstats: bool = True,
    index_name: str = "",
    digits: dict[str, int] | None = None,
    shorten: bool = False,
) -> str:
    """Formatted HTML table: tabular figures, |t| >= 2 in bold, highlighted rows (selected / benchmark)."""
    if df is None or df.empty:
        return "<p class='empty'>No data.</p>"
    highlight = highlight or {}
    digits = digits or {}
    head = "".join(f"<th scope='col'>{_esc(c)}</th>" for c in df.columns)
    rows = []
    for idx, row in df.iterrows():
        cls = highlight.get(str(idx), "")
        cells = []
        for c, v in row.items():
            d = digits.get(c, _digits(str(c)))
            txt = (
                _num(v, d)
                if not isinstance(v, pd.Timestamp | datetime)
                else _esc(pd.Timestamp(v).date())
            )
            cs = str(c)
            is_t = tstats and (
                cs == "t" or cs.startswith(("t (", "t_")) or cs.endswith(" t") or "t-stat" in cs
            )
            strong = (
                is_t and isinstance(v, float | np.floating) and not math.isnan(v) and abs(v) >= 2
            )
            cells.append(f"<td class='{'strong' if strong else ''}'>{txt}</td>")
        label = short(str(idx)) if shorten else str(idx)
        title = f" title='{_esc(idx)}'" if shorten and label != str(idx) else ""
        rows.append(
            f"<tr class='{cls}'><th scope='row'{title}>{_esc(label)}</th>{''.join(cells)}</tr>"
        )
    return (
        f"<div class='tbl-wrap'><table class='tbl'><thead><tr><th scope='col'>{_esc(index_name)}</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def tile(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (
        f"<div class='tile {tone}'><div class='tile-label'>{_esc(label)}</div><div class='tile-value'>{value}</div>"
        f"{f'<div class=tile-sub>{sub}</div>' if sub else ''}</div>"
    )


def section(title: str, body: str, note: str = "", wide: bool = False) -> str:
    return (
        f"<section class='block{' wide' if wide else ''}'><h3>{_esc(title)}</h3>"
        f"{f'<p class=note>{note}</p>' if note else ''}{body}</section>"
    )


def details(summary: str, body: str) -> str:
    return f"<details class='more'><summary>{_esc(summary)}</summary>{body}</details>"


STATUS = {
    "green": (
        "good",
        "Healthy",
        "<path d='M3 8.5l3 3 7-7' fill='none' stroke='currentColor' stroke-width='2'/>",
    ),
    "amber": (
        "warn",
        "Weakening",
        "<path d='M8 2l7 12H1z' fill='none' stroke='currentColor' stroke-width='1.8'/>"
        "<path d='M8 6v4M8 12v.5' stroke='currentColor' stroke-width='1.8'/>",
    ),
    "red": (
        "crit",
        "Broken",
        "<path d='M3 3l10 10M13 3L3 13' stroke='currentColor' stroke-width='2'/>",
    ),
}


def status_badge(status: str) -> str:
    tone, label, icon = STATUS.get(status, STATUS["amber"])
    return f"<span class='badge {tone}'><svg viewBox='0 0 16 16' aria-hidden='true'>{icon}</svg>{label}</span>"


# ----------------------------------------------------------------------------- chart specs
def _xy(s: pd.Series, scale: float = 1.0) -> tuple[list, list]:
    s = s.dropna()
    x = [
        d.strftime("%Y-%m-%d")
        if isinstance(d, pd.Timestamp)
        else (d if isinstance(d, str) else str(d))
        for d in s.index
    ]
    return x, [None if not np.isfinite(v) else float(v) * scale for v in s.to_numpy(dtype=float)]


def line(s: pd.Series, name: str, color: str, scale: float = 1.0, width: float = 2, **kw) -> dict:
    x, y = _xy(s, scale)
    return {
        "type": "scatter",
        "mode": "lines",
        "x": x,
        "y": y,
        "name": name,
        "line": {"color": color, "width": width, "shape": "linear"},
        **kw,
    }


def bar(x: list, y: list, name: str, color: str, orientation: str = "v", **kw) -> dict:
    vals = [None if v is None or not np.isfinite(v) else float(v) for v in y]
    spec = {
        "type": "bar",
        "name": name,
        "marker": {"color": color, "cornerradius": 4},
        "orientation": orientation,
    }
    spec.update({"x": vals, "y": list(x)} if orientation == "h" else {"x": list(x), "y": vals})
    return {**spec, **kw}


def chart(cid: str, data: list, title: str, height: int = 340, **layout) -> tuple[str, dict]:
    """Plotly spec + its container. The title is an HTML caption, so it never collides with the legend."""
    top = 10 if layout.get("showlegend") is False else 34
    layout.setdefault("margin", {})
    layout["margin"] = {"t": top, **layout["margin"]}
    spec = {"data": data, "layout": {"height": height, **layout}}
    cap = f"<figcaption>{_esc(title)}</figcaption>" if title else ""
    return (
        f"<figure class='chart'>{cap}<div id='{cid}' class='plot' data-chart='{cid}' style='min-height:{height}px'></div>"
        f"</figure>"
    ), {cid: spec}


def strategy_color(name: str, selected: str) -> tuple[str, float]:
    if name == selected:
        return "var(--s1)", 2.0
    if name == BENCH:
        return "var(--ref)", 2.0
    if name == EW:
        return "var(--s2)", 2.0
    return "var(--de)", 1.25


# ----------------------------------------------------------------------------- tabs
def tab_today(res: dict, specs: dict) -> str:
    sig = res.get("signals")
    paper = res.get("paper")
    if sig is None or sig.empty:
        ticket = "<p class='empty'>No signals yet: run the signals step.</p>"
        date_txt, regime, p_st, strat = res["last_date"], "–", float("nan"), res.get("selected", "")
    else:
        date_txt = str(sig["signal_date"].iloc[0])
        regime, p_st, strat = (
            str(sig["hmm_regime"].iloc[0]),
            float(sig["hmm_p_stressed"].iloc[0]),
            str(sig["strategy"].iloc[0]),
        )
        buys = sig[sig["target_weight_%"] > 0].sort_values("target_weight_%", ascending=False)
        wmax = max(float(buys["target_weight_%"].max()), 1.0) if len(buys) else 1.0
        has_stop = buys["stop_price"].notna().any() if len(buys) else False
        rows = "".join(
            f"<tr><th scope='row'>{_esc(str(r['Ticker']).replace('.BK', ''))}</th><td class='l'>{_esc(r['sector'])}</td>"
            f"<td><span class='wbar' style='--w:{100 * r['target_weight_%'] / wmax:.0f}%'></span>{r['target_weight_%']:.1f}%</td>"
            f"<td>{int(r['shares (100-lot)']):,}</td><td>{r['last_close']:,.2f}</td>"
            + (f"<td>{_num(float(r['stop_price']), 2)}</td>" if has_stop else "")
            + f"<td>{_num(float(r['exp_alpha_5d_%']), 2, sign=True, pct=True)}</td></tr>"
            for r in buys.to_dict("records")
        )
        ticket = (
            "<div class='tbl-wrap'><table class='tbl ticket'><thead><tr><th scope='col'>Stock</th><th class='l' scope='col'>Sector</th>"
            "<th scope='col'>Target weight</th><th scope='col'>Shares (lots of 100)</th><th scope='col'>Last close</th>"
            + ("<th scope='col'>Stop price</th>" if has_stop else "")
            + "<th scope='col'>Expected 5-day alpha</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
        score_col = "pred_blend" if "pred_blend" in sig.columns else "pred_ens"
        watch = sig[sig["target_weight_%"] <= 0][
            ["Ticker", "sector", score_col, "exp_alpha_5d_%"]
        ].copy()
        if len(watch):
            watch["Ticker"] = watch["Ticker"].str.replace(".BK", "", regex=False)
            ticket += details(
                f"Next {len(watch)} candidates (not bought)",
                table(
                    watch.set_index("Ticker").rename(
                        columns={score_col: "score", "exp_alpha_5d_%": "expected 5d alpha %"}
                    ),
                    tstats=False,
                    index_name="Stock",
                ),
            )
    mon = (paper or {}).get("monitor")
    regime_tone = {"calm": "good", "normal": "", "stressed": "crit"}.get(regime, "")
    health = ""
    if mon:
        th = mon["thresholds"]
        roll = mon["rolling"].dropna()
        roll = roll[roll.index >= roll.index.max() - pd.Timedelta(days=3 * 365)]
        live0 = pd.Timestamp(paper["start"])
        html_c, spec = chart(
            "ic-health",
            [
                line(roll[roll.index < live0], "Backtest (walk-forward)", "var(--ref)"),
                line(roll[roll.index >= live0], "Live (published scores)", "var(--s1)"),
            ],
            "",
            height=200,
            showlegend=False,
            margin={"t": 8, "b": 28, "l": 40, "r": 8},
            shapes=[
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": th["amber"],
                    "y1": th["amber"],
                    "line": {"color": "var(--warn)", "width": 1},
                },
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": th["red"],
                    "y1": th["red"],
                    "line": {"color": "var(--crit)", "width": 1},
                },
            ],
        )
        specs.update(spec)
        ic = mon["ic"].dropna()
        facts = pd.DataFrame(
            {"mean daily IC": [ic[:"2019"].mean(), ic["2020":"2021"].mean(), ic["2022":].mean()]},
            index=["2013-19", "2020-21 (validation)", "2022 onward (test)"],
        )
        low = roll.idxmin()
        health = (
            f"<div class='health'><div class='health-head'>{status_badge(mon['status'])}"
            f"<span>Rolling {th['window']}-day IC <b>{mon['last_rolling_ic']:.3f}</b></span></div>{html_c}"
            + table(facts, tstats=False, index_name="Period")
            + f"<p class='note'>Lowest point in the last three years: {roll.min():.3f} on {_day(low)}.</p>"
            f"<p class='note'>Amber line {th['amber']}, red line {th['red']}. Live days measured so far: "
            f"{mon['live_days']}. History before {paper['start']} is the walk-forward backtest.</p></div>"
        )
    hero = f"""
<div class='hero'>
  <div class='hero-main'>
    <p class='kicker'>Orders for the next open, from signals at the close on {_esc(_day(date_txt))}</p>
    <h2>{len(sig[sig['target_weight_%'] > 0]) if sig is not None and not sig.empty else 0} stocks to hold, equal weight</h2>
    <div class='facts'>
      <span><span class='dot s1'></span>Deployed: {_esc(_strategy_desc(strat))}</span>
      <span class='badge {regime_tone}'>Market regime: {_esc(regime)}</span>
      <span>P(stressed) {_num(p_st, 2)}</span>
    </div>
    {ticket}
    <p class='note'>The 30-rank buffer uses the holdings of the deployed paper book (S5), so this list matches what it trades
    at its weekly rebalance; before the ledger starts it is the book from cash. E2 stays in the backtest tabs for comparison.</p>
  </div>
  <aside class='hero-side'><h3>Is the signal still working?</h3>{health}</aside>
</div>"""
    return hero + paper_block(paper, specs)


def _strategy_desc(strat: str) -> str:
    from setalpha import paper

    return paper.DEPLOYED_DESC if strat == paper.DEPLOYED else short(strat)


def paper_block(paper: dict | None, specs: dict) -> str:
    if not paper:
        return ""
    reg = paper["registry"].copy()
    reg["stop"] = reg["stop"].map(
        lambda v: "–"
        if v is None or (isinstance(v, float) and math.isnan(v))
        else f"{100 * v:.0f}%"
    )
    reg_html = table(reg, tstats=False, index_name="Shadow")
    if paper["perf"].empty:
        body = (
            f"<div class='empty-state'><p><b>The ledger starts with the first live daily run on or after {_esc(paper['start'])}.</b></p>"
            "<p>Unpause <code>set_quant_alpha_daily</code> in Airflow (localhost:8080) or trigger it once by hand. Each run fills "
            "yesterday's orders at today's open, marks the six books at the close and adds a row here.</p></div>"
        )
    else:
        nav = paper["nav"]
        colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)"]
        data = [
            line(
                nav[c] / nav[c].iloc[0],
                short(c) if not c.startswith("S") else c,
                "var(--ref)" if "TDEX" in c else colors[i % 6],
            )
            for i, c in enumerate(nav.columns)
        ]
        html_c, spec = chart(
            "paper-nav",
            data,
            "Growth of 1 since the paper start",
            height=300,
            hovermode="x unified",
        )
        specs.update(spec)
        body = html_c + table(paper["perf"], tstats=False, index_name="Shadow")
    return section(
        "Paper trading: six frozen candidates",
        body + details("Registry (frozen " + paper["registered"] + ")", reg_html),
        "The only out-of-sample evidence no experiment has used. Point-in-time scores, fills at the next open, "
        f"{100 * config.FEE_PER_SIDE:.3f}% per side all-in, THB 20M per book.",
        wide=True,
    )


DEP = "Deployed"


def _focus(res: dict) -> tuple[str, pd.Series, pd.Series | None, dict | None]:
    """The strategy the dashboard reports on: the deployed walk-forward strategy when published, else E2."""
    d = res.get("deployed")
    if d:
        bt = d["daily"]
        test = bt[bt.index >= pd.Timestamp(res["test_start"])]
        return DEP, test["net"], test["gross"], d["meta"]
    sel = res["selected"]
    return short(sel), res["daily_returns"][sel], None, None


def _cagr(r: pd.Series) -> float:
    r = r.dropna()
    return 100 * ((1 + r).prod() ** (252 / len(r)) - 1)


def tab_performance(res: dict, specs: dict) -> str:
    from setalpha import evaluate, risk

    sel, m = res["selected"], res["metrics"]
    name, r, gross, meta = _focus(res)
    rets = res["daily_returns"]
    b = rets[BENCH].reindex(r.index)
    st = evaluate.perf_stats(r, b)
    bm = m.loc[BENCH]
    e2 = m.loc[sel]
    tiles = [
        tile(
            "CAGR since 2022",
            _num(st["CAGR %"], 1, pct=True),
            f"TDEX {_num(bm['CAGR %'], 1, pct=True)}; E2 {_num(e2['CAGR %'], 1, pct=True)}",
            "bad" if st["CAGR %"] < bm["CAGR %"] else "good",
        ),
        tile("Sharpe ratio", _num(st["Sharpe"], 2), f"TDEX {_num(bm['Sharpe'], 2)}"),
        tile(
            "Worst drawdown",
            _num(st["max DD %"], 1, pct=True),
            f"TDEX {_num(bm['max DD %'], 1, pct=True)}",
        ),
        tile(
            "Alpha vs TDEX (CAPM)",
            _num(st["alpha % p.a."], 1, sign=True, pct=True),
            f"t-stat {_num(st['alpha t-stat'], 2)}",
        ),
    ]
    if meta:
        tiles += [
            tile(
                "Walk-forward 2016-26",
                _num(meta["CAGR %"], 1, pct=True),
                f"TDEX {_num(meta['TDEX CAGR %'], 1, pct=True)}; CAPM t {_num(meta['CAPM alpha t'], 2)}",
                "good" if meta["CAGR %"] > meta["TDEX CAGR %"] else "bad",
            ),
            tile(
                "Trading cost drag since 2022",
                _num(_cagr(gross) - st["CAGR %"], 1, pct=True),
                f"2016-26: {meta['cost drag %']:.1f}% at {meta['turnover 2-way x/yr']:.0f}× turnover; DSR {meta['DSR']:.2f}",
            ),
        ]
    else:
        tiles += [
            tile("Deflated Sharpe", _num(e2["DSR"], 2), "probability the Sharpe beats luck"),
            tile(
                "Trading cost drag",
                _num(e2["cost drag %"], 1, pct=True),
                f"turnover {e2['turnover x/yr']:.0f}× a year",
            ),
        ]

    eq = (1 + rets.fillna(0)).cumprod()
    data = []
    for c in [c for c in eq.columns if c not in (sel, BENCH, EW)] + [EW, BENCH, sel]:
        col, w = strategy_color(c, sel)
        if meta and c == sel:
            col, w = "var(--s3)", 2.0
        label = short(c) + (" (backtest pick)" if c == sel else "")
        data.append(line(eq[c], label, col, width=w, hovertemplate="%{y:.3f}"))
    focus_eq = (1 + r.fillna(0)).cumprod()
    if meta:
        data.append(
            line(
                focus_eq,
                "Deployed (integrated walk-forward)",
                "var(--s1)",
                width=2.5,
                hovertemplate="%{y:.3f}",
            )
        )
    ends = [
        {
            "x": _xy(s_)[0][-1],
            "y": float(s_.iloc[-1]),
            "text": f"{lab} {s_.iloc[-1]:.2f}",
            "showarrow": False,
            "xanchor": "left",
            "xshift": 6,
            "font": {"color": "var(--ink-2)", "size": 12},
        }
        for lab, s_ in ((name, focus_eq), ("TDEX", eq[BENCH]))
    ]
    eq_html, spec = chart(
        "equity",
        data,
        "Growth of 1 THB since January 2022 (test period, after costs)",
        height=400,
        hovermode="x unified",
        annotations=ends,
        margin={"r": 110},
    )
    specs.update(spec)
    dd = (1 + b.fillna(0)).cumprod()
    dd_html, spec = chart(
        "drawdown",
        [
            line(dd / dd.cummax() - 1, "TDEX", "var(--ref)", 100),
            line(focus_eq / focus_eq.cummax() - 1, name, "var(--s1)", 100),
        ],
        "Drawdown from the previous peak, %",
        height=260,
        hovermode="x unified",
    )
    specs.update(spec)
    z = (risk.monthly_table(r) - risk.monthly_table(b)).drop(columns="Year", errors="ignore")
    lim = float(np.nanmax(np.abs(z.to_numpy()))) or 1.0
    heat, spec = chart(
        "heat",
        [
            {
                "type": "heatmap",
                "z": [
                    [None if pd.isna(v) else round(float(v), 1) for v in row]
                    for row in z.to_numpy()
                ],
                "x": list(z.columns),
                "y": [str(i) for i in z.index],
                "zmid": 0,
                "zmin": -lim,
                "zmax": lim,
                "colorscale": [[0, "var(--neg)"], [0.5, "var(--mid)"], [1, "var(--pos)"]],
                "xgap": 2,
                "ygap": 2,
                "texttemplate": "%{z:.1f}",
                "hovertemplate": "%{y} %{x}: %{z:.1f}% vs TDEX<extra></extra>",
                "colorbar": {"title": {"text": "%"}, "thickness": 10},
            }
        ],
        f"Monthly return of the {name.lower()} strategy minus TDEX, %",
        height=280,
        yaxis={"autorange": "reversed", "type": "category"},
    )
    specs.update(spec)

    if meta:
        full = res["deployed"]["daily"]
        yr = lambda s_: (1 + s_.dropna()).groupby(s_.dropna().index.year).prod().sub(1).mul(100)  # noqa: E731
        cal = pd.DataFrame(
            {
                "Deployed": yr(full["net"]),
                "TDEX": yr(full["tdex"]),
                "E2 (backtest pick)": yr(rets[sel]),
            }
        ).sort_index()
        cal["Deployed minus TDEX"] = cal["Deployed"] - cal["TDEX"]
        cal_note = (
            "Deployed = walk-forward out of sample: each year's design was chosen only on earlier years. 2026 runs "
            "to the last walk-forward date. E2 is shown from 2022, its test period."
        )
    else:
        cal = res["risk"]["calendar"].rename(
            columns={
                sel: short(sel),
                BENCH: "TDEX",
                EW: "EW top-100",
                f"{sel} active": f"{short(sel)} minus TDEX",
                f"{EW} active": "EW minus TDEX",
            }
        )
        cal_note = "2020-21 are the validation years, 2022 onward the test years."
    factsheet = table(
        m.drop(columns=["selected"]),
        {sel: "sel", BENCH: "ref"},
        shorten=True,
        index_name="Strategy",
    )
    eq_note = (
        (
            "Blue is the deployed strategy (integrated walk-forward). Aqua is E2, the strategy the backtest chose on "
            "2020-21. Grey lines are the other backtest designs; click a legend item to hide or show it."
        )
        if meta
        else (
            "Blue is the strategy chosen on 2020-21 before the test years were seen. Grey lines are the other five designs."
        )
    )
    return (
        f"<div class='tiles'>{''.join(tiles)}</div>"
        + section("Equity curve", eq_html + f"<p class='note'>{eq_note}</p>", wide=True)
        + "<div class='grid2'>"
        + section("Drawdowns", dd_html)
        + section("Monthly returns against TDEX", heat)
        + "</div>"
        + section(
            "Calendar-year returns, %",
            table(cal, tstats=False, index_name="Year", digits=dict.fromkeys(cal.columns, 1)),
            cal_note,
        )
        + section(
            "Backtest factsheet: E1-E6 (test period)",
            details("Show all 46 metrics for every backtest strategy", factsheet),
        )
    )


def tab_factors(res: dict, specs: dict) -> str:
    sel = res["selected"]
    th, em = res.get("factors_th", pd.DataFrame()), res.get("factors_em", pd.DataFrame())
    blocks = ""
    meta = (res.get("deployed") or {}).get("meta")
    if meta:
        rows = {
            "CAPM (TDEX)": ("CAPM alpha", "gross CAPM alpha"),
            "Fama-French 3 (EM)": ("FF3-EM alpha", "gross FF3-EM alpha"),
            "Carhart (EM)": ("Carhart-EM alpha", "gross Carhart-EM alpha"),
            "Carhart (local Thai)": ("TH-Carhart alpha", "gross TH-Carhart alpha"),
            "Carhart (local Thai, factors also pay costs)": (
                "TH-Carhart (net factors) alpha",
                None,
            ),
        }
        dep = pd.DataFrame(
            {
                k: {
                    "net alpha % a year": meta.get(f"{n} %"),
                    "net t": meta.get(f"{n} t"),
                    "gross alpha % a year": meta.get(f"{g} %") if g else np.nan,
                    "gross t": meta.get(f"{g} t") if g else np.nan,
                }
                for k, (n, g) in rows.items()
            }
        ).T
        blocks += section(
            "Deployed strategy: alpha against each benchmark (walk-forward, out of sample 2016-26)",
            table(dep, index_name="Benchmark model"),
            f"Net = after {100 * config.FEE_PER_SIDE:.3f}% per side all-in; gross = before costs, the like-for-like skill test against "
            "frictionless factors. Every alpha is positive but none is significant (|t| < 2).",
        )
    for key, df, title in (
        ("th", th, "Local Thai factors (daily, THB): market = TDEX, size, momentum"),
        ("em", em, "Fama-French emerging-market Carhart factors (monthly, USD)"),
    ):
        if df.empty:
            continue
        d = df.dropna(subset=["alpha % p.a."])
        names = [short(i) for i in d.index]
        sel_col = "var(--s3)" if meta else "var(--s1)"  # blue is reserved for the deployed strategy
        colors = [
            sel_col if i == sel else "var(--ref)" if i == BENCH else "var(--de)" for i in d.index
        ]
        b = bar(
            names,
            d["alpha % p.a."].tolist(),
            "alpha % p.a.",
            "var(--de)",
            orientation="h",
            customdata=d["alpha t"].round(2).tolist(),
            hovertemplate="%{y}: %{x:.1f}% a year (t %{customdata})<extra></extra>",
        )
        b["marker"]["color"] = colors
        html_c, spec = chart(
            f"alpha-{key}",
            [b],
            "Alpha after factors, % a year",
            height=270,
            bargap=0.45,
            yaxis={"autorange": "reversed"},
            xaxis={"zeroline": True},
            margin={"l": 92},
        )
        specs.update(spec)
        blocks += section(
            f"Backtest strategies E1-E6: {title}",
            html_c + table(df, {sel: "sel", BENCH: "ref"}, shorten=True, index_name="Strategy"),
            "Bold t-statistics are significant at about 5% (|t| ≥ 2). An alpha near zero means the return is "
            "explained by market, size and momentum exposure, not skill.",
        )
    return blocks


def tab_signal(res: dict, specs: dict) -> str:
    sig = res.get("signal")
    if not sig:
        return "<p class='empty'>Run the simulate step to build the signal research.</p>"
    t, v = sig["test"], sig["valid"]
    raw, neu = t["ic_summary"].loc["raw IC"], t["ic_summary"].loc["factor-neutral IC"]
    tiles = "".join(
        [
            tile(
                "Rank IC, test",
                _num(raw["IC mean"], 3),
                f"validation {_num(v['ic_summary'].loc['raw IC', 'IC mean'], 3)}",
            ),
            tile("ICIR, annualised", _num(raw["ICIR annualised"], 2), "mean IC ÷ its volatility"),
            tile(
                "t-stat (overlap-robust)",
                _num(raw["t (non-overlapping)"], 1),
                f"Newey-West {_num(raw['t (Newey-West)'], 1)}",
            ),
            tile(
                "IC after style factors",
                _num(neu["IC mean"], 3),
                f"t {_num(neu['t (non-overlapping)'], 1)}",
            ),
            tile(
                "Decile monotonicity",
                _num(t["monotonicity"], 2),
                "1 = alpha rises with every decile",
            ),
        ]
    )
    mon = (res.get("paper") or {}).get("monitor")
    roll_html = ""
    if mon:
        r = mon["rolling"].dropna()
        roll_html, spec = chart(
            "ic-roll",
            [line(r, "Rolling 60-day IC", "var(--s1)")],
            "Rolling 60-day IC since 2013: the signal is real but has weakened",
            height=300,
            shapes=[
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "var(--axis)", "width": 1},
                }
            ],
        )
        specs.update(spec)
    q = t["quantiles"].loc[[i for i in t["quantiles"].index if i.startswith("D") and "-" not in i]]
    dec_html, spec = chart(
        "deciles",
        [
            bar(
                q.index,
                q["mean alpha %"].tolist(),
                "test",
                "var(--s1)",
                hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
            )
        ],
        "Mean 5-day alpha by score decile, % (test)",
        height=280,
        showlegend=False,
    )
    specs.update(spec)
    dec = t["ic_decay"]
    decay_html, spec = chart(
        "decay",
        [
            bar(dec.index, dec["IC mean"].tolist(), "test 2022+", "var(--s1)"),
            bar(
                v["ic_decay"].index,
                v["ic_decay"]["IC mean"].tolist(),
                "validation 2020-21",
                "var(--s2)",
            ),
        ],
        "IC by horizon: alpha keeps coming after day 5",
        height=280,
        barmode="group",
        bargap=0.35,
    )
    specs.update(spec)
    yr = t["ic_by_year"]
    year_html, spec = chart(
        "ic-year",
        [bar([str(i) for i in yr.index], yr["IC mean"].tolist(), "IC", "var(--s1)")],
        "Mean IC by test year",
        height=240,
        showlegend=False,
    )
    specs.update(spec)
    fl = pd.Series(
        {
            k: _num(float(v), 3 if k.startswith(("IC", "transfer", "implied")) else 2)
            for k, v in sig["fundamental_law"].items()
        }
    ).to_frame("value")
    return (
        f"<div class='tiles'>{tiles}</div>"
        + section("Signal decay monitor", roll_html, wide=True)
        + "<div class='grid2'>"
        + section("Deciles", dec_html, "Gross of costs; top-minus-bottom spread in the table.")
        + section("Decay by horizon", decay_html)
        + "</div>"
        + "<div class='grid2'>"
        + section("By year", year_html + table(yr, index_name="Year"))
        + section(
            "Does it survive known factors? (Fama-MacBeth)",
            table(t["fama_macbeth"], index_name="Characteristic"),
        )
        + "</div>"
        + "<div class='grid2'>"
        + section("Persistence and turnover", table(t["persistence"], index_name="Lag (days)"))
        + section(
            "Fundamental law of active management",
            table(fl, tstats=False),
            "IR ≈ IC × √breadth × transfer coefficient.",
        )
        + "</div>"
        + section("Deciles, full table", table(t["quantiles"], index_name="Decile"))
    )


# Launch gates: fixed before reading the results, in line with common buy-side practice (HLZ t > 3 for a new signal,
# IR 0.5 as a typical hurdle for an active book, DSR 0.95, PBO 0.5, a live record before capital).
GATES = [
    ("Net alpha vs TDEX (CAPM)", "t-stat > 2"),
    ("Net alpha vs Carhart, local Thai factors", "t-stat > 2"),
    ("Net alpha vs Carhart, Fama-French EM factors", "t-stat > 2"),
    ("Information ratio vs TDEX, net", "> 0.5"),
    ("Deflated Sharpe ratio (all trials counted)", "> 0.95"),
    ("Track record vs minimum track record length", "track record ≥ MinTRL"),
    ("Probability of backtest overfitting, 2020-26", "< 0.5"),
    ("Capacity: fund size before impact eats the gross alpha", "≥ THB 100m"),
    ("Live paper-trading record", "≥ 6 months"),
]
SIGNAL_GATES = [
    ("Signal IC after the multiple-testing haircut", "t > 3"),
    ("Signal IC positive in every test year", "every year > 0"),
]


def _gates(values: list, shown: list[str], live: int) -> pd.DataFrame:
    """values: CAPM t, local t, EM t, IR, DSR, track - MinTRL, PBO, capacity THB m (None = not measured)."""
    cap = values[7]
    ok = [
        values[0] > 2,
        values[1] > 2,
        values[2] > 2,
        values[3] > 0.5,
        values[4] > 0.95,
        values[5] >= 0,
        values[6] < 0.5,
        "pending" if cap is None else cap >= 100,
        "pass" if live >= 126 else "pending",
    ]
    state = [
        o if isinstance(o, str) else ("pass" if o else "fail") for o in ok
    ]  # explicit states: no NaN-truthiness
    return pd.DataFrame(
        {"gate": [g for _, g in GATES], "value": shown, "pass": state}, index=[n for n, _ in GATES]
    )


def launch_decision(res: dict) -> tuple[pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    """Go / no-go tables: the deployed strategy (walk-forward 2016-26), E2 (backtest pick) and the signal."""
    sel, m, val, rk = (
        res["selected"],
        res["metrics"].loc[res["selected"]],
        res["validity"],
        res["risk"],
    )
    live = (res.get("paper") or {}).get("monitor", {}).get("live_days", 0)
    cap = rk["capacity_at"].get("impact = gross alpha (THB m)", float("nan"))
    e2v = [
        m["alpha t-stat"],
        res["factors_th"].loc[sel, "alpha t"],
        res["factors_em"].loc[sel, "alpha t"],
        m["info ratio"],
        m["DSR"],
        m["track record (yrs)"] - m["MinTRL (yrs)"],
        val.get("pbo_2020_26", {}).get("PBO", float("nan")),
        cap,
    ]
    e2 = _gates(
        e2v,
        [
            f"{e2v[0]:.2f}",
            f"{e2v[1]:.2f}",
            f"{e2v[2]:.2f}",
            f"{e2v[3]:.2f}",
            f"{e2v[4]:.2f}",
            f"{m['track record (yrs)']:.1f} vs {m['MinTRL (yrs)']:.0f} years",
            f"{e2v[6]:.2f}",
            f"THB {cap:.1f}m",
            f"{live} trading days",
        ],
        live,
    )
    dep = None
    d = (res.get("deployed") or {}).get("meta")
    if d:
        dv = [
            d["CAPM alpha t"],
            d["TH-Carhart alpha t"],
            d["Carhart-EM alpha t"],
            d["IR vs TDEX"],
            d["DSR"],
            d["years"] - d["MinTRL yrs"],
            d["pbo_alpha_t"],
            None,
        ]
        dep = _gates(
            dv,
            [
                f"{dv[0]:.2f}",
                f"{dv[1]:.2f}",
                f"{dv[2]:.2f}",
                f"{dv[3]:.2f}",
                f"{dv[4]:.2f} ({d['trials']} trials)",
                f"{d['years']:.1f} vs {d['MinTRL yrs']:.1f} years",
                f"{dv[6]:.2f}",
                "not measured",
                f"{live} trading days",
            ],
            live,
        )
    h = val["haircuts"]
    ic_t = float(h.loc[[i for i in h.index if "raw IC" in i][0], "haircut t bonferroni"])
    yr_min = float(res["signal"]["test"]["ic_by_year"]["IC mean"].min())
    sig = pd.DataFrame(
        {
            "gate": [g for _, g in SIGNAL_GATES],
            "value": [f"{ic_t:.2f}", f"lowest {yr_min:.3f}"],
            "pass": ["pass" if ic_t > 3 else "fail", "pass" if yr_min > 0 else "fail"],
        },
        index=[n for n, _ in SIGNAL_GATES],
    )
    return dep, e2, sig


def _verdict_badge(state: str) -> str:
    if state == "pending":
        return status_badge("amber").replace("Weakening", "Pending")
    return (
        status_badge("green").replace("Healthy", "Pass")
        if state == "pass"
        else status_badge("red").replace("Broken", "Fail")
    )


def _gate_table(df: pd.DataFrame) -> str:
    rows = "".join(
        f"<tr><th scope='row'>{_esc(i)}</th><td class='l'>{_esc(r['gate'])}</td><td>{_esc(r['value'])}</td>"
        f"<td class='l'>{_verdict_badge(r['pass'])}</td></tr>"
        for i, r in df.iterrows()
    )
    return (
        "<div class='tbl-wrap'><table class='tbl gates'><thead><tr><th scope='col'>Criterion</th><th class='l' scope='col'>Gate</th>"
        f"<th scope='col'>Value</th><th class='l' scope='col'>Result</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def launch_card(res: dict) -> str:
    try:
        dep, e2, sig = launch_decision(res)
    except (KeyError, IndexError, TypeError):
        return ""
    main, label = (
        (dep, "Deployed strategy: integrated walk-forward, out of sample 2016-26")
        if dep is not None
        else (e2, "Strategy (E2)")
    )
    n_ok, n = int((main["pass"] == "pass").sum()), len(main)
    n_pend = int((main["pass"] == "pending").sum())
    e2_html = (
        details("E2 (backtest pick) for comparison", _gate_table(e2)) if dep is not None else ""
    )
    return (
        f"<section class='block launch'><h3>Launch decision: no-go for capital, signal kept, strategy in paper trading</h3>"
        f"<p class='note'>The deployed strategy passes {n_ok} of {n} launch gates ({n_pend} pending); the signal passes "
        f"{int((sig['pass'] == 'pass').sum())} of {len(sig)}. Gates were set before reading the results. Review again "
        "after six months of live paper trading or when new information (foreign / NVDR flows) is added.</p>"
        f"<h4>{_esc(label)}</h4>{_gate_table(main)}{e2_html}"
        f"<h4>Signal (the model's ranking)</h4>{_gate_table(sig)}</section>"
    )


def tab_trust(res: dict, specs: dict) -> str:
    val, rk, sel, m = res.get("validity"), res.get("risk"), res["selected"], res["metrics"]
    if not val or not rk:
        return "<p class='empty'>Run the simulate step to build the validity and risk layers.</p>"
    vt = val["table"]
    h = val["haircuts"]
    ic_row = [i for i in h.index if "raw IC" in i]
    verdicts = [
        (
            "good",
            "The signal is real",
            f"IC t-stat stays {h.loc[ic_row[0], 'haircut t bonferroni']:.1f} after a Bonferroni haircut for {val['n_trials']} trials "
            f"(hurdle {h.iloc[0]['t hurdle (Bonferroni, 5%)']:.2f})."
            if ic_row
            else "",
        ),
        (
            "crit",
            "The strategy's Sharpe ratio is not proven",
            f"After the haircut it is {vt.loc[sel, 'haircut SR Bonferroni']:.2f}. Proving the test Sharpe needs "
            f"{vt.loc[sel, 'MinTRL SR>0 @95% (yrs)']:.0f} years of record; there are {vt.loc[sel, 'track record (yrs)']:.1f}.",
        ),
        (
            "warn",
            "Design choices do not carry over",
            f"Probability of backtest overfitting {val.get('pbo_test', {}).get('PBO', float('nan')):.2f} on the test years and "
            f"{val.get('pbo_2020_26', {}).get('PBO', float('nan')):.2f} on 2020-26 (below 0.5 is acceptable).",
        ),
        (
            "warn",
            "Capacity is tiny",
            f"Square-root market impact uses up the whole gross alpha at about THB "
            f"{rk['capacity_at'].get('impact = gross alpha (THB m)', float('nan')):.1f}m (k = {rk['impact_k']}).",
        ),
    ]
    d = (res.get("deployed") or {}).get("meta")
    if d:  # report on the deployed strategy; E2 stays in the tables below
        verdicts[1] = (
            "crit",
            "The deployed strategy's edge is not proven",
            f"Walk-forward 2016-26: CAGR {d['CAGR %']:.1f}% vs TDEX {d['TDEX CAGR %']:.1f}%, CAPM alpha t "
            f"{d['CAPM alpha t']:.2f}, deflated Sharpe {d['DSR']:.2f} after {d['trials']} trials. It does meet the "
            f"minimum track record ({d['years']:.1f} vs {d['MinTRL yrs']:.1f} years).",
        )
        verdicts[2] = (
            "warn",
            "Tuning does not carry over",
            f"Probability of backtest overfitting {d['pbo_alpha_t']:.2f} across the 288 tuned designs; each year's "
            "pick ranked mid-table the following year.",
        )
        verdicts[3] = (
            "warn",
            "Capacity measured for E2 only",
            f"E2's gross alpha is used up by market impact at about THB "
            f"{rk['capacity_at'].get('impact = gross alpha (THB m)', float('nan')):.1f}m; the deployed strategy "
            "trades less (14× vs 27× a year) but has not been measured.",
        )
    icons = {"good": STATUS["green"][2], "warn": STATUS["amber"][2], "crit": STATUS["red"][2]}
    vlist = "".join(
        f"<li class='verdict {tone}'><svg viewBox='0 0 16 16' aria-hidden='true'>{icons[tone]}</svg>"
        f"<div><b>{_esc(head)}</b><p>{_esc(txt)}</p></div></li>"
        for tone, head, txt in verdicts
    )
    cap = rk["capacity"]
    cap_html, spec = chart(
        "capacity",
        [
            line(
                cap["net CAGR %"],
                f"impact k = {rk['impact_k']}",
                "var(--s1)",
                mode="lines+markers",
                marker={"size": 8, "line": {"width": 2, "color": "var(--surface)"}},
            ),
            line(
                cap["net CAGR % (k=0.5)"],
                "impact k = 0.5",
                "var(--s2)",
                mode="lines+markers",
                marker={"size": 8, "line": {"width": 2, "color": "var(--surface)"}},
            ),
        ],
        f"Net CAGR of {short(sel)} as the fund grows, %",
        height=300,
        xaxis={"type": "log", "title": {"text": "Fund size, THB m"}},
        hovermode="x unified",
        shapes=[
            {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "y0": float(m.loc[BENCH, "CAGR %"]),
                "y1": float(m.loc[BENCH, "CAGR %"]),
                "line": {"color": "var(--ref)", "width": 1},
            }
        ],
        annotations=[
            {
                "xref": "paper",
                "x": 1,
                "y": float(m.loc[BENCH, "CAGR %"]),
                "text": "TDEX",
                "showarrow": False,
                "xanchor": "right",
                "yshift": 10,
                "font": {"color": "var(--ink-2)"},
            }
        ],
    )
    specs.update(spec)
    rd = rk["risk_decomp"]
    comp = [
        ("MKT %", "Market", "var(--s1)"),
        ("SMB %", "Size", "var(--s2)"),
        ("WML %", "Momentum", "var(--s3)"),
        ("specific %", "Stock-specific", "var(--s4)"),
    ]
    rd_data = [
        bar([short(i) for i in rd.index], rd[c].tolist(), lab, col, orientation="h")
        for c, lab, col in comp
    ]
    for t in rd_data:  # 2px surface gap between stacked segments
        t["marker"]["line"] = {"color": "var(--surface)", "width": 2}
    rd_html, spec = chart(
        "riskdec",
        rd_data,
        "Where the risk comes from: share of daily variance, %",
        height=300,
        barmode="relative",
        bargap=0.45,
        yaxis={"autorange": "reversed"},
        margin={"l": 92},
    )
    specs.update(spec)
    stress = rk["stress"].copy()
    keep = ["from", "to", "days"] + [
        c for c in stress.columns if c.endswith(" %") and "active" not in c and "max DD" not in c
    ]
    stress = stress[keep].rename(
        columns=lambda c: short(c.removesuffix(" %")) if c.endswith(" %") else c
    )
    te = pd.Series(rk["te"], dtype=object).to_frame("value")
    return (
        launch_card(res)
        + f"<ul class='verdicts'>{vlist}</ul>"
        + ("<h3 class='sub'>Backtest strategies (E1-E6): risk detail</h3>" if d else "")
        + "<div class='grid2'>"
        + section("Capacity", cap_html)
        + section("Risk decomposition", rd_html)
        + "</div>"
        + section(
            "Stress windows: compounded return over each episode, %",
            table(stress, tstats=False, index_name="Episode"),
        )
        + "<div class='grid2'>"
        + section(
            "Tracking error: forecast vs realised",
            table(te, tstats=False),
            "A bias statistic below 1 means the risk model over-forecasts.",
        )
        + section(
            "Return decomposition, % a year",
            table(rk["return_decomp"].T.rename(columns=short), tstats=False),
        )
        + "</div>"
        + section(
            "Sharpe validity per strategy",
            table(
                pd.concat([vt, val.get("long_history", pd.DataFrame())]),
                {sel: "sel", BENCH: "ref"},
                shorten=True,
                index_name="Strategy",
            ),
            f"Trials counted: {val['n_trials']} ({', '.join(f'{k} {n}' for k, n in val['trials'].items())}).",
        )
        + section(
            "Multiple-testing haircuts",
            details("Bonferroni, Holm and BHY detail", table(h.T, tstats=False)),
        )
    )


def tab_diagnostics(res: dict, specs: dict) -> str:
    sel = res["selected"]
    reg = res["regimes"]
    p_html, spec = chart(
        "regime-p",
        [
            line(
                reg["hmm_p_stressed"],
                "P(stressed)",
                "var(--s3)",
                fill="tozeroy",
                fillcolor="var(--s3-wash)",
            )
        ],
        "Gaussian HMM: probability of the stressed regime",
        height=220,
        showlegend=False,
        yaxis={"range": [0, 1]},
    )
    specs.update(spec)
    px_html, spec = chart(
        "regime-px",
        [line(reg["bench_close"], "TDEX", "var(--ref)")],
        "TDEX price (same dates)",
        height=200,
        showlegend=False,
    )
    specs.update(spec)
    ro = res["risk"]["rolling"]
    ra_html, spec = chart(
        "roll-alpha",
        [line(ro["alpha % p.a."], "alpha", "var(--s1)")],
        f"Rolling 12-month CAPM alpha of {short(sel)}, % a year",
        height=240,
        showlegend=False,
    )
    specs.update(spec)
    ri_html, spec = chart(
        "roll-ir",
        [line(ro["IR"], "IR", "var(--s1)")],
        "Rolling 12-month information ratio vs TDEX",
        height=240,
        showlegend=False,
    )
    specs.update(spec)
    imp = res["importance"].sort_values(ascending=False)
    imp_html, spec = chart(
        "importance",
        [bar(imp.index, imp.tolist(), "importance", "var(--s1)", orientation="h")],
        "XGBoost feature importance (top 25)",
        height=620,
        showlegend=False,
        bargap=0.4,
        yaxis={"autorange": "reversed"},
        margin={"l": 150},
    )
    specs.update(spec)
    mv = res["metrics_valid"].drop(columns=["selected"], errors="ignore")
    cols = [
        c
        for c in [
            "CAGR %",
            "Sharpe",
            "max DD %",
            "alpha % p.a.",
            "alpha t-stat",
            "DSR",
            "turnover x/yr",
        ]
        if c in mv.columns
    ]
    return (
        "<div class='grid2'>"
        + section(
            "Market regime",
            p_html + px_html,
            "Two panels on one time axis instead of a dual-axis chart.",
        )
        + section("Rolling alpha and information ratio", ra_html + ri_html)
        + "</div>"
        + section(
            "How the selection was made: validation years 2020-21",
            table(mv[cols], {sel: "sel", BENCH: "ref"}, shorten=True, index_name="Strategy"),
            "The strategy with the highest validation Sharpe was frozen before the test years were run.",
        )
        + section(
            "Where the picks work: top-decile picks by sector, year and regime (test)",
            table(res["by_sector"], tstats=False, index_name="Sector")
            + table(res["by_year"].sort_index(), tstats=False, index_name="Year")
            + table(res["by_regime"], tstats=False, index_name="Regime"),
            "Lift = precision minus the base rate, in percentage points.",
        )
        + section(
            "Model quality, test years",
            table(res["scorecard"], index_name="Model"),
            "IC t-stats here treat overlapping 5-day targets as independent; the Signal tab has the corrected ones.",
        )
        + section(
            "Vector simulations (THB 100 per signal, 5-day hold)",
            table(res["vector"], tstats=False, index_name="Rule"),
        )
        + section("What the model looks at", imp_html)
    )


# ----------------------------------------------------------------------------- page
CSS = """
:root{--font:"IBM Plex Sans Thai",system-ui,-apple-system,"Segoe UI",sans-serif;
--plane:#edf1ee;--surface:#fcfcfb;--ink:#17211e;--ink-2:#4c5752;--muted:#7a827e;--line:#dce2de;--grid:#e1e0d9;--axis:#c3c2b7;
--brand:#3f7566;--brand-wash:#dfe9e4;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s3-wash:rgba(27,175,122,.12);--ref:#52514e;--de:#cfcec8;
--pos:#2a78d6;--neg:#e34948;--mid:#f0efec;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--good-ink:#006300;--crit-ink:#b42d2d;--warn-ink:#8a5a00;
--sel-row:#e8f0fb;--ref-row:#f1f1ee;color-scheme:light}
:root[data-theme=dark]{--plane:#0f1312;--surface:#1a1a19;--ink:#f4f6f5;--ink-2:#c3c2b7;--muted:#9a9d98;--line:#2c312f;--grid:#2c2c2a;--axis:#383835;
--brand:#7fb3a2;--brand-wash:#1f2b27;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s3-wash:rgba(25,158,112,.18);--ref:#c3c2b7;--de:#4a4a46;
--pos:#3987e5;--neg:#e66767;--mid:#383835;--good-ink:#0ca30c;--crit-ink:#e66767;--warn-ink:#fab219;--sel-row:#1d2a3a;--ref-row:#252523;color-scheme:dark}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--plane:#0f1312;--surface:#1a1a19;--ink:#f4f6f5;--ink-2:#c3c2b7;--muted:#9a9d98;
--line:#2c312f;--grid:#2c2c2a;--axis:#383835;--brand:#7fb3a2;--brand-wash:#1f2b27;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;
--s3-wash:rgba(25,158,112,.18);--ref:#c3c2b7;--de:#4a4a46;--pos:#3987e5;--neg:#e66767;--mid:#383835;--good-ink:#0ca30c;--crit-ink:#e66767;
--warn-ink:#fab219;--sel-row:#1d2a3a;--ref-row:#252523;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.55 var(--font);-webkit-font-smoothing:antialiased}
a{color:var(--brand)}
code{font-size:.9em;background:var(--brand-wash);padding:1px 5px;border-radius:4px}
.top{position:sticky;top:0;z-index:5;background:var(--plane);border-bottom:1px solid var(--line)}
.bar{max-width:1240px;margin:auto;padding:14px 20px 0;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
.brand{font-size:19px;font-weight:600;letter-spacing:-.01em;margin:0}
.brand span{color:var(--brand);font-weight:500;margin-left:6px}
.meta{color:var(--muted);font-size:13px;flex:1}
.theme{border:1px solid var(--line);background:var(--surface);color:var(--ink-2);border-radius:8px;padding:5px 10px;font:inherit;font-size:13px;cursor:pointer}
nav.tabs{max-width:1240px;margin:auto;padding:6px 14px 0;display:flex;gap:2px;overflow-x:auto;scrollbar-width:none}
nav.tabs button{border:0;background:none;color:var(--ink-2);font:inherit;font-size:14px;padding:10px 12px 11px;cursor:pointer;
border-bottom:2px solid transparent;white-space:nowrap}
nav.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--brand);font-weight:600}
button:focus-visible,summary:focus-visible{outline:2px solid var(--brand);outline-offset:2px}
main{max-width:1240px;margin:auto;padding:22px 20px 60px}
.panel{display:none}.panel.on{display:block}
.hero{display:grid;grid-template-columns:minmax(0,1.9fr) minmax(280px,1fr);gap:20px;margin-bottom:22px}
.hero-main,.hero-side,.block{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 20px}
.hero-main{border-top:4px solid var(--brand)}
.hero-main,.hero-side{min-width:0}
.ticket td:nth-child(3){padding-right:18px}
.kicker{margin:0;color:var(--muted);font-size:13px}
.hero h2{font-size:32px;line-height:1.15;letter-spacing:-.02em;font-weight:600;margin:6px 0 10px}
.hero-side h3,.block h3{font-size:16px;font-weight:600;margin:0 0 8px;letter-spacing:-.005em}
.facts{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;color:var(--ink-2);font-size:14px;margin-bottom:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:1px}.dot.s1{background:var(--s1)}
.badge{display:inline-flex;gap:6px;align-items:center;font-size:13px;font-weight:600;padding:3px 10px;border-radius:999px;
background:var(--ref-row);color:var(--ink-2)}
.badge svg{width:14px;height:14px}
.badge.good{color:var(--good-ink);background:color-mix(in srgb,var(--good) 14%,transparent)}
.badge.warn{color:var(--warn-ink);background:color-mix(in srgb,var(--warn) 20%,transparent)}
.badge.crit{color:var(--crit-ink);background:color-mix(in srgb,var(--crit) 14%,transparent)}
.health-head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:14px;color:var(--ink-2)}
.note{color:var(--muted);font-size:13px;margin:8px 0 0;max-width:78ch}
.block{margin-bottom:18px;min-width:0}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.grid2>.block{margin-bottom:18px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile-label{color:var(--ink-2);font-size:13px}
.tile-value{font-size:28px;font-weight:600;letter-spacing:-.02em;margin-top:2px}
.tile-sub{color:var(--muted);font-size:12.5px}
.tile.bad .tile-value{color:var(--crit-ink)}.tile.good .tile-value{color:var(--good-ink)}
.tbl-wrap{overflow-x:auto;margin-top:8px}
.tbl{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:max-content;max-width:none}
.tbl.ticket{width:100%}
.tbl thead th:first-child{text-align:left}
figcaption{font-size:13.5px;color:var(--ink-2);margin:4px 0 2px}
.launch{border-top:4px solid var(--brand)}h3.sub{font-size:15px;margin:6px 0 10px;color:var(--ink-2)}.launch h4{font-size:14px;font-weight:600;margin:16px 0 0}
.tbl th,.tbl td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
.tbl thead th{color:var(--muted);font-weight:500;font-size:12.5px;border-bottom:1px solid var(--axis);vertical-align:bottom;white-space:normal;min-width:64px}
.tbl tbody th{text-align:left;font-weight:500;position:sticky;left:0;background:var(--surface)}
.tbl .l{text-align:left}
.tbl td.strong{font-weight:700;color:var(--ink)}
.tbl tr.sel th,.tbl tr.sel td{background:var(--sel-row)}.tbl tr.sel th{box-shadow:inset 3px 0 var(--s1)}
.tbl tr.ref th,.tbl tr.ref td{background:var(--ref-row)}
.ticket td,.ticket th{font-size:14px;padding:8px 10px}
.wbar{display:inline-block;height:6px;width:calc(var(--w) * .6);max-width:60px;background:var(--s1);border-radius:3px;margin-right:8px;vertical-align:2px}
details.more{margin-top:10px}details.more summary{cursor:pointer;color:var(--brand);font-size:14px}
.empty,.empty-state{color:var(--ink-2)}.empty-state{background:var(--brand-wash);border-radius:10px;padding:12px 16px}
.empty-state p{margin:4px 0}
.verdicts{list-style:none;padding:0;margin:0 0 18px;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.verdict{display:flex;gap:10px;background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.verdict svg{flex:none;width:18px;height:18px;margin-top:3px}.verdict p{margin:2px 0 0;color:var(--ink-2);font-size:13.5px}
.verdict.good svg{color:var(--good-ink)}.verdict.warn svg{color:var(--warn-ink)}.verdict.crit svg{color:var(--crit-ink)}
figure.chart{margin:4px 0 0;min-width:0;max-width:100%;overflow:hidden}.plot{width:100%;max-width:100%}
.tbl-wrap{max-width:100%}
main,.panel,.block,.grid2>*{min-width:0}
@media (max-width:900px){.hero,.grid2{grid-template-columns:minmax(0,1fr)}.hero h2{font-size:26px}}
@media (max-width:520px){main,.bar{padding-left:16px;padding-right:16px}.tile-value{font-size:24px}.meta{order:3;flex-basis:100%;font-size:12px}.brand span{display:block;margin:0}.hero-main,.hero-side,.block{padding:14px}}
"""

JS = """
(function(){
const root=document.documentElement;
try{const t=localStorage.getItem('sqa-theme');if(t)root.dataset.theme=t;}catch(e){}
const SPECS=JSON.parse(document.getElementById('chart-specs').textContent);
const drawn=new Set();
function tok(name){return getComputedStyle(root).getPropertyValue(name).trim();}
function resolve(v){
  if(typeof v==='string'){const m=v.match(/^var\\((--[\\w-]+)\\)$/);return m?tok(m[1]):v;}
  if(Array.isArray(v))return v.map(resolve);
  if(v&&typeof v==='object'){const o={};for(const k in v)o[k]=resolve(v[k]);return o;}
  return v;
}
function base(){return{font:{family:tok('--font'),color:tok('--ink-2'),size:12},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',
  margin:{t:34,r:16,b:36,l:56},
  legend:{orientation:'h',y:1.0,yanchor:'bottom',x:0,font:{size:12},itemclick:'toggle'},
  xaxis:{gridcolor:tok('--grid'),linecolor:tok('--axis'),zerolinecolor:tok('--axis'),tickfont:{color:tok('--muted')}},
  yaxis:{gridcolor:tok('--grid'),linecolor:tok('--axis'),zerolinecolor:tok('--axis'),tickfont:{color:tok('--muted')}},
  hoverlabel:{bgcolor:tok('--surface'),bordercolor:tok('--line'),font:{color:tok('--ink'),family:tok('--font')}}};}
function merge(a,b){for(const k in b){if(b[k]&&typeof b[k]==='object'&&!Array.isArray(b[k])&&a[k]&&typeof a[k]==='object')merge(a[k],b[k]);else a[k]=b[k];}return a;}
function draw(id){const el=document.getElementById(id);if(!el||!window.Plotly)return;
  const s=resolve(SPECS[id]);const layout=merge(base(),s.layout);
  Plotly.react(el,s.data,layout,{displaylogo:false,responsive:true,modeBarButtonsToRemove:['lasso2d','select2d']});drawn.add(id);}
function drawPanel(p){p.querySelectorAll('[data-chart]').forEach(el=>draw(el.id));}
const tabs=[...document.querySelectorAll('nav.tabs button')];
function show(id,push){tabs.forEach(b=>b.setAttribute('aria-selected',b.dataset.tab===id));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id==='p-'+id));
  const p=document.getElementById('p-'+id);if(p)drawPanel(p);if(push)history.replaceState(null,'','#'+id);window.scrollTo(0,0);}
tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab,true)));
const start=(location.hash||'').slice(1);show(tabs.some(b=>b.dataset.tab===start)?start:tabs[0].dataset.tab,false);
document.getElementById('theme').addEventListener('click',()=>{
  const dark=getComputedStyle(root).getPropertyValue('--surface').trim()==='#1a1a19';
  root.dataset.theme=dark?'light':'dark';try{localStorage.setItem('sqa-theme',root.dataset.theme);}catch(e){}
  drawn.forEach(id=>draw(id));});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>drawn.forEach(id=>draw(id)));
})();
"""

TABS = [
    ("today", "Today", tab_today),
    ("performance", "Performance", tab_performance),
    ("factors", "Factors", tab_factors),
    ("signal", "Signal", tab_signal),
    ("trust", "Trust & risk", tab_trust),
    ("diagnostics", "Diagnostics", tab_diagnostics),
]


def _clean(v):
    """JSON-safe: NaN / inf -> null, numpy scalars -> Python."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_clean(x) for x in v]
    if isinstance(v, float | np.floating):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def render(res: dict) -> str:
    specs: dict = {}
    panels = "".join(
        f"<div class='panel' id='p-{tid}' role='tabpanel' aria-label='{_esc(label)}'>{fn(res, specs)}</div>"
        for tid, label, fn in TABS
    )
    buttons = "".join(
        f"<button role='tab' data-tab='{tid}' aria-selected='false'>{_esc(label)}</button>"
        for tid, label, _ in TABS
    )
    spec_json = json.dumps(_clean(specs), allow_nan=False, default=str).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SET alpha monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style><script src="{PLOTLY_JS}" defer></script></head><body>
<header class="top"><div class="bar"><h1 class="brand">SET alpha monitor<span>หุ้นไทย top-100</span></h1>
<span class="meta">Prices to {_esc(_day(res['last_date']))}. Test period from {_esc(_day(res['test_start']))}. Page built {datetime.now():%d %b %Y, %H:%M}.</span>
<button id="theme" class="theme" type="button" aria-label="Switch light or dark theme">Light / dark</button></div>
<nav class="tabs" role="tablist" aria-label="Dashboard sections">{buttons}</nav></header>
<main>{panels}</main>
<script id="chart-specs" type="application/json">{spec_json}</script>
<script>window.addEventListener('load',function(){{{JS}}});</script>
</body></html>"""
