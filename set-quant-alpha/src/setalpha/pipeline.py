"""The end-to-end workflow as seven step functions.

The same functions are called by the CLI (`scripts/run_pipeline.py`), the master notebook
and the Airflow DAG, so there is exactly one implementation of every step:

    ingest -> transform -> train -> simulate -> signals -> paper -> report

Each step reads its inputs from and writes its outputs to the data / models / reports
folders, which is what lets Airflow run them as separate tasks.
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from setalpha import (
    config,
    evaluate,
    factors,
    ingest,
    model,
    paper,
    report,
    risk,
    signal_eval,
    strategy,
    transform,
    validity,
)

log = logging.getLogger("setalpha")
SIM_RESULTS = config.PROCESSED_DIR / "sim_results.joblib"
TRIAL_RETURNS = config.PROCESSED_DIR / "trial_returns.joblib"  # written by scripts/trial_returns.py
EXPERIMENTS = config.PROJECT_DIR / "experiments"
BEST_PARAMS = config.MODELS_DIR / "best_params.json"
STEPS = ["ingest", "transform", "train", "simulate", "signals", "paper", "report"]


def step_ingest(mode: ingest.Mode = "file") -> dict:
    raw = ingest.ingest_all(mode)
    log.info(
        "ingest: %s price rows, %d tickers, last date %s",
        f"{len(raw['prices']):,}",
        raw["prices"]["Ticker"].nunique(),
        raw["prices"]["Date"].max().date(),
    )
    return raw


def step_transform(raw: dict | None = None) -> pd.DataFrame:
    raw = raw or ingest.ingest_all("file")
    df = transform.build_dataset(raw)
    transform.save(df)
    return df


def _model_frame() -> tuple[pd.DataFrame, list[str]]:
    df = transform.add_dummies(transform.load())
    sets = transform.feature_sets(df)
    return df, sets["NUMERIC"] + sets["DUMMIES"]


def step_train(retune: bool = False) -> dict:
    df, feats = _model_frame()
    if retune or not BEST_PARAMS.exists():
        best, logs = model.tune(model.time_split(df), feats)
        BEST_PARAMS.write_text(json.dumps(best, indent=1))
        logs.to_csv(config.REPORTS_DIR / "tuning_log.csv", index=False)
    best = json.loads(BEST_PARAMS.read_text())
    preds = model.walk_forward(df, feats, best)
    return finalize_models(df, feats, best, preds)


def finalize_models(df: pd.DataFrame, feats: list[str], best: dict, preds: pd.DataFrame) -> dict:
    """Validation IC + decision thresholds -> meta.json, and the production refit on all data."""
    valid = preds[preds["split"] == "valid"].merge(
        df[["Date", "Ticker", "tradable", "fwd_alpha_5"]], on=["Date", "Ticker"]
    )
    trad = valid[valid["tradable"]]
    members, val_alpha = model.choose_ensemble(preds, df)
    preds = model.add_ensemble(preds, members)
    preds.to_parquet(model.PREDICTIONS, index=False)
    valid = preds[preds["split"] == "valid"].merge(
        df[["Date", "Ticker", "tradable", "fwd_alpha_5"]], on=["Date", "Ticker"]
    )
    trad = valid[valid["tradable"]]
    meta = {
        "ensemble": members,
        "valid_top_decile_alpha_%": {n: 100 * a for n, a in val_alpha.items()},
        "features": feats,
        "best_params": best,
        "ic_valid": float(model.daily_ic(valid, "pred_ens").mean()),
        "thr": {
            n: float(trad[f"pred_{n}"].quantile(0.95)) for n in members
        },  # custom rule: top 5% probability
        "trained_through": str(df.loc[df["is_alpha_5"].notna(), "Date"].max().date()),
    }
    model.fit_final(df, feats, best, members)
    report.save_meta(meta)
    log.info("train: IC(valid) %.4f, thresholds %s", meta["ic_valid"], meta["thr"])
    return meta


def _exact(f: pd.DataFrame, prices: pd.DataFrame, bench: pd.DataFrame, ic: float):
    """Run every exact strategy on one split; returns results, daily returns, the stats table and
    the zero-cost (gross) daily returns."""
    start, end = f["Date"].min() - pd.Timedelta(days=250), f["Date"].max() + pd.Timedelta(days=10)
    px = strategy.wide_prices(
        prices[
            (prices["Date"] >= start)
            & (prices["Date"] <= end)
            & prices["Ticker"].isin(f["Ticker"].unique())
        ]
    )
    results = {
        c.name: strategy.exact_sim(c, f, px, bench, ic) for c in strategy.default_exact_configs()
    }
    idx = next(iter(results.values())).equity.index
    daily = pd.DataFrame({k: r.returns for k, r in results.items()})
    daily[f"{config.BENCHMARK} buy & hold"] = bench["Close"].reindex(idx).ffill().pct_change()
    daily["EW tradable top-100 (no costs)"] = strategy.universe_benchmark(px["Close"], f).reindex(
        idx
    )
    daily = daily.iloc[1:]
    metrics = evaluate.compare(
        {c: daily[c] for c in daily.columns},
        daily[f"{config.BENCHMARK} buy & hold"],
        trials=list(results),
    )
    metrics["avg exposure %"] = [
        100 * results[k].exposure.mean() if k in results else 100.0 for k in metrics.index
    ]
    metrics["trades"] = [len(results[k].trades) if k in results else 0 for k in metrics.index]

    # implementation metrics: turnover, cost drag (same strategies re-run at zero cost), break-even cost
    years = len(daily) / 252
    bench_cagr = metrics.loc[f"{config.BENCHMARK} buy & hold", "CAGR %"]
    ew_cagr = metrics.loc["EW tradable top-100 (no costs)", "CAGR %"]
    gross_daily = {}
    for k, r in results.items():
        traded = float((r.trades["qty"] * r.trades["price"]).abs().sum()) if len(r.trades) else 0.0
        turnover = traded / r.equity.mean() / years
        gross = strategy.exact_sim(
            next(c for c in strategy.default_exact_configs() if c.name == k),
            f,
            px,
            bench,
            ic,
            fee=0.0,
        )
        gross_cagr = 100 * ((gross.equity.iloc[-1] / gross.equity.iloc[0]) ** (1 / years) - 1)
        gross_daily[k] = gross.returns.reindex(daily.index)
        metrics.loc[k, "turnover x/yr"] = turnover
        metrics.loc[k, "gross CAGR %"] = gross_cagr
        metrics.loc[k, "cost drag %"] = gross_cagr - metrics.loc[k, "CAGR %"]
        # fee per side at which the strategy would only match the benchmark (cost/yr ~ turnover x fee)
        metrics.loc[k, "break-even cost vs TDEX (%/side)"] = (
            (gross_cagr - bench_cagr) / turnover if turnover else np.nan
        )
        # turnover-adjusted: the signal's own benchmark (EW universe) and gross CAPM alpha per unit of turnover
        metrics.loc[k, "break-even cost vs EW universe (%/side)"] = (
            (gross_cagr - ew_cagr) / turnover if turnover else np.nan
        )
        g_alpha = evaluate.capm(gross_daily[k], daily[f"{config.BENCHMARK} buy & hold"])[
            "alpha % p.a."
        ]
        metrics.loc[k, "gross alpha % p.a."] = g_alpha
        metrics.loc[k, "gross alpha per 1x turnover (bp)"] = (
            100 * g_alpha / turnover if turnover else np.nan
        )
    return results, daily, metrics, pd.DataFrame(gross_daily)


def step_simulate() -> dict:
    df = transform.load()
    preds = pd.read_parquet(model.PREDICTIONS)
    meta = report.load_meta()
    raw = ingest.ingest_all("file")
    market, prices = raw["market"], raw["prices"]

    bench = market[market["Ticker"] == config.BENCHMARK].set_index("Date").sort_index()
    f = strategy.sim_frame(df, preds, market, split="test")
    vec = strategy.vector_sims(f, meta["thr"])
    vector = strategy.summarize_vector(vec)
    results, daily, metrics, gross = _exact(f, prices, bench, meta["ic_valid"])
    _, daily_valid, metrics_valid, _ = _exact(
        strategy.sim_frame(df, preds, market, split="valid"), prices, bench, meta["ic_valid"]
    )
    selected = metrics_valid.loc[list(results), "Sharpe"].idxmax()  # chosen on 2020-2021 only
    for table in (metrics, metrics_valid):
        table.insert(0, "selected", ["*" if k == selected else "" for k in table.index])
    meta["selected_strategy"] = selected
    report.save_meta(meta)

    full = preds.merge(
        df[["Date", "Ticker", "tradable", "fwd_alpha_5", "is_alpha_5"]], on=["Date", "Ticker"]
    )
    scorecard = model.scorecard(
        full[full["split"] == "test"], [f"pred_{n}" for n in model.MODEL_NAMES] + ["pred_ens"]
    )
    f["year"] = f["Date"].dt.year
    xgb = model.load_final("xgb") if (config.MODELS_DIR / "xgb.joblib").exists() else None
    regimes = (
        df.groupby("Date")[["hmm_p_stressed"]]
        .first()
        .join(bench["Close"].rename("bench_close"), how="left")
    )

    # factor attribution of the test-period daily returns (strategies + benchmarks)
    usdthb = market[market["Ticker"] == "THB=X"].set_index("Date").sort_index()["Close"]
    local = factors.local_factors(df, prices, bench["Close"])
    rets = {c: daily[c] for c in daily.columns}
    factors_th = factors.factor_table(rets, factors.thai_local, local)
    factors_th.loc[f"{config.BENCHMARK} buy & hold"] = np.nan  # TDEX is the MKT factor itself
    factors_em = (
        factors.factor_table(rets, factors.carhart_em, raw["fama_french"], usdthb)
        if raw.get("fama_french") is not None
        else pd.DataFrame()
    )

    res = {
        "last_date": str(df["Date"].max().date()),
        "test_start": config.TEST_START,
        "selected": selected,
        "metrics": metrics,
        "metrics_valid": metrics_valid,
        "vector": vector,
        "scorecard": scorecard,
        "factors_th": factors_th,
        "factors_em": factors_em,
        "local_factors": local,
        "daily_returns": daily,
        "trades": {k: r.trades for k, r in results.items()},
        "exposure": pd.DataFrame({k: r.exposure for k, r in results.items()}),
        "by_sector": evaluate.by_group(f, "sector"),
        "by_year": evaluate.by_group(f, "year"),
        "by_regime": evaluate.by_group(f, "hmm_state"),
        "by_month": evaluate.by_group(f, "month"),
        "by_size": evaluate.by_group(f, "size_q"),
        "worst_calls": evaluate.worst_calls(f),
        "importance": model.feature_importance(xgb["model"], xgb["features"])
        if xgb
        else pd.Series(dtype=float),
        "regimes": regimes[regimes.index >= "2013-01-01"],
        "vector_frame": vec,
        "daily_valid": daily_valid,
        "gross_returns": gross,
        "equity": pd.DataFrame({k: r.equity for k, r in results.items()}),
    }
    sig = _signal_research(
        df, preds, prices, results[selected], res["equity"][selected], daily, gross, selected
    )
    res["signal"] = sig
    res["validity"] = _validity(daily, daily_valid, sig["test"], selected)
    res["risk"] = _risk_capacity(df, prices, bench, results[selected], res, selected)
    joblib.dump(res, SIM_RESULTS)
    log.info(
        "simulate (selected on validation: %s):\n%s",
        selected,
        metrics[["CAGR %", "Sharpe", "max DD %", "alpha % p.a.", "DSR"]].round(3).to_string(),
    )
    return res


# ----------------------------------------------------------------------------- research layer
def _signal_research(
    df,
    preds,
    prices,
    sel: strategy.ExactResult,
    equity: pd.Series,
    daily: pd.DataFrame,
    gross: pd.DataFrame,
    selected: str,
) -> dict:
    """Alphalens-style tear-sheet on validation and test + the fundamental-law check of the selected book."""
    fwd = signal_eval.forward_alphas(prices, df)
    frames = {s: signal_eval.research_frame(preds, df, fwd, s) for s in ("valid", "test")}
    out = {s: signal_eval.tearsheet(f) for s, f in frames.items()}
    close = prices[prices["Ticker"].isin(sel.trades["Ticker"].unique())].pivot_table(
        index="Date", columns="Ticker", values="Close"
    )
    w = signal_eval.weights(sel.trades, close, equity)
    tc = signal_eval.transfer_coefficient(w, frames["test"])
    active_gross = (gross[selected] - daily["EW tradable top-100 (no costs)"]).dropna()
    t = out["test"]
    out["tc"] = tc
    out["weights"] = w
    out["fundamental_law"] = signal_eval.fundamental_law(
        t["ic_summary"].loc["raw IC", "IC mean"],
        t["names_per_day"],
        float(tc.mean()),
        realised_ir=float(active_gross.mean() / active_gross.std() * np.sqrt(252)),
    )
    return out


def _trial_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """Test-period daily returns of every saved design: pipeline E1-E6 + strategy_v2 + strategy_v3."""
    cols = {f"pipe: {c}": daily[c] for c in daily.columns if c[:1] == "E" and c[1].isdigit()}
    v2, v3 = (
        EXPERIMENTS / "strategy_v2/output/test_daily_returns.joblib",
        EXPERIMENTS / "strategy_v3/output/test_daily_returns.parquet",
    )
    if v2.exists():
        cols.update({f"v2: {k}": v for k, v in joblib.load(v2).items()})
    if v3.exists():
        cols.update({f"v3: {k}": v for k, v in pd.read_parquet(v3).items()})
    return pd.DataFrame(cols).loc[daily.index[0] :]


def _validity(
    daily: pd.DataFrame, daily_valid: pd.DataFrame, sig_test: dict, selected: str
) -> dict:
    """Lo Sharpe, MinTRL, HLZ haircuts and PBO (CSCV) on the saved trial returns."""
    trials = validity.dedupe(_trial_matrix(daily))
    p_test = {k: validity.p_value(validity.sharpe_tstat(trials[k])) for k in trials.columns}
    out = {
        "trials": validity.TRIALS,
        "n_trials": validity.N_TRIALS,
        "trial_matrix_test": list(trials.columns),
        "pbo_test": validity.pbo_cscv(trials),
        "table": validity.validity_table(
            {c: daily[c] for c in daily.columns}, list(p_test.values())
        ),
    }
    short, key = selected.split()[0], f"pipe: {selected}"
    rows = {
        f"{short} Sharpe, test 2022-26": validity.haircut(
            validity.sharpe_tstat(daily[selected]),
            len(daily[selected].dropna()) / 252,
            [p for k, p in p_test.items() if k != key],
        )
    }
    for name in ("raw IC", "factor-neutral IC"):  # the signal itself, overlap-robust t
        rows[f"signal {name}, test (non-overlapping t)"] = validity.haircut(
            float(sig_test["ic_summary"].loc[name, "t (non-overlapping)"])
        )
    if TRIAL_RETURNS.exists():  # rebuilt by scripts/trial_returns.py
        out.update(
            _long_history_validity(
                joblib.load(TRIAL_RETURNS), daily, daily_valid, list(p_test.values()), rows
            )
        )
    out["haircuts"] = pd.DataFrame(rows).T
    return out


def _long_history_validity(
    tr: dict, daily: pd.DataFrame, daily_valid: pd.DataFrame, others: list[float], rows: dict
) -> dict:
    """E2 over its 2013-21 selection period (strategy_v3 run) and PBO on 2020-26 (validation + test)."""
    sel = tr["selection_2013_2021"]
    e2 = sel[[c for c in sel.columns if c.startswith("E2")][0]]
    sel_csv = EXPERIMENTS / "strategy_v3/output/select_2013_2021.csv"
    others_sel = (
        [
            validity.p_value(t)
            for k, t in pd.read_csv(sel_csv, index_col=0)["CAPM alpha t"].dropna().items()
            if not k.startswith("E2")
        ]
        if sel_csv.exists()
        else []
    )
    rows["E2 CAPM alpha t, 2013-21"] = validity.haircut(
        evaluate.capm(e2, sel[f"{config.BENCHMARK} buy & hold"])["alpha t-stat"], None, others_sel
    )
    rows["E2 Sharpe, 2013-21"] = validity.haircut(
        validity.sharpe_tstat(e2), len(e2) / 252, others_sel
    )

    test = _trial_matrix(daily)
    pipe_valid = daily_valid[
        [c for c in daily_valid.columns if c[:1] == "E" and c[1].isdigit()]
    ].add_prefix("pipe: ")
    valid = pd.concat([pipe_valid, tr["strategy_v2"].add_prefix("v2: ")], axis=1).loc[
        : config.TEST_START
    ]
    long = validity.dedupe(pd.concat([valid, test[valid.columns]]).dropna())
    return {
        "long_history": validity.validity_table({"E2 (strategy_v3 run), 2013-21": e2}, others),
        "pbo_2020_26": validity.pbo_cscv(long),
        "trial_matrix_2020_26": list(long.columns),
    }


def _risk_capacity(df, prices, bench, sel: strategy.ExactResult, res: dict, selected: str) -> dict:
    """ES, stress windows, ex-ante vs realised TE, risk/return decomposition, capacity, reporting tables."""
    daily, bname, ew = (
        res["daily_returns"],
        f"{config.BENCHMARK} buy & hold",
        "EW tradable top-100 (no costs)",
    )
    r, b = daily[selected], daily[bname]
    keep = [selected, "E1 equal-weight top-10", bname, ew]
    both = pd.concat([res["daily_valid"][keep], daily[keep]]).sort_index()
    w, equity = res["signal"]["weights"], res["equity"][selected]
    tickers = list(w.columns)
    close = prices[prices["Ticker"].isin(tickers)].pivot_table(
        index="Date", columns="Ticker", values="Close"
    )
    stock_rets = close.pct_change(fill_method=None).reindex(columns=tickers)
    bench_ret = bench["Close"].reindex(stock_rets.index).ffill().pct_change()
    te = risk.ex_ante_te(
        w, stock_rets, bench_ret, (r - b).dropna(), equity.index[:: config.HORIZON]
    )

    panel = df[df["Ticker"].isin(tickers) & (df["Date"] >= equity.index[0] - pd.Timedelta(days=10))]
    adv = panel.pivot_table(index="Date", columns="Ticker", values="dvol60")
    vol = panel.pivot_table(index="Date", columns="Ticker", values="vol_20")
    tl = risk.trade_liquidity(sel.trades, adv, vol)
    curve = risk.capacity_curve(r, equity, tl, w, adv)
    g_alpha = float(res["metrics"].loc[selected, "gross alpha % p.a."])

    loc = res["local_factors"]
    strategies = [c for c in daily.columns if c != bname]
    return {
        "stress": risk.stress_table(both, bname),
        "te": risk.te_summary(te, (r - b).dropna()),
        "te_series": te,
        "risk_decomp": pd.DataFrame(
            {c: risk.risk_decomposition(daily[c], loc) for c in strategies}
        ).T,
        "return_decomp": pd.DataFrame(
            {
                c: risk.return_decomposition(daily[c], res["gross_returns"][c], loc)
                for c in res["gross_returns"].columns
            }
        ).T,
        "capacity": curve,
        "impact_k": risk.IMPACT_K,
        # capacity = AUM at which square-root impact eats 1% a year / half / all of the gross CAPM alpha
        "capacity_at": {
            "impact drag 1% p.a. (THB m)": risk.capacity_at(curve, "impact drag % p.a.", 1.0),
            "impact = 50% of gross alpha (THB m)": risk.capacity_at(
                curve, "impact drag % p.a.", g_alpha / 2
            ),
            "impact = gross alpha (THB m)": risk.capacity_at(curve, "impact drag % p.a.", g_alpha),
            "impact = gross alpha, k/2 (THB m)": risk.capacity_at(
                curve.assign(half=curve["impact drag % p.a."] / 2), "half", g_alpha
            ),
        },
        "trade_liquidity": tl,
        "monthly": {
            selected: risk.monthly_table(r),
            bname: risk.monthly_table(b),
            "active (strategy - TDEX)": risk.monthly_table(r) - risk.monthly_table(b),
        },
        "calendar": risk.calendar_years(both[[selected, bname, ew]], bname),
        "rolling": risk.rolling_alpha_ir(r, b),
    }


def step_signals() -> pd.DataFrame:
    df, feats = _model_frame()
    signals = report.latest_signals(df, ingest.ingest_all("file")["prices"], feats)
    log.info(
        "signals for %s (%s):\n%s",
        signals["signal_date"].iloc[0],
        signals["strategy"].iloc[0],
        signals[signals["target_weight_%"] > 0][
            ["Ticker", "pred_blend", "target_weight_%", "shares (100-lot)"]
        ].to_string(),
    )
    return signals


def step_paper() -> dict:
    """Live paper trading of the pre-registered shadow portfolios + the signal-decay monitor (setalpha.paper)."""
    df, feats = _model_frame()
    raw = ingest.ingest_all("file")
    return paper.step(df, feats, raw["prices"], raw["market"])


def step_report() -> str:
    res = joblib.load(SIM_RESULTS)
    if report.SIGNALS.exists():
        res["signals"] = pd.read_csv(report.SIGNALS)
    if paper.SUMMARY.exists():
        res["paper"] = joblib.load(paper.SUMMARY)
    res["deployed"] = report.load_deployed_backtest()
    path = report.build_dashboard(res)
    log.info("report: %s", path)
    return path


def run(steps: list[str] | None = None, mode: ingest.Mode = "file", retune: bool = False) -> None:
    steps = steps or STEPS
    raw = step_ingest(mode) if "ingest" in steps else None
    if "transform" in steps:
        step_transform(raw)
    if "train" in steps:
        step_train(retune)
    if "simulate" in steps:
        step_simulate()
    if "signals" in steps:
        step_signals()
    if "paper" in steps:
        step_paper()
    if "report" in steps:
        step_report()
