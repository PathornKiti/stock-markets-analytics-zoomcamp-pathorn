"""strategy_v2: pre-registered portfolio / strategy designs for the `pred_ens` signal.

Stages (run in this order; the test stage refuses to run until the validation stage froze a pick):

  uv run python set-quant-alpha/experiments/strategy_v2/run.py --stage valid
  uv run python set-quant-alpha/experiments/strategy_v2/run.py --stage test
  uv run python set-quant-alpha/experiments/strategy_v2/run.py --stage secondary \
        --preds set-quant-alpha/experiments/model_v2/output/predictions_v2.parquet --label v2

  --preds : predictions file (default data/processed/predictions.parquet), same schema, `pred_ens` = score.

PRE-REGISTERED PROTOCOL (written before any strategy_v2 simulation was run)
  Designs: `sv2.grid()` (E1->E2 ablation lattice AB0-AB3, single modifications D01-D10, D12 = V8 score)
  plus two derived trials:
    D11 combination rule - every *composable* single modification (D01 cooldown, D02 breadth,
        D03 EMA score, D06 sqrt-traded-value weights, D10 vol-scaled stop, D12 V8 score) whose
        validation local-Thai alpha t-stat beats E2's is applied together to E2 (if two score
        changes qualify, only the one with the higher t is kept). Built only if >= 2 qualify.
    P-CS core-satellite wrapper of the pick: stock sleeves scaled to s = min(1, 4% / validation TE of the
        pick), rest in TDEX (enhanced-index tracking-error budget). Skipped if s == 1.
  Selection rule (validation 2020-21 only): among pickable designs (E2 = AB3, D01-D12) whose validation
  one-way turnover <= 1.1 x E2's, pick the highest validation alpha t-stat in the LOCAL THAI factor model
  (MKT = TDEX, SMB, WML; daily, Newey-West). Rationale: the previous experiment showed that selecting on raw
  IR vs TDEX selects a size/style bet (mid caps +18.5 %/yr vs TDEX in 2020-21, -12 %/yr in 2022-26), so the
  rule ranks on style-adjusted alpha. E2 is in the pool, so "no change" is a possible outcome.
  Trial count for DSR: 37 (cost_aware) + 6 (pipeline E1-E6) + every strategy_v2 trial simulated on validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sv2  # noqa: E402
from setalpha import config, evaluate, model  # noqa: E402

log = logging.getLogger("strategy_v2")
OUT = sv2.OUT
FROZEN = OUT / "picks_frozen_before_test.txt"
FROZEN_JSON = OUT / "frozen_pick.json"
SUPERSEDED = OUT / "picks_frozen_v1_superseded.txt"
TE_BUDGET = 0.04
TURNOVER_TOL = 1.1
E2_NAME = "AB3 E2: buffer 30 + 8% stop"
E1_NAME = "AB0 E1: top-10 weekly"
FEES = (0.0, 0.0010, 0.0015, 0.0020, 0.0025)


# ---------------------------------------------------------------------------- helpers
def design_from_json(d: dict) -> sv2.Design:
    d = dict(d)
    d["sleeves"] = tuple(
        sv2.Sleeve(**{**s, "universe": tuple(s["universe"])}) for s in d["sleeves"]
    )
    return sv2.Design(**d)


def run_designs(designs, inp, split, gross=True, fee=config.FEE_PER_SIDE):
    f = sv2.make_frame(inp, split)
    px = sv2.price_panel(inp, f)
    res, rows = {}, {}
    ew = None
    for des in designs:
        net = sv2.simulate(des, f, px, inp["bench"], fee=fee)
        g = sv2.simulate(des, f, px, inp["bench"], fee=0.0, tdex_fee=0.0) if gross else None
        b = sv2.bench_returns(inp["bench"], net.equity.index)
        if ew is None:
            ew = sv2.universe_ew(px["Close"], f)
        rows[des.name] = {
            "family": des.family,
            "pickable": des.pickable,
            **sv2.stats(net, g, inp, b, ew),
            "re-entries after stop (<=10d)": reentries(net.trades),
        }
        res[des.name] = net
    return pd.DataFrame(rows).T, res, f, px


def reentries(tr: pd.DataFrame) -> int:
    """Stop-outs followed by a re-purchase of the same name within 10 trading days (~14 calendar days)."""
    if not len(tr):
        return 0
    stops = tr[tr["reason"] == "stop-loss"]
    buys = tr[(tr["reason"] == "rebalance") & (tr["qty"] > 0)]
    n = 0
    for _, s in stops.iterrows():
        b = buys[
            (buys["Ticker"] == s["Ticker"])
            & (buys["Date"] > s["Date"])
            & (buys["Date"] <= s["Date"] + pd.Timedelta(days=14))
        ]
        n += int(len(b) > 0)
    return n


def ablation(t: pd.DataFrame) -> pd.DataFrame:
    """Two-factor (buffer, stop) Shapley decomposition of E2 - E1, split into gross return and cost saving."""
    a = {
        k: t.loc[k]
        for k in ["AB0 E1: top-10 weekly", "AB1 E1 + 8% stop", "AB2 E1 + 30-rank buffer", E2_NAME]
    }
    v = lambda k, c: float(a[k][c])  # noqa: E731
    k0, k1, k2, k3 = list(a)
    rows = {}
    for c in [
        "CAGR %",
        "gross CAGR %",
        "cost drag %",
        "CAPM alpha %",
        "TH alpha %",
        "CAPM beta",
        "Sharpe",
        "turnover 2-way x/yr",
        "stock exposure %",
    ]:
        buf = 0.5 * ((v(k2, c) - v(k0, c)) + (v(k3, c) - v(k1, c)))
        stp = 0.5 * ((v(k1, c) - v(k0, c)) + (v(k3, c) - v(k2, c)))
        rows[c] = {
            "E1": v(k0, c),
            "E1+stop": v(k1, c),
            "E1+buffer": v(k2, c),
            "E2": v(k3, c),
            "E2-E1": v(k3, c) - v(k0, c),
            "buffer share": buf,
            "stop share": stp,
        }
    return pd.DataFrame(rows).T


def trial_lists(val: pd.DataFrame) -> tuple[list[float], list[float], dict]:
    prev = pd.read_csv(sv2.HERE.parent / "output" / "validation_all.csv", index_col=0)
    pipe = joblib.load(config.PROCESSED_DIR / "sim_results.joblib")["metrics_valid"]
    pipe = pipe.loc[
        [k for k in pipe.index if k[:1] == "E" and k[1:2].isdigit()]
    ]  # E1-E6 only (not benchmarks)
    srs = (
        prev["sr_daily"].astype(float).tolist()
        + (pipe["Sharpe"].astype(float) / np.sqrt(252)).tolist()
        + val["sr_daily"].astype(float).tolist()
    )
    irs = (
        prev["ir_daily"].astype(float).tolist()
        + (pipe["info ratio"].astype(float) / np.sqrt(252)).tolist()
        + val["ir_daily"].astype(float).tolist()
    )
    return (
        srs,
        irs,
        {
            "cost_aware": len(prev),
            "pipeline": len(pipe),
            "strategy_v2": len(val),
            "total": len(srs),
        },
    )


COLS_MAIN = [
    "CAGR %",
    "Sharpe",
    "max DD %",
    "turnover 2-way x/yr",
    "cost drag %",
    "gross CAGR %",
    "IR vs TDEX",
    "TE %",
    "CAPM alpha %",
    "CAPM alpha t",
    "CAPM beta",
    "break-even vs TDEX %/side",
]
COLS_FACT = [
    "FF3 alpha %",
    "FF3 alpha t",
    "FF3 b_Mkt",
    "FF3 b_SMB",
    "FF3 b_HML",
    "C4 alpha %",
    "C4 alpha t",
    "C4 b_Mkt",
    "C4 b_SMB",
    "C4 b_HML",
    "C4 b_WML",
    "TH alpha %",
    "TH alpha t",
    "TH b_MKT",
    "TH b_SMB",
    "TH t_SMB",
    "TH b_WML",
    "TH t_WML",
]
COLS_IMPL = [
    "stock exposure %",
    "TDEX weight %",
    "avg liq rank (w)",
    "stop-outs",
    "re-entries after stop (<=10d)",
    "vs EW top-100 % p.a.",
    "gross TH alpha %",
    "break-even TH alpha %/side",
    "PSR",
]


def bench_row(inp, idx) -> dict:
    b = sv2.bench_returns(inp["bench"], idx).dropna()
    s = evaluate.perf_stats(b)
    row = {
        "CAGR %": s["CAGR %"],
        "Sharpe": s["Sharpe"],
        "max DD %": s["max DD %"],
        "PSR": s["PSR (SR>0)"],
        "turnover 2-way x/yr": 0.0,
        "cost drag %": 0.0,
        "gross CAGR %": s["CAGR %"],
        "CAPM beta": 1.0,
    }
    row.update(sv2.attribution(b, inp))
    return row


# ---------------------------------------------------------------------------- stages
def stage_valid(inp: dict) -> None:
    if FROZEN.exists():
        raise SystemExit(
            f"{FROZEN} exists - the pick is already frozen; delete it only to restart the whole protocol"
        )
    rep = sv2.replication_check(inp)
    log.info("replication check (validation):\n%s", rep.to_string())
    assert (rep["max abs diff"] < 1e-6).all(), "simulate() does not reproduce strategy.exact_sim"
    rep.to_csv(OUT / "replication_check.csv")

    designs = sv2.grid(has_v8="pred_v8" in inp["preds"].columns)
    val, _, _, _ = run_designs(designs, inp, "valid")
    by = {d.name: d for d in designs}

    # D11 combination rule
    e2t = float(val.loc[E2_NAME, "TH alpha t"])
    good = [
        (by[k].composable, float(val.loc[k, "TH alpha t"]))
        for k in val.index
        if by[k].composable and float(val.loc[k, "TH alpha t"]) > e2t
    ]
    scores = sorted([g for g in good if g[0].startswith("score:")], key=lambda x: -x[1])[:1]
    mods = [g[0] for g in good if not g[0].startswith("score:")] + [g[0] for g in scores]
    log.info(
        "modifications beating E2 on validation TH alpha t (%.2f): %s -> combo %s", e2t, good, mods
    )
    if len(mods) >= 2:
        combo = sv2.combine(by[E2_NAME], mods, "D11 combo: E2 + " + " + ".join(mods))
        cv, _, _, _ = run_designs([combo], inp, "valid")
        val = pd.concat([val, cv])
        designs.append(combo)
        by[combo.name] = combo

    # selection rule
    e2_to = float(val.loc[E2_NAME, "turnover 1-way x/yr"])
    pool = val[
        val["pickable"].astype(bool)
        & (val["turnover 1-way x/yr"].astype(float) <= TURNOVER_TOL * e2_to)
    ]
    pick = pool["TH alpha t"].astype(float).idxmax()
    pick_des = by[pick]
    s = float(min(1.0, TE_BUDGET / (float(val.loc[pick, "TE %"]) / 100)))
    cs = sv2.core_satellite(pick_des, s) if s < 0.995 else None
    if cs is not None:
        cv, _, _, _ = run_designs([cs], inp, "valid")
        val = pd.concat([val, cv])
        designs.append(cs)
    val["pick"] = ""
    val.loc[pick, "pick"] = "PICK"
    if cs is not None:
        val.loc[cs.name, "pick"] = "P-CS"
    val.loc[pool.index, "eligible"] = True

    srs, irs, counts = trial_lists(val)
    val.to_csv(OUT / "validation_all.csv")
    FROZEN_JSON.write_text(
        json.dumps(
            {
                "pick": asdict(pick_des),
                "pcs": asdict(cs) if cs else None,
                "combo": asdict(by[[k for k in by if k.startswith("D11")][0]])
                if any(k.startswith("D11") for k in by)
                else None,
                "trials": counts,
                "preds": inp["preds_path"],
            },
            indent=1,
            default=str,
        )
    )
    FROZEN.write_text(
        f"frozen {datetime.now(UTC).isoformat(timespec='seconds')} (before any strategy_v2 test-split simulation)\n"
        f"rule = max validation local-Thai (MKT/SMB/WML) alpha t among pickable designs with 1-way turnover <= "
        f"{TURNOVER_TOL} x E2 ({TURNOVER_TOL * e2_to:.1f}x/yr)\n"
        f"PICK = {pick}  (valid TH alpha {float(val.loc[pick, 'TH alpha %']):.2f}%/yr, t {float(val.loc[pick, 'TH alpha t']):.2f}; "
        f"E2 t {e2t:.2f})\n"
        f"P-CS = {cs.name if cs else 'not built (s = 1)'}\n"
        f"D11 combination = {[k for k in by if k.startswith('D11')] or 'not built (< 2 qualifying modifications)'}\n"
        f"trials for DSR = {counts}\n"
        + (
            f"NOTE: re-frozen; superseded freeze kept in {SUPERSEDED.name} (fixed before any test run: P-CS min-trade "
            f"threshold blocked 1%-slot buys; pipeline trial count included a benchmark row)\n"
            if SUPERSEDED.exists()
            else ""
        )
    )
    log.info("FROZEN:\n%s", FROZEN.read_text())
    pd.set_option("display.width", 250, "display.max_columns", 40)
    log.info(
        "\n%s",
        val[
            [
                "pick",
                "CAGR %",
                "Sharpe",
                "turnover 1-way x/yr",
                "IR vs TDEX",
                "CAPM alpha %",
                "TH alpha %",
                "TH alpha t",
                "TH b_SMB",
                "C4 alpha %",
                "C4 alpha t",
            ]
        ]
        .astype(str)
        .to_string(),
    )


def stage_test(inp: dict) -> None:
    if not FROZEN.exists():
        raise SystemExit("run --stage valid first (no frozen pick)")
    fz = json.loads(FROZEN_JSON.read_text())
    val = pd.read_csv(OUT / "validation_all.csv", index_col=0)
    designs = sv2.grid(has_v8="pred_v8" in inp["preds"].columns)
    if fz["combo"]:
        designs.append(design_from_json(fz["combo"]))
    if fz["pcs"]:
        designs.append(design_from_json(fz["pcs"]))
    pick_name = fz["pick"]["name"]
    assert [d.name for d in designs] == list(
        val.index
    ), "design list differs from the frozen validation run"

    t0 = time.time()
    test, tres, f, px = run_designs(designs, inp, "test")
    log.info("test simulations in %.0fs", time.time() - t0)
    srs, irs, counts = trial_lists(val)
    for split, tab, rr in (("test", test, tres),):  # noqa: B007
        for k in tab.index:
            r = rr[k].returns
            b = sv2.bench_returns(inp["bench"], r.index)
            tab.loc[k, "DSR"] = evaluate.dsr(r, srs)
            tab.loc[k, "DSR active"] = evaluate.dsr((r - b.reindex(r.index)).dropna(), irs)
    test["pick"] = val["pick"].reindex(test.index).fillna("")
    test.loc["TDEX buy & hold", list(bench_row(inp, next(iter(tres.values())).equity.index))] = (
        list(bench_row(inp, next(iter(tres.values())).equity.index).values())
    )
    ew = sv2.universe_ew(px["Close"], f).reindex(next(iter(tres.values())).equity.index).dropna()
    ews = evaluate.perf_stats(ew)
    test.loc["EW tradable top-100 (no costs)", ["CAGR %", "Sharpe", "max DD %"]] = [
        ews["CAGR %"],
        ews["Sharpe"],
        ews["max DD %"],
    ]
    test.loc["EW tradable top-100 (no costs)", list(sv2.attribution(ew, inp))] = list(
        sv2.attribution(ew, inp).values()
    )

    # validation DSR for the pick(s)
    vf = sv2.make_frame(inp, "valid")
    vpx = sv2.price_panel(inp, vf)
    for k in [k for k in val.index if val.loc[k, "pick"] in ("PICK", "P-CS")] + [E1_NAME, E2_NAME]:
        des = next(d for d in designs if d.name == k)
        r = sv2.simulate(des, vf, vpx, inp["bench"]).returns
        b = sv2.bench_returns(inp["bench"], r.index)
        val.loc[k, "DSR"] = evaluate.dsr(r, srs)
        val.loc[k, "DSR active"] = evaluate.dsr((r - b.reindex(r.index)).dropna(), irs)

    # cost sensitivity: E1, E2, pick, P-CS
    sens_names = [E1_NAME, E2_NAME, pick_name] + ([fz["pcs"]["name"]] if fz["pcs"] else [])
    sens_designs = [d for d in designs if d.name in sens_names]
    sens = []
    for fee in FEES:
        for split in ("valid", "test"):
            t, _, _, _ = run_designs(sens_designs, inp, split, gross=False, fee=fee)
            for k in t.index:
                sens.append(
                    {
                        "split": split,
                        "fee %/side": 100 * fee,
                        "design": k,
                        "CAGR %": t.loc[k, "CAGR %"],
                        "TDEX CAGR %": t.loc[k, "TDEX CAGR %"],
                        "IR vs TDEX": t.loc[k, "IR vs TDEX"],
                        "TH alpha %": t.loc[k, "TH alpha %"],
                        "CAPM alpha %": t.loc[k, "CAPM alpha %"],
                    }
                )
    sens = pd.DataFrame(sens)

    abl_v, abl_t = ablation(val), ablation(test)
    test.to_csv(OUT / "test_all.csv")
    val.to_csv(OUT / "validation_all_with_dsr.csv")
    sens.to_csv(OUT / "cost_sensitivity.csv", index=False)
    abl_v.to_csv(OUT / "ablation_valid.csv")
    abl_t.to_csv(OUT / "ablation_test.csv")
    joblib.dump({k: r.returns for k, r in tres.items()}, OUT / "test_daily_returns.joblib")
    write_tables(val, test, sens, abl_v, abl_t, counts)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    log.info(
        "\n%s",
        test[["pick"] + COLS_MAIN + ["TH alpha %", "TH alpha t", "C4 alpha %", "C4 alpha t", "DSR"]]
        .astype(str)
        .to_string(),
    )


def stage_secondary(inp: dict, label: str) -> None:
    """Re-run the frozen pick, P-CS, E1 and E2 on another predictions file (no re-selection)."""
    fz = json.loads(FROZEN_JSON.read_text())
    designs = [d for d in sv2.grid(False) if d.name in (E1_NAME, E2_NAME)]
    pick = design_from_json(fz["pick"])
    if any(s.score == "pred_v8" for s in pick.sleeves):
        log.warning(
            "frozen pick uses the V8 score, not pred_ens: it is shown unchanged (reads pred_v8 column)"
        )
    designs += [pick] + ([design_from_json(fz["pcs"])] if fz["pcs"] else [])
    designs = list({d.name: d for d in designs}.values())
    srs, irs, counts = trial_lists(pd.read_csv(OUT / "validation_all.csv", index_col=0))
    tabs = {}
    for split in ("valid", "test"):
        t, res, _, _ = run_designs(designs, inp, split)
        for k in t.index:
            r = res[k].returns
            b = sv2.bench_returns(inp["bench"], r.index)
            t.loc[k, "DSR"] = evaluate.dsr(r, srs)
            t.loc[k, "DSR active"] = evaluate.dsr((r - b.reindex(r.index)).dropna(), irs)
        tabs[split] = t
    out = pd.concat(tabs, names=["split", "design"])
    out.to_csv(OUT / f"secondary_{label}.csv")
    cols = [
        "CAGR %",
        "Sharpe",
        "max DD %",
        "turnover 2-way x/yr",
        "cost drag %",
        "gross CAGR %",
        "IR vs TDEX",
        "CAPM alpha %",
        "CAPM alpha t",
        "FF3 alpha %",
        "FF3 alpha t",
        "C4 alpha %",
        "C4 alpha t",
        "TH alpha %",
        "TH alpha t",
        "TH b_SMB",
        "DSR",
    ]
    (OUT / f"secondary_{label}.md").write_text(
        f"# Secondary comparison on `{inp['preds_path']}` (label {label})\n\nFrozen designs re-run unchanged; "
        f"no selection on this file. DSR trials = {counts['total']} (+ the ML agent's model trials, not deflated here).\n\n"
        + sv2.to_md(out[cols].astype(float))
        + "\n"
    )
    log.info("\n%s", out[cols].astype(float).round(2).to_string())


def write_tables(val, test, sens, abl_v, abl_t, counts) -> None:
    def sub(t, cols):
        c = ["pick"] + [x for x in cols if x in t.columns]
        x = t[c].copy()
        for col in c[1:]:
            x[col] = pd.to_numeric(x[col], errors="coerce")
        x["pick"] = x["pick"].fillna("")
        return sv2.to_md(x)

    piv = sens.pivot_table(
        index=["split", "design"], columns="fee %/side", values=["CAGR %", "TH alpha %"]
    ).round(2)
    text = (
        f"# strategy_v2 generated tables\n\nTrials for DSR: {counts}\n\n"
        "## Validation 2020-21 (selection data)\n\n"
        + sub(val, COLS_MAIN)
        + "\n\n### Validation factor alphas\n\n"
        + sub(val, COLS_FACT)
        + "\n\n### Validation implementation\n\n"
        + sub(val, COLS_IMPL)
        + "\n\n### Validation DSR (pick, P-CS, E1, E2)\n\n"
        + sv2.to_md(val.loc[val["DSR"].notna(), ["pick", "PSR", "DSR", "DSR active"]].fillna(""))
        + "\n\n## Test 2022-01 to 2026-10 (run once, after freezing)\n\n"
        + sub(test, COLS_MAIN + ["PSR", "DSR", "DSR active"])
        + "\n\n### Test factor alphas (FF3 / Carhart: FF-EM monthly USD; TH: local Thai daily)\n\n"
        + sub(test, COLS_FACT)
        + "\n\n### Test implementation\n\n"
        + sub(test, COLS_IMPL)
        + "\n\n## E1 -> E2 ablation (Shapley over buffer and stop)\n\n### Validation\n\n"
        + sv2.to_md(abl_v)
        + "\n\n### Test\n\n"
        + sv2.to_md(abl_t)
        + "\n\n## Cost sensitivity\n\n"
        + sv2.to_md(piv)
        + "\n"
    )
    (OUT / "tables.md").write_text(text)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("setalpha").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["valid", "test", "secondary", "tables"], required=True)
    ap.add_argument("--preds", default=str(model.PREDICTIONS))
    ap.add_argument("--label", default="alt")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.stage == "tables":
        return stage_tables()
    t0 = time.time()
    inp = sv2.load_inputs(a.preds)
    if a.stage == "valid":
        stage_valid(inp)
    elif a.stage == "test":
        stage_test(inp)
    else:
        stage_secondary(inp, a.label)
    log.info("done in %.0fs", time.time() - t0)


def stage_tables() -> None:
    """Rebuild tables.md from the saved CSVs (no simulation)."""
    val = pd.read_csv(OUT / "validation_all_with_dsr.csv", index_col=0)
    test = pd.read_csv(OUT / "test_all.csv", index_col=0)
    sens = pd.read_csv(OUT / "cost_sensitivity.csv")
    abl_v = pd.read_csv(OUT / "ablation_valid.csv", index_col=0)
    abl_t = pd.read_csv(OUT / "ablation_test.csv", index_col=0)
    _, _, counts = trial_lists(pd.read_csv(OUT / "validation_all.csv", index_col=0))
    write_tables(val, test, sens, abl_v, abl_t, counts)


if __name__ == "__main__":
    main()
