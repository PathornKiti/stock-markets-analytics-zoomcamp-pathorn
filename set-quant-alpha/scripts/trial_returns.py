#!/usr/bin/env python
"""Rebuild the daily returns of earlier research trials for the overfitting statistics (~15 s).

    uv run set-quant-alpha/scripts/trial_returns.py

The experiments saved only their *test-period* daily returns. PBO (CSCV) and the long-history
track-record statistics also need the periods the designs were selected on, so this script
re-runs the frozen designs read-only (no selection, nothing in experiments/ is written):

* strategy_v2: all 17 designs on 2020-21 (validation) + the saved 2022-26 test returns;
  the test run is repeated as a reproduction check against the saved file;
* strategy_v3 baselines: E1 / E2 on 2013-2021 (its selection period, THB 20M) + TDEX.

Output: data/processed/trial_returns.joblib, read by the simulate step if present.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings

import joblib
import pandas as pd
from setalpha import config

EXP = config.PROJECT_DIR / "experiments"
OUT = config.PROCESSED_DIR / "trial_returns.joblib"
log = logging.getLogger("trial_returns")


def strategy_v2() -> tuple[pd.DataFrame, float]:
    sys.path.insert(0, str(EXP / "strategy_v2"))
    import run
    import sv2

    inp = sv2.load_inputs()
    designs = sv2.grid(has_v8="pred_v8" in inp["preds"].columns)
    frozen = json.loads((EXP / "strategy_v2/output/frozen_pick.json").read_text())
    designs += [run.design_from_json(frozen[k]) for k in ("combo", "pcs") if frozen.get(k)]
    saved = joblib.load(EXP / "strategy_v2/output/test_daily_returns.joblib")
    parts, diff = {}, 0.0
    for split in ("valid", "test"):
        f = sv2.make_frame(inp, split)
        px = sv2.price_panel(inp, f)
        for d in designs:
            r = sv2.simulate(d, f, px, inp["bench"]).returns
            if split == "test":
                diff = max(diff, float((r - saved[d.name]).abs().max()))
            else:
                parts[d.name] = r
    valid = pd.DataFrame(parts)
    test = pd.DataFrame({k: v for k, v in saved.items()})
    return pd.concat([valid, test]).sort_index(), diff


def strategy_v3_selection() -> pd.DataFrame:
    sys.path.insert(0, str(EXP / "strategy_v3"))
    import sv2
    import v3

    inp, f, px, _, _ = v3.setup("sel")
    out = {
        d.name: sv2.simulate(d, f, px, inp["bench"], capital=v3.CAPITAL).returns
        for d in v3.BASELINES
    }
    out = pd.DataFrame(out)
    out[f"{config.BENCHMARK} buy & hold"] = sv2.bench_returns(inp["bench"], out.index)
    return out.loc[: v3.SEL_END]


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    v2, diff = strategy_v2()
    log.info(
        "strategy_v2: %d designs x %d days; test reproduction max |diff| = %.2e",
        v2.shape[1],
        len(v2),
        diff,
    )
    sel = strategy_v3_selection()
    log.info(
        "strategy_v3 selection period: %s, %s -> %s",
        list(sel.columns),
        sel.index[0].date(),
        sel.index[-1].date(),
    )
    joblib.dump(
        {"strategy_v2": v2, "strategy_v2_test_repro_diff": diff, "selection_2013_2021": sel}, OUT
    )
    log.info("wrote %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
