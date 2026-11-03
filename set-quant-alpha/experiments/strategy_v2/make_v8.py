"""Regenerate the ML agent's V8 score (XGB regression on the per-date rank of 20-day beta-adjusted
alpha) as a walk-forward predictions file with the production schema, for use as ONE pre-registered
strategy design (D12). Uses experiments/model_v2/model_v2.py read-only (imported, not modified).

  valid rows (2020-21): model trained on data < 2020-01-01 (20-day embargo)
  test rows, year Y   : model refit on data < Y-01-01 (20-day embargo)  - same walk-forward as production

Output: experiments/strategy_v2/output/predictions_v8.parquet  (pred_ens = daily percentile rank of V8)
Run: uv run python set-quant-alpha/experiments/strategy_v2/make_v8.py      (several minutes)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_v2"))
import model_v2 as mv  # noqa: E402
from setalpha import config, model  # noqa: E402

OUT = HERE / "output" / "predictions_v8.parquet"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df, feats, rank_feats = mv.load()
    base = pd.read_parquet(model.PREDICTIONS)
    parts = []
    v = (
        base[base.split == "valid"][["Date", "Ticker", "split"]]
        .merge(df, on=["Date", "Ticker"], how="left")
        .reset_index(drop=True)
    )
    v["pred_v8"] = mv.scorer("V8", df, feats, rank_feats, config.VALID_START, v)
    parts.append(v[["Date", "Ticker", "split", "pred_v8"]])
    logging.info("valid done")
    te = base[base.split == "test"][["Date", "Ticker", "split"]].merge(
        df, on=["Date", "Ticker"], how="left"
    )
    for yr in sorted(te.Date.dt.year.unique()):
        sub = te[te.Date.dt.year == yr].reset_index(drop=True)
        sub["pred_v8"] = mv.scorer("V8", df, feats, rank_feats, f"{yr}-01-01", sub)
        parts.append(sub[["Date", "Ticker", "split", "pred_v8"]])
        logging.info("walk-forward %d done", yr)
    new = base.drop(columns=["pred_ens"]).merge(pd.concat(parts), on=["Date", "Ticker", "split"])
    new["pred_ens"] = new.pop("pred_v8")
    new.to_parquet(OUT, index=False)
    logging.info("wrote %s (%d rows)", OUT, len(new))


if __name__ == "__main__":
    main()
