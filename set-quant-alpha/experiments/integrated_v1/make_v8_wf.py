"""Walk-forward V8 score (XGB regression on the per-date rank of 20-day beta-adjusted alpha), 2013-2026.

For each year Y the model is fit on all liquid rows dated before Y-01-01 minus a 20-trading-day embargo
(model_v2.train_window), then scores the predictions_ext rows of year Y. Output: per-date percentile rank.
Uses experiments/model_v2/model_v2.py read-only.

    uv run python set-quant-alpha/experiments/integrated_v1/make_v8_wf.py      (~10 min)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_v2"))
import model_v2 as mv  # noqa: E402

PREDS_EXT = HERE.parent / "strategy_v3" / "output" / "predictions_ext.parquet"
OUT = HERE / "output" / "predictions_v8_wf.parquet"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df, feats, rank_feats = mv.load()
    keys = pd.read_parquet(PREDS_EXT, columns=["Date", "Ticker"])
    rows = keys.merge(df, on=["Date", "Ticker"], how="left")
    parts = []
    for yr in sorted(rows.Date.dt.year.unique()):
        sub = rows[rows.Date.dt.year == yr].reset_index(drop=True)
        sub["pred_v8"] = mv.scorer("V8", df, feats, rank_feats, f"{yr}-01-01", sub)
        parts.append(sub[["Date", "Ticker", "pred_v8"]])
        logging.info("V8 walk-forward %d done (%d rows)", yr, len(sub))
    pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
    logging.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
