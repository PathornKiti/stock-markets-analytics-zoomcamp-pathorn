#!/usr/bin/env python
"""Command-line entry point for the whole workflow (also what cron / Airflow call).

Examples
--------
Full run from the local data store (no network, reproducible for a reviewer):
    python set-quant-alpha/scripts/run_pipeline.py --mode file

Daily update after the Thai close: new data -> features -> next-day signals -> dashboard:
    python set-quant-alpha/scripts/run_pipeline.py --mode live --steps ingest transform signals paper report

Weekly retrain + backtest (add --retune to redo the hyperparameter search):
    python set-quant-alpha/scripts/run_pipeline.py --steps train simulate report

Cron (Mon-Fri 18:30 Bangkok time):
    30 18 * * 1-5 cd /path/to/repo && .venv/bin/python set-quant-alpha/scripts/run_pipeline.py \
        --mode live --steps ingest transform signals paper report >> set-quant-alpha/reports/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

from setalpha import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="set-quant-alpha pipeline")
    parser.add_argument(
        "--mode",
        choices=["file", "live"],
        default="file",
        help="file = use the local data store; live = download new data incrementally",
    )
    parser.add_argument("--steps", nargs="+", choices=pipeline.STEPS, default=pipeline.STEPS)
    parser.add_argument("--retune", action="store_true", help="redo the hyperparameter grid search")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    pipeline.run(args.steps, args.mode, args.retune)
    return 0


if __name__ == "__main__":
    sys.exit(main())
