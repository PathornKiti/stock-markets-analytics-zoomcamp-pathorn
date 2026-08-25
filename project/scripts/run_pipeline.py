#!/usr/bin/env python
"""End-to-end capstone pipeline: ingest -> transform -> model -> simulate.

This is the single entry point the rubric's automation criteria ask for. It is
cron-safe (no interactive prompts, non-zero exit on failure) and supports both
data regimes:

    uv run project/scripts/run_pipeline.py --mode file    # replay local data
    uv run project/scripts/run_pipeline.py --mode live    # refresh from the APIs

Cron example -- refresh and re-predict every weekday at 18:30:

    30 18 * * 1-5 cd /path/to/repo && /path/to/uv run project/scripts/run_pipeline.py --mode live >> logs/pipeline.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys

from algotrade import ingest, model, strategy, transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("pipeline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["file", "live"], default="file")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    args = parser.parse_args(argv)

    log.info("1/4 ingest (mode=%s)", args.mode)
    raw = ingest.ingest_all(mode=args.mode)

    log.info("2/4 transform")
    df = transform.build_dataset(raw["prices"], raw["macro"])
    path = transform.save(df)
    log.info("    dataset: %s rows -> %s", f"{len(df):,}", path)

    log.info("3/4 model (%s)", args.model)
    features = transform.feature_columns(df)
    target = transform.TARGET
    labelled = df.dropna(subset=[target])
    split = model.time_split(labelled, args.valid_start, args.test_start)
    clf = model.fit_classifier(split.train, features, target, kind=args.model)
    log.info("    valid:\n%s", model.evaluate(clf, split.valid, features, target).to_string())
    log.info("    test:\n%s", model.evaluate(clf, split.test, features, target).to_string())
    model.save(clf, args.model)

    log.info("4/4 simulate")
    test = split.test.copy()
    test["proba"] = model.predict_proba(clf, test, features)
    test["signal"] = strategy.threshold_signal(test["proba"], args.threshold)

    strat = strategy.portfolio_returns(test, "signal", fee_bps=args.fee_bps)
    bench = strategy.benchmark_returns(test, ingest.BENCHMARK)
    log.info(
        "\n%s",
        strategy.report({"strategy": strat, f"{ingest.BENCHMARK} buy&hold": bench}).to_string(),
    )

    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
