#!/usr/bin/env python
"""Refresh the research event store (Yahoo earnings calendars + dividends) on demand.

    uv run set-quant-alpha/scripts/fetch_events.py            # skips if refreshed in the last 7 days
    uv run set-quant-alpha/scripts/fetch_events.py --force

Not part of the daily Airflow run: these are research features (setalpha.events), kept out of the production model.
"""

import argparse
import logging

from setalpha import ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="refresh even if the store is less than 7 days old"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    if args.force:
        ingest.EVENTS_REFRESH_DAYS = 0
    tickers = ingest.discover_universe("file")["Ticker"].tolist()
    ev = ingest.ingest_events(tickers, "live")
    e = ev["earnings"]
    print(
        f"earnings: {0 if e is None else len(e):,} rows; dividends: {0 if ev['dividends'] is None else len(ev['dividends']):,} rows"
    )
