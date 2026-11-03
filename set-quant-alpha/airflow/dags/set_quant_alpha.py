"""Airflow DAGs for set-quant-alpha.

* ``set_quant_alpha_daily``  - Mon-Fri 18:30 Bangkok, after the SET close:
      ingest (live, incremental) -> transform -> signals -> paper -> report
* ``set_quant_alpha_weekly`` - Saturday 06:00 Bangkok:
      ingest (live) -> transform -> train (walk-forward + production refit) -> simulate -> publish the deployed
      walk-forward backtest -> rebuild research-trial returns -> report.
      It starts with its own data refresh, so on a fresh clone triggering this DAG once builds everything.
* ``set_quant_alpha_lseg`` - Mon-Fri 19:30 Bangkok, optional:
      scripts/fetch_lseg.py update (research data from LSEG Workspace into data/raw/lseg/). It is SKIPPED (exit
      code 99) when LSEG_APP_KEY is not set in set-quant-alpha/.env or Workspace is not running on the host.

Each task runs one pipeline step in the project's own virtualenv, so a failed step can be
retried on its own and the Airflow UI shows where the run stopped.
"""

from __future__ import annotations

import os

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator

PY = os.environ.get("SETALPHA_PYTHON", "python")
HOME = os.environ.get("SETALPHA_HOME", "/opt/setalpha")
TZ = pendulum.timezone("Asia/Bangkok")
DEFAULTS = {"owner": "set-quant-alpha", "retries": 2, "retry_delay": pendulum.duration(minutes=10)}


def script(task_id: str, path: str) -> BashOperator:
    """Run one project script; exit code 99 marks the task skipped (Airflow default)."""
    return BashOperator(task_id=task_id, bash_command=f"cd {HOME} && {PY} {path}")


def step(task_id: str, steps: str, mode: str = "file") -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {HOME} && {PY} scripts/run_pipeline.py --mode {mode} --steps {steps}",
    )


with DAG(
    dag_id="set_quant_alpha_daily",
    description="Incremental data update and next-day signals for Thai equities",
    schedule="30 18 * * 1-5",
    start_date=pendulum.datetime(2026, 11, 1, tz=TZ),
    catchup=False,
    default_args=DEFAULTS,
    tags=["set-quant-alpha"],
) as daily:
    ingest = step("ingest_live", "ingest", mode="live")
    transform = step("transform", "transform")
    signals = step("signals", "signals")
    paper = step("paper", "paper")  # shadow portfolios + signal-decay monitor
    report = step("report", "report")
    ingest >> transform >> signals >> paper >> report


with DAG(
    dag_id="set_quant_alpha_weekly",
    description="Weekly walk-forward retrain, backtest and dashboard refresh",
    schedule="0 6 * * 6",
    start_date=pendulum.datetime(2026, 11, 1, tz=TZ),
    catchup=False,
    default_args=DEFAULTS,
    tags=["set-quant-alpha"],
) as weekly:
    ingest_w = step("ingest_live", "ingest", mode="live")
    transform_w = step("transform", "transform")
    train = step("train", "train")
    simulate = step("simulate", "simulate")
    publish = script("publish_deployed", "experiments/integrated_v1/publish_deployed.py")
    trials = script("trial_returns", "scripts/trial_returns.py")
    report_w = step("report", "report")
    ingest_w >> transform_w >> train >> simulate >> publish >> trials >> report_w


with DAG(
    dag_id="set_quant_alpha_lseg",
    description="Optional LSEG Workspace research pull (skipped without LSEG_APP_KEY or a running Workspace)",
    schedule="30 19 * * 1-5",
    start_date=pendulum.datetime(2026, 11, 1, tz=TZ),
    catchup=False,
    default_args={**DEFAULTS, "retries": 0},
    tags=["set-quant-alpha", "lseg"],
) as lseg_dag:
    script("lseg_update", "scripts/fetch_lseg.py update")
