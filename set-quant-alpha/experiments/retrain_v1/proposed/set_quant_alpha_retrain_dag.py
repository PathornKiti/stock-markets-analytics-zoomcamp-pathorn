"""PROPOSED Airflow DAG (not installed in airflow/dags): model-health monitor + champion/challenger retraining.

Does NOT modify set_quant_alpha_daily / _weekly.  The production `train` task (weekly fit_final) stays as it is because the
four frozen paper shadows (registered 2026-11-02) depend on it.  This DAG only OBSERVES production and builds CHALLENGERS in a
separate registry; a challenger can reach the paper book only as a new shadow with its own registration date.

  monitor (daily 19:15, after the daily DAG's `paper`)  ->  branch
        green/amber without trigger -> end                         (amber/red alerts are sent by monitor itself)
        trigger (ic_120d < 0 and >= 60 trading days since last fit) -> fit_challenger
  schedule: first Saturday of Jan/Apr/Jul/Oct 07:00 -> fit_challenger (reason=schedule)
  fit_challenger -> score_challengers (daily, appended) ; evaluate_and_gate (daily, after 120 shadow days) -> manual approval Variable
Rollback: `python registry.py`-style call `rollback()` from the `rollback` DAG (manual trigger), artefacts are immutable.
"""

from __future__ import annotations

import os

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

PY = os.environ.get("SETALPHA_PYTHON", "python")
HOME = os.environ.get("SETALPHA_HOME", "/opt/setalpha")
EXP = f"{HOME}/experiments/retrain_v1/proposed"  # move to src/setalpha/retrain/ once approved
TZ = pendulum.timezone("Asia/Bangkok")
DEFAULTS = {
    "owner": "set-quant-alpha",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=10),
    "email_on_failure": True,
    "email": [os.environ.get("SETALPHA_ALERT_EMAIL", "")],
}


def sh(task_id: str, code: str, **kw) -> BashOperator:
    return BashOperator(
        task_id=task_id, bash_command=f'cd {HOME} && cd {EXP} && {PY} -c "{code}"', **kw
    )


FIT = (
    "import policy, registry, pandas as pd; from setalpha import transform, report; import json;"
    "df=transform.add_dummies(transform.load()); s=transform.feature_sets(df); f=s['NUMERIC']+s['DUMMIES'];"
    "best=json.load(open('{home}/models/best_params.json')); registry.register_current();"
    "print(registry.fit_challenger(df,f,best,df.Date.max()+pd.Timedelta(days=1),'{reason}'))"
)

with DAG(
    dag_id="set_quant_alpha_monitor",
    description="Daily model-health metrics, alerts and retrain trigger",
    schedule="15 19 * * 1-5",
    start_date=pendulum.datetime(2026, 11, 2, tz=TZ),
    catchup=False,
    default_args=DEFAULTS,
    tags=["set-quant-alpha", "proposed"],
) as monitor_dag:
    wait_paper = ExternalTaskSensor(
        task_id="wait_for_daily_paper",
        external_dag_id="set_quant_alpha_daily",
        external_task_id="report",
        execution_delta=pendulum.duration(minutes=-45),
        timeout=3600,
        mode="reschedule",
    )
    monitor = BashOperator(task_id="monitor", bash_command=f"cd {HOME} && {PY} {EXP}/monitor.py")

    def _branch(**_):
        import json

        s = json.load(
            open(
                os.environ.get("SETALPHA_RETRAIN_HOME", f"{HOME}/data/monitor")
                + "/monitor_status.json"
            )
        )
        return "fit_challenger_triggered" if s["trigger_challenger_retrain"] else "no_retrain"

    branch = BranchPythonOperator(task_id="branch_on_trigger", python_callable=_branch)
    no_retrain = BashOperator(task_id="no_retrain", bash_command="true")
    fit_trig = sh("fit_challenger_triggered", FIT.format(home=HOME, reason="trigger:ic_120d<0"))
    score = sh(
        "score_challengers",
        "import policy, registry, pandas as pd; from setalpha import transform;"
        "df=transform.add_dummies(transform.load()); s=transform.feature_sets(df); f=s['NUMERIC']+s['DUMMIES'];"
        "d=df[df.liquid & (df.Date==df.Date.max())]; registry.score_day(d,f)",
        trigger_rule="none_failed_min_one_success",
    )
    wait_paper >> monitor >> branch >> [fit_trig, no_retrain]
    [fit_trig, no_retrain] >> score

with DAG(
    dag_id="set_quant_alpha_retrain_scheduled",
    description="Quarterly expanding-window challenger",
    schedule="0 7 1-7 1,4,7,10 6",
    start_date=pendulum.datetime(2027, 1, 1, tz=TZ),
    catchup=False,  # first Saturday of the quarter
    default_args=DEFAULTS,
    tags=["set-quant-alpha", "proposed"],
) as sched_dag:
    sh("fit_challenger", FIT.format(home=HOME, reason="schedule:quarterly"))

with DAG(
    dag_id="set_quant_alpha_promote",
    description="Gate + manual promotion / rollback of the champion",
    schedule=None,
    start_date=pendulum.datetime(2026, 11, 2, tz=TZ),
    catchup=False,
    params={"challenger": "", "approved": False, "kind": "no_harm", "rollback": False},
    default_args=DEFAULTS,
    tags=["set-quant-alpha", "proposed"],
) as promote_dag:
    BashOperator(
        task_id="gate_and_promote",
        bash_command=(
            f'cd {HOME} && cd {EXP} && {PY} -c "import registry, policy, pandas as pd; from setalpha import transform;'
            "p={{ params | tojson }}; df=transform.load();"
            "print(registry.rollback() if p['rollback'] else registry.promote(p['challenger'], registry.evaluate(df,p['challenger']), p['approved'], p['kind']))\""
        ),
    )
