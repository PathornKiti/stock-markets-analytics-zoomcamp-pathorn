"""Proposed retrain / monitoring policy for set-quant-alpha (config only; nothing here is wired into production).

Numbers come from experiments/retrain_v1 (see REPORT.md); the evidence for each is noted beside it.
Runtime artefacts default to experiments/retrain_v1/output/runtime so that running these files cannot touch
models/, data/processed or the frozen paper-trading state.  Point SETALPHA_RETRAIN_HOME at data/monitor in deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path(os.getenv("SETALPHA_RETRAIN_HOME", HERE.parent / "output" / "runtime"))
METRICS = HOME / "monitor_metrics.parquet"  # append-only, one row per trading day
STATUS = HOME / "monitor_status.json"  # last status, read by the dashboard / Airflow branch
REGISTRY = (
    HOME / "registry"
)  # registry/<version>/{logit,xgb}.joblib + meta.json ; registry/registry.json
CHALLENGER_SCORES = (
    HOME / "challenger_scores.parquet"
)  # point-in-time scores of challengers (append-only, like paper scores)
ALERT_WEBHOOK_ENV = (
    "SETALPHA_ALERT_WEBHOOK"  # optional Slack/Teams-style incoming webhook (via config.get_secret)
)

# ---- scheduled retrain (challenger) -------------------------------------------------------------------------------
SCHEDULE = {
    "window": "expanding from 2010-01-01",  # rolling 3y/5y lost 0.006-0.016 IC (t -1.5..-2.7) on 2016-21
    "cadence": "quarterly: first Saturday of Jan/Apr/Jul/Oct",  # annual/quarterly/monthly are indistinguishable (|dIC|<0.002)
    "members": ["logit", "xgb"],  # production hyperparameters, models/best_params.json
    "recency_weights": False,  # half-life 2y: -0.007 IC (t -1.6) on 2016-21
    "embargo_days": 5,  # model._embargo
}

# ---- retrain TRIGGERS (each one only ever creates a CHALLENGER; promotion is gated, see registry.PROMOTION) --------
TRIGGERS = {
    "schedule": "SCHEDULE.cadence",
    "ic_red": {"metric": "ic_120d", "below": 0.0, "min_days_since_last_fit": 60},
    "schema_change": "feature list / column hash differs from the champion",
}

# ---- monitoring thresholds ---------------------------------------------------------------------------------------
# IC is lagged: a day's IC is known 5 trading days after the signal date.  SE values: daily IC sd 0.15, 5-day overlap.
IC_THRESHOLDS = {
    "ic_60d": {
        "amber": 0.02,
        "red": None,
        "note": "dashboard only: SE 0.043, flags 16% of days at 0.02",
    },
    "ic_120d": {
        "amber": 0.02,
        "red": 0.0,
        "note": "SE 0.031; red hit 1.9% of days 2016-26 (2023 trough ~0.02 never red)",
    },
    "ic_250d": {"amber": 0.03, "red": 0.02, "note": "SE 0.021; red hit 4% of days 2016-26"},
}
GUARDS = {  # data/model-integrity guards: histories are 2013-2026 monthly percentiles; these are NOT retrain triggers
    "stock_feature_psi_max_monthly": 4.0,  # ~p99 of the monthly max PSI over the 39 stock-level features
    "stock_feature_psi_mean_monthly": 0.90,  # ~p99
    "label_base_rate_trailing_year": (0.42, 0.52),  # yearly range 2011-26: 0.435-0.507
    "score_xgb_std_cross_section": (
        0.035,
        0.088,
    ),  # p1..p99 of the monthly std of raw XGB probability
    "rank_autocorr_1d_min": 0.60,  # non-refit days run 0.76-0.80
    "refit_day_rank_corr_min": 0.50,  # champion vs challenger score continuity on the switch day
    "feature_nan_share_max": 0.20,
}
# Fraction of a champion's training rows after which a feature-drift review is requested (informational only)
PSI_LEVELS = {"watch": 0.10, "alert": 0.25}

PROMOTION = {
    "min_shadow_trading_days": 120,  # >= 24 non-overlapping 5-day targets
    "no_harm": {
        "max_paired_dIC": -0.005,
        "reason": "scheduled / data-fix retrains: promote unless worse by > 0.005 IC",
    },
    "superiority": {
        "min_paired_t": 2.0,
        "reason": "any change of architecture/window/features: needs a pre-registered paired t > 2",
    },
    "needs_human_approval": True,  # the frozen paper shadows must never see an unannounced signal change
}
