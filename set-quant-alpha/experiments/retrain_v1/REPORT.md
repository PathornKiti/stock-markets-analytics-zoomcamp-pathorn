# retrain_v1 - model degradation and retraining policy (signal level)

Reproduce (repo root): `fit_grid.py` (520 fits, ~25 min, cached in output/fits, 187 MB, regenerable) -> `staleness.py`, `drift.py`,
`policies.py`, `leading.py`, `diagnostics.py`. Proposed wiring: `proposed/`. Nothing in src/, airflow/, models/, data/ was changed.

Setup: logit + XGB (production hyperparameters, rank-average), all liquid labelled rows, no row subsampling (<=0.45M rows, 2-8 s/fit).
Every model trains only on rows with Date < refit date minus `model._embargo` (5 trading days). IC = daily Spearman vs `fwd_alpha_5` on the
tradable top-100. A-exp reproduces the strategy_v3 annual walk-forward (0.0954 vs 0.0954 in 2016, same in all years to 3 decimals).

## 1. Staleness: the IC decline is NOT model age
* Existing annual predictions, IC by months since the January refit: no within-year decay. Year-demeaned slope +0.0009 IC/month
  (90% year-bootstrap CI -0.0021..+0.0033); H2-minus-H1 IC is -0.009 in 2013-19 and +0.017 in 2020-26 (noise). Month-level IC is dominated by calendar noise
  (age 10: 0.037, age 9: 0.104).
* Paired test (same dates, expanding-window model of age k vs a model refit that month): 2016-21 loss -0.0003 IC/month (age 11: -0.0025, no age
  with |t|>2 except age 3 at t=-1.96); 2022-26 -0.0001/month (age 11: -0.0003). A whole year of staleness costs <= 0.004 IC.
* Look-ahead upper bound (diagnostic only, not tradable): leave-one-year-out models trained on all other years, past AND future (`diagnostics.py`):
  mean IC 2013-19 0.0993 vs walk-forward 0.0917 (+0.008); 2020-26 0.0544 vs 0.0513 (+0.003); 2023: 0.027 vs 0.019. Even a model that has seen the
  test year's regime gets only 0.027 in 2023. The 0.09-0.10 -> 0.05 decline (-0.04 walk-forward, -0.045 LOYO) is alpha decay / crowding / a lower signal-to-noise
  market, with at most ~0.003-0.008 IC (<10% of the decline) attributable to anything retraining could recover.

## 2. Policy comparison (IC, daily mean; ICIR = daily mean/std; TD = top-decile 5d alpha %; t = paired non-overlapping t of daily IC difference)
Decision window 2016-21 only; 2022-26 shown once for information. Policies fixed before running.

| policy | IC 16-21 | ICIR | TD % | dIC vs A-exp (t) | IC 22-26 | TD % | dIC vs A-exp (t) | rank autocorr on refit day / other days |
|---|---|---|---|---|---|---|---|---|
| A-exp  annual, expanding | 0.0765 | 0.496 | 0.405 | - | 0.0503 | 0.265 | - | 0.66 / 0.78 |
| Q-exp  quarterly, expanding | 0.0774 | 0.497 | 0.435 | +0.001 (-0.1) | 0.0498 | 0.271 | -0.000 (-0.6) | 0.72 / 0.78 |
| M-exp  monthly, expanding (~production weekly) | 0.0780 | 0.493 | 0.411 | +0.002 (0.0) | 0.0490 | 0.279 | -0.001 (0.0) | 0.74 / 0.79 |
| A-r5   annual, rolling 5y | 0.0709 | 0.478 | 0.372 | -0.006 (-2.1) | 0.0459 | 0.248 | -0.004 (-0.6) | 0.61 / 0.76 |
| Q-r5   quarterly, rolling 5y | 0.0692 | 0.449 | 0.361 | -0.007 (-1.7) | 0.0478 | 0.203 | -0.003 (-0.2) | 0.67 / 0.77 |
| M-r5   monthly, rolling 5y | 0.0711 | 0.449 | 0.346 | -0.006 (-1.5) | 0.0469 | 0.206 | -0.003 (-0.1) | 0.71 / 0.78 |
| M-r3   monthly, rolling 3y | 0.0609 | 0.391 | 0.306 | -0.016 (-2.7) | 0.0449 | 0.216 | -0.005 (-0.6) | 0.72 / 0.80 |
| M-expW monthly, expanding, half-life 2y | 0.0696 | 0.430 | 0.362 | -0.007 (-1.6) | 0.0491 | 0.254 | -0.001 (-0.5) | 0.71 / 0.78 |

Per-year IC (2016..2026; file output/policy_ic_by_year.csv):
```
        2016   2017   2018   2019   2020   2021   2022   2023   2024   2025   2026
A-exp  .0954  .1088  .0835  .0684  .0455  .0575  .0644  .0187  .0461  .0506  .0769
Q-exp  .0975  .1173  .0834  .0683  .0440  .0535  .0609  .0213  .0459  .0507  .0754
M-exp  .0951  .1212  .0866  .0673  .0458  .0520  .0655  .0139  .0456  .0522  .0725
A-r5   .0967  .1084  .0806  .0647  .0305  .0442  .0459  .0289  .0387  .0439  .0787
Q-r5   .0983  .1148  .0845  .0682  .0261  .0230  .0454  .0326  .0412  .0541  .0698
M-r5   .0961  .1190  .0897  .0677  .0278  .0256  .0475  .0251  .0395  .0576  .0691
M-r3   .0838  .1082  .0883  .0646  .0128  .0072  .0207  .0267  .0344  .0741  .0743
M-expW .0866  .1154  .0907  .0636  .0328  .0281  .0503  .0254  .0386  .0573  .0799
```
Findings. (a) Cadence does not matter: annual/quarterly/monthly differ by <= 0.0015 IC with |t| < 0.3 in both windows. (b) Shortening memory hurts: every rolling or
recency-weighted variant is below expanding on 2016-21 (rolling 3y -0.016, t -2.7), worst in 2020-21 (the COVID/low-IC years, where a short window forgets
the pre-2020 relationships); the 2022-26 ordering is the same but within noise (-0.001..-0.005). Rolling windows only win in 2025-26 (r3: 0.074 in 2025)
- an uncorrected single-period result. (c) Refit-day score continuity is a small cost of frequent refits (day-to-day rank autocorr 0.66 at an annual refit
vs 0.78 normally; monthly 0.74) - it raises top-decile turnover slightly; this is a reason NOT to refit more often than needed.
Pre-registered rule (a challenger must beat the incumbent by paired t > 2, ties -> cheapest): no policy beats A-exp/M-exp; the decision is **expanding window,
cadence immaterial**. Top-decile alpha is much noisier than IC (e.g. 2021 M-exp 0.247 vs A-exp 0.168) and was not used for the decision. Note that production's
weekly fit_final is, at signal level, equivalent to M-exp, so it neither helps nor hurts.

## 3. Drift diagnostics (output/drift_*.csv, monthly_panel.csv, lead_lag_*.csv, loyo_vs_walkforward.csv)
* Feature PSI vs the training window (all liquid rows before 1 Jan): the 26 date-level macro/regime series (tbill3m, credit_spread, vix, hmm_p_*, usdthb_vol_20, skew...)
  have mean PSI 0.5-2.1 every year - they are non-stationary by construction (a single time series, so any new regime is "new") and are not informative about stock selection.
  The 39 stock-level features are mild: mean PSI 0.02-0.16 (yearly), 0-9 features above 0.25 (2020: 9, 2022: 0). Top stock-level drifters are volatility/risk (atr_14, vol_20/60, beta_eem,
  beta_thb in 2013-2020) and then price-level features (dist_52w_low, sma200_ratio, ret_120, dist_52w_high) from 2023 on, i.e. market-trend state, not data issues.
  2026 includes liquidity (amihud_20, log_dvol60).
* Label base rate (is_alpha_5): 0.435-0.507 by year, no trend (2016-21: 0.435-0.507; 2022-26: 0.456-0.490). Cross-sectional alpha std: 0.039-0.060, high in 2020-21 and 2013-14.
* Score distribution: raw XGB probability mean/std move with the training base rate and with market dispersion (std range 0.035-0.088); PSI of the monthly score vs the previous 12 months has no relation to IC.
* Rolling IC: daily IC sd 0.150, so a 60-day mean has SE ~0.043 (non-overlapping equivalent), 120d ~0.031, 250d ~0.021. Monthly IC ranges -0.07..+0.16 inside 2023 alone.
* What leads IC drops? (monthly, 156 months, predictors known at month-end, target = mean IC of the next 3 months; HAC t, sign stability across 2013-19 / 2020-26, year-demeaned version)
  - Nothing is a validated leading indicator of IC *drops*. Significant-looking items fail robustness: score_xgb_mean (rho -0.36, t -3.1) disappears within-year (rho -0.02) - it is a
    cross-year level effect of the training base rate; psi_stock_mean/max and #features PSI>0.10 correlate **positively** with future IC (rho +0.29..+0.34; in 2020-26 the first two are ~0) - more
    stock-level drift came with better, not worse, IC; alpha_xs_std / mkt_dispersion (high dispersion -> lower IC, rho -0.31, HAC t -2.8) is stable in 2013-19 (-0.33) but flips sign in 2020-26 (+0.05).
    Macro PSI, base rate, VIX, score PSI: |t| < 1.7, no signal. With ~18 predictors x 4 targets and overlapping 3-month targets, p-values around 0.01 are not evidence.
  - The trailing 60d IC is coincident (rho 0.57 with the month's IC) but only weakly predictive (rho 0.22 for the next 3 months, 0.03/0.04 in each half); trailing 250d IC mean-reverts (rho -0.37 with the next-3m change).
  - Rolling-IC monitor (2016-26): a 60d IC < 0.02 flags 16% of days (next-60d IC 0.041 vs 0.071 when not flagged, but 15% vs 6% negative); 120d < 0.02 flags 13% of days; 120d < 0 only 1.9%; 250d < 0.02 4.2%; 250d never < 0.
    Even 2023 (annual IC 0.019) never turned the 120d red. The current paper.py thresholds (60d amber < 0.02, red < 0) are inside one SE of the typical IC (0.05) and will fire often by chance.
  - Does retraining help when drift is high? M-exp minus A-exp monthly IC: mean +0.0003, sd 0.020; larger when stock/macro PSI is high (+0.0025 vs -0.0020; +0.0044 vs -0.0038, rho ~ +0.2, p~0.02-0.03) - a hint, 5x smaller than monthly noise,
    not tested out of sample.
  => Use drift for data integrity and for *requesting a challenger*, not as an evidence-based retrain trigger. The only statistic that is both lagged-but-reliable and decision relevant is the long-window IC.

## 4. Recommendation
**Policy.** Keep the expanding window from 2010, production hyperparameters and the logit+XGB rank average. Cadence is statistically immaterial, so choose by operations: refit **quarterly** (first Saturday of Jan/Apr/Jul/Oct)
instead of weekly - same IC, fewer score discontinuities, 4 fits/year to audit and register (the pre-registered tie-break would say annual; quarterly is a judgment call to cap staleness at 3 months at a cost <= 0.001 IC).
Never use a rolling window <= 5y or recency weights (IC -0.006..-0.016, ICIR -0.02..-0.1). Do not expect retraining to recover the 2013-19 IC level: 90%+ of the decline is alpha decay (cap the expected IC at ~0.05
when sizing: `portfolio.alpha_from_scores` already uses a measured IC).
**Triggers** (each only creates a *challenger*, never changes the champion): (T1) schedule above; (T2) 120d IC < 0 and >= 60 trading days since the last fit; (T3) feature-schema change. Drift measures are data guards, not triggers.
**Monitoring (daily, IC lagged 5 days; thresholds in proposed/policy.py)**
| metric | amber | red / action |
|---|---|---|
| IC 120d (SE 0.031) | < 0.02 | < 0.0 -> challenger + human review |
| IC 250d (SE 0.021) | < 0.03 | < 0.02 -> review whether the signal still earns its costs (top-decile alpha 0.27%/5d vs 0.30% round trip) |
| IC 60d | dashboard only (16% false flags) | - |
| stock-level feature PSI, latest month vs champion training set | max > 4 or mean > 0.9 (p99 of 2013-26) | NaN share > 0.20 -> block the signals task |
| label base rate (trailing 1y) | outside 0.42-0.52 | - |
| score std, rank autocorr 1d | autocorr < 0.60 on a non-refit day; XGB score std outside 0.035-0.088 | refit-day champion/challenger rank corr < 0.50 -> do not promote |
**Promotion (champion-challenger).** A challenger is scored point-in-time next to the champion for >= 120 trading days. Scheduled/data-fix retrains: promote unless paired dIC < -0.005 (no-harm, since cadence has no
detectable effect, demanding t > 2 would never pass). Any change of window/architecture/features: pre-registered paired t > 2. Human approval is always required, and a promoted model reaches the paper book only as a **new shadow
with its own registration date**; S1-S4 (frozen 2026-11-02) keep using today's weekly `fit_final` until their evaluation ends.

## 5. Proposed wiring (files in experiments/retrain_v1/proposed/, tested by hand, not installed)
* `policy.py` - schedule, triggers, thresholds, artifact paths (default output/runtime; set SETALPHA_RETRAIN_HOME=data/monitor in deployment).
* `monitor.py` - daily `monitor` task: reuses `paper.ic_monitor` for the IC series, computes IC 60/120/250 (5d lag), stock-level PSI, base rate, score autocorr; appends to `monitor_metrics.parquet`
  (idempotent per day), writes `monitor_status.json`, alerts through log + optional webhook (`SETALPHA_ALERT_WEBHOOK`) + Airflow failure e-mail. Run on current data: status green (IC120 0.095, IC250 0.079; psi_stock_mean 0.19).
* `registry.py` - versioned registry (`registry/<vYYYYMMDD>/{logit,xgb}.joblib + meta.json` with trained_through, n_train, feature hash, params, git sha; `registry.json` with champion pointer and event history),
  `fit_challenger` (embargoed expanding fit), `score_day` (append-only point-in-time scores for champion+challengers), `evaluate` (paired IC), `promote` (gated, needs approval), `rollback` (pointer move; artifacts immutable).
  `register_current` bootstraps the production models as champion without touching `models/`.
* `set_quant_alpha_retrain_dag.py` - three DAGs, none of which edits `set_quant_alpha_daily/_weekly`: `set_quant_alpha_monitor` (sensor on the daily `report` -> monitor -> branch -> fit_challenger_triggered | no_retrain -> score_challengers),
  `set_quant_alpha_retrain_scheduled` (first Saturday of each quarter -> fit_challenger), `set_quant_alpha_promote` (manual trigger with params challenger/approved/kind/rollback).
  Smoke-tested outside Airflow: monitor on live data; register_current -> fit_challenger -> score_day -> evaluate -> promote refused without approval. The DAG file itself was not run under Airflow.

## 6. Caveats
* Signal-level only; no cost/portfolio effect of the extra score churn was simulated. Policy choice rests on 6 years (2016-21; ~290 independent 5-day targets/year); differences < 0.005 IC are undetectable. 2022-26 was seen
  by other experiments (reported once, not used).
* Hyperparameters were tuned on 2020-21 with a train-only fit (production `tune`); I did not retune per window (a 3y window might prefer a stronger regulariser). XGB/logit only (RF/tree not in the ensemble).
* Training pool is the liquid universe (~0.1-0.45M rows), so no subsampling was needed. Feature float32 cast has negligible effect (A-exp matches strategy_v3).
* Monthly drift/lead-lag panel: 156 overlapping months, ~18 predictors; treat all correlations as hypotheses. PSI uses a month of data against a fixed Jan-1 reference, so it rises mechanically within the year.
* The demo challenger-vs-production evaluation I ran was invalid on purpose-built grounds (production models are trained through Oct-2026, so scoring them on Jul-Sep 2026 is in-sample); the registry must score the champion live, point-in-time only. The demo outputs were deleted.
* output/fits (187 MB) is regenerable cache; do not commit.
