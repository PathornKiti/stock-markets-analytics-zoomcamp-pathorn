# model_v2 report

Run: `uv run set-quant-alpha/experiments/model_v2/model_v2.py` (about 1 minute; `--phase valid|test` to split).
Selection used 2020-21 only; the selection rule is in the script docstring and was coded before any result. Test was run once.
Trials: 12 (B0 + V1..V11), so use N=12 for deflation.

## Validation (2020-21, tradable names, IC vs fwd_alpha_5)
See `output/valid_comparison.csv` and `output/valid_ic_by_year.csv`. Headlines:
- baseline ens: IC 0.0511 (t 7.1), top-decile alpha +0.15%, lag-5 rank autocorr 0.43
- V3 (vol-scaled, winsorised, date-demeaned alpha regression): IC 0.0638 (t 9.9), top-decile +0.29%, lag-5 autocorr 0.45, top-decile 5d overlap 0.36 (baseline 0.31)
- V2 (CS-rank target) IC 0.0551; V1 (winsorised alpha) IC 0.0517 but top-decile +0.62%
- Slow-signal variants: V7 (10d target) ac5 0.61 / IC 0.0484; V8 (20d) ac5 0.75 / IC 0.0483 / top-decile +0.39%
- Not helpful: V5 (CS-ranked features, 0.0397), V9 (ridge, 0.0365), V6 (tradable-only training), V4 (pairwise ranker, 0.0525), V11 EMA3 smoothing (cuts IC to 0.0576)

Rule outcome: eligible = V2, V3, V10, V11; chosen = V3 (max valid IC).

## Test (2022-26, walk-forward, one shot), `output/test_comparison.csv`
| | IC | t | top-decile 5d alpha | AUC | lag-5 rank ac | top-decile overlap |
|---|---|---|---|---|---|---|
| baseline | 0.0503 | 11.9 | +0.26% | 0.5303 | 0.566 | 0.41 |
| V3 | 0.0527 | 14.1 | +0.38% | 0.5316 | 0.500 | 0.43 |

IC by year (baseline / V3): 2022 .064/.058, 2023 .019/.038, 2024 .046/.055, 2025 .051/.050, 2026 .077/.066.
The test gain (+0.002 IC, +0.12pt top-decile) is far smaller than the validation gain (+0.013 IC) and not uniform across years,
so treat it as modest at best. The score is slightly LESS persistent than the baseline on test (lag-5 ac 0.50 vs 0.57), so V3 does not help turnover.

## Caveats
- Baseline IC on test here (0.0503) differs from README (0.056) because the metric is computed on tradable names and on the prediction rows only; same function for both columns.
- pred_ens in predictions_v2.parquet is the daily percentile rank (0-1) of V3 scores, not a probability. pred_tree/rf/logit/xgb are copied from production; `pred_v2` duplicates pred_ens.
- Untested: sample weights, early stopping, calibration, hyperparameter retuning per formulation.
