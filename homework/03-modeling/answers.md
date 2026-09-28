# Module 03 homework - answers

Submitted at: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw03
Date submitted:

| Q | Question | Computed | Choice to submit |
| --- | --- | --- | --- |
| 1 | Max absolute correlation of a `<month>_w<week_of_month>` dummy with `is_positive_growth_30d_future` | 0.025 | **0.025** (exact) |
| 2 | Precision of the best new hand rule (`pred3` / `pred4`) on TEST | 0.580 | **0.580** (exact) |
| 3 | TEST records where only `pred5_clf_10` is correct | 3802 | **3770** (nearest; gap 32) |
| 4 | Optimal `max_depth` (1..20) for the `DecisionTreeClassifier` | 5 | **5** (exact) |
| 5 | What data is missing? | free text - measured experiment in `notebook.ipynb`, Question 5 | free text |

All four validated against the executed `notebook.ipynb` (0 execution errors).

## Working notes

**Dataset.** Same source as the lecture Colab, not a rebuild: the Google Drive parquet
`stocks_df_combined_2025_06_13.parquet.brotli` (file id `1mb0ae2M5AouSDlqcUnIwaHq7avwGNrmB`),
downloaded once into the gitignored `data/raw/` cache. 230,262 rows x 203 columns, 33 tickers
(US / EU / India), 1972-06-01 -> 2025-06-13, truncated to `Date >= 2000-01-01` for modelling
(191,795 rows) exactly as the lecture does. Feature matrix = 184 numerical + 115 dummies = 299
columns.

**Splits.** The lecture's `temporal_split` at 70/15/15 by calendar date, giving
train 67.6% / validation 16.0% / test 16.4%. TEST is 2021-08-20 -> 2025-06-13, 31,408 rows.
Base rate of `is_positive_growth_30d_future` on TEST = 0.551 - the number every precision
below has to beat to be worth anything.

**Q1 - 0.025.** `CATEGORICAL` extended to
`['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']` with
`month_wom = Month + '_w' + (Date.day - 1) // 7 + 1`. `get_dummies` yields 115 dummies, 60 of
them from `month_wom`. The winner is `month_wom_October_w4` at +0.024968 -> **0.025**;
`November_w3` (+0.0221) and `November_w2` (+0.0188) follow, so the October/November hint holds.
All new dummies are kept in the feature set for Q2-Q4.

**Q2 - 0.580.** `pred3_manual_dgs10_5` = `(DGS10 <= 4) & (DGS5 <= 1)` makes 997 positive calls
on TEST, 578 correct -> precision 0.57974 -> **0.580**. `pred4_manual_dgs10_fedfunds` =
`(DGS10 > 4) & (FEDFUNDS <= 4.795)` makes 5,660 calls at 0.466 precision, i.e. *worse* than the
0.551 base rate, so `pred3` is the answer.

Why the thresholds had to be loosened from the tree's own splits: on the TEST window `DGS5`
never goes below 0.77 and `FEDFUNDS` sits at 5.33 whenever `DGS10` exceeds 4.825. Both original
conditions (`DGS10 <= 4.825 & DGS5 <= 0.745`, `DGS10 > 4.825 & FEDFUNDS <= 4.795`) therefore
produce **zero** positive calls on TEST and precision is undefined. The tree learned those
cut-points on the ZIRP years that live in TRAIN/VALIDATION.

For reference, the lecture rules on the same TEST set: `pred0_manual_cci` 0.558 (794 calls),
`pred1_manual_prev_g1` 0.542, `pred2_manual_prev_g1_and_snp` 0.522.

**Q3 - 3802 computed; submit 3770.** The offered options are 1770 / 2770 / 3770 / 4770, spaced
1000 apart, so 3770 is unambiguously the intended bucket - but it is not an exact match and is
not presented as one.

A sweep over the pipeline choices that could change the fitted tree shows the count is genuinely
sensitive, and that every spec-compliant reading lands in the 3770 bucket:

| variant | count | nearest option |
| --- | --- | --- |
| baseline: 115 dummies, fit TRAIN+VALIDATION | **3802** | 3770 (gap 32) |
| lecture's `ln_volume = np.log(Volume)` with `-inf` | 3802 | 3770 |
| without the `month_wom` dummies (239 features) | 3950 | 3770 |
| NUMERICAL only, no dummies (184 features) | 3774 | 3770 (gap 4) |
| fit on TRAIN only - violates the spec | 3272 | 3770 |
| no 2000-01-01 truncation - violates the lecture | 6973 | 4770 |

Two conclusions. First, the `-inf` -> NaN change made to `ln_volume` is **not** the cause: both
spellings give exactly 3802. Second, the closest variant to the grader's number (3774, gap 4) is
the one that fits on NUMERICAL features with no dummies at all - which contradicts both the
lecture (`features_list = NUMERICAL + DUMMIES`) and Q1's instruction to leave the new dummies in
the dataset. That variant was **not** adopted: matching the answer key by dropping features the
task says to keep would be fitting to the grader rather than solving the problem.

The residual 32-row gap (0.1% of the 31,408 TEST rows) is most likely scikit-learn version
tie-breaking - this repo runs sklearn 1.9.0 against a 2025-vintage course notebook, and the
`best` splitter's feature permutation under `random_state=42` has changed between versions. A
handful of tied splits resolving differently is enough to move the count by this much, and no
amount of matching the course's code would close it.

**Q3 detail.** `DecisionTreeClassifier(max_depth=10, random_state=42)` fitted on TRAIN+VALIDATION,
predicted over the whole frame (same length and row order, so the result is a plain column
assignment). TEST precision 0.589. `only_pred5_is_correct` is computed from the `is_correct_*`
columns rather than hard-coded per rule, so it scales to `pred0..pred99`. Count on TEST =
**3802** (~12% of the TEST rows).

**Q4 - 5.** Depths 1..20, all with `random_state=42`, trained on TRAIN+VALIDATION and scored by
precision on TEST. Best is **max_depth = 5** at precision **0.6278**, well past the 0.58 the task
suggests and ahead of every hand rule and of the depth-10 tree (0.589). The curve is not
monotonic: depths 1-4 are effectively "always predict up" (~31k of 31,408 TEST rows called
positive, precision = the base rate), depth 5 turns selective (18,452 calls), and depths 6-20
oscillate in 0.57-0.60 while validation precision keeps rising - overfitting. Retrained at
depth 5 and stored as `pred6_clf_best`.

**Caveat on all four numbers.** The target overlaps: rows one trading day apart share 29 of the
30 days of their forward window, so ~31k TEST rows carry on the order of ~1k independent
observations. The ranking among depths 6-20, and gaps of one or two precision points between
models generally, are inside that noise. The depth-5 vs depth-10 gap (0.628 vs 0.589) is large
enough to survive it; the rest of the table is not.


---

## Advanced (optional) sections

All optional extensions in the task are implemented in the notebook.

**Q3 advanced - generalising to `pred0`..`predN`.** Two implementations, cross-checked against
each other with an `assert`:
- `uniquely_correct_row(row, target_prefix, all_prefixes)` applied with `df.apply(..., axis=1)`,
  which is the row-wise form the task asks for and takes any number of prediction columns with
  no hard-coded conditions.
- `unique_correctness_table(df, prefixes, mask)`, the vectorised generalisation, which reports
  correct / uniquely-correct / unique-share for *every* predictor in one pass.

**Q4 advanced - saturation and the complexity trade-off.** The depth sweep records precision,
accuracy and AUC on train/validation/test plus `n_leaves` at every depth. Two plots (precision
and accuracy, all three splits, leaf count on a log twin axis), a `complexity` table with
`leaves_per_1k_rows` and the train-test accuracy gap, and the optional tree-head visualisation
at depths 1/2/3/best via both `export_text` and `plot_tree`. Train accuracy climbs steadily while
test accuracy does not - the tree buys complexity it cannot spend.

---

## Question 5 - measured, not asserted

Four feature blocks were engineered and evaluated rather than merely proposed. Everything is
derived from the parquet already on disk: no downloads, no external APIs, no network.

| Block | Blind spot addressed | Features |
| --- | --- | --- |
| A | Rates appear only as levels | term spreads, real rates, policy gap, 1m/3m/6m rate changes, cycle flags (17) |
| B | Every feature is absolute, none relative | per-date percentile ranks (all tickers and within region), excess growth vs S&P and vs home index - US->S&P, EU->DAX, India->EPI (22) |
| C | No sense of position in the ticker's own range | 52-week high/low distance, close vs own SMAs, vol/volume/RSI z-scored against own history (9) |
| D | No market-regime signal | index *levels* rebuilt by cumprod of `growth_*_1d`, giving drawdown and realised-vol regime; gold/oil ratio; correction flag (10) |

**Protocol fix.** Q4 selects `max_depth` by TEST precision, which is selection on the test set.
The Q5 protocol tunes depth on VALIDATION, refits on TRAIN+VALIDATION and touches TEST once.
Under it the baseline single tree scores **0.5785**, not Q4's 0.628 - a 5-point drop that is
purely the cost of the homework's protocol.

### Single decision tree

| feature set | TEST precision | TEST AUC |
| --- | --- | --- |
| BASE (homework) | 0.5785 | 0.5400 |
| + A macro curve | 0.5362 | 0.5271 |
| **+ B cross-sectional** | **0.5869** | **0.5467** |
| + C own-history | 0.5832 | 0.5378 |
| + D market regime | 0.5466 | 0.5432 |
| + ALL blocks | 0.5388 | 0.4827 |

Only block B improves both metrics. ALL blocks together give an AUC *below* 0.5 - worse than
random.

### Ensembles, same features (TEST base rate 0.5511)

| feature set | model | precision@0.5 | AUC | precision on top decile |
| --- | --- | --- | --- | --- |
| BASE | RandomForest | 0.5508 | 0.5081 | 0.5632 |
| BASE | HistGradientBoosting | 0.5642 | 0.5420 | 0.6065 |
| BASE+B+C | HistGradientBoosting | 0.5693 | 0.5479 | 0.6192 |
| ALL | HistGradientBoosting | 0.5660 | **0.5514** | 0.6552 |
| ALL | **RandomForest** | 0.5537 | 0.5411 | **0.7077** |

### What the experiment actually shows

1. **The blocks that wreck a single tree are the ones the ensembles want.** A and D are
   date-level series, identical across all 33 tickers on a day. One tree splits on them and
   memorises 2000-2021 macro regimes that never recur in 2021-2025 (BASE+D picks depth 3 and
   collapses to "always up", 31,053 of 31,408 rows). Three hundred bagged trees with
   `min_samples_leaf=50` average that away and keep what generalises - ALL is the best feature
   set for both ensembles and the worst for a lone tree.
2. **The uplift lives in the ranking, not at the 0.5 threshold.** No ensemble beats the single
   tree's 0.5869 precision@0.5, but that metric compares models making 19,896 vs 29,217 calls.
   Matched on selectivity, RandomForest on ALL gets **70.8% of its top-decile signals right
   against a 55.1% base rate: +15.7pp, of which +14.5pp comes from the engineered features**
   (the same model on BASE alone manages 0.5632).
3. **The importance audit agrees.** The 58 engineered columns are 16% of the 357 features but
   hold **39.4% of the RandomForest's total importance**, and 8 of its top 20 features are
   engineered - led by `cpi_core_yoy_chg_126d`, `real_10y`, `fedfunds_chg_126d`,
   `fedfunds_chg_63d`. That is block A, the block that was *worst* for a single tree. Rate
   dynamics carry real signal; one tree just could not use them without memorising the era.
4. **Two different things were missing.** Relative/cross-sectional information and regime
   context were genuinely absent from the frame. But the bigger share of the uplift came from
   *how the prediction is consumed* - rank by probability, act on the top decile - rather than
   from any column. A tree emits one probability per leaf, so thresholding at 0.5 discards the
   ranking, which is why every Q1-Q4 precision sits in such a narrow band.

### Still missing, would need downloads (not tested here)

Earnings calendar (`days_to_next_earnings` - a 30d window almost always straddles a release);
region-matched policy rates (ECB/RBI - a third of tickers are non-US and every macro column is
US); FX (USDINR, EURUSD); true implied vol and credit spreads (VIX/VSTOXX/India VIX, HY OAS -
block D's realised-vol rebuild is backward-looking where those are forward-looking);
fundamentals (P/E, EV/EBITDA, margins, earnings surprise - not one valuation feature exists);
GICS sector (`ticker_type` is geography only).
