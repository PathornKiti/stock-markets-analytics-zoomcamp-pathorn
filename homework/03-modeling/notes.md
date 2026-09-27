# Module 03 notes - Analytical Modeling

## Lecture takeaways

- **Temporal split, never a shuffle.** `temporal_split` cuts train/validation/test by calendar
  date so no future row can leak into training. The date cut does not give exactly 70/15/15 of
  the *rows* (we got 67.6/16.0/16.4) because ticker coverage grows over time - that is fine and
  expected.
- **Precision, not accuracy, is the metric that matters here.** We only act when the model says
  "1", so `TP / (TP + FP)` on the positive calls is the thing being optimised. Accuracy rewards
  a model for correctly declining to trade, which is worth nothing.
- **Always quote the base rate next to the precision.** `is_positive_growth_30d_future` is
  positive 55.1% of the time on TEST, so a "0.55 precision" model has learned nothing, and
  `pred4` at 0.466 is actively worse than a coin weighted to "up".
- **Positive-call count is half the story.** A shallow tree hits 0.55 precision by calling
  almost every row positive. Depth 5 gets 0.628 by calling only 18,452 of 31,408. Precision
  without volume is meaningless, and vice versa.
- **Read rules off the fitted tree.** `export_text(clf, feature_names=..., max_depth=3)` is far
  easier to read than `plot_tree` and is where the `pred3`/`pred4` hand rules came from.
- Feature importance for `clf10` is dominated by macro rates (`DGS10`, `DGS5`, `FEDFUNDS`),
  not by any technical indicator - worth remembering before adding a 60th candle pattern.

## Gotchas hit while doing the homework

- **A tree's own thresholds can be unreachable on TEST.** `DGS5 <= 0.745` never happens after
  2021-08 (min on TEST is 0.77), so the rule fires zero times and precision is `0/0`. The tree
  learned the split on the ZIRP era sitting in TRAIN/VALIDATION. This is regime shift in its
  purest form, and it is why the homework tells you to round the cut-points outward.
- **`random_state=42` is not optional.** Without it `DecisionTreeClassifier` breaks ties
  randomly among equally-good splits and Q3/Q4 move between runs.
- `ln_volume = log(Volume)` produces `-inf` wherever `Volume == 0` (halted/untraded days).
  Replace `+-inf` with `NaN` and then fill with 0 *before* fitting, or sklearn raises.
- **Predict on the whole frame, not just TEST**, when the prediction has to become a column.
  `X_all` keeps the row order of `new_df`, so `new_df['pred5_clf_10'] = clf.predict(X_all)`
  needs no join. Predicting only on TEST leaves you joining on a non-unique index.
- `gdown --fuzzy` does not exist in gdown 6.x - pass the bare file id instead
  (`gdown.download(id=..., output=...)`).
- `pd.get_dummies` on `month_wom` gives 60 columns, not 12x5=60 exactly by luck: months with a
  31st day get a `w5`, February usually does too. Check the count is ~115 total before moving on.

## From the Q5 experiment (the part that generalises)

- **Tuning on the test set costs ~5 precision points here.** Selecting `max_depth` by TEST
  precision gives 0.628; selecting on VALIDATION and touching TEST once gives 0.578 for the
  same pipeline. Whenever a tutorial says "pick the best on test", that gap is what it is
  hiding.
- **A feature that helps an ensemble can destroy a single tree.** Date-level macro features
  (identical across all tickers on a day) let one tree memorise regimes: AUC fell to 0.4827,
  *below random*, while the same columns gave the best AUC and top-decile precision of any
  configuration once bagged/boosted. Never conclude "this feature is useless" from one tree.
- **Precision at a fixed 0.5 threshold is not comparable across models** that make different
  numbers of calls (19,896 vs 29,217 here). Compare at matched selectivity - precision on the
  top decile of predicted probability - or compare AUC.
- **Ranking beats thresholding.** The single largest uplift in the whole notebook came not
  from a feature but from consuming the model as a ranking: base rate 0.551 -> 0.708 on the
  top decile. A decision tree has one probability per leaf, so a 0.5 cut throws that away.
- **Rebuild what you cannot download.** With no network allowed, `growth_snp500_1d` cumprod'd
  over the unique-date series recovers a synthetic index level, and from a level you get
  drawdown and realised vol - a serviceable local stand-in for VIX.
- `-inf` from `log(0)` silently poisons every rolling window it touches. `np.log(x.replace(0,
  np.nan))` is identical once NaNs are filled, and keeps rolling z-scores finite.

## Ideas worth reusing in the capstone project

- The `is_correct_<pred>` / `only_<pred>_is_correct` pattern generalises: build one boolean
  column per model, then ask which model is *uniquely* right. That is the honest way to decide
  whether a new model earns its place in an ensemble rather than duplicating an existing one.
  `unique_correctness_table()` in the notebook does this for every predictor in one pass.
- `evaluate_feature_set(name, feature_list)` - tune on validation, refit on train+validation,
  score once on test, return a dict. Ablating a feature block becomes a one-line call, which is
  the only way feature work stays honest as the block count grows.
- `precision_table(df, prediction_columns, split=...)` - one call, every rule and model side by
  side with its positive-call count. Lift this straight into the project's evaluation module.
- The depth-vs-precision curve with validation plotted alongside is the cheapest overfitting
  diagnostic available; do the same for any `n_estimators` / `max_depth` sweep later.
- **Overlapping targets are the project's biggest statistical risk.** A 30-day forward return
  sampled daily means consecutive rows share 29/30 of their window, so ~31k rows behave like
  ~1k. Before trusting any precision comparison in the capstone, move to purged and embargoed
  walk-forward CV and report an interval, not a point estimate.
- The macro block in this dataset is US-only while a third of the tickers are EU/India. If the
  project reuses this frame, region-matched rates (ECB, RBI) and FX are the first things to add.
