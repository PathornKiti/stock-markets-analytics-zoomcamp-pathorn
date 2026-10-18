# Module 04 homework - answers

Submitted at: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw04
Date submitted:

| Q | Question | Computed | Choice to submit |
| --- | --- | --- | --- |
| 1 | CAGR of `pred10_rf_best_rule_55` with fewer dummies (208 features) | 1.1291 | **1.126** (nearest; gap 0.003) |
| 2 | Threshold of the second-best RF rule with average trades/day < 5 | 0.57 (4.55/day, CAGR 1.103) | **0.57** (exact) |
| 3 | Number of trades of `pred5_clf_10` after redefining the target | 7.85 per day (7,201 in total) | **8** |
| 4 | [EXPLORATORY] Ideal trading strategy | see below | free text |

All validated against the executed `notebook.ipynb` (0 execution errors, scikit-learn 1.7.2).

## Q4 - my ideal strategy: "Relative-strength RF ladder" (beats the baseline)

1. **Signal.** The same Random Forest (`max_depth=13`, 100 trees, 208 features), retrained every 6 months with a 45-day gap (walk-forward), but trained to find stocks that finish in the **top third of the day's 30-day returns** instead of stocks that merely go up. Buy $100 when its score is above a threshold chosen on the 2018-2022 validation years so that it selects the share of rows you can fund.
2. **Execution.** Hold 30 trading days, no stop-loss, one position per ticker; optionally split each $100 into five $20 tranches (DCA, same CAGR).
3. **Not re-tested on this signal:** the alpha-gated 2x leverage and the Kelly circuit breaker below.

Walk-forward results (thresholds from validation, TEST read once; baseline = static RF >= 0.57, flat $100):

| target / selection share | TEST trades/day | net $ | reserve capital | CAGR | alpha-CAGR |
| --- | --- | --- | --- | --- | --- |
| **baseline (static RF >= 0.57)** | 4.62 | 8,669 | 18,000 | 1.1033 | 1.0159 |
| walk-forward "goes up", 6.4% | 5.51 | 4,927 | 27,000 | 1.0428 | 1.0151 |
| walk-forward "beats S&P", 6.4% | 3.90 | 7,506 | 18,000 | 1.0910 | 1.0349 |
| **top-third-of-day, 6.4%** | 1.76 | 6,047 | 6,000 | **1.1904** | 1.1147 |
| top-third-of-day, 10% | 2.33 | 7,355 | 9,000 | 1.1610 | 1.0703 |
| **top-third-of-day, 20% (same capital as baseline)** | 4.52 | **13,552** | 18,000 | **1.1506** | 1.0421 |

Reading it: the top-third target is best at every selection share (CAGR 1.15-1.19; validation 1.11-1.14 vs ~1.02 for "goes up"). At 6.4% its high CAGR is mostly capital efficiency (sparse signals, small reserve); the like-for-like check is the 20% row, which uses the same $18k reserve and earns $13.6k vs $8.7k. Resampling the 48 entry months, it beats the static baseline in 96% of resamples (95% interval of the CAGR gap [-0.014, +0.183], touching zero) and the walk-forward "goes up" model in 99% (interval [+0.040, +0.237]). It makes money in all 5 TEST years (2026 partial) and 3 of 5 validation years, trading 25 of 33 tickers. The walk-forward "goes up" model is worse than the static baseline (1.043 vs 1.103), so part of the static baseline's strength is a lucky single fit. Caveats: four TEST years, 33 tickers, no margin / slippage / short costs modelled.

### Earlier candidate on the old signal: "Confident-core RF ladder with alpha-gated leverage"

1. **Signal.** Random Forest (static, `max_depth=13`, 100 trees, 208-feature setup); buy $100 of every stock with `P(up in 30d) >= 0.57` (~4.6 trades/day, CAGR 1.103).
2. **Hold** 30 trading days, no stop-loss, one position per ticker.
3. **CAPM monitor.** Beta vs the S&P 500 per stock (estimated before the TEST window) and the trailing alpha of the last 200 closed trades (only trades older than ~42 days). A diagnostic, not a hedge: hedging the market away leaves CAGR 1.016.
4. **Leverage when efficient.** Double to $200 only while that trailing alpha is positive, financed at Fed funds + 1% on the extra $100, equity fixed: CAGR 1.136 vs 1.100 at 1x. By CAGR alone, doubling every signal is best (1.168); the gate switches leverage off when the model stops working. Margin calls, liquidation and drawdowns are not modelled.

All ideas backtested on TEST, same fees; CAGR is the main metric, alpha is the second lens (alpha-CAGR = CAGR of the beta-hedged P&L):

| strategy | trades/day | CAGR | alpha % per trade | alpha-CAGR |
| --- | --- | --- | --- | --- |
| **0  baseline RF >= 0.57, $100** | 4.62 | 1.1033 | +0.92 | 1.0159 |
| 1  conviction: $200 if p>=0.65 else $100 (p>=0.58) | 4.16 | 1.0950 | +0.84 | 1.0129 |
| 2  consensus: RF >= 0.55 AND decision tree | 3.67 | 1.0720 | +0.36 | 1.0000 |
| 3  top-5 per day, p >= 0.60 | 2.51 | 1.0378 | +0.82 | 1.0048 |
| 4a RF >= 0.57 + stop -8% / take-profit +15% | 4.62 | 1.0730 | -0.41 | 0.9778 |
| 4b top-3/day p>=0.65 + stop -5% / take-profit +10% | 1.56 | 1.0035 | -0.61 | 0.9984 |

| leverage (equity fixed at 1x capital) | doubled share | CAGR |
| --- | --- | --- |
| 1x | 0% | 1.1001 |
| 2x on every signal | 100% | 1.1680 |
| 2x when p >= 0.60 | 32% | 1.1283 |
| 2x when trailing hit-rate >= 60% | 45% | 1.1348 |
| **2x when trailing CAPM alpha > 0** | 40% | **1.1358** |

Creative extensions (own flat twin first; equity fixed at the flat strategy's capital; CAGR / alpha per trade):

| strategy | CAGR | alpha % per trade |
| --- | --- | --- |
| Flat $100, RF >= 0.60 | 1.0620 | +1.16 |
| Kelly-sized (rolling win rate and win/loss ratio of closed trades) | 1.0564 | +2.41 |
| Flat RF >= 0.57 -> skip days when trailing Kelly <= 0 | 1.0921 (vs 1.1033) | +1.62 (vs +0.92) |
| ... and double when relative Kelly >= 1 | 1.0949 | +1.62 |
| Risk parity, all RF >= 0.57 (flat 1.1033) | 1.0995 | +0.92 |
| Fundamental overlay on RF >= 0.55 (flat 1.1291): value proxy / dividend proxy / both | 1.0834 / 1.1194 / 1.0762 | +0.78 / +0.85 / +0.69 |
| Monday signals only / next-Monday entry (any-day 1.1033) | 1.0223 / 1.0980 | +0.75 / +0.84 |
| DCA: 5 x $20 on consecutive days (lump 1.1033) | 1.1008 | +0.84 |

Kelly works as a circuit breaker (better alpha quality, ~1 CAGR point lower), not a growth engine. Risk parity and the fundamental overlay lowered CAGR; P/E, free cash flow and sector are not in the dataset, so that overlay used proxies (price below its 250-day average, dividend payer). There is no Monday effect in this data (Monday signals 3.7% over 30 days vs 4.3% average); DCA matches the lump sum, so it is a free execution choice.

CAPM split of the baseline signals: average 30-day return 4.28% = 3.36 points market beta (beta 0.74) + 0.92 points alpha; the whole universe earns +0.50% alpha. The model selects better stocks, but most of the profit is market exposure. Idea 1's ">0.70" tier never fires (max RF probability on TEST is 0.694). Logistic Regression / XGBoost consensus was not tested, and arbitrage-style long/short was tried and dropped (it added nothing at low short thresholds and lost money at 0.45). Optional extras of the strategy: a Kelly circuit breaker (skip trades while the trailing Kelly fraction <= 0) and DCA execution (5 x $20).

## Working notes

**Dataset.** The official notebook's parquet `stocks_df_combined_2026_09_18.parquet.brotli` (Drive file id
`1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5`), cached in the gitignored `data/raw/`. Truncated to `Date >= 2000-01-01`;
temporal split 70/15/15 gives train -> 2018-09-13, validation -> 2022-09-15, TEST 2022-09-16 -> 2026-09-18
(33,163 rows). Feature matrix 186 numerical + 115 dummies = **301** originally and 186 + 22 = **208** after
dropping `Ticker` and `month_wom` from `CATEGORICAL` - both counts match the task text.

**Port, not a rewrite.** The notebook's cells were ported into functions (`build_dataset`, `add_model_predictions`,
`simulate`) and cross-checked by running the official cells verbatim: the fixed rules `pred0..pred4` give identical
CAGRs (e.g. `pred1_manual_prev_g1` 1.119072) in both. One quirk of the official `simulate` is reproduced on purpose: the
net-revenue column is `gross + fees` per row, so the ~500 latest rows with unknown 30-day growth (NaN) fall out of the
sum together with their fees. Not reproducing it moves CAGRs by ~0.0002 - 0.0005.

**Q1 - computed 1.1291, submit 1.126.** With the reduced dummies `pred10_rf_best_rule_55` makes 3,976 trades at
6.03/day, net $15,013, capital $24,000 -> CAGR **1.1291**. It overtakes `pred5_clf_10` (1.1270), which matches the
task text ("the best ML predictor has changed"). None of the four options is exact; 1.126 is the nearest (gap 0.003,
next 1.136 at 0.007). The original 301-feature setup gives 1.0954 here, not the 1.1195 quoted in the task - so the
instructor's run and this one differ by a small, systematic amount (see the sklearn note).

**Q2 - 0.57.** Thresholds 0.56 ... 0.64 added after `pred11`. Average trades/day: 0.56 -> 5.21 (fails), **0.57 -> 4.55**
(CAGR 1.103), 0.58 -> 4.11 (1.092), 0.59 -> 3.62 (1.068). 0.57 is the highest-CAGR rule under 5/day. Precision rises
with the threshold (0.61 at 0.50 -> 0.66 at 0.60) while trades fall from 12,036 to 668, as the task predicts.

**Q3 - 8.** Target redefined as `growth_future_30d > 1.022` (class 1 drops 59% -> 49%). `pred5_clf_10` is then the best
predictor by CAGR (1.1200). It makes **7,201** trades in the 4 TEST years = **7.85 per day** (9.03 with the original
target) at a similar CAGR (1.119). The options 6-9 only fit a per-day figure, so **8** is submitted; if the form wanted
the raw count it would be 7,201. This number comes from a decision tree and is identical on every scikit-learn version.

**Caveat - scikit-learn version.** A Random Forest with `random_state=42` is not reproducible across scikit-learn
versions. Same code and data:

| scikit-learn | `pred10` CAGR (Q1) | Q2 answer |
| --- | --- | --- |
| 1.4.2, 1.5.2 | 1.1189 | 0.59 (not an option) |
| **1.6.1, 1.7.2** | **1.1291** | **0.57** |
| 1.8.0, 1.9.0 (repo venv) | 1.1189 | 0.59 (not an option) |

Only the 1.6/1.7 results are consistent with the task (Q2's answer is an offered option and `pred10` becomes the best
ML predictor), so the notebook is executed with 1.7.2 and says so at the top. Q1's 1.126 is therefore a nearest-option
choice rather than an exact match, the same way HW3 Q3 was handled.
