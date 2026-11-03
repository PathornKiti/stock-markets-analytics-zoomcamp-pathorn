# integrated_v1 - all signals combined, parameters tuned by walk-forward optimisation (2026-11-02)

Code: `integrated_v1.py` (stages `panel`, `grid`, `wf`), `make_v8_wf.py`, `diagnostics.py`. Tables are in
`output/tables.md`; the protocol was frozen in `output/protocol_frozen.txt` (09:40Z, before any selection).

**Verdict:** no integrated, honestly tuned strategy beats TDEX or the factor models with significance. The
walk-forward strategy earned 7.8% a year net against TDEX's 7.2% (2016-01 to 2026-10). Every net alpha is positive but
not significant (t 0.3-1.1), DSR is 0.16 and PBO is about 0.6.

## Signals and composites

**Signals:** the ML score; V8, a 20-day-target ML score (regenerated walk-forward for 2013-26); low beta; low
volatility; residual momentum; 52-week high; EPS surprise; dividend yield. Each is ranked daily within the top-100.

| Composite | Definition |
| --- | --- |
| C1 | ML score alone |
| C2 | average rank of ML and V8 |
| C3 | average rank of the 4 literature factors |
| C4 | equal-weight average of all 8 signals |
| C5 | IC-weighted, expanding window (past data only) |
| C6 | as C5, rolling 3-year window |
| C7 | ML + V8 + EPS surprise + dividend yield |
| C8 | C5 made neutral to size, beta and momentum |

## Grid and objective

* **Grid:** 288 configs = 8 composites × N {10, 20, 30} × rebalance every {5, 10, 21} days × buffer {none, min(3N, 80)}
  × weighting {equal, score-tilted}.
* **Extended grid:** adding an 8% stop gives 576 configs.
* **Objective:** net CAPM-alpha t-stat vs TDEX (Newey-West), on an expanding window from 2013. Configs above 40×
  two-way turnover a year are ineligible.
* **Walk-forward:** each year from 2016 to 2026, the config is chosen on 2013 to Y−1 only, then traded in year Y.
* **Trials:** 89 earlier + 576 + 6 overlays = **671**. The Harvey-Liu-Zhu (Bonferroni) hurdle at that count is t = 3.96.
* **Simulators:** the fast (proportional-cost) and exact (THB 1M, 100-share lots) simulators agree: correlation of
  daily returns 0.9998, CAGR 7.90% vs 7.78%.

## Walk-forward picks

| Year | Pick | OOS net (exact) | TDEX |
| --- | --- | --- | --- |
| 2016-2020 | C3 literature factors, N10, monthly, buffered, equal weight | 14.8 / 17.6 / 0.9 / 14.5 / **−23.2%** | 23.7 / 18.7 / −4.6 / 6.2 / −11.0% |
| 2021-2026 | C2 ML + V8, N10-20, weekly, buffered, score-tilted | 27.4 / 2.7 / 10.4 / −13.9 / 10.9 / 32.1% | 13.4 / 9.0 / −8.4 / 9.1 / −4.2 / 30.8% |

On average, each year's pick ranked only at the 52nd percentile of the grid in the following year: tuning added
nothing out of sample.

## Out-of-sample result, 2016-01-04 → 2026-10-22

| | Walk-forward net | Walk-forward gross | TDEX |
| --- | --- | --- | --- |
| CAGR | **7.78%** | 10.08% | **7.16%** |
| Cost drag / two-way turnover | 2.3% / 13.7× | | |
| Sharpe / max drawdown | 0.58 / −34.9% | | 0.53 / −36.5% |
| Information ratio vs TDEX | 0.04 | | |
| CAPM α (t), beta 0.65 | +3.4% (0.97) | +5.5% (1.58) | |
| FF3-EM α (t) | +1.4% (0.29) | +3.4% (0.72) | |
| Carhart-EM α (t) | +5.4% (1.07) | +7.5% (1.47) | |
| Local-Thai Carhart α (t) | +2.7% (0.75) | +4.8% (1.36) | |
| Local Carhart with the factors charged their own costs | +3.3% (0.96) | | |
| PSR / DSR (671 trials) | 0.97 / **0.16** | | |

* **Sub-periods (net):** 2016-21 7.6% vs TDEX 7.3%; 2022-26 8.1% vs TDEX 7.0% (E2 made 1.6%).
* **PBO** (CSCV over the 288 configs) is about 0.6, with a degradation slope of −0.98.

## Ablations (pre-declared)

| Variant | CAGR | CAPM α t |
| --- | --- | --- |
| 50% / 70% TDEX core | 7.7% / 7.6% | 0.98 |
| 8% stop forced on | 6.4% | 0.88 |
| IR objective instead of alpha t | 12.8% | 1.84 |
| ML composites only (C1, C2) | 13.4% | 2.02 |
| Literature factors only (C3) | 6.9% | 0.96 |

The IR-objective and ML-only variants look better. They were not the pre-registered primary, so choosing them now
would be selecting on the out-of-sample period, and even they fail the DSR and the t-hurdle.

## Cost sensitivity (same picks; TDEX 7.16%)

| Cost per side | CAGR | CAPM t / local t |
| --- | --- | --- |
| 0.05% | 9.41% | 1.37 / 1.14 |
| 0.10% | 8.65% | 1.18 / 0.95 |
| 0.15% | 7.90% | 0.98 / 0.76 |

Re-optimising at lower costs did worse, because it picked wide, unbuffered books. No cost level gives significant
alpha.

## In-sample / overfit, for comparison only

The hindsight-best config (C2: ML + V8, 10 names, weekly, buffered, score-tilted) shows:
* **2013-26:** 16.3% net CAGR, CAPM α t 3.17, local Carhart t 2.84.
* **2016-26:** 15.4% net CAGR, CAPM α t 2.36.
* **DSR:** about 0.55, below the bar even in hindsight.

**Why the hindsight best beats the walk-forward by about 7.5 points a year:**
* **Timing.** In 2016-20 the walk-forward chose the low-beta literature book, which led in-sample but lost 24% in
  2020, while C2 gained 26%.
* **Selection noise.** The best and second-best configs differ by at most 0.27 in t, so the choice is near random.
* **Multiple testing.** Even the hindsight best misses its DSR and t bars.
* **Caveats that flatter every C2 number:** ML hyperparameters were tuned on 2020-21, V8's behaviour was already known,
  2022-26 has been reviewed many times, and the universe is survivorship-biased.

## Recommendation

Do not deploy capital. If anything goes to paper trading, register C2 N10 R5 B T (or its 50%-TDEX-core version,
tracking error about 6%) as a new shadow with its own registration date. Judge it only on future data.
