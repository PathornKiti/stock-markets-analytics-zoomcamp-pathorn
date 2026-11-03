# adaptive_v1 - adaptive signal weights, wider books, earnings-surprise overlay (2026-11-03)

Code: `adaptive_v1.py` (stages `ml3`, `discover`, `check`). Tables: `output/tables.md`.

* **Protocol.** Frozen in `output/protocol_frozen.txt` at 2026-11-03T02:28:30Z, before any of these strategies was
  computed.
* **Decision.** Written to `output/discovery_decision.txt` (02:30:23Z) from 2013-21 only.
* **2022-26 is burned.** The signal pool, the surprise signal and the wider-book grid had all been seen on it before,
  so it is reported as a consistency check, not a holdout.
* **Universe.** Survivorship-corrected L, tradable top-100; weekly, buffer, equal weight, net of 0.168% per side.

**Verdict: none of the 6 strategies passes the pre-registered 2013-21 rule.**

* **S5's headline backtest number depends on which weekday it rebalances.** S5 decides on one fixed day of each
  weekly cycle. Running it on each of the 5 possible decision days:
  * 2013-21: 16.0% on the standard day, 14.4% on average (spread ±2.4).
  * 2022-26: 7.7% vs 6.0% on average, below TDEX's 6.7%.
  * 2013-26: 13.0% vs 11.4%.
  * The standard day happened to be about 1.6 points a year better than average, so the phase average is the more
    honest backtest figure.

## Results

| Strategy | 2013-21 CAGR / Sharpe | 2013-21 avg over 5 days ± sd | 2022-26 CAGR / Sharpe / max DD (burned) | 2013-26 CAGR / Sharpe / α t | Decision |
| --- | --- | --- | --- | --- | --- |
| A: S5 (C2, top-10) | 16.0% / 0.91 | 14.4 ± 2.4 | 7.7% / 0.54 / −27.6% | 13.0% / 0.79 / 2.65 | baseline |
| S1a: adaptive weights (trailing 24-month IC) | 14.7% / 0.82 | 13.3 ± 3.5 | **11.6% / 0.89 / −12.0%** | **13.6% / 0.83 / 3.12** | fail |
| S1b: ML refit on last 3 years only | 14.4% / 0.80 | 10.9 ± 2.9 | 6.8% / 0.48 / −24.1% | 11.7% / 0.70 / 2.29 | fail |
| S2a: C2, top-20 | 14.4% / 0.88 | 10.8 ± 3.0 | 5.6% / 0.44 / −27.5% | 11.2% / 0.74 / 2.56 | fail |
| S2b: C2, top-30 | 9.0% / 0.60 | 8.2 ± 1.9 | 0.3% / 0.09 / −32.8% | 5.9% / 0.44 / 0.81 | fail |
| S3a: ML + V8 + surprise | 14.4% / 0.84 | 13.2 ± 2.3 | 8.9% / 0.66 / −26.1% | 12.4% / 0.78 / 2.79 | fail |
| S3b: ML + V8 + fresh surprise (10 days after report) | **16.4% / 0.93** | 14.3 ± 2.6 | 9.1% / 0.63 / −24.7% | 13.8% / 0.83 / 2.85 | fail (avg 14.3 < 14.4) |
| TDEX | 3.9% / 0.31 | | 6.7% / 0.54 | 4.8% / 0.38 | |

Every stationary-bootstrap Sharpe-difference CI vs A includes zero, except S2b (worse in both periods).

### 1. Adaptive weighting (S1a, S1b)

* **The weights rotate as intended.**
  * 2014-21: weight goes to ML, V8 and analyst revisions (rev3, ptrev3, dispersion); value gets 0.
  * 2025-26: weight on bp, ep and forward dividend yield rises to 0.30-0.40.
  * See `tables.md`.
* **It is late.** 24 months of IC history means it starts on value only after the switch has happened. So it lost to S5 in
  2013-21 (13.3% vs 14.4%, averaged over the 5 decision days) and won in 2022-26.
  * 2022-26: 11.6% vs 7.7%, max drawdown −12% vs −28%.
  * 2013-26 overall: 13.6% vs 13.0%; alpha t 3.12 vs 2.65.
  * Under the pre-registered rule this is a fail. The 2022-26 win is in the burned period and cannot be used to select
    it.
* **The ML refit on only 3 years is clearly worse.** Daily IC 0.070 vs 0.081 in 2013-21 (paired t −3.7), and 0.039 vs
  0.055 in 2022-26 (t −2.9). The pooled all-history refit wastes rotating signals, but a shorter window loses more by
  having less data.

### 2. Wider books (S2a, S2b)

* **They do not reduce path dependence.** The spread across decision days is 3.0 points for top-20 vs 2.4 for top-10;
  only top-30 is lower (1.9).
* **They cost return.** The score's edge is concentrated at the very top: top-20 loses 3.6 points a year (averaged over
  the 5 days) in 2013-21, and top-30 loses 6.
* **The noise comes from the score itself**, which changes ranks a lot week to week, not from the number of names.

### 3. Earnings surprise (S3a, S3b)

* **Neither beats S5.** The always-on blend (S3a) dilutes ML + V8. The fresh-surprise version (S3b) ties S5:
  * 16.4% vs 16.0% on the standard day, 14.3 vs 14.4 averaged over the 5 days;
  * Sharpe difference +0.02 (CI −0.13 to +0.18).
* **The signal is real, but S5 already captures most of it.** Its information beyond S5 (lseg_signals_v1: FM t 3.1 and
  3.9) is too small to move a 10-stock book by a measurable amount.

## What it means

* **No change to S5, and no S7 from this experiment.**
* **The S5 backtest figure should be read as ~11.4% for 2013-26, not 13.0%.** That is the average over the 5 decision
  days. It still beats TDEX's 4.8%, but in 2022-26 alone the average (6.0%) is below TDEX (6.7%).
* **If you register a forward shadow anyway, S1a has the most interesting profile**, with two caveats:
  * it is a post-hoc choice;
  * its edge came in the burned period.
  * Its profile: it adapts to rotation and had the smallest 2022-26 drawdown. Only live data could show whether that is
    real. S3b is the low-risk alternative: nearly identical to S5, with a clean point-in-time signal.

## Deviations

None.

## Trials

7 (6 strategies + ML3). `validity.TRIALS["adaptive_v1"] = 7`, so `N_TRIALS` = 795. The 5 decision-day phases
measure noise and select nothing.

## Reproduce

    cd ~/stock-markets-analytics-zoomcamp-pathorn
    for s in ml3 discover check; do uv run python set-quant-alpha/experiments/adaptive_v1/adaptive_v1.py $s; done
