# fundamentals_v1 - do analyst-consensus and fundamental data improve the ML model and S5? (2026-11-03)

Code: `fundamentals_v1.py` (stages `features`, `scores`, `discover`, `holdout`). Tables: `output/tables.md`.

* **Protocol.** Frozen in `output/protocol_frozen.txt` at 2026-11-02T17:34:52Z, before any fundamental value was seen.
* **Decision.** Written to `output/discovery_decision.txt` (17:40:45Z) before any 2022-26 number was computed; the
  holdout was computed once.
* **Universe.** Survivorship-corrected L (survivorship_v1), delisted stocks included.
* **Data.** LSEG monthly as-of history for the 263 SET100-ever stocks: I/B/E/S FY1 EPS consensus, number of
  estimates, recommendation mean, P/B, P/E, ROE and market cap (`scripts/fetch_lseg.py fundamentals`).

**Verdict: no pass, and no S7 proposed.**

* **ML model.** Fundamentals do not improve it. The daily IC of ML with fundamentals is +0.0015 above ML in 2013-21
  (t 0.86) and 0.0000 in 2022-26, so the information gate G1 fails.
* **Strategy.** The pre-registered pick D (ML_F score, rebalanced every 10 days) passed both performance gates.
  * 2013-21: 17.3% net vs A's 16.0%.
  * 2022-26: 10.6% vs 7.7%, TDEX 6.7%.
  * But the holdout Sharpe difference vs A is 0.16 with a 95% CI of −0.34 to +0.64.
  * Rebalancing every 10 days without fundamentals made S5 *worse* in both periods (see "Why D is noise").
  * So D's lead is noise from a 10-stock book, not a repeatable improvement.
* **The new information is real but rotates.**
  * Earnings revisions and recommendation upgrades predicted alpha in 2013-21 and faded in 2022-26.
  * Value (book-to-price, earnings-to-price) did the opposite.
  * Only the equal-weight blend of all six (FC) predicted in both periods, with IC t 4.1 and 4.3.

## Point-in-time handling (from the protocol)

| Field | Use |
| --- | --- |
| I/B/E/S FY1 EPS mean, number of estimates, recommendation mean | as-of history; used from the next trading day |
| P/B, P/E | not trusted as point-in-time: book / earnings per share derived from the ratio 63 trading days earlier, divided by today's close |
| ROE | LSEG dates it by **fiscal period end**: asking in Q1 2014 returns full-year 2014 ROE. Used only from period end + 90 days |
| Market cap | month-end value, used from the next trading day |

FY1 revisions are set to missing when the FY1 fiscal period changes inside the window, or when fewer than 2 analysts
cover the stock.

## Stage 1 - signals (tradable top-100; IC vs forward alpha, Newey-West t; FM = Fama-MacBeth slope controlling for C2, size, beta and momentum)

| Signal | Discovery 2013-21: IC 5d (t) / FM t | Discovery IC 20d (t) | Holdout 2022-26: IC 5d (t) / FM t | Holdout IC 20d (t) |
| --- | --- | --- | --- | --- |
| rev1: 1-month FY1 revision | **0.027 (4.1)** / 2.4 | 0.044 (3.3) | 0.014 (1.7) / 0.6 | 0.012 (0.7) |
| rev3: 3-month FY1 revision | **0.032 (3.8)** / 1.8 | 0.045 (2.5) | 0.015 (1.4) / 0.7 | 0.034 (1.9) |
| recup3: rating moved toward buy | 0.015 (3.0) / **2.8** | 0.023 (2.3) | −0.000 (−0.0) / 0.3 | −0.016 (−1.1) |
| bp: book-to-price | −0.012 (−1.7) / 0.2 | −0.020 (−1.4) | **0.027 (3.0) / 3.0** | 0.048 (2.9) |
| ep: earnings-to-price | −0.001 (−0.2) / 0.2 | −0.003 (−0.2) | **0.033 (3.7) / 3.3** | 0.063 (3.6) |
| roe | 0.014 (2.8) / 1.0 | 0.014 (1.3) | 0.002 (0.2) / −0.3 | −0.005 (−0.3) |
| **FC: equal blend of the six** | **0.021 (4.1)** / 2.0 | 0.030 (2.9) | **0.035 (4.3) / 3.2** | 0.052 (3.5) |

* **Analyst signals carried information in 2013-21.** Revisions are positive at both horizons, and upgrades add
  information beyond S5's own score (FM t 2.8). Both faded after 2022.
* **Value was dead in 2013-21 and strong in 2022-26.** This matches the 2022-26 Thai market, where cheap, high-yield
  stocks held up (alpha_portfolio_v1 found the same for dividend yield).
* **Only the diversified blend FC is stable,** because its parts take turns. It is the most robust new piece of
  evidence in this experiment.

## Stage 2 - ML with fundamental features (ML_F)

| | 2013-21 | 2022-26 |
| --- | --- | --- |
| IC of ML (pred_ens) | 0.0811 | 0.0551 |
| IC of ML_F | 0.0826 | 0.0551 |
| Difference (paired NW t) | +0.0015 (0.86) | 0.0000 (−0.02) |
| IC of C2 (S5 score) → C2F | 0.0729 → 0.0743 | 0.0637 → 0.0640 |

* **The model uses the wrong parts of the data.** The only fundamental features in XGBoost's 2026 top 25 are
  coverage and size: has_ibes, log market cap and log number of estimates. The revision and value signals are not
  there.
* **Why.** A model refit each January on all past data averages over the regimes in which these signals take turns,
  so it learns almost nothing from them.

## Stage 3 - strategies on L (top-10, buffer 30, EW, net of 0.168% per side)

| Strategy | 2013-21 CAGR / Sharpe / CAPM α t | 2022-26 CAGR / Sharpe / CAPM α t / max DD | 2013-26 CAGR / Sharpe |
| --- | --- | --- | --- |
| A: C2 = ML + V8 (S5 baseline) | 16.0% / 0.91 / 3.06 | 7.7% / 0.54 / 0.36 / −27.6% | 13.0% / 0.79 |
| B: ML_F + V8 | 16.7% / 0.93 / 3.25 | 6.7% / 0.49 / 0.40 / −24.6% | 13.1% / 0.79 |
| C: ML + V8 + FC | 10.3% / 0.65 / 2.09 | 9.7% / **0.72** / **1.07** / **−17.8%** | 10.1% / 0.67 |
| **D: B, rebalanced every 10 days (pick)** | 17.3% / 0.99 / **3.56** | **10.6%** / 0.70 / 0.78 / −27.6% | **14.9%** / **0.89** |
| TDEX | 3.9% / 0.31 | 6.7% / 0.54 | 4.8% / 0.38 |

| Gate | Result |
| --- | --- |
| G1 information (ML_F IC > ML, t ≥ 2) | **fail**: +0.0015, t 0.86 |
| G2 pick beats A in 2013-21 | pass: 17.3% vs 16.0%, Sharpe 0.99 vs 0.91 |
| G3 holdout: Sharpe ≥ A and CAGR > TDEX | pass: Sharpe 0.70 vs 0.54; 10.6% vs 6.7% |
| Pass all | **no** |

### Why D is noise

This check reused existing integrated_v1 configs (Yahoo universe), so it adds no trials.

| S5 score (C2), top-10, buffer 30, EW | 2013-21 CAGR / Sharpe | 2022-26 CAGR / Sharpe |
| --- | --- | --- |
| Rebalance every 5 days | 19.2% / 1.04 | 7.9% / 0.57 |
| every 10 days | 13.0% / 0.76 | 6.5% / 0.48 |
| every 21 days | 13.2% / 0.76 | 9.7% / 0.68 |

* **Slowing S5 down does not help consistently.** Results move by 3-6 points a year between rebalance schedules
  without a pattern.
* **The cause is path dependence.** A 10-stock book makes the timing of the decision days matter as much as the
  signal.
* **So D's +2.9 points over A in 2022-26 is luck.** It is the same size as this noise, and the Sharpe CI includes
  zero.

## What this means for "beating the benchmark"

* **The baseline already beats TDEX in every period here.** S5 (A) earned 13.0% against 4.8% over 2013-26, with CAPM
  α t 2.65, even after the survivorship correction.
* **What is missing is proof, not performance.**
  * A 10-stock book is noisy enough that ±3 points a year between variants is indistinguishable.
  * After 757 trials no backtest variant can clear the multiple-testing hurdle.
  * More strategy search on 2013-26 cannot fix that. Only out-of-sample (live) data can.
* **The most consistent new evidence across all of tonight's experiments is diversified, slow information:**
  * FC here (IC t 4.1 and 4.3);
  * the slow composites in alpha_portfolio_v1 (twice C2's net spread);
  * possibly foreign-board activity after 2022 (foreign_v1).

  Each is a candidate for a *forward-only* paper shadow, registered before any further data is seen.

## Deviations

* **D1.** One bug was fixed before any statistic was computed: the period-end conversion crashed on missing dates.
* **D2.** "Why D is noise" was not in the protocol. It reads existing integrated_v1 grid results (no new
  configurations, no new trials).

## Trials

16 (12 signal tests + ML_F + strategies B, C, D). `validity.TRIALS["fundamentals_v1"] = 16`, so `N_TRIALS` = 757.

## Reproduce

    cd ~/stock-markets-analytics-zoomcamp-pathorn
    uv run --group lseg python set-quant-alpha/scripts/fetch_lseg.py fundamentals   # LSEG Workspace + key; ~3 min
    for s in features scores discover holdout; do
      uv run python set-quant-alpha/experiments/fundamentals_v1/fundamentals_v1.py $s; done   # holdout runs once
