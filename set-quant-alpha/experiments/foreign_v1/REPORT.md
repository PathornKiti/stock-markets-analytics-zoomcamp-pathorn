# foreign_v1 - does foreign-board demand predict Thai stock alpha? (2026-11-03)

Code: `foreign_v1.py` (stages `discover`, `holdout`). Tables: `output/tables.md`. The protocol was frozen in
`output/protocol_frozen.txt` at 2026-11-02T17:06:52Z, before any signal was computed. The discovery decision was
written to `output/discovery_decision.txt` before the holdout was run, and the holdout was run once.
Data: LSEG foreign-board lines (`<SYM>_f.BK`) for 191 live stocks that were ever in SET100, 2009-26.

**Verdict: no signal passes the pre-registered gates.** In discovery (2013-21) none of the 3 foreign-board signals
predicts 5- or 20-day alpha with IC t ≥ 2.

The holdout (2022-26) then shows a different picture:
* foreign-board activity (`fshare20`) and the premium level (`prem`) predict positively, with IC t 3.3-3.6;
* `prem` has the *opposite* sign to discovery;
* the result is not consistent across periods, so it cannot be claimed. It is a hypothesis for a forward test.

## Background and signals

* **The foreign board.** Thai stocks have a foreign ownership limit. Once foreign holdings reach it, foreigners can buy
  only from other foreigners on the foreign board, often at a premium (Bailey & Jagtiani 1994).
* **New information.** Every earlier foreign-flow feature in this project was market level (baht, EEM). This is the
  first per-stock foreign-demand measure.

| Signal | Definition | Prior |
| --- | --- | --- |
| prem | foreign close / local close − 1 on foreign-board trading days, carried ≤ 20 days | two-sided |
| dprem20 | 20-day change in prem | + |
| fshare20 | 20-day foreign-board value / 20-day local value | + |

* **Coverage is thin.** On an average day only about 7 stocks trade on the foreign board, mostly banks and large caps
  (KBANK, BBL, SCC, LH, PTT).
* **Covered names per day** inside the tradable top-100:
  * 16-27 for prem and dprem20;
  * 43-63 for fshare20, which counts zeros for stocks that have a foreign line.

## Results

| Signal | h | Discovery 2013-21 IC (t) | FM t (controls: C2, size, beta, momentum) | Holdout 2022-26 IC (t) | Holdout FM t | Gates G1 / G2 / G3 |
| --- | --- | --- | --- | --- | --- | --- |
| prem | 5 | −0.003 (−0.3) | −1.3 | +0.017 (1.8) | 1.0 | no / no / no |
| prem | 20 | −0.015 (−1.1) | −1.6 | **+0.048 (3.5)** | 2.7 | no / no / no |
| dprem20 | 5 | +0.011 (1.3) | **2.8** | −0.006 (−0.7) | 0.6 | no / yes / no |
| dprem20 | 20 | +0.016 (1.2) | 1.3 | +0.014 (1.1) | 0.7 | no / no / no |
| fshare20 | 5 | +0.011 (1.8) | 0.6 | **+0.031 (3.6)** | 1.2 | no / no / yes |
| fshare20 | 20 | +0.005 (0.4) | 0.2 | **+0.058 (3.3)** | 2.0 | no / no / yes |

* **Discovery fails.**
  * dprem20 at 5 days is the closest: its marginal slope over S5's score is significant (t 2.8), but the raw IC t is
    only 1.3.
  * No signal passes G1, so per the protocol nothing can pass, whatever the holdout shows.
* **The holdout disagrees with discovery.**
  * fshare20 predicts positively in 4 of 5 holdout years (20-day IC 0.09-0.11 in 2022, 2025 and 2026), and its
    top-minus-bottom quintile earns +1.8% per 20 days (t 3.2).
  * In discovery the same quintile spread was −2.1% (t −5.3). That number is distorted by ties, because most covered
    stocks have zero foreign-board value. Its sign still shows the effect was not stable.
  * prem's 20-day IC flips from −0.015 to +0.048.
* **Interpretation.** Foreign-board demand may have become informative after 2022, as foreign flows dominated Thai
  equity moves. Equally, a 5-year window with 20-60 names a day can produce t ≈ 3.5 for a time-varying effect by
  chance (6 tests here). Only data no one has seen yet can tell these apart.

## What to do with it

* **Not a trading change.** Nothing passes, so S5 is unchanged and no shadow is proposed from this experiment.
* **For the thesis.** The foreign-board signal is a clean, pre-registered negative-then-positive finding. A new,
  separate pre-registration could test the hypothesis "fshare20 (20-day) predicts alpha after 2022" on **future data
  only**, for example a paper shadow S7 = C2 + fshare20 from its registration date. It would also need per-stock NVDR
  flows (SET website templates in `docs/templates/`), which cover all stocks, not only those at the foreign limit.

## Deviations

* **D1.** The LSEG nullable dtypes crashed the first `discover` run before any statistic was computed. They are now
  cast to float64, and nothing in the method changed.

## Trials

6 (3 signals × 2 horizons). `validity.TRIALS["foreign_v1"] = 6`, so `N_TRIALS` = 741.

## Reproduce

    cd ~/stock-markets-analytics-zoomcamp-pathorn
    uv run python set-quant-alpha/scripts/fetch_lseg.py members foreign prices     # LSEG Workspace + key needed
    uv run python set-quant-alpha/experiments/foreign_v1/foreign_v1.py discover
    uv run python set-quant-alpha/experiments/foreign_v1/foreign_v1.py holdout      # refuses to run twice
