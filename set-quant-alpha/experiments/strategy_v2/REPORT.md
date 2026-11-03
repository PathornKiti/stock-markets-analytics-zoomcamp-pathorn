# strategy_v2 - can a refined strategy earn factor alpha? (2026-11-02)

**Answer: no.** None of 17 pre-registered designs earns significant alpha against CAPM (TDEX), Fama-French EM 3-factor,
Carhart FF-EM or local-Thai Carhart in the 2022-01 → 2026-10 test. The design frozen on validation (D05) failed on
test. The useful results are diagnostic: what makes E2 better than E1, and why validation rankings did not carry over.

Run: `uv run set-quant-alpha/experiments/strategy_v2/run.py {valid,test,secondary,tables} [--preds <predictions.parquet>]`.
Tables: `output/tables.md`, `output/diagnostics.md`, `output/secondary_v2.md`; raw: `output/*.csv`.

## Protocol

* The simulator reproduces `strategy.exact_sim` for E1 and E2 to within 1e-9 THB (`output/replication_check.csv`).
* Designs and the selection rule were written in `run.py` before any simulation.
* Selection rule: the highest validation (2020-21) local-Thai Carhart alpha t (MKT/SMB/WML, Newey-West), among designs
  with one-way turnover ≤ 1.1 × E2.
* Pick frozen 2026-11-02T03:43:19Z (`output/picks_frozen_before_test.txt`), then the test was run once. A first freeze
  had two bugs, both fixed before any test run; the pick did not change.
* DSR trial count: 60 (37 cost_aware + 6 pipeline + 17 here).
* Caveat: the design *ideas* (size neutrality, TDEX core) were motivated by the known 2022-26 megacap regime. Selection
  used validation data only.

## E1 vs E2: what does the work

Buffer × stop ablation, effects averaged over both orders, % per year:

| | valid E2−E1 | buffer | stop | **test E2−E1** | **buffer** | **stop** |
| --- | --- | --- | --- | --- | --- | --- |
| net CAGR | +3.9 | +3.0 | +0.9 | **+2.0** | **+4.8** | **−2.9** |
| cost drag | −5.0 | −5.7 | +0.8 | −5.1 | −5.4 | +0.3 |
| local alpha | +6.4 | +3.0 | +3.4 | +2.0 | +4.5 | −2.5 |
| turnover (2-way, ×/yr) | −28.6 | −32.7 | +4.1 | −31.7 | −35.4 | +3.7 |

* **The 30-rank buffer is the active ingredient.** It cuts turnover from about 59× to 23-27× a year at almost no cost in
  gross return, in both periods.
* **The 8% stop is not robust.** It helped in 2020-21 by de-risking the book (beta 0.87 → 0.60), and hurt in 2022-26.
  It fights the reversal signal: 37-50% of stop-outs are re-bought within 10 days (E2 test: 59 of 160), so the spread is
  paid twice.

## Frozen pick (D05: 50% E2 + 50% residual-momentum sleeve)

| | CAGR | Sharpe | local-Thai α (t) | CAPM α (t) | DSR |
| --- | --- | --- | --- | --- | --- |
| validation 2020-21 | 45.9% | 1.58 | +23.3% (2.37) | +39.2% (2.77) | 0.76 |
| **test 2022-26** | **−6.5%** | −0.29 | −10.0% (−1.78) | −11.2% (−1.87) | 0.00 |

Why it failed (post-hoc): the residual-momentum sleeve alone made +73.6% a year in validation and −14.1% a year in test.
Local 12-1 momentum does not span residual momentum, so the validation "alpha" was an unpriced style bet.

## Test results (2022-01 → 2026-10, t-stats in brackets)

| design | CAGR | Sharpe | max DD | 2-way turnover | CAPM α (t) | FF3-EM α (t) | Carhart FF-EM α (t) | local-Thai α (t) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TDEX | 6.45 | 0.53 | −26.6 | 0 | – | −3.0 (−0.38) | +0.1 (0.02) | – |
| E1 | −0.39 | 0.07 | −45.7 | 58.6 | −4.0 (−0.64) | −6.4 (−0.75) | −3.1 (−0.34) | −3.0 (−0.49) |
| E2 | 1.61 | 0.18 | −36.3 | 26.9 | −1.5 (−0.29) | −4.8 (−0.58) | −1.7 (−0.19) | −1.1 (−0.20) |
| **D05 (pick)** | −6.48 | −0.29 | −54.3 | 16.4 | −11.2 (−1.87) | −12.2 (−1.33) | −12.1 (−1.18) | −10.0 (−1.78) |
| P-CS (20% pick + 80% TDEX) | 4.17 | 0.38 | −27.3 | 3.0 | −1.9 (−1.63) | −4.3 (−0.56) | −1.8 (−0.22) | −1.6 (−1.47) |
| *info: AB2 = E1 + buffer, no stop* | 4.69 | 0.37 | −31.0 | 23.2 | +1.1 (0.21) | −1.3 (−0.15) | +1.9 (0.23) | +1.7 (0.32) |
| *info: D10 = vol-scaled stop + cooldown* | 5.00 | 0.39 | −31.0 | 26.7 | +1.5 (0.28) | −1.8 (−0.22) | +1.2 (0.13) | +2.2 (0.41) |
| *info: D08 = 70% TDEX + 30% top-30 liquid* | 7.36 | 0.62 | −24.2 | 8.7 | +1.4 (0.85) | −1.2 (−0.18) | +2.1 (0.31) | +1.4 (0.81) |

*info* rows are test-period observations, not selections. All DSRs ≤ 0.18.

**Validation rank did not predict test rank** across the 17 designs: Spearman ρ is −0.60 for Sharpe (p 0.01) and −0.43
for local-alpha t. Only turnover carries over (ρ 0.91). With these data, design choices beyond "trade less" are not
identifiable from two validation years.

Cost sensitivity, test CAGR at 0 / 0.10 / 0.15 / 0.20 / 0.25% per side:

| design | 0% | 0.10% | 0.15% | 0.20% | 0.25% |
| --- | --- | --- | --- | --- | --- |
| E1 | 8.98 | 2.82 | −0.39 | −3.17 | −5.97 |
| E2 | 5.85 | 3.07 | 1.61 | 0.31 | −1.32 |
| P-CS | 4.78 | 4.44 | 4.17 | 4.02 | 3.87 |

## Secondary run on the model_v2 score (V3; frozen designs only, no re-selection)

| test | CAGR | CAPM α (t) | local-Thai α (t) |
| --- | --- | --- | --- |
| E1 | 0.42 | −3.3 (−0.48) | −0.7 (−0.11) |
| E2 | −2.10 | −5.7 (−0.89) | −3.9 (−0.62) |
| D05 | −9.23 | −14.4 (−2.27) | −12.5 (−2.07) |
| P-CS | 3.77 | −2.3 (−1.87) | −1.9 (−1.61) |

V3 looked much stronger on validation (E2 CAGR 34.9%, local alpha t 2.67) but is worse than production on test.

## Recommendation

1. Keep E2 in production. Do not ship D05, P-CS or V3. Switching to a design that looked good on test (AB2, D03, D08,
   D10) would be selecting on test data.
2. Document that the buffer is the active ingredient and the stop is a regime-dependent de-risking overlay. If the
   stop stays, the D01 re-entry cooldown (about 10 lines) is cost-neutral and removes stop-then-re-buy round trips.
3. Paper-trade from 2026-11 as hypotheses: E2 without the stop (AB2), and D08 (70% TDEX + large-cap overlay, size
   loading ≈ 0).
4. Lower-turnover scores (V8, EMA) cut turnover by 25-50% but did not improve test alpha. The binding constraint is the
   regime-dependent size/momentum style, not costs alone.
