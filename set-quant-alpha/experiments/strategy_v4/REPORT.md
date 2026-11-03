# strategy_v4 - widening the tradable universe from the top-100 to the top-200 (2026-11-02)

Run: `uv run set-quant-alpha/experiments/strategy_v4/v4.py {select,test}`. This reuses strategy_v3's extended
walk-forward predictions (2013-2026). Outputs are in `output/`.

## Question

The raw IC rises as liquidity falls. On validation (2020-21) it is 0.047 in liquidity ranks 1-50, 0.059 in 51-100 and
0.087 in 101-200. Does trading the wider universe earn more after realistic costs?

Note: only about **115 names a day** pass the 20M THB liquidity filter, so "top-200" means every liquid SET stock.
Names below the filter were never in training and are not considered.

## Design (pre-registered)

* **Cost model, the same for every design:**
  * 0.11% per side (commission + VAT), plus
  * square-root impact Y·σ·√(trade / ADV), with Y = 0.5.
  * At THB 20M this is about the old flat 0.15% for top-50 names. Y = 1.0 is run as a stress test.
* **Caps:** position ≤ min(5% of the book, 20% of the 60-day median daily traded value).
* **Execution:** next open, 100-share lots, THB 20M. No stop-loss (strategy_v2: the stop is not robust).
* **Designs:**
  * **R100:** top-10 equal weight with a 30-rank buffer, ranks 1-100.
  * **R200:** the same, with a 60-rank buffer, ranks 1-200.
  * **N100 / N200:** strategy_v3's TDEX + factor-neutral cost-aware optimiser on the same universes, with the
    convex impact term in the objective and λ = 1.
* **Selection:** highest 2013-2021 local-Thai Carhart α t. Frozen 2026-11-02T04:54Z, then the test was run once.
* **DSR trials:** 67 earlier + 4 here.

## Selection period 2013-2021

| | CAGR | gross | cost drag | impact/yr | turnover | CAPM α (t) | Carhart FF-EM α (t) | local-Thai α (t) | b_SMB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TDEX | 3.8% | – | – | – | – | – | +2.2 (0.42) | – | – |
| **R100 (pick)** | **12.6%** | 19.8% | 7.2% | 4.2% | 20× | +9.6 (2.16) | +3.3 (0.55) | **+6.2 (1.49)** | 0.04 |
| R200 | 7.0% | 10.4% | 3.3% | 2.3% | 9× | +4.4 (1.05) | −1.1 (−0.17) | +1.2 (0.32) | 0.13 |
| N100 | 8.1% | 10.4% | 2.3% | 1.0% | 11× | +4.6 (1.46) | +1.0 (0.17) | +2.6 (0.87) | −0.01 |
| N200 | 8.9% | 11.5% | 2.6% | 1.1% | 11× | +5.4 (1.67) | +1.5 (0.24) | +3.3 (1.07) | 0.02 |

## Test 2022-01 → 2026-10 (one shot)

| | CAGR | gross | cost drag | turnover | IR vs TDEX | CAPM α (t) | Carhart FF-EM α (t) | local-Thai α (t) | DSR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TDEX | 6.8% | – | – | – | – | – | +0.1 (0.02) | – | – |
| **R100 (pick)** | **5.5%** | 9.6% | 4.1% | 16× | −0.05 | +1.8 (0.33) | +2.0 (0.22) | +2.4 (0.45) | 0.09 |
| R200 | 5.3% | 7.8% | 2.6% | 9× | −0.09 | +1.6 (0.32) | +0.8 (0.10) | +2.2 (0.43) | 0.09 |
| N100 | 5.6% | 7.8% | 2.2% | 12× | −0.09 | −0.5 (−0.14) | −0.5 (−0.05) | −0.6 (−0.18) | 0.10 |
| N200 | 5.7% | 8.1% | 2.4% | 12× | −0.07 | −0.3 (−0.08) | −0.5 (−0.06) | −0.2 (−0.06) | 0.10 |
| R100 with impact Y = 1.0 (stress) | 3.6% | 9.6% | 6.0% | 16× | −0.18 | −0.1 (−0.02) | +0.2 (0.02) | +0.6 (0.11) | – |

TDEX shows 6.8% here, not 6.4%, because the test window ends at the last prediction date, 2026-10-22.

## Findings

1. **Widening to 200 did not help the rule-based book.**
   * The extra IC in ranks 101-200 does not reach the top-10 portfolio.
   * R200's gross return is lower in both periods: 19.8% vs 10.4% in 2013-21, 9.6% vs 7.8% on test.
   * Caveat: R200 also uses a wider buffer (60 vs 30), so universe and buffer effects are mixed.
2. **The optimiser gains only marginally** from the wider universe: local α t 0.87 → 1.07 in 2013-21, −0.18 → −0.06
   on test. Its impact and ADV caps keep most of the weight in liquid names anyway (average held liquidity rank about 45).
3. **The extra IC sits in names too thin to trade at size.** Median daily traded value is 26-42M THB in ranks 101-200.
   Square-root impact and the 20%-of-ADV cap absorb the advantage.
4. **No design earns significant factor alpha in 2022-26**, consistent with strategy_v2 and v3.
   * R100 (E2 without the stop, under this cost model) reaches 5.5% vs TDEX 6.8%, with a positive but insignificant
     local α of +2.4% (t 0.45).
   * It is cost-sensitive: at Y = 1.0 it falls to 3.6%.

## Recommendation

Keep the top-100 tradable universe. To use the stronger signal in smaller names, a strategy needs a different
execution style (longer holding periods, smaller positions, or patient limit orders), not just a wider universe.
