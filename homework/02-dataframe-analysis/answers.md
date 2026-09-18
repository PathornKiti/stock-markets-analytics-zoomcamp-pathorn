# Module 02 homework — answers

Submitted at: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw02
Date submitted:

Data as of **2026-09-11** (notebook run 2026-09-18). Working: [`notebook.ipynb`](notebook.ipynb).

| Q | Question | Computed | Answer |
| --- | --- | --- | --- |
| 1 | Total withdrawn IPO value of the largest company class | Acquisition Corp, $499.99M | **500** |
| 2 | Median Sharpe ratio on 2026-09-11, IPOs before 2025-09-01 | 0.0501 | **0.04** |
| 3 | Holding period maximising median growth | 1 month, median 0.9354 | **1** |
| 4 | Net income from the RSI < 30 strategy ($ thousands) | $65,805.59 | **65** |
| 5 | Increasing IPO strategy profitability | trade `RSI<30 & natr>3 & slowk<20`: +62.7% profit at equal capital, 78.1% vs 27.8% alpha | *free text* |

## Working notes

### Live-data drift

Both iposcoop tables are scraped live and change daily, so raw row counts no longer match
the ones quoted in the task. The notebook was run a week after the homework's anchor date:

| Checkpoint | Task | This run | Why |
| --- | --- | --- | --- |
| Withdrawn IPOs | 32 | 34 → **32** | Two withdrawals filed after 2026-09-11; fixed by filtering `File Date < 2026-09-11` |
| 2025 IPOs after filtering | 148 | 146 | Two more names went stale (`Return` froze at 0.00%) in the intervening week |
| Tickers with yfinance data | ~134 | 132 | Follows from the 146 above |

None of it moves an answer.

### Q1 — Withdrawn IPOs by company type

32 withdrawn deals filed before 2026-09-11. Classification rules applied **in order**,
first match wins — that ordering is load-bearing: `EUPEC International Group Ltd.` is a
`Group`, not a `Limited`, and matching is exact, so `Xinxu Copper Industry Technology Ltd.`
falls through `Technologies` to `Limited`.

| Company Type | Total $M | Deals |
| --- | --- | --- |
| **Acquisition Corp** | **499.99** | 5 |
| Inc. | 351.00 | 1 |
| Holdings | 311.66 | 4 |
| Other | 290.44 | 8 |
| Limited | 203.85 | 8 |
| Technologies | 184.90 | 3 |
| Group | 32.50 | 3 |

`Avg_price` is the midpoint of the parsed low/high range. Six rows have zero shares *and*
no price range, so `Shares x Avg_price` is NaN and `Est $ Vol (millions)` takes over.

Acquisition Corp wins on **count × uniform ticket size** — SPACs all price at exactly
$10.00 — not on one large deal, unlike `Inc.` which is a single company (Clear Street
Group, $351M). The bare `Corp` pattern also sweeps in `Helio Corp.`, a $15M uplisting unit
deal that is not a SPAC; excluding it leaves $485M, still the winner.

### Q2 — Median Sharpe ratio

132 tickers downloaded, 130 reached the 252-day milestone. **Median Sharpe 0.0501** →
nearest option **0.04**.

- `growth_252d`: **median 0.59 vs mean 1.06**. The typical 2025 IPO is worth 41% less than
  a year ago; the mean is dragged to break-even by one 33x survivor. Mean is unusable here.
- Three tickers (EFTY, MAMK, MAGH) have a **flat price for 30 straight sessions**, so
  `volatility` is 0 and Sharpe is `inf`. Mean Sharpe is literally `inf`; excluding them it
  is 0.186. The median is immune.
- **Caveat on the prescribed formula:** `Close.rolling(30).std()` is the stdev of the
  *price in dollars*, not of returns, so `volatility` scales with the share price and the
  result is not a textbook Sharpe ratio. Reproduced as specified, but the level is not
  comparable to a return-based Sharpe.
- Best *finite* Sharpes are SPAC units (HCMAU, CEPF) grinding $10.00 → $10.47 — risk
  adjustment rewards the boring names, not the high-growth ones.

### Q3 — Fixed-months holding strategy

21 trading days per month, measured from each ticker's first close.
**Best: 1 month, median growth 0.9354** → option **1**.

| Months | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Median | **0.935** | 0.893 | 0.827 | 0.730 | 0.691 | 0.726 | 0.662 | 0.604 | 0.580 | 0.533 | 0.482 | 0.492 |

The "optimal" holding period is the **shortest one offered, and it still loses 6.5%**.
Every horizon is below break-even and the median decays almost monotonically to 0.48 at
eleven months — holding the median 2025 IPO for a year loses half the capital. The
investor conclusion is not "sell after a month", it is that buying the median IPO at the
first close loses at every horizon and tuning the exit only loses less.

**Two data problems found and fixed:** the `mean` column peaks at a nonsensical 95x,
caused by PPCB (Propanc Biopharma) — a $0.02 stock whose reverse split is unadjusted in
this history, giving a 12,500x "growth". It also exposed that `min_date` is *not* the IPO
date for the five uplistings (PPCB, AVBH, ALM, CIIT, CAPS), whose history starts at the
download window in January 2025, up to 225 days before their offer date. Re-anchoring
entry on the first session on/after the true offer date leaves the answer unchanged
(month 1, median 0.924) and drops the mean from 95x to a believable 1.06x.

### Q4 — RSI < 30 strategy

5,206 trades between 2000-01-01 and 2025-06-01, $1,000 each.
**Net income $65,805.59 → option 65.** All three of the task's sanity checks reproduce
exactly (5,206 trades, 1.2640% average 30-day return, 55.13% win rate).

Caveats the headline hides: $66k of profit came from **$5.2M of deployed capital** — the
edge is 1.26% per ticket, not 66x anything. The rule also assumes unlimited free capital
with unbounded concurrent positions, and ignores costs; at 1.26% gross, a 20bps round trip
on an illiquid oversold name eats ~16% of the edge.

**`growth_future_30d` is mislabelled — it is a 5-trading-day forward return, not 30-day.**
Verified two ways: it matches `Close_x.shift(-5) / Close_x` to ten decimal places across all
229,767 rows (a 21- or 30-row shift is off by >3.7), and its dispersion matches a 5-day
horizon independently (AAPL log-sd 0.0627 vs 0.1582 for a true 30-day return; medians
1.0051 vs 1.0302). The sibling `growth_30d` *is* genuinely 30 rows, which is what makes the
name misleading.

This does not change the answer — the arithmetic is unchanged and all three sanity checks
still reproduce — but it changes the interpretation substantially: the 1.264% is earned per
**week**, so capital turns over ~50x a year, not ~8x, and the implied annualised rate is
**~88%**, not ~11%. It makes Q4 a much higher bar than its headline suggests.

### Q5 — Increasing profitability

**Short answer: stop buying IPOs, and add a volatility filter to the Q4 signal.**
`RSI < 30 AND natr > 3 AND slowk < 20` earns **+62.7% more profit on the same capital**
with **78.1% annualised alpha against Q4's 27.8%**. Full working in the notebook.

**Why not fix the IPO strategy.** Q3 settled it: the best of twelve fixed holding periods
still has a median growth of **0.9354**, so every horizon loses for the typical name. Q4's
mean-reversion rule has positive expectancy over 5,206 trades and 25 years — that is the
thing worth improving.

#### The two metrics, fixed before searching

- **Profit at equal capital.** Q4's metric (total dollars at $1,000/signal) is
  `n × mean return`, so it rewards trade *count* as much as edge. Everything is scored on
  Q4's own budget instead: $1,000 × 5,206 = **$5.206M**.
- **Alpha.** Intercept of `r ~ benchmark`, benchmark = equal-weight forward 5-day return of
  the same region on the same date, errors clustered by date, annualised at 50.4/year.

Calibration: always-invested gives beta 1.0000 and a machine-zero intercept. The bar is
Q4's **$65,806 and 27.8% alpha (t = 6.24)**.

#### The search, with a real holdout

5 base triggers × 70 confirming conditions (oscillators, trend, volatility, macro regime,
candlestick reversals, region) = **264 rules** with ≥300 trades. Selection ran on
**2000–2014 only**; 2015–2025 never influenced the choice.

- 46 rules beat Q4 in-sample on both metrics.
- **13 survive the holdout.** The top one is not another oscillator — it is **`natr > 3`**:
  take the oversold signal only when average true range exceeds 3% of price. Mean reversion
  pays in proportion to how far prices travel; an oversold reading on a quiet stock has
  nothing to revert.
- The holdout kills the rest. The three best in-sample rules — `rsi<25 & oil30d-down`
  (3.62%/trade), `rsi<25 & mfi<20` (3.45%), `rsi<25 & reg=INDIA` (3.40%) — fall to
  **+0.19%, −0.93%, −0.90%** after 2015. Fitted to 2008–09, like `rsi<20`.

#### Result

| | Q4 `rsi<30` | **`rsi<30 & natr>3 & slowk<20`** |
| --- | --- | --- |
| trades | 5,206 | 1,622 |
| return per trade | 1.264% | **2.058%** |
| **profit at equal capital** | $65,806 | **$107,052 (+62.7%)** |
| **annualised alpha** | 27.8% | **78.1%** |
| alpha t (clustered) | 6.24 | **8.04** |
| annualised Sharpe | 0.95 | **1.48** |
| win rate | 55.1% | **60.4%** |
| CVaR₂₀ | **−0.083** | −0.108 |

Out of sample (never fitted): **2.36%/trade vs Q4's 0.82%**, Sharpe 1.83, alpha 68.5%.

Robustness, all agreeing:

- **Not a knife-edge** — raising the `natr` cut 2 → 5 improves the OOS edge monotonically
  (1.24% → 5.56%) as trade count falls. `natr > 3` is the capacity/edge compromise.
- **Wins all five 5-year blocks and all three regions.** In 2020–2026 **Q4 decays to
  0.36%/trade and 3.6% alpha while the rule holds 2.11% and 79%** — the plain RSI edge is
  being competed away; the volatility-filtered one is not.
- **Survives costs** — at a 50bp round trip, $81,022 vs Q4's $39,776.

#### Two honest deductions

1. **Worse tail.** CVaR₂₀ −8.3% → −10.8%: a volatility filter buys volatile names, so
   losers lose more. Sharpe and win rate still improve, so the trade is worth making — but
   position-size down rather than keeping the flat $1,000 ticket.
2. **Not alpha in a factor sense.** Against market + momentum the intercept is +86.6%
   (t = 8.72); adding short-term reversal turns it negative (−51%, t = −2.88). The rule is a
   **concentrated dose of the reversal premium**, not a new return source. (Partly circular:
   `rev_1m` nearly *is* the signal, and a linear fit extrapolates badly into the tail the
   rule lives in. The trades really did earn 2.06% each.)

#### Recommendation and limits

1. Trade **`RSI < 30 AND natr > 3 AND slowk < 20`**, position-sized to hold risk per trade
   constant.
2. Never rank rules by total dollars — hold capital constant, or a rule firing on 46% of
   bar-days (`fastk<50`: $488k, 7.4× Q4) wins with 3.6% alpha and −55.6% on equal capital.
3. Limits: ~65 trades/year, so capacity is well below Q4's; the stacking step reused the
   holdout once; and all of it lives on 33 tickers — re-test outside them before sizing up.
