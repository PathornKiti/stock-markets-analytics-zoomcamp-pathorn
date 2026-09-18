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
| 5 | Increasing IPO strategy profitability | *free text — see below* | — |

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
eleven months — hold the median 2025 IPO for a year and you lose half your money. The
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

### Q5 — Increasing profitability

Full reasoning in the notebook. Short version:

**CAPM and APT cannot be applied to an IPO at entry.** Both price expected return off
factor loadings, and loadings are estimated from return history — a company that listed
this morning has none. APT is the better framework in principle (multiple factors, no
efficient-market-portfolio assumption) but has the same problem plus one more: it does not
tell you what the factors are.

The workable substitute is a **cross-sectional characteristic model** — price a new IPO
from observables available at listing rather than from its own (nonexistent) history.
Testing that on this cohort, **deal size is the strongest single filter**:

| Screen | n | Median 12m | Win rate | sd |
| --- | --- | --- | --- | --- |
| All IPOs | 129 | 0.475 | 27.1% | 2.15 |
| Deal > $25M | 51 | 0.870 | 43.1% | 0.92 |
| Deal > $100M | 41 | 0.842 | 41.5% | **0.56** |

Screening out sub-$25M deals lifts the median from 0.48 to 0.87, the win rate from 27% to
43%, and cuts cross-sectional dispersion by four-fifths — which is a Sharpe improvement
via the denominator. Industry adds a weaker second cut (Financials 0.82, Health Care 0.68
vs Consumer Services 0.12); first-day pop is worthless (Spearman 0.07).

**But no screen flipped the sign.** Even the best filter leaves the median below 1.0 and
the equal-weight mean at −2% to −4% at every horizon. Changes I would make, in order:
fix the entry filter (drop micro-caps) before touching portfolio construction; replace the
"buy at first close" entry (buy at offer, or wait out the ~180-day lock-up washout); build
the characteristic model as module-3 work; only then apply mean-variance sizing — and with
Ledoit-Wolf shrinkage, since a covariance matrix over 40 listings with <1y of history is
badly conditioned, and equal-weighting beats a fitted optimiser at this sample size.

The honest conclusion: across every screen, horizon and entry rule tried, the 2025 IPO
cohort's median outcome stayed negative. The Q4 mean-reversion rule has positive
expectancy and 5,206 observations behind it — a far better base for a Sharpe-optimised
portfolio than a 129-name cohort whose base rate is a loss.
