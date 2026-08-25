# Module 01 homework — answers

Submitted at: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw01
Date submitted:

Data as of **2026-08-21**. Working: [`notebook.ipynb`](notebook.ipynb).

| Q | Question | Answer |
| --- | --- | --- |
| 1 | Year with the highest number of S&P 500 additions (from 2020) | **2025** |
| 2 | Indexes (of 10) with better YTD returns than the S&P 500 | **2** |
| 3 | Median drawdown of corrections ≥5% | **7.99%** |
| 4 | Median 2-day return after a positive earnings surprise (AMZN) | **0.35%** |
| 5 | Capstone brainstorm | *free text — see notes.md* |
| 6 | New metrics | *free text — see notes.md* |

## Working notes

### Q1 — S&P 500 additions

Scraped `List_of_S&P_500_companies` from Wikipedia with a browser `User-Agent`
(the default pandas agent is blocked). 503 companies, all `Date added` values
parsed cleanly.

Additions per year: 2020: 10, 2021: 10, 2022: 15, 2023: 15, 2024: 16,
**2025: 18**, 2026: 13 (partial).

2026 is excluded — it is not yet a full year, so it isn't comparable.

**Caveat:** the table lists only *current* constituents, so this counts
additions that survived to today. Companies added and later removed are
invisible, which biases recent years upward. The 2020–2025 ranking is still
safe, but these are not true per-year addition counts.

*Additional:* **224 of 503** companies (44.5%) have been in the index more than
20 years, i.e. added on or before 2006-08-21.

### Q2 — World indexes YTD

Window 2026-01-01 → 2026-08-21 (`end='2026-08-22'`, since yfinance's `end` is
exclusive). Each index uses its own first/last available close, because market
holiday calendars differ by country.

S&P 500 YTD: **+11.90%**. Beating it:

| Index | YTD |
| --- | --- |
| Japan — Nikkei 225 | +27.36% |
| Canada — S&P/TSX | +14.86% |
| *US — S&P 500* | *+11.90%* |

Laggards: India −7.25%, China −2.94%, Hong Kong −1.25%.

*Additional (3/5/10 years):* **2 / 2 / 1** indexes beat the S&P 500. Only Japan
wins on every horizon (+299% over 10y vs the S&P's +251%); Canada beats it at 3
and 5 years but not 10. The "diversify away from the US" case is **not**
supported by price returns over the last decade — US outperformance has been
persistent rather than a YTD artifact.

Local currency, price-only (no FX, no dividends) — the latter penalises
high-yield markets like the FTSE 100 most.

### Q3 — Market corrections

19,281 daily closes, 1950-01-03 → 2026-08-21. 1,537 all-time-high days, 681
dips between consecutive highs, of which **74** were ≥5%.

Duration is measured **peak → trough**, which is what the prompt's reference
table uses (`2007-10-09 to 2009-03-09 ... 517 days` — the second date is the
bottom, not the recovery; the index didn't regain its 2007 peak until 2013).

| Percentile | Drawdown | Duration |
| --- | --- | --- |
| 25th | 6.23% | 22 days |
| **50th (median)** | **7.99%** | **40.5 days** |
| 75th | 14.02% | 86 days |

All 10 rows of the prompt's reference list reproduce exactly (drawdowns *and*
durations), confirming the methodology.

Distribution is strongly right-skewed — the typical correction is a shallow ~8%
dip under six weeks, but the tail holds 2007–09 (56.8%) and the dot-com bust
(49.1%). Median is the right summary; the mean would be dragged by those.

### Q4 — AMZN earnings surprises

25 earnings entries from 2020-10-29 (matching the prompt), 24 with a reported
surprise — the most recent is a scheduled future date. 20 positive, 4 negative.

2-day return anchored on the announcement day: `Close[t+1] / Close[t-1] - 1`.
Amazon reports after the close, so the announcement-day close is pre-reaction
and the move lands on Day 3 — which this window captures. `auto_adjust=True`
handles the June 2022 20:1 split.

**Median after a positive surprise: +0.35%** (mean +0.46%).

Context that the median hides:

- The all-history baseline for *any* random 2-day window is **+0.16%**, so
  +0.35% is barely above chance.
- Only **55%** of positive surprises produced a positive move.
- Range: −10.59% (Oct 2022) to +19.82% (Jul 2026).
- Worst counter-example: 2026-02-05, a +0.22% surprise (effectively in-line)
  drew **−9.73%**.

**Correlation:** +0.2217 (Pearson, all 24 events).

But this is **not robust** — it collapses to **−0.04** once the two >200%
surprises (Feb 2022: 641.9%; Jul 2026: 215.0%) are excluded. Rank-based
Spearman holds at **+0.28**. So there is a weak *monotonic* tendency for bigger
beats to be rewarded, but no reliable linear relationship.

Reading: beating consensus isn't tradeable on its own. The market prices in an
expected beat (Amazon beat in 20 of 24 quarters); what moves the stock is the
size of the beat relative to that expectation, plus guidance — which isn't in
this dataset. n=24 is also far too small for confidence.

*Bull vs bear regimes* would need a regime label (e.g. S&P 500 above/below its
200-day MA); with ~24 events, splitting further leaves too few per bucket.
