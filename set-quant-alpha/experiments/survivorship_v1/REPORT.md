# survivorship_v1 - how much does survivorship bias flatter the S5 rule? (2026-11-03)

Code: `survivorship_v1.py` (stages `data`, `scores Y|L`, `sim`). Tables: `output/tables.md`. Protocol frozen in
`output/protocol_frozen.txt` at 2026-11-02T17:04:09Z, before any dataset, score or simulation was built.
Data: LSEG Workspace via `scripts/fetch_lseg.py` (licensed; kept in gitignored `data/raw/lseg/`).

**Verdict: survivorship bias is small and not distinguishable from noise.**

* **Strategy.** Adding the 50 SET100 stocks that Yahoo's current listing misses changes the S5 rule's net return by
  −0.4 points a year over 2013-26 (NW t −0.16).
  * 2016-26: −1.7 points a year (t −0.69).
  * 2022-26: −1.5 points a year (t −0.39).
  * The paired-bootstrap 95% CI of every Sharpe difference includes zero.
* **Signal.** It is unaffected: the C2 daily rank IC is 0.070 on the corrected data against 0.071 on Yahoo.
* **Index.** An equal-weight SET100 built from true point-in-time members (delisted names included) differs from its
  Yahoo-available subset by +1.3 to +1.6 points a year in 2013-15 and by about 0 from 2016 on.
* **Under the pre-registered rule** (replace if the CAGR differs by > 0.5 point), the corrected 2016-26 figure is the
  honest one: **11.2% instead of 13.1%** for the fixed S5 rule.

## What was missing from Yahoo

* **Coverage.** SET100 has had 263 distinct members since 2009. 53 of them are not in the Yahoo data.
  * 3 are renamed tickers that Yahoo has under the new name (BMCL→BEM, TRUEE→TRUE, and BPP, see below).
  * 50 were added from LSEG, 44 of them delisted.
* **Missing member-days by year.** 19.5% in 2013, 12% in 2016, 8-10% in 2017-23, 0.7% in 2026.
* **A Yahoo data error.** Yahoo's `BANPU.BK` history from 2016 is Banpu Power's (BPP) price series.
  * Daily-return correlation with BPP is 0.988, but only 0.44 with Banpu's own old RIC `BANPUU.BK^G26`.
  * So the Yahoo data never contained Banpu itself, a large SET50 stock.
  * The corrected data adds Banpu from LSEG and drops BPP as a duplicate.
* **LSEG vs Yahoo on 208 shared names.** Median daily-return correlation is 0.989, and the median cumulative
  difference over 2009-26 is 2.8%.
  * The outliers are small caps with rights or warrant adjustments (TWZ, MJD, BTS).
  * LSEG is used only for the 50 names Yahoo lacks.

## S5 rule, Yahoo (Y) vs survivorship-corrected (L), net of 0.168% per side

S5 rule = C2 (mean rank of the ML and V8 scores), top-10, buffer 30, weekly, equal weight, next-open fills.

* **One code path.** The same code builds the dataset, the walk-forward ML and V8 scores (refit every January) and the
  simulation for both universes.
* **Delistings.** A held stock is sold at its last LSEG close.

| Period | Series | CAGR % | Sharpe | Max DD % | CAPM α % (t) |
| --- | --- | --- | --- | --- | --- |
| 2013-26 | S5 rule, Y | 13.5 | 0.83 | −34.8 | 10.1 (2.73) |
| | S5 rule, **L** | **13.0** | 0.79 | −37.8 | 9.4 (2.65) |
| | EW tradable top-100, Y / L | 5.2 / 4.7 | 0.37 / 0.34 | | |
| | TDEX | 4.8 | 0.38 | −36.5 | |
| 2016-26 | S5 rule, Y | 13.1 | 0.83 | −34.8 | 8.1 (2.04) |
| | S5 rule, **L** | **11.2** | 0.71 | −37.8 | 6.0 (1.55) |
| | TDEX | 7.0 | 0.52 | −36.5 | |
| 2022-26 | S5 rule, Y | 9.4 | 0.66 | −23.4 | 4.2 (0.70) |
| | S5 rule, **L** | **7.7** | 0.54 | −27.6 | 2.2 (0.36) |
| | TDEX | 6.7 | 0.54 | −26.6 | |

| Is L − Y more than noise? | L − Y % a year | NW t | Sharpe L − Y (95% bootstrap CI) |
| --- | --- | --- | --- |
| 2013-26 | −0.36 | −0.16 | −0.05 (−0.29, +0.19) |
| 2016-26 | −1.66 | −0.69 | −0.12 (−0.39, +0.14) |
| 2013-21 | +0.25 | +0.10 | −0.01 (−0.30, +0.28) |
| 2022-26 | −1.50 | −0.39 | −0.12 (−0.55, +0.30) |

* **Year by year the gap is large and changes sign** (+18.8 points in 2014, −18.2 in 2020, −13.5 in 2025; see
  `per_year.csv`).
  * A 10-stock book is path dependent: a slightly different universe, and models retrained on it, change the picks.
  * So a single-year difference is mostly noise, and the multi-year averages above are the estimate.
* **The level of the fixed S5 rule is not out of sample.** C2 top-10 weekly was chosen with hindsight (integrated_v1),
  so read the L − Y difference, not the 13% level.
* **The deployed walk-forward record (7.4% in result.md) was not re-run on L.** It held the literature book (C3) in
  2016-20 and C2 books from 2021.
  * For 2021-26 the fixed-rule gap averages about −0.3 points a year.
  * An exact correction would need integrated_v1's selection re-run on L. EPS and dividend signals for the delisted
    names are not available, so that is not done here.

## Index-level bias (LSEG total return, one source)

EW return of all point-in-time SET100 members vs the members present in the Yahoo data (renamed tickers count as
present):

| | 2009-12 | 2013 | 2014 | 2015 | 2016-26 mean |
| --- | --- | --- | --- | --- | --- |
| Yahoo subset − all members, points | −2.9 to +0.3 | +1.3 | +1.6 | +1.6 | −0.1 |

The bias exists where coverage is worst (2013-15) and is about 1.5 points a year there. It is gone once coverage
passes 90%.

## Replication

The Y run reproduces the existing S5-rule simulation (integrated_v1, `C2|N10|R5|B|E`) with a daily-return correlation
of 0.88 and CAGR of 14.6% vs 15.2% over 2013-26. The gap is expected:

* This run scores *all* liquid rows. integrated_v1 scored only rows with a known 5-day target, which silently drops a
  stock in its last 5 trading days before a delisting or data gap.
* Here every year is refit in one consistent pass.

## Deviations

* **D1.** Two crashes were fixed before any result was read: a path bug that nested the variant data dir, and
  duplicate feature columns. Nothing in the method changed.
* **D2.** LSEG prices arrive as nullable Float64. They are cast to float64 before simulation; values are unchanged.
* **D3.** The index-level table was first produced with the 95 RICs downloaded at that time and re-run on all 263. Only
  the full version is reported.

## Trials

1 (one evaluation of an existing rule on a corrected dataset; no search). `validity.TRIALS["survivorship_v1"] = 1`.

## Reproduce

    cd ~/stock-markets-analytics-zoomcamp-pathorn
    uv sync --group lseg     # plus LSEG Workspace running and LSEG_APP_KEY in set-quant-alpha/.env
    uv run python set-quant-alpha/scripts/fetch_lseg.py members missing prices
    uv run python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py data
    uv run python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py scores Y   # ~2 min each
    uv run python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py scores L
    uv run python set-quant-alpha/experiments/survivorship_v1/survivorship_v1.py sim
