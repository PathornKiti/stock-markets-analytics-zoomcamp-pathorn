# events_v1 - earnings events and post-earnings drift (2026-11-02)

Run: `uv run set-quant-alpha/experiments/events_v1/events_v1.py` (tables in `output/tables.md`).

## Data

* **Source:** Yahoo earnings calendars (date, EPS estimate, reported EPS, surprise %) and dividends for all 853 universe
  tickers. Fetched by `ingest.ingest_events` and stored in `data/raw/earnings.parquet` and `dividends.parquet`.
  * Earnings history reaches back to 2004 for large caps.
  * 42-81% of tradable stock-years have at least one record in 2013-2026.
  * In tradable stock-days covered after a ticker's first record, coverage is 86-98%.
  * Surprise % is filled for 91% of rows since 2021.
* **Features** (`setalpha.events`, research only, not in the production model):
  * days since the last report;
  * a point-in-time "expected report in the next 5 days" flag (last year's date + 364 days);
  * the earnings reaction: alpha over days E and E+1, known at the close of E+1;
  * the last EPS surprise.
  * The look-ahead flag "actual report inside the 5-day hold" is used for diagnostics only.

## Protocol

Diagnostics only: no strategy design is selected here. 2013-19 and 2020-21 decide. 2022-26 is shown for information,
because every earlier experiment has used it. Scores are the walk-forward ensemble (strategy_v3 predictions_ext),
tradable top-100. Four feature hypotheses were examined: the hold-window flag, the expected-report flag, the
earnings reaction and the EPS surprise.

## Results

**A. Reports inside the 5-day hold** (look-ahead; about 7% of top-decile picks)

| | P(alpha < −10%), with vs without a report | mean 5d alpha, with vs without |
| --- | --- | --- |
| 2013-19 | 1.5% vs 0.6% | +0.48% vs +0.47% |
| 2020-21 | 1.8% vs 2.0% | +0.51% vs +0.13% |
| 2022-26 (info) | 2.6% vs 0.9% | +0.17% vs +0.29% |

Reports add variance (std is 10-25% higher) and, in 2 of 3 periods, crash risk. They do not lower the mean alpha in
the decision periods.

**B. The point-in-time flag**

* It anticipates the actual window with about 64% precision and 60% recall.
* Flagged top picks had *higher* mean alpha in 2013-19 (+0.57% vs +0.46%) and 2020-21 (+0.25% vs +0.14%).
* Only 2022-26 shows them worse (−0.06% vs +0.30%).
* Excluding flagged names would have hurt in the decision periods, so the filter is **not adopted**.

**C. Post-earnings drift**

| Feature | IC (t), 2013-19 | IC (t), 2020-21 | IC (t), 2022-26 info | Rank correlation with the model score | Fama-MacBeth slope t added to the score |
| --- | --- | --- | --- | --- | --- |
| EPS surprise | 0.019 (2.1) | 0.007 (0.0) | 0.026 (3.5) | 0.04-0.12 | 1.96 / −0.08 / 1.39 |
| earnings reaction | 0.014 (1.5) | 0.016 (1.1) | 0.009 (0.7) | 0.04-0.12 | 1.73 / 0.37 / 0.20 |

* The EPS surprise is nearly orthogonal to the model and positive in 2 of 3 periods. It fails in 2020-21.
* The earnings reaction is too weak.

## Conclusion

* No new paper shadow is registered: the decision-period evidence does not support an earnings filter or a drift
  overlay.
* The earnings-window effect is a **risk** effect (fatter tails), not an alpha effect. If used later, it belongs in
  position sizing, not in exclusion.
* **The EPS surprise is the best candidate for further research.** It is orthogonal to the price signal. Its weakness may
  come from Yahoo's thin consensus coverage for Thai names, so it should be retested with better consensus data
  (SETSMART / Bloomberg).
