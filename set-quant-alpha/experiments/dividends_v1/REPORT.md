# dividends_v1 - ex-dividend timing and dividend yield (free quick-win check, 2026-11-02)

Run: `uv run set-quant-alpha/experiments/dividends_v1/dividends_v1.py` (tables in `output/tables.md`). Data: the Yahoo
dividends already stored by `ingest.ingest_events` (13.9k rows, 824 tickers). Diagnostics only: 2013-19 and 2020-21
decide, 2022-26 is shown for information.

## A. Pre-ex-dividend window (ex-date in the next 5 trading days, about 3% of rows)

| Mean 5d alpha | 2013-19 | 2020-21 | 2022-26 (info) |
| --- | --- | --- | --- |
| all tradable, window vs not | +0.27% vs 0.00% | **−0.44%** vs −0.02% | +0.02% vs −0.01% |
| top-decile picks, window vs not | +0.48% vs +0.48% | +0.50% vs +0.14% | +0.50% vs +0.26% |

No consistent run-up: the sign flips between periods. Picks inside the window do no worse, so no filter is needed.
Assumption: ex-dates are treated as known 5 days ahead (Thai ex-dates are announced weeks before).

## B. Trailing 12-month dividend yield

| | IC (t) | rank correlation with the model score | Fama-MacBeth slope t added to the score |
| --- | --- | --- | --- |
| 2013-19 | 0.033 (4.8) | 0.21 | −0.49 |
| 2020-21 | 0.013 (0.6) | 0.11 | −1.06 |
| 2022-26 (info) | 0.056 (6.4) | 0.16 | 3.95 |

On its own, yield predicts alpha. In the decision periods, though, the model score already contains that information:
the slope added to the score is about zero or negative. Only 2022-26, a high-dividend / value regime, shows extra
power, and the test years cannot be used to choose.

## Conclusion

Nothing to promote. Dividend yield behaves like a **value / income style factor** that the score partly captures and
that paid off in 2022-26. Its best use is on the evaluation side: a local high-minus-low dividend-yield factor could
stand in for the missing local HML in the Thai Carhart attribution (book values are not free).
