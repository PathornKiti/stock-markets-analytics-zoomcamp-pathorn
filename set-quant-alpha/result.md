# Result: does strategy S5 beat the Thai stock market?

## The answer in one minute

**What S5 is.** A computer model that every week buys the 10 Thai stocks it expects to do best over the next few
days. It picks them from the 100 most-traded stocks on the SET. It holds them in equal amounts and swaps a stock only
when it drops well down the ranking.

**What we compared it with.** Simply buying and holding TDEX, an ETF that tracks the SET50. This is the cheap, passive
alternative any investor could buy instead.

**Result over 10 years (2016-2026), after real trading costs:**

| | S5 | TDEX (buy and hold) |
| --- | --- | --- |
| Return per year | **7.4%** | 7.2% |
| Worst fall from a peak | **−35%** | −37% |
| Return per unit of risk (Sharpe) | **0.57** | 0.53 |
| Exposure to market swings (beta) | 0.62 (moves about 60% as much as the market) | 1.00 |

*How the 10-year record was built:*
* *It is a walk-forward record. Each January, the rules were re-chosen using only earlier years.*
* *It held a literature-factor portfolio in 2016-20 and S5's exact rules from 2021.*
* *How S5 works is explained in [strategy.md](strategy.md).*

**What this means:**

1. **S5 roughly matched the market with less risk. It did not clearly beat it.** The 0.2-point lead is too small to
   rule out luck.
2. **The stock-picking signal is real.**
   * The model's rankings predict which stocks will do better, every year from 2013 to 2026.
   * This holds even after allowing for the 795 versions we tested.
   * In 2022-26, S5's picks beat its own pool of 100 stocks by about 13% a year (t 2.0).
3. **Trading costs eat most of the edge.**
   * S5 trades a lot (it turns the portfolio over about 11 times a year).
   * Thai brokerage and exchange fees then take about 2% a year, roughly 40% of the extra return before costs.
4. **Realistic expectation: around market return, not above it.**
   * Two effects made the backtest look 1-3 points a year better than reality (section 4): which weekday the
     portfolio rebalances, and stocks that were later delisted.
5. **Decision: not ready for real money.**
   * S5 has been paper-traded with fake money since 2 November 2026, under rules fixed in advance.
   * It is reviewed about May 2027.

## 1. What could make it a "go"

| Condition | Why |
| --- | --- |
| 6-12 months of live paper trading at or above TDEX after costs | Only data no test has seen can prove the edge |
| A cheaper broker (about 0.08% commission instead of the 0.15% average) | Costs would roughly halve, which matters more than any model change |
| No live drawdown much worse than TDEX's | The backtest's worst fall was −35%. An investor must be able to sit through that |

## 2. Can it be improved? What we tried (November 2026)

We used professional data from LSEG (Refinitiv) to test whether more information makes S5 better. Every test had its
rules written down **before** looking at the results. Decisions used 2013-21 only, and 2022-26 was a final check.

| Idea | Result | Kept? |
| --- | --- | --- |
| Include delisted stocks (remove "survivor" bias) | Changes S5 by −0.4 points a year (2013-26). Not significant | Noted as a caveat |
| Foreign-investor demand per stock | Not predictive in 2013-21. Predictive in 2022-26 only | No (idea for future research) |
| Analyst forecasts, value ratios (P/B, P/E) | Signals take turns working: forecasts before 2022, value after. Inside the model they add nothing | No |
| 13 more signals (price targets, earnings surprises, dividends, ESG, ...) | A blend beat S5 in 2013-21 (17.3% vs 16.0%) but lost in 2022-26 (4.5% vs 7.7%) | No |
| Earnings surprise alone | The one new signal that worked in both periods. S5 already captures most of it | No (candidate to watch) |
| Signal weights that adapt to recent performance | Adapts, but too late: lost in 2013-21 | No |
| Holding 20 or 30 stocks instead of 10 | Lower return, and no steadier | No |

**Conclusion: S5 stays exactly as it is.** The extra data did not produce an improvement that held up in both
periods. The sensible next step is to let the live paper record accumulate, not to search for more variants.

## 3. How sure are we?

* **Many versions were tried.** That makes it easy to find something that worked by chance, so the bar for "real" is
  raised to match: a t-statistic of 4.0 instead of the usual 2.
  * In total, 795 strategy and model versions were tested (up to 3 November 2026).
* **The stock-ranking signal clears that bar** (t 4.3).
* **The strategy's extra return over TDEX and over factor models does not.**
  * All estimates are positive, but every t-statistic is below 2.
  * 10 years is too short to prove an edge of this size.

## 4. Caveats in plain words

* **Rebalance-day luck.** S5 rebalances once a week, on a fixed day.
  * Re-running the same rules on each of the 5 possible weekdays, the usual day turned out about 1.6 points a year
    luckier than the average (2013-26: 13.0% vs 11.4%; [adaptive_v1](experiments/adaptive_v1/REPORT.md)).
  * In 2022-26 alone, the average-day result (6.0%) was slightly below TDEX (6.7%).
  * This was measured on S5's fixed rules. The deployed track record was not re-run this way, but it probably shares
    part of the luck.
* **Delisted stocks.** The free Yahoo data only lists today's stocks.
  * Adding the 50 missing SET100 stocks (44 of them delisted) from LSEG lowers results by 0.4-1.7 points a year. This
    is not significant ([survivorship_v1](experiments/survivorship_v1/REPORT.md)).
  * Treat 7.4% as up to 1-2 points optimistic.
* **The test years were used more than once.** 2022-26 was examined by several experiments before the final design.
  This is why the significance bar counts all 795 versions, and why live paper trading is the real test.
* **Costs are an average retail estimate** (0.168% per side, below). Larger investors or cheaper brokers pay less.
* **Losses can be large.** The backtest's worst fall was −35%, similar to the market's.

---

## Appendix A. Detailed results (for reviewers)

**Deployed strategy:** the integrated walk-forward pick.
* It blends the 5-day ML score with a 20-day-target ML score.
* It holds the top 10 of the liquidity top-100 Thai stocks and keeps a held name while it ranks in the top 30.
* It rebalances weekly, uses equal weights and has no stop-loss.
* Its rules are re-selected each January using only earlier years.

**Period:** 2016-01-04 to 2026-10-22 (10.4 years), walk-forward out of sample, unless noted. "Net" = after 0.168% per
side; "gross" = before trading costs.

### A.1 Against the benchmark an investor could buy instead

| Comparison | Deployed | Benchmark | Significant? |
| --- | --- | --- | --- |
| **vs TDEX (SET50 ETF), buy and hold** | **7.4% CAGR**, Sharpe 0.57, max DD −34.9% | 7.2% CAGR, Sharpe 0.53, max DD −36.5% | No (active t ≈ 0); beats TDEX in 6 of 11 years |
| **vs its own universe** (equal-weight tradable top-100, no costs), 2022-26 | **+7.3% a year** | **−5.9% a year** | **Yes, t 2.04** (weekly active returns) |

### A.2 Against factor models (alpha = return not explained by the factors)

| Benchmark model | Net alpha % a year (t) | Gross alpha % a year (t) |
| --- | --- | --- |
| **CAPM** (market = TDEX) | +3.2 (0.93) | +5.2 (1.52) |
| **Fama-French 3-factor** (emerging markets: Mkt, SMB, HML) | +1.2 (0.26) | +3.1 (0.68) |
| **Carhart 4-factor** (emerging markets: + momentum WML) | +5.2 (1.08) | +7.1 (1.43) |
| **Carhart 4-factor, local Thai factors** (TDEX, size, momentum) | +1.9 (0.55) | +3.9 (1.13) |
| Carhart local Thai, with the factors also charged trading costs | +2.8 (0.82) | – |

Market beta to TDEX is 0.62.

### A.3 The signal behind the strategy

| Measure | Value |
| --- | --- |
| Daily rank IC vs 5-day alpha, test 2022-26 | 0.050 (overlap-robust t 5.6), positive in every year 2013-2026 |
| After the multiple-testing haircut for all 795 trials | t 4.3 (Bonferroni / Holm; BHY 3.8): still significant (hurdle 4.00) |

### A.4 Statistical checks on the strategy

| Check | Value | Threshold | Result |
| --- | --- | --- | --- |
| Probabilistic Sharpe ratio (SR > 0) | 0.96 | > 0.95 | pass |
| Track record vs minimum track record length | 10.4 vs 8.6 years | track record ≥ MinTRL | pass |
| Deflated Sharpe ratio | 0.16 (computed at 671 trials; more trials only lower it) | > 0.95 | fail |
| Probability of backtest overfitting | 0.61 | < 0.5 | fail |
| Information ratio vs TDEX | 0.01 | > 0.5 | fail |

### A.5 Verdict

| Question | Answer |
| --- | --- |
| Does the model pick stocks well? | **Yes.** The signal is significant after 795 trials, and the strategy beats its own universe with t 2.04 |
| Does it beat TDEX after realistic costs? | **Yes on CAGR, Sharpe and drawdown**, but by too little to be statistically certain |
| Does it earn significant alpha against CAPM, Fama-French or Carhart? | **No.** Every alpha is positive, every t-stat is below 2 |
| Why not? | Trading costs (2.1% a year at 11× turnover) absorb about 40% of the gross alpha, and 10 years is too short to prove an edge this size |
| Is it ready for real capital? | **No-go yet.** It is being paper-traded live as shadow S5. Review about May 2027 |

## Appendix B. How the test was done

### B.1 Data and target

* **Universe:** 853 Thai stocks from Yahoo Finance, daily, 2010 onward (dividend-adjusted).
  * Liquid = 60-day median traded value ≥ THB 20M.
  * Tradable = the top 100 by traded value, recomputed every day (a point-in-time SET100).
* **Target:** 5-day alpha = return from the next open to the close 5 days later, minus beta × the equal-weight market
  return over the same days.
* **No look-ahead:**
  * US, FRED and Cboe data are lagged one day;
  * regime probabilities are forward-filtered;
  * every decision uses data up to the close and fills at the next open.
* **LSEG research data** (Nov 2026 experiments only):
  * point-in-time index membership, delisted prices, I/B/E/S estimates, financials with announcement times;
  * licensed, kept out of the repository, never used by the daily pipeline.

### B.2 Splits and walk-forward

| Tier | Period | Use |
| --- | --- | --- |
| Train | 2010-2019 | Fit models |
| Validation | 2020-2021 | Choose hyperparameters, ensemble members, the original strategy E2 |
| Out-of-sample backtest | 2022-01 to 2026-10 (one-shot test); 2013-2026 walk-forward | Judge models and strategies |
| Live forward test | from 2026-11-02 | Paper-traded shadows S1-S6, point-in-time scores, append-only ledger |

* **Model refits:** the models are refit every January on all earlier data. A 5-day embargo at each boundary stops
  overlapping targets leaking across splits.
* **Strategy re-selection:** each January from 2016, the best design among 576 (8 signal blends × holdings, rebalance
  frequency, buffer, weighting, stop) is chosen **only on 2013 to the previous year**, by net CAPM-alpha t-stat. That
  design is then traded for the year. The yearly results are stitched into one 2016-2026 track record, so the
  reported performance is out of sample with respect to every choice.
* **Pre-registration:** the protocol was frozen with a timestamp before the first selection
  (`experiments/integrated_v1/output/protocol_frozen.txt`). Each November 2026 experiment has its own frozen protocol.

### B.3 Execution and costs

* **Simulation:** exact day-by-day, THB 1M, 100-share lots, sells before buys, fills at the next open.
* **Stock cost: 0.168% per side**, the researched all-in Thai cost:
  * average online commission of 30 brokers, 0.150%;
  * SET 0.005% + TSD 0.001% + SEC 0.001% fees;
  * 7% VAT on both.
* **TDEX legs:** 0.05% per side.
* Gross results re-run the same trades at zero cost.

### B.4 Benchmarks and factor regressions

| Benchmark | Question it answers | Method |
| --- | --- | --- |
| TDEX buy and hold | Better than what I could buy cheaply? | CAGR, Sharpe, drawdown; active return t-stat |
| Equal-weight tradable top-100 | Does stock selection add value within the universe? | Active return vs the universe (no costs charged to the universe) |
| CAPM | Alpha after market exposure | r − rf = α + β(TDEX) + ε; daily; Newey-West (5 lags) t-stats |
| Fama-French 3 (EM) | Alpha after market, size, value | Monthly, USD (THB returns converted with USD/THB); Ken French EM factors; Newey-West (3 lags) |
| Carhart 4 (EM) | + momentum | As FF3, + WML |
| Carhart 4 (local Thai) | Same idea, Thai factors | Daily; MKT = TDEX, SMB = small − big liquid stocks, WML = 12-1-month winners − losers |
| Carhart local, factors net of costs | Fair "costs on both sides" version | Each factor charged 0.15% per side on its own daily membership turnover (about 3.8% a year for SMB, 8.3% for WML) |

### B.5 Significance and overfitting checks

| Check | What it guards against |
| --- | --- |
| t-stats with Newey-West / non-overlapping returns | Overstated significance from overlapping 5-day targets |
| Harvey-Liu-Zhu multiple-testing haircut, **795 trials** counted (`validity.TRIALS`) | Finding something by chance after many tries |
| Deflated Sharpe ratio | Selecting the best of many backtests |
| Probability of backtest overfitting (CSCV) | In-sample winners failing out of sample |
| Minimum track record length | Too short a record to trust the Sharpe |
| Rebalance-phase average (5 weekdays) | A result that depends on the decision day |
| Live paper trading with pre-registered rules | Everything above: the only data no experiment has seen |

## Appendix C. How to reproduce

```bash
cd ~/stock-markets-analytics-zoomcamp-pathorn
uv run set-quant-alpha/scripts/run_pipeline.py --mode file --steps simulate signals paper report   # backtests, orders, dashboard
uv run set-quant-alpha/experiments/integrated_v1/integrated_v1.py wf      # walk-forward selection at config.FEE_PER_SIDE
uv run set-quant-alpha/experiments/integrated_v1/publish_deployed.py      # publish the deployed backtest to the dashboard
uv run set-quant-alpha/scripts/run_pipeline.py --mode file --steps report
open set-quant-alpha/reports/dashboard.html                             # or http://localhost:8000 with docker compose
```

* **Signal results:** `experiments/integrated_v1/output/tables.md`; **test-period results:** the dashboard's
  Performance, Factors and Trust & risk tabs.
* **November 2026 experiments:** `experiments/{survivorship_v1, foreign_v1, fundamentals_v1, lseg_signals_v1,
  adaptive_v1}/REPORT.md`. These need LSEG Workspace and a licence.
* **Live record:** unpause `set_quant_alpha_daily` in Airflow (localhost:8080).

## Glossary

| Term | Meaning |
| --- | --- |
| CAGR | Average yearly growth rate |
| Sharpe ratio | Return per unit of volatility; higher is better |
| Max drawdown (DD) | Largest fall from a previous high |
| Alpha | Return left after removing what the market (and other common factors) explain |
| t-statistic | How many "standard errors" a result is from zero; about 2 means unlikely to be luck for a single test |
| Beta | How much the portfolio moves when the market moves 1% |
| Paper trading | Following the strategy live with fake money, to test it on new data |
