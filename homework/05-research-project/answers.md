# Module 05 homework - answers

Submit: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05
Free-data proofs of concept: [notebook.ipynb](notebook.ipynb) · Long version: [answers_detailed.md](answers_detailed.md)

---

## Q1 - SET100: do foreign flows predict next-day returns?

1. **Market & scope:** SET100 stocks (Thailand), hold 1-5 days.
2. **Thesis:** When foreigners sell for reasons unrelated to the company (global risk-off, weak baht), prices fall too far and bounce back. Whoever buys from them earns the rebound.
3. **Edge:** Thai-specific flow data (NVDR trading, flows by investor type) that global ML studies don't use.
4. **Data:** yfinance `.BK` prices; SET daily trading by investor type; SET / Thai NVDR daily reports; full per-stock NVDR history from SETSMART or CMRI (to verify).
5. **Approach:** Walk-forward ML (XGBoost) predicting next-day relative return, with vs without flow features; buy the top 10% each day.
6. **Validation & risks:** Must beat SET100 and a plain reversal rule after ~0.3% round-trip cost. Reject if flow features don't improve forecasts significantly. Risks: costs, bid-ask bounce, survivorship bias, data published late.

**POC result:** ML does predict (daily IC 0.06), but flow proxies add little (t = 1.4), and costs turn +30% a year into -19% a year. Next: real NVDR data and slower trading.

---

## Q2 - SET50 options: better volatility forecasts and cheaper hedging

1. **Market & scope:** SET50 index options and futures on TFEX, 1-30 days.
2. **Thesis:** In a thin options market, (a) implied vol drifts away from realised vol, so a good vol forecast finds mispriced options; (b) hedging is costly (wide spreads, whole contracts), so a cost-aware hedge should lose less than standard delta hedging.
3. **Edge:** No study applies ML vol forecasting or deep hedging to SET50 options; most work uses the S&P 500.
4. **Data:** TDEX ETF (tracks SET50) on yfinance; TFEX settlement prices by series (collect daily, request history); broker fee schedule.
5. **Approach:** (a) HAR vs ML forecast of 20-day vol; signal = forecast - implied vol -> delta-hedged straddle. (b) Compare Black-Scholes delta vs a no-trade-band hedge on the tail of hedging losses (CVaR).
6. **Validation & risks:** Reject if the vol signal doesn't predict straddle P&L, or the band doesn't cut tail risk after costs. Risks: wide spreads, sparse quotes, short history.

**POC result:** HAR beat ML for vol forecasting. The cost-aware band cut tail risk more as costs rose: 1.8% at 2 bps up to 15.8% at 40 bps for a 100-option book, and nothing for small books.

---

## Q3 - S&P 500: when to sell put options

1. **Market & scope:** S&P 500 index options (SPX/XSP, available via Webull Thailand), monthly decisions.
2. **Thesis:** Investors overpay for crash insurance (the variance risk premium), so put sellers earn a premium but lose badly in crashes. Selling only in calm markets should keep most of the premium with smaller losses.
3. **Edge:** Simple, rule-based timing tested from the view of a Thai retail trader (costs, FX, defined-risk limits).
4. **Data:** Free Cboe indices (PUT, PUTY, WPUT, CNDR, VIX, VIX3M, VVIX, SKEW) and the FRED T-bill rate.
5. **Approach:** Each month, sell puts only if VIX < VIX3M (or if an ML forecast is positive); otherwise hold T-bills.
6. **Validation & risks:** Must beat always-selling and SPY on Sharpe and drawdown over 2014-2026, with a bootstrap significance test. Risks: crashes, overfitting, broker limits, FX.

**POC result:** ML timing failed (Sharpe 0.28 vs 0.70). VIX < VIX3M on 2% OTM puts: Sharpe 0.66 -> 0.90 and max loss -20% -> -10% (not yet statistically significant). Iron condors earned less than T-bills.

---

## Bottom line

- All three ideas run on free data, and each has a first test.
- Simple models beat ML on small data. Better data matters more than a more complex model.
- Consistent with efficient markets after costs: returns are somewhat predictable, but profits need better data, liquidity provision, or bearing risk.
