# Module 05 homework - research ideas

Submit: https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw05
Background and gap analysis: [thesis_literature_review.md](thesis_literature_review.md)

| # | Market | Economic mechanism | Data | Horizon |
| --- | --- | --- | --- | --- |
| 1 | SET100 stocks | Foreign flows create temporary price pressure; locals who provide liquidity get paid | NVDR / investor-type flows | 1-5 days |
| 2 | SET50 index options (TFEX) | Thin option market + costly hedging -> mispriced vol and expensive hedges | Option quotes, SET50 futures | 1-30 days |
| 3 | S&P 500 index options (SPX/XSP) | Variance risk premium; its size depends on the state of the vol market | Cboe vol indices, option chains | 7-45 days |

Legend: [checked] = I looked at the source. [verify] = still needs checking.
All three ideas use free (or university-library) data only. Proofs of concept: `notebook.ipynb`.

---

## Question 1 - Daily ML on SET100 with local investor-flow features

**1. Market and scope.** SET100 constituents (point-in-time membership, including stocks later removed), 2012-2025. Daily rebalancing with a 1-5 day holding period. Long-only, or hedged with SET50 futures, because short selling is restricted for many Thai stocks.

**2. Investment thesis.** Foreign investors hold a large share of Thai free float and trade in waves driven by global risk appetite, USD/THB and EM fund flows rather than company news. Heavy foreign selling should push prices below fair value for a few days, and the stocks that provide liquidity should earn the rebound. Studies of Asian markets find that foreign flows have large, partly temporary price impact (Richards, 2005, JFQA), and US reversal profits behave like a reward for providing liquidity that grows when VIX is high (Nagel, 2012, RFS). Hypotheses:
- **H1:** Adding local flow features to a baseline of price and characteristic features improves the out-of-sample daily forecast (R², long-short Sharpe).
- **H2:** The gain comes from *reversal after foreign selling* (price pressure) and is stronger in high-VIX regimes. If it comes instead from *continuation after foreign buying*, flows carry information.

**3. Potential edge.** ML studies of emerging markets (e.g. Hanauer & Kalsbach, 2023: 32 countries, monthly) use generic characteristics, and Thailand is a small slice of the pooled sample. Existing Thai flow studies are linear and index-level. Thailand-specific data is mostly ignored: **NVDR trading** (the receipts foreigners use to get around foreign ownership limits), distance to the foreign limit, and proximity to the ±30% price limit. Reading those signals correctly is local knowledge.

**4. Data and sources.**
All free or available through the university; no paid terminal needed.
- Daily prices and volume: yfinance `.BK` [checked: 95 current SET100 names back to 2012].
- Point-in-time SET100 membership: rebuilt from SET's semi-annual constituent announcements (two lists a year) [verify].
- SET market-level trading by investor type (foreign / local institution / proprietary / retail), daily, on the SET website [checked that it is published; it blocks scripts, so download by hand]. TFEX publishes the same split for futures [checked].
- Per-stock NVDR net buy/sell: public daily SET / Thai NVDR reports list only the top 30 [checked via press reprints]. The full cross-section via SETSMART (university library) or a data request to SET's research institute CMRI [verify; main data risk].
- Controls: VIX, USD/THB, EEM (yfinance) [checked]; SET50 futures basis (TFEX).

**5. Research approach.** Target: next-day and next-5-day return relative to SET100. Features: price/technical + characteristics (baseline) vs baseline + flows. Models: elastic net -> LightGBM -> small MLP, all trained walk-forward with purged and embargoed splits (López de Prado). An LSTM on Korean flow data was recently **withdrawn for leakage** (Kang, 2026, arXiv), so leakage control is a core design choice. Decision: each evening, buy the top-decile predicted stocks at the next open. SHAP analysis then tests H2 by showing *which* flow pattern drives the predictions.

**6. Validation and risks.**
- Benchmarks: SET100 TRI, a plain 1-week reversal sort, and the baseline model without flows.
- Reject if the flow features do not improve the forecast significantly (Clark-West test), or if the net alpha disappears after round-trip costs (commission + VAT + half-spread, roughly 0.2-0.4% for SET100).
- Risks: flow data published after the close (use it only for the next day's open), NVDR being a noisy proxy (Thai investors use NVDRs too), regime breaks (2020, rule changes), and capacity in mid-caps.

**First test (free-data POC, `notebook.ipynb`).** 95 current SET100 names, 2017-2026 walk-forward, XGBoost.
- Market-level flow proxies (USD/THB, EEM, VIX) lift the daily IC from 0.057 to 0.063, but not significantly (t = 1.39), so H1 needs real stock-level NVDR data.
- The gross top-decile edge (~30% p.a.) turns into -19% p.a. after costs, because the book turns over 130% a day.
- Today's losers rebound 11.7 bps the next day, more in illiquid names and after the baht weakens, but not when VIX is high.
- Next steps: stock-level flows, open-to-close returns to remove bid-ask bounce, and slower rebalancing.

---

## Question 2 - SET50 options: ML volatility forecasts and deep hedging in a thin market

**1. Market and scope.** SET50 index options on TFEX, hedged with SET50 index futures, 2015-2025. Holding period is 1-30 days (front and second month).

**2. Investment thesis.** SET50 options are thinly traded: few active strikes, wide spreads, and hedging in whole futures contracts. Two consequences are testable:
- Implied volatility is often set by hedging demand rather than expected volatility, so the gap between a good realized-vol forecast and IV should predict **delta-hedged straddle returns**. This is the same logic as my RITC volatility-trading bot, applied to real data.
- Black-Scholes delta hedging is expensive in such a market. **Deep hedging** (Buehler et al., 2019) learns a hedge that trades off risk against costs (e.g. a no-transaction band) and should reduce the tail of hedging losses.

**3. Potential edge.** ML option-pricing and deep-hedging studies use S&P 500, simulated data or Bitcoin. Thai SET50 option research uses Black-Scholes, binomial trees, GARCH, and descriptive smile studies [checked: ThaiJO / Chula theses]. I found no study that applies ML vol forecasting or deep hedging to SET50 options. The frictions that make the market hard are exactly what these methods are built for.

**4. Data and sources.**
- SET50 level and realized vol: the TDEX SET50 ETF on yfinance, daily from 2008 [checked; Yahoo has no SET50 index history].
- SET50 option and futures settlement prices, volume and open interest by series: tfex.co.th series pages [checked that they exist]. They show recent data, so start collecting daily now and ask TFEX or SETSMART for the back history [verify depth; main data risk].
- Costs: TFEX member-broker fee schedule (commission per contract) plus the tick size of 0.1 index points.
- Transfer source: SPX implied vol from the free Cboe indices (VIX, VIX9D, VIX3M, SKEW) to pre-train surface models, then fine-tune on sparse SET50 quotes (following Conti & Morelli, 2025, arXiv).

**5. Research approach.**
- (a) Forecast 20-day realized vol with HAR-RV vs LightGBM using Thai features (foreign flows, USD/THB, VIX). Signal = forecast RV - ATM IV. Trade: buy (sell) a delta-hedged ATM straddle when the signal is above (below) a threshold.
- (b) Hedging horse race on the same option book: BS delta vs Leland vs Whalley-Wilmott vs a no-transaction-band neural net. The objective is CVaR of hedging P&L net of futures costs.

**6. Validation and risks.**
- Benchmarks: unhedged and BS-delta-hedged positions; always selling (or always buying) the straddle.
- Reject if the vol-gap signal has no out-of-sample correlation with straddle P&L, or if deep hedging does not lower CVaR after costs on test years.
- Risks: wide bid-ask spreads can wipe out any edge, sparse quotes make IVs noisy, the futures contract size limits how finely positions can be hedged, liquidity disappears in a crisis, and the data history may be too short for a neural net (mitigated by pre-training on simulated or SPX data).

**First test (free-data POC).**
- (a) Using the TDEX SET50 ETF, HAR-RV forecasts 20-day vol best. Adding Thai/global features or switching to XGBoost makes the forecast worse (Diebold-Mariano t = -2.3 and -3.2). Better inputs (intraday data, TFEX IV) matter more than model complexity.
- (b) Simulated short-call book hedged in whole SET50 futures contracts. A cost-aware no-trade band cuts tail risk (CVaR95) relative to BS delta by 1.8% at 2 bps, 4.8% at 10 bps and 15.8% at 40 bps for a 100-call book. A 10-call book gains nothing below 20 bps, because whole-contract rounding already acts as a band. The direction supports the hypothesis; the size depends on book size and costs.

---

## Question 3 - Timing the S&P 500 variance risk premium with ML (SPX/XSP put spreads)

**1. Market and scope.** S&P 500 index options. XSP (1/10 size) is suited to a small account: European-style and cash-settled, so there is no early assignment. Tradable from Thailand: Webull Thailand offers SPX/XSP index options [checked: Webull TH press releases]. Dime offers US stock options at USD 0.55 per contract [checked]; [verify] whether it offers index options and allows selling or spreads. Expiries from 7 to 45 days; monthly decisions.

**2. Investment thesis.** Option buyers overpay for crash insurance, so implied variance is on average above later realized variance: the variance risk premium (Carr & Wu, 2009; Bollerslev, Tauchen & Zhou, 2009). Selling puts harvests it but suffers in crashes. **H1:** an ML model that predicts *when* the premium is large and safe raises Sharpe and cuts drawdown versus selling every month. Inputs: VIX vs a HAR-RV forecast, VIX/VIX3M term structure, VVIX, SKEW, credit spreads. **H0:** the premium is fair payment for crash risk that timing cannot avoid.

**3. Potential edge.** Bali et al. (2023, RFS) show ML predicts *individual stock* option returns. The 0DTE literature shows retail 0DTE traders lose money (Beckmeyer, Branger & Gayda, 2023). Less studied: ML timing of *index* short-vol trades under realistic retail constraints (defined risk only, USD/THB conversion, Thai tax on foreign income remitted into Thailand [verify current rule]). Most of the evidence is about the US market maker or institution, not the overseas retail trader.

**4. Data and sources.**
- Free [checked]: Cboe's CSV feed for VIX, VIX3M, VVIX, SKEW and the strategy indices **PUT** (ATM monthly put-write), **PUTY** (2% OTM), **WPUT** (weekly) and **CNDR** (iron condor, i.e. defined risk); FRED `DTB3` T-bill; yfinance `^GSPC` for realized vol.
- Option chains with bid/ask, only for a strike-level backtest: OptionsDX (paid, about USD 50 per year of data) [verify], or snapshot yfinance chains daily from now.

**5. Research approach.**
- Stage 1 (free data, done in the POC): a gradient-boosted model forecasts next month's excess return of each Cboe strategy index (PUT, PUTY, WPUT, CNDR) from the vol-market state. Rule: sell this month only when the forecast is positive, otherwise hold T-bills. The simple VIX < VIX3M rule is the benchmark it has to beat.
- Stage 2 (chains, OptionsDX): simulate 30-45 DTE 0.20/0.05-delta put **spreads** with bid-ask costs, sized so the max loss is ≤ 2% of the account.

**6. Validation and risks.**
- Benchmarks: SPY buy-and-hold, the PUT index (always sell), and a simple linear rule (sell only when VIX < VIX3M).
- Reject if ML does not beat the linear term-structure rule out of sample (train 2006-2017, test 2018-2025, which covers Feb 2018, Mar 2020 and 2022), or if the Sharpe gain is not significant under block bootstrap.
- Risks: tail risk (report max drawdown and worst month), few crash events to learn from (class imbalance), overfitting thresholds, XSP spreads, broker restrictions on selling, FX. Paper-trade for 3 months before using real money.

**First test (free-data POC).** Cboe strategy indices, test period 2014-01 to 2026-09 (153 months).
- XGBoost timing **fails**: Sharpe 0.28 vs 0.70 for always selling ATM puts, beating it in only 1% of bootstraps. With about 85 monthly training points the model overfits.
- The VIX < VIX3M rule raises Sharpe to 0.80 and cuts the max drawdown (-15% vs -21%) and the worst month (-7.6% vs -13.4%). The Sharpe gain is not significant (P = 0.56).
- The rule works best on **2% OTM** put-writing: Sharpe 0.66 -> 0.90, max drawdown -20% -> -10%.
- Weekly put-writing earns little (Sharpe ~0.23). The defined-risk **iron condor** returns ~1% a year, less than T-bills: the protective wings cost most of the premium. That matters for a retail account limited to spreads.
