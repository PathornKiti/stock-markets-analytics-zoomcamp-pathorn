# Thesis scoping: literature review and research gaps

> Superseded by the Google Scholar gap analysis (1 Nov 2026), kept as a Claude Doc for the discussion:

Profile: data scientist doing a Finance/Economics master's. Interests: (1) daily ML trading in SET100,
(2) option pricing, derivatives and algorithmic hedging (built on the RITC 2026 volatility and market-making work).
Data: free sources plus university-library data (SETSMART, TFEX / CMRI data requests).

Status: a first scoping pass (November 2026) based on web search. Every reference below was found online.
Before citing, read the original and check authors, year and venue. Entries marked (verify) were not
fully confirmed.

---

## Stream A - Machine learning for the cross-section of stock returns (-> SET100 daily)

### What is known
| Paper | Setting | Finding | Relevance |
| --- | --- | --- | --- |
| Gu, Kelly & Xiu (2020, RFS) "Empirical Asset Pricing via Machine Learning" | US, monthly, ~900 predictors | Trees and neural nets roughly double the Sharpe ratio of linear models; the top predictors are momentum, liquidity and volatility | The reference framework |
| Leippold, Wang & Zhou (2022, JFE) "Machine learning in the Chinese stock market" | China A-shares, retail-dominated | Predictability is larger than in the US and comes mostly from small, illiquid stocks and short horizons | Shows ML gains are *market-structure dependent* |
| Hanauer & Kalsbach (2023, Emerging Markets Review) | 32 EM countries, 1990-2021, 36 characteristics, monthly | Random forest / GBRT around 1.0%/month and neural-net ensembles around 1.2%/month in long-short returns, beating OLS | Thailand is just **one slice** of a pooled sample, with no local features |
| Richards (2005, JFQA) "Big fish in small ponds" | Six Asian markets incl. Thailand, daily foreign flows | Foreign flows have a large price impact; part of it is temporary | Economic mechanism for a flow feature |
| Nagel (2012, RFS) "Evaporating liquidity" | US short-term reversal | Reversal profits = return to liquidity provision; higher when VIX is high | Explains *why* a daily reversal/flow signal could pay |
| Thai studies: Chula thesis on net foreign purchases vs SET returns (1997-2001, monthly); Chula study of daily investor-type flows (2015-2020, OLS, index level); Rattanaphongchirakun (2025) NVDR ownership vs crash risk, SET100 2005-2024 | Thailand | Foreign trading follows positive feedback; proprietary traders dominate the index; NVDR ownership is linked to crash risk in pooled OLS, but the link disappears with firm fixed effects | All **linear**, mostly **index-level or annual**, none forecast returns at a daily horizon |
| Kang (2026, arXiv 2601.07131, **withdrawn**) | Korea, stock-level investor-type flows, LSTM vs linear | Withdrawn because of **look-ahead bias / data leakage** | Cautionary example: daily flow-ML is easy to get wrong |
| López de Prado (2018) *Advances in Financial Machine Learning* | Method | Purged/embargoed CV, meta-labeling, triple barrier | Already in `trading_helper_ml` |

### Gap
1. ML cross-section studies of emerging markets are **monthly** and use **generic characteristics**. Nobody tests whether **Thailand-specific flow and microstructure signals** improve *daily* forecasts out of sample. Examples of such signals: stock-level NVDR net flow, market-level investor-type flows, distance to the foreign ownership limit, the TFEX SET50 futures basis, and proximity to the ±30% price limit.
2. Thai flow studies are linear and descriptive. None asks whether a flow signal is **information or liquidity provision** in a forecasting setting, with transaction costs.
3. The only close analogue (Korea) was withdrawn for leakage. A design that is **leakage-safe by construction** and fully reproducible is itself a contribution.

### Candidate thesis question A
> **Do local investor-flow signals improve machine-learning forecasts of daily SET100 returns, and is the gain compensation for liquidity provision or informed trading?**

- **H1 (predictive):** Adding flow features (NVDR flow, investor-type flow, foreign-limit proximity) to a baseline of price and characteristic features raises the out-of-sample R² and the long-short Sharpe ratio. Test: Clark-West / Diebold-Mariano for the forecasts, and bootstrap tests for Sharpe differences.
- **H2 (mechanism):** The gain comes from **reversal after foreign selling** and grows when VIX or USD/THB volatility is high (Nagel-type liquidity provision). Under the information alternative, the gain would come from **continuation after foreign buying**. Test: SHAP / ALE interactions and portfolio sorts by regime.
- **H3 (implementability):** The net-of-cost alpha survives only in the most liquid tercile. Costs are commission + VAT + half-spread, and positions are long-only or hedged with SET50 futures because short selling is restricted.
- Models: elastic net -> LightGBM -> small MLP, plus a ranking objective. Walk-forward with purged and embargoed splits.
- Data: yfinance / SET / SETSMART for prices and flows. Public daily NVDR reports show only top-30 lists, so (verify) whether the **full per-stock NVDR history** is available via SETSMART or a CMRI data request. This is the main data risk; check it first.
- Why it fits you: it extends `trading_helper_ml` (21-day horizon, generic features) to a daily, locally informed, hypothesis-driven design.

---

## Stream B - ML option pricing and deep hedging (-> SET50 options on TFEX)

### What is known
| Paper | Setting | Finding | Relevance |
| --- | --- | --- | --- |
| Buehler, Gonon, Teichmann & Wood (2019, Quantitative Finance) "Deep Hedging" | Simulated markets with frictions | Neural nets learn hedges that optimize a risk measure including transaction costs | Core method |
| Imaki et al. (2021) "No-Transaction Band Network" | Simulation | Architecture that learns a no-trade band; faster and more accurate | A natural fit for costly futures hedging |
| arXiv 2504.06208 (2025) deep hedging of S&P 500 index option portfolios | Realistic simulator of SPX returns + IV surface, trained 1996-2020, tested 2021-2023 | Beats delta-gamma hedging out of sample | The state of the art uses **only US data** |
| Deep hedging under frictions on Bitcoin options (arXiv 2608.29025, verify) | Deribit, 5 years | Compares BS delta, cost-aware benchmarks and deep hedging | Another liquid, global market |
| Horvath, Muguruza & Tomas (2021, QF) "Deep learning volatility"; Ruf & Wang (2020) literature review on neural nets for option pricing | Method | Neural nets for calibration and pricing | Toolkit |
| Conti & Morelli (2025/26, arXiv 2512.11731) "Transfer Learning (Il)liquidity" | Risk-neutral density from sparse quotes, tested on SPX | Recovers the density from as few as 3 quotes via transfer learning | Tested on **simulated illiquidity within SPX**, not a truly illiquid market |
| VolNP: meta-learning neural process for IV surfaces with SABR priors (arXiv 2509.11928) | Sparse quotes | SABR pre-training cuts RMSE by around 40% | Same idea |
| Thai SET50 option studies: GARCH vs historical vol for pricing low-liquidity calls (KJSS); binomial tree comparison (Chula 2009); NIDA (2025) volatility smile structure; studies on SET50 futures efficiency | TFEX | Classical models; smiles are described, not modelled with ML; futures market found inefficient | **No ML pricing or deep hedging on SET50** |

### Gap
1. Deep hedging is tested on **S&P 500, simulated data or crypto**. These are liquid markets with continuous hedging. SET50 options have few strikes, wide spreads and lumpy hedging through **SET50 futures contracts**: exactly the frictions deep hedging is meant to handle. No study tests it there.
2. Transfer learning for sparse option markets so far uses *artificially* thinned SPX data. Moving from a **liquid source (SPX) to a genuinely illiquid target (SET50)** is untested.

### Candidate thesis question B
> **Can deep hedging and transfer learning reduce hedging error and pricing error for SET50 index options relative to Black-Scholes and cost-aware classical hedges?**

- **H1:** Deep hedging (no-transaction-band net) using SET50 futures achieves lower CVaR of hedging P&L, net of costs, than BS delta, Leland, and Whalley-Wilmott hedging.
- **H2:** An IV surface model pre-trained on SPX (or SABR synthetic data) and fine-tuned on SET50 quotes fits and forecasts SET50 IVs better than SVI or a model trained only on SET50.
- **H3 (trading):** The gap between ML-forecast realized vol and IV predicts delta-hedged straddle returns (the RITC volatility case, on real data).
- Data risk: (verify) whether TFEX (or SETSMART) provides **intraday or end-of-day bid/ask by strike** for SET50 options back to around 2015. Daily settlement prices alone weaken H2.
- Why it fits you: a direct continuation of the RITC volatility and market-making bots. The method is more novel than in A, but the data risk is higher.

---

## Stream C - Variance risk premium and ML on index options (-> SPX/XSP, personal trading)

### What is known
- Carr & Wu (2009, RFS) and Bollerslev, Tauchen & Zhou (2009, RFS): the variance risk premium exists and the VRP predicts equity returns. Corsi (2009): HAR-RV forecasts.
- Carr, Wu & Zhang (arXiv 1909.10035, verify venue): ML for realized variance forecasting.
- Bali, Beckmeyer, Moerke & Weigert (2023, RFS) "Option Return Predictability with Machine Learning and Big Data": 12M+ delta-hedged **individual-stock** option returns, 1996-2020; option returns are more predictable than stock returns; profits above 2%/month after costs.
- Bandi, Fusari & Renò (2023) "0DTE Option Pricing" and Beckmeyer, Branger & Gayda (2023) "Retail Traders Love 0DTE Options... But Should They?": retail 0DTE trades lose money on average.

### Gap (smaller)
ML timing of **index** short-volatility trades (defined-risk put spreads) across tenors including 0DTE, evaluated under **retail constraints for a Thai investor** (broker limits, FX, tax on remitted foreign income). This is useful for trading, but the academic novelty is lower than in A or B. Best treated as the homework idea and a personal research project.

---

## Recommendation

| | A: SET100 daily ML + flows | B: SET50 deep hedging | C: SPX VRP timing |
| --- | --- | --- | --- |
| Novelty | High (local + mechanism) | High (method x market) | Medium-low |
| Data risk | Medium (NVDR per-stock history) | **High** (SET50 option quotes) | Low (Cboe free) / medium (option chains) |
| Finance contribution | Asset pricing + market microstructure | Derivatives / risk management | Empirical options |
| Builds on your code | `trading_helper_ml` | `ritc26` volatility / market making | `ritc26` volatility |

**Primary thesis: A**, with B as the alternative if deep SET50 option history can be obtained from TFEX. **Next step this week:**
check (1) per-stock NVDR or foreign-flow history for SET100 and (2) SET50 option bid/ask history by strike. Those two checks decide between A and B.
