# Which information could add independent alpha? (research note, 2026-11-02)

**Goal.** The price-based signal has a test IC of about 0.050. To beat the factor models at retail costs it needs IC of
roughly 0.066 (break-even) to 0.097 (+1% a year); see README section 12, finding 5. So any new data must carry
information that is **not** already in prices and known factors: incremental IC of roughly +0.02 to +0.05.

Ranked by expected value × feasibility. "Thai evidence" means studies on the SET itself.

| # | Information | Thai evidence | Data source and access | Point-in-time | Fit with this project |
| --- | --- | --- | --- | --- | --- |
| 1 | **NVDR trading by stock** (net buying by mostly-foreign NVDR holders) | NVDR order imbalances predict daily returns, and an uninformed trader could profit from them (Likitapiwat & Johnson, 2016) | SET NVDR statistics, "Trading by Stock" (web view); history via **SETSMART** (investor package about 5 years, longer in professional packages) or the **SMART Marketplace** API (infoproducts@set.or.th) | published after the close; lag 1 day | **highest**. The hook exists (`data/external/nvdr_by_stock.csv`, `events.nvdr_features`). It also tests the thesis question: is the reversal alpha pay for providing liquidity to foreign sellers? |
| 2 | **Insider trades** (directors' and executives' Form 59 reports to the SEC) | Insider buys and sells are followed by abnormal returns (transaction data 2009-12). Effects are larger for top executives, and leakage persists despite the 3-day reporting rule | SEC Thailand website (Form 59-2 reports); check the SEC's open-data services before collecting | report date + 1 day (not trade date) | high: event-driven, unrelated to price patterns, free. Needs a new hook |
| 3 | **Trading by investor type** (foreign, institution, proprietary, retail; whole market) | Foreign order-flow innovations predict future returns across Thai FX, equity and bond markets; foreigners trade momentum and locals provide liquidity (Krohn, Sushko & Synsatayakul, BIS WP 1154, 2023). Older Thai studies find causality mostly from returns to flows | SET website (daily); SMART Marketplace API (JSON); SETSMART | after the close; lag 1 day | medium: market-level, so mainly a **timing / regime** input and an interaction with stock betas to the baht and EM. The hook exists (`set_investor_type.csv`) |
| 4 | **Analyst forecast revisions** | Revision strategies earn abnormal returns in Asia-Pacific emerging markets; strongest in Korea, India, Taiwan (S&P research). Thai coverage is thin | I/B/E/S (Refinitiv) or Bloomberg; **paid** (university terminals) | revision date | medium-high if the data is accessible. It extends the EPS-surprise finding (events_v1: weak but orthogonal) |
| 5 | **Foreign ownership room** (foreign holding vs the foreign limit) | Less room left means stronger foreign demand and a higher foreign-share premium; NVDRs made foreign demand more elastic (Chulalongkorn theses) | SET company profiles (snapshot); SETSMART for history | as published | low-medium: slow-moving, close to a style characteristic |
| 6 | **Investor attention** (Google search volume) | High search volume predicts high returns for 1-2 weeks, then a reversal; Thailand appears in panel studies at market level | Google Trends (unofficial clients; Thai company names, noisy) | weekly | low-medium: cheap to test, fits the retail-heavy SET |
| 7 | **Market-surveillance events** (cash-balance lists, trading alerts) | Cash-balance announcements show small *positive* abnormal returns (2011-13, 146 events) | SET news / surveillance pages (manual) | announcement date | a **risk filter** for the worst blow-ups rather than alpha |
| 8 | **Short sales / securities lending** | Shortability does not speed up price adjustment in Thailand, because of high fees and scarce lendable supply | SET daily short-sales statistics | after the close | low |
| 9 | **Tax-saving fund season** (LTF → SSF / Thai ESG) | December inflows go mostly to SET50 large caps, and have shrunk since LTFs ended (press reports) | calendar | known in advance | low: month dummies already in the model |

## Recommended order

1. **Get history.** Ask the university library / thesis advisor for SETSMART, Bloomberg or Refinitiv access, or ask SET
   Information Services (infoproducts@set.or.th) about SETSMART / SMART Marketplace educational terms. This one step
   unlocks #1, #3, #4 and #5.
2. **Start forward collection today.** Copy NVDR trading by stock and investor-type trading into the CSV templates
   (`docs/templates/`) after each close. This is point-in-time by construction and runs alongside the paper shadows.
3. **Insider trades (#2)** are the best free source to build next: a Form 59 hook plus an event study on 2013-21.
4. Every candidate goes through the same protocol: pre-registered, decided on 2013-21, test period for information
   only. Report its IC, its Fama-MacBeth slope added to the model score, and its correlation with the score; then a
   new paper shadow if it passes. Count every test in the trial ledger.

## Sources

* Likitapiwat, T. & Johnson, W. F. (2016). Market segmentation in developing markets: evidence from Thai non-voting
  depository receipts. *Journal of Applied Economic Sciences*. <https://aquila.usm.edu/fac_pubs/19551>
* Krohn, I., Sushko, V. & Synsatayakul, W. (2023). Foreign investor feedback trading in an emerging financial market.
  BIS Working Paper 1154. <https://econpapers.repec.org/RePEc:bis:biswps:1154>
* Rattanaphongchirakun, P. (2025). Disaggregating foreign investor influence: direct holdings, NVDRs, and stock price
  crash risk in the Thai equity market. *Procedia of Multidisciplinary Research* 3(11).
  <https://so09.tci-thaijo.org/index.php/PMR/article/view/7893>
* SEC Thailand working paper on insider trading. <https://www.sec.or.th/TH/Documents/SECWorkingPapersForum/working-paper-forum-25570917-insider.pdf>
* Corporate governance and insider trading: evidence from Thailand (Chulalongkorn thesis).
  <https://doi.nrct.go.th/ListDoi/listDetail/10.14457%2FCU.the.2009.1852>
* Foreign share premium and NVDRs (Chulalongkorn thesis). <https://cuir.car.chula.ac.th/xmlui/handle/123456789/52243>
* Earnings revision strategies in Asia (S&P Dow Jones Indices). <https://www.indexologyblog.com/2018/02/08/earnings-revision-strategies-in-asia/>
* Analyst forecast revisions and asset allocation in Asia Pacific markets (ANU).
  <https://researchportalplus.anu.edu.au/en/publications/analyst-forecast-revisions-and-asset-allocation-in-asia-pacific-m/>
* Investor attention and stock market activities: new evidence from panel data (MDPI, 2019). <https://www.mdpi.com/2227-7072/7/2/30>
* Stock price impacts from the cash balance measure (Mahidol). <https://archive.cm.mahidol.ac.th/bitstream/123456789/739/1/TP%20FM.006%202557.pdf>
* Short sales and speed of price adjustment on the SET (Chulalongkorn thesis). <https://cuir.car.chula.ac.th/xmlui/handle/123456789/36503>
* SET NVDR statistics. <https://www.set.or.th/en/market/statistics/nvdr/main>
* SETSMART for investors (SET). <https://media.set.or.th/set/Documents/2023/Jun/setsmart_for_investor_th.pdf>
* SMART Marketplace (SET). <https://set.or.th/en/services/connectivity-and-data/data/smart-marketplace>
* Tax-saving fund inflows (Bangkok Post). <https://www.bangkokpost.com/business/1943676/fund-inflows-into-extra-ssf-units-estimated-at-b10bn>
