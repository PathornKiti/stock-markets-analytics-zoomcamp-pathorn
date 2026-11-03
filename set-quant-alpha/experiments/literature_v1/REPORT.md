# literature_v1 - published market-beating strategies on Thai stocks (2026-11-02)

Run: `uv run set-quant-alpha/experiments/literature_v1/literature_v1.py {select,test}` (outputs in `output/`).

## Question

Do strategies that research papers report as beating the market also beat TDEX and the CAPM / Fama-French /
Carhart models on the SET after costs?

## Designs (pre-registered; parameters from the papers, not tuned)

| | Strategy | Paper | Score |
| --- | --- | --- | --- |
| L1 | Low beta | Frazzini & Pedersen (2014); Thai low-risk anomaly (AAMJAF 2017) | −120-day beta |
| L2 | Low volatility | Blitz & van Vliet (2007); Blitz, Pang & van Vliet (2012) | −250-day volatility |
| L3 | Residual momentum | Blitz, Huij & Martens (2011) | 12-1 month residual return / its volatility |
| L4 | 52-week high | George & Hwang (2004) | price / 250-day high |
| L5 | Multi-factor | composite | average rank of L2, L3, L4 |
| L6 | Multi-factor + ML | composite | average rank of L5 and the ML score |

**Rules for all designs:**
* Top-100 liquidity universe; long the top quintile (20 names), equal weight.
* Rebalance monthly (21 trading days). A held name stays while it ranks in the top 40 (buy/hold buffer; Novy-Marx &
  Velikov 2016).
* Next-open fills, 100-share lots, 0.15% per side, THB 1M.

**Selection:** highest 2013-21 worst-case net alpha t across CAPM, local-Thai Carhart and FF-EM Carhart. Frozen
2026-11-02T09:17Z, then 2022-26 was run once. The trial count is now 89.

**Caveat:** which papers to test was chosen knowing that 2022-26 favoured megacaps / low risk.

## Results

| | 2013-21 CAGR | CAPM α (t) | local Carhart α (t) | **2022-26 CAGR** | **CAPM α (t)** | **local Carhart α (t)** | cost drag | turnover |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TDEX | 3.8% | – | – | 6.8% | – | – | – | – |
| L1 low beta | 8.3% | +5.5 (1.88) | +4.1 (1.46) | 2.3% | −1.2 (−0.38) | −0.9 (−0.27) | 0.3% | 2.3× |
| L2 low volatility | 5.2% | +2.3 (1.06) | +2.1 (0.95) | 3.0% | −1.0 (−0.30) | −0.1 (−0.02) | 0.2% | 1.2× |
| L3 residual momentum | 7.4% | +5.0 (0.81) | −3.2 (−0.76) | −4.5% | −9.4 (−1.63) | −9.3 (−1.73) | 0.7% | 3.4× |
| L4 52-week high | 15.2% | +11.9 (2.47) | +7.0 (1.74) | −1.9% | −5.9 (−1.47) | −6.3 (−1.61) | 0.6% | 3.8× |
| **L5 multi-factor (pick)** | **12.8%** | **+9.6 (2.87)** | **+6.3 (2.19)** | **2.3%** | **−1.8 (−0.53)** | **−1.9 (−0.57)** | 0.3% | 2.3× |
| L6 multi-factor + ML | 10.8% | +7.7 (2.27) | +4.7 (1.56) | 0.3% | −3.6 (−0.99) | −3.5 (−0.95) | 0.6% | 3.1× |

All DSRs ≤ 0.29 (2013-21) and ≤ 0.06 (2022-26). Turnover is two-way, × the portfolio per year.

## Findings

1. **Monthly, buffered factor strategies solve the cost problem.** Turnover is 1-5× a year and cost drag 0.1-1.0%,
   against 27× and 4.2% for E2.
2. **They do not solve the alpha problem.**
   * The pick beat TDEX by 9% a year in 2013-21 (CAPM t 2.9), as the papers predict.
   * It lost by 4.5% a year in 2022-26, with alpha about zero or negative everywhere.
   * Momentum and 52-week high turned clearly negative. Low beta and low volatility were about zero, with lower drawdowns.
3. **This is the published-anomaly pattern:** strong in history, weak after publication and in new regimes (McLean &
   Pontiff 2016; Hou, Xue & Zhang 2020). The 2013-21 numbers also carry survivorship bias.
4. **Contrast with E2:** the ML signal has gross alpha (+3% a year) but high costs. The published factors have low
   costs but no alpha in 2022-26. Neither passes the launch gates.

## Decision

No design passes the launch gates. Nothing is registered as a new paper shadow: the 2022-26 failure gives no reason
to expect forward success. If one were tested forward, low beta / low volatility (about zero alpha, lower drawdowns)
would be the candidates, as defensive TDEX alternatives rather than alpha strategies.
