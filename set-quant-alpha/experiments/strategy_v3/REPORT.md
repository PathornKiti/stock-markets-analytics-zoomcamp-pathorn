# strategy_v3 - fixing the four leaks between the signal and factor-adjusted alpha (2026-11-02)

Run: `uv run set-quant-alpha/experiments/strategy_v3/v3.py {preds,select,test}`. Outputs are in `output/`.

## The leaks and the fixes

| Leak | Evidence | Fix in v3 |
| --- | --- | --- |
| 1. Costs | Top-decile alpha is +0.27% per 5 days; a round trip costs 0.30% | Costs inside the optimiser: trade only when expected alpha beats cost |
| 2. Construction adds factor tilts | Equal-weight top-10 has b_SMB of +0.12 to +0.24 against a megacap SET50 | TDEX core + stock tilts, with local MKT/SMB/WML loadings held at TDEX's (1, 0, 0) ± 0.05 |
| 3. Too few bets | 10 names, so a low transfer coefficient | Grinold alpha / residual-risk sized positions (about 25-35 names, max 5% each) |
| 4. Test too short | 4.8 years; alpha t > 2 needs IR ≈ 0.9 | Walk-forward predictions back to 2013. Designs selected on 2013-2021 (9 years); 2022-26 is a one-shot test |

## Design

* **Optimiser** (weekly, SLSQP), maximising

  α'w − λ/2 Σ σ²_ε w² − κ · cost(|Δw|)

  subject to:
  * the book is fully invested and long-only;
  * w ≤ 5% per stock;
  * |active loading| ≤ 0.05 on each of MKT, SMB and WML.
* **Inputs:**
  * α: Grinold 5-day alpha = IC 0.051 × residual vol × z(score).
  * Loadings: rolling 250-day OLS on the local factors.
  * Costs: 0.15% per side on stocks, 0.05% on TDEX.
* **Execution:** at the next open, in 100-share lots.
* **Capital:** THB 20M, so lots don't block 1-3% tilts. E1 and E2 are re-run at the same capital.
* **λ calibration:** uses ex-ante risk only, no returns. It is set to the median ex-ante tracking error of the cost-free optimum.
  * With the factor bands, 8% TE is not reachable: λ hits its floor of 1 and the realised ex-ante TE is about 6.4%.
* **Pre-registered:** 3 pickable designs (F1-F3) and 2 ablations (A1, A2).
* **Selection rule:** highest 2013-2021 local-Thai Carhart alpha t among E2, F1, F2 and F3.
* **Frozen:** 2026-11-02T04:10:43Z, before any test simulation.
* **DSR trials:** 60 earlier + 7 here.

## Leak 4: the longer history (IC of the walk-forward ensemble score)

| 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .092 | .104 | .091 | .095 | .109 | .084 | .068 | .046 | .057 | .064 | .019 | .046 | .051 | .077 |

* The IC is positive in all 14 years.
* It roughly **halved after 2019**. This is alpha decay, a common fate of published price-based anomalies.

## Selection period 2013-2021

| | CAGR | cost drag | turnover | CAPM α (t) | Carhart FF-EM α (t) | local-Thai α (t) | b_SMB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TDEX | 3.8% | – | – | – | +2.2 (0.42) | – | – |
| E1 | 15.1% | 12.1% | 66× | +11.8 (2.55) | +5.7 (0.90) | +8.1 (1.91) | 0.06 |
| **E2 (pick)** | **16.0%** | 6.1% | 35× | **+12.9 (2.97)** | +6.3 (0.95) | **+10.2 (2.51)** | 0.12 |
| F1 neutral, TE 4% | 7.3% | 1.5% | 9× | +3.6 (1.84) | +2.1 (0.40) | +2.3 (1.21) | 0.00 |
| F2 neutral, TE ~6% | 10.3% | 2.8% | 17× | +6.6 (2.07) | +2.6 (0.45) | +4.6 (1.49) | 0.02 |
| F3 all-stock neutral | 9.7% | 2.9% | 17× | +6.2 (1.84) | +2.0 (0.34) | +4.1 (1.27) | 0.02 |
| A1 = F2 without bands | 15.6% | 3.9% | 22× | +12.0 (2.48) | +3.2 (0.48) | +6.9 (1.66) | 0.11 |
| A2 = F2 cost-unaware | 9.7% | 7.8% | 45× | +6.1 (1.85) | +4.2 (0.72) | +4.1 (1.28) | 0.02 |

In 2013-2021 the concentrated E2 book earned significant CAPM and local-Thai alpha. Neutralising size and momentum
removed about half of it, so the rule picked E2.

## Test 2022-01 → 2026-10 (one shot)

| | CAGR | gross | cost drag | turnover | IR vs TDEX | CAPM α (t) | FF3-EM α (t) | Carhart FF-EM α (t) | local-Thai α (t) | b_SMB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TDEX | 6.4% | – | – | – | – | – | −3.0 (−0.38) | +0.1 (0.02) | – | – |
| E1 | −0.3% | 9.0% | 9.3% | 60× | −0.36 | −4.0 (−0.63) | −6.4 (−0.74) | −3.1 (−0.33) | −3.0 (−0.48) | 0.24 |
| **E2 (pick)** | **1.9%** | 6.2% | 4.3% | 27× | −0.30 | −1.3 (−0.25) | −4.7 (−0.56) | −1.6 (−0.18) | −0.9 (−0.17) | 0.12 |
| F1 | 5.1% | 6.3% | 1.3% | 8× | −0.30 | −1.2 (−0.60) | −3.6 (−0.46) | −0.6 (−0.07) | −1.1 (−0.55) | 0.03 |
| *F2* | *5.6%* | *8.0%* | *2.3%* | *14×* | *−0.07* | *−0.5 (−0.13)* | *−2.2 (−0.25)* | *+0.6 (0.07)* | *−0.5 (−0.13)* | *0.06* |
| F3 | 4.2% | 6.4% | 2.2% | 14× | −0.20 | −1.7 (−0.41) | −3.1 (−0.37) | −0.3 (−0.03) | −1.5 (−0.36) | 0.07 |
| A1 (no bands) | −0.9% | 2.3% | 3.2% | 21× | −0.56 | −6.1 (−1.17) | −4.6 (−0.54) | −2.3 (−0.25) | −4.8 (−0.93) | 0.25 |
| A2 (cost-unaware) | 3.9% | 10.4% | 6.6% | 41× | −0.24 | −2.1 (−0.54) | −3.8 (−0.45) | −1.2 (−0.13) | −2.1 (−0.53) | 0.06 |

*Italic* = not the pick; it is shown for information only. All DSRs ≤ 0.09. (TDEX's own local regression is degenerate
because MKT is TDEX itself.)

## What the fixes did

* **Leak 1, costs: fixed.**
  * The cost-aware optimiser cuts turnover from 41× to 14× and cost drag from 6.6% to 2.3% (F2 vs A2), at the same
    signal and risk settings.
  * Net CAGR gains about 1.8 pp a year on test and 0.6 pp in 2013-21.
* **Leak 2, factor tilts: fixed.** b_SMB falls from 0.12-0.25 to 0.03-0.07 and the beta-to-TDEX moves to about 1.
  * On test, neutrality turns A1's −0.9% into F2's +5.6%.
  * In 2013-21 it cost about 5 pp, because small-cap and momentum exposure *paid* in those years.
  * Neutrality removes a bet; it does not add alpha.
* **Leak 3, too few bets: diversifies but dilutes.**
  * Alpha is concentrated in the top decile (the decile chart is convex), so 25-35 Grinold-sized names earn less gross
    alpha than the top 10.
* **Leak 4, short history: fixed, and it changes the story.**
  * The signal and E2 *did* earn significant alpha in 2013-21: CAPM t 3.0, local-Thai t 2.5.
  * 2022-26 is a decayed-signal regime, not proof that the method never worked.
  * The Carhart FF-EM alpha is not significant in either period (t 0.95 in 2013-21), so part of E2's earlier edge was
    momentum (b_WML 0.22).

## Bottom line

The fixes closed the gap to TDEX from about −4.5% a year (E2) to −0.8% a year (F2, information only), with near-zero
factor loadings. **No design earns significant factor alpha in 2022-26.**

After neutralisation the remaining gross alpha is about +1.5% a year (F2 gross 8.0% vs TDEX 6.4%). Its break-even cost
is about **0.11% per side**, below the 0.15% assumed. Winning against the factor models needs **more style-free gross
alpha**, from information that isn't price-based (foreign flows, NVDR, earnings events), and/or **institutional
costs** of ≤ 0.10% per side. Optimisation alone is not enough.

Do not switch production to F2 on this evidence: it was not the pre-registered pick.
