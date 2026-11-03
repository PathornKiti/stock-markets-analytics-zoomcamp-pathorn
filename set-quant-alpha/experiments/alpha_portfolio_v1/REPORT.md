# alpha_portfolio_v1 - factor diagnostics, correlation control, portfolio construction, robustness (2026-11-02)

Code: `alpha_portfolio_v1.py` (stages `diag`, `corr`, `build`, `robust`, plus the post-hoc check `c3check`). Tables are
in `output/diag.md`, `output/corr.md`, `output/tables.md` and `output/check_c3.md`. The protocol was frozen in
`output/protocol_frozen.txt` at 2026-11-02T16:05:17Z, before any code of this experiment ran.

**Verdict: does not improve on S5 under the pre-registered gates.** The new walk-forward strategy (AP) earned
**9.98% a year net against S5's 7.38% and TDEX's 7.16%** (exact lots, 0.168% per side, 2016-01-04 to 2026-10-22).
Its Sharpe was 0.73 against 0.57. All of that edge came in 2016-21. In 2022-26 AP trailed S5 by 0.43 points a year,
so the time-stability gate G3 fails. The Sharpe difference is not significant (bootstrap p 0.13), and DSR is 0.29 at
734 trials. **No S7 shadow is proposed.**

What the process did improve is the evidence. Every signal now has a full diagnostic (Stage 1), the signal
correlation is measured and controlled (Stage 2), and the cost-aware constructions were tested and shown not to help
here (Stage 3).

## Decision table (gates frozen before any simulation)

| Gate | Pass | Value |
| --- | --- | --- |
| G1 beats S5: net CAGR and Sharpe | yes | 9.98% vs 7.38%; Sharpe 0.73 vs 0.57 |
| G2 beats TDEX: CAGR and IR > 0 | yes | 9.98% vs 7.16%; IR 0.20 |
| **G3 time-stable: AP − S5 > 0 in 2016-21 and in 2022-26** | **no** | **+5.09 / −0.43 points a year** |
| G4 neighbour-stable: ≥ 2/3 of the 2026 pick's neighbours beat S5 | yes | 5 of 5 |
| G5 PBO < 0.5 (Sharpe, 24 configs, 2013-26) | yes, barely | 0.495 (0.75 on 2016-26 alone) |
| G6 Sharpe(AP) > Sharpe(S5) significant, p < 0.05 | no | difference 0.16, 95% CI −0.10 to +0.46, p 0.13 |
| G7 DSR > 0.95 | no | 0.29 at 734 trials (S5: 0.14) |

* **"Improves"** required G1-G5 to pass, and G3 fails.
* **"Significant / capital-ready"** required G1-G7 to pass, and G3, G6 and G7 fail.

## Out-of-sample result, 2016-01-04 → 2026-10-22 (walk-forward, exact 100-share lots, THB 1M)

| | AP (new) | S5 deployed record | TDEX |
| --- | --- | --- | --- |
| Net CAGR | **9.98%** | 7.38% | 7.16% |
| Gross CAGR / cost drag / two-way turnover | 12.78% / 2.79% / 15.5× | 9.52% / 2.14% / 11.2× | |
| Sharpe / max drawdown | 0.73 / −36.4% | 0.57 / −34.9% | 0.53 / −36.5% |
| IR vs TDEX / tracking error | 0.20 / 12.6% | 0.01 / 12.3% | |
| CAPM α (t), beta | +5.6% (1.53), 0.62 | +3.2% (0.93), 0.62 | |
| FF3-EM α (t) | +2.7% (0.65) | +1.2% (0.26) | |
| Carhart-EM α (t) | +7.4% (1.67) | +5.2% (1.08) | |
| Local-Thai Carhart α (t) | +4.5% (1.22) | +1.9% (0.55) | |
| Local-Thai Carhart, factors also charged costs | +5.6% (1.54) | +2.8% (0.82) | |
| Local b_SMB / b_WML | 0.07 / 0.10 | 0.01 / 0.10 | |
| PSR / DSR (734 trials) / MinTRL | 0.99 / 0.29 / 5.4 yrs | 0.96 / 0.14 / 8.6 yrs | |
| HLZ (Bonferroni, 734 trials) | Sharpe t 2.34 vs hurdle 3.98: haircut to 0 | t 1.83: haircut to 0 | |

* **Simulator checks.**
  * The harness reproduces integrated_v1's fast simulator for the S5 rule to 1e-8.
  * It reproduces the S5 deployed record exactly, in both the fast and the exact-lot versions.
  * AP's fast and exact daily returns correlate 0.9999.
* **No significant alpha.** Every alpha is positive but none is significant. The net local-Thai alpha more than
  doubles against S5 (+4.5% vs +1.9%) while size and momentum loadings stay close to zero.

### Walk-forward picks and where the difference comes from

| Year | AP pick | S5 record pick | AP net | S5 net | TDEX | AP − S5 |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | WFC EW N10 | C3 N10 monthly EW | 26.4% | 14.7% | 23.7% | **+11.6** |
| 2017 | WFC EW N10 | C3 N10 monthly EW | 21.9% | 17.6% | 18.7% | +4.4 |
| 2018 | WFC EW N10 | C3 N10 monthly EW | −2.6% | 0.9% | −4.6% | −3.5 |
| 2019 | WFC EW N10 | C3 N10 monthly EW | 14.5% | 14.5% | 6.2% | +0.1 |
| 2020 | WFC EW N10 | C3 N10 monthly EW | −8.1% | −23.2% | −11.0% | **+15.1** |
| 2021 | C2 TILT N10 | C2 TILT N10 | 25.5% | 26.7% | 13.4% | −1.3 |
| 2022 | C2 TILT N20 | C2 TILT N20 | 2.5% | 2.5% | 9.0% | 0.0 |
| 2023 | C2 TILT N10 | C2 TILT N10 | 9.9% | 9.8% | −8.4% | 0.0 |
| 2024 | C2 TILT N10 | C2 TILT N10 | −14.3% | −14.4% | 9.1% | +0.1 |
| 2025 | WFC EW N10 | C3 N10 monthly EW | 9.0% | 8.0% | −4.2% | +1.0 |
| 2026 | C2 IV N10 | C2 EW N10 | 29.3% | 33.1% | 30.8% | −3.8 |

* **2016-20: the composite made the difference.** AP held WFC, which in those years was the Löwdin-orthogonalised
  IC-weighted blend of all 8 signals (K4, about 35% ML). The S5 record held the low-risk literature book (C3).
  * 2016 and 2020 alone account for most of the gap.
  * In 2020 C3 lost 23% in the COVID year, while AP lost 8%.
* **2021-24: the same strategy.** Both processes picked the same C2 score-tilted book.
* **2026: AP was worse.** It picked inverse-vol weighting over S5's equal weight and lost 3.8 points.
* **The 2016-20 gap is not just C3 being left out of the grid** (post-hoc check D3).
  * C3 was added under all 12 constructions and the walk-forward re-run. It picked exactly the same configs every year.
  * In every selection window, WFC EW N10 had a higher net α t than any weekly C3 book.
  * Caveat: the S5 record's C3 pick rebalanced monthly, which the weekly-only grid cannot reproduce.

### Sub-periods and regimes (net, exact)

| | AP | S5 | TDEX | AP − S5 |
| --- | --- | --- | --- | --- |
| 2016-19 CAGR / Sharpe | 15.0% / 1.47 | 12.1% / 1.25 | 10.8% / 0.94 | +2.9 |
| 2020-21 | 7.7% / 0.46 | −1.4% / 0.03 | 0.5% / 0.14 | +9.1 |
| 2016-21 | 12.5% / 0.88 | 7.4% / 0.58 | 7.3% / 0.51 | **+5.1** |
| 2022-26 | 6.9% / 0.53 | 7.3% / 0.56 | 7.0% / 0.57 | **−0.4** |
| HMM calm / normal / stressed, AP − S5 % a year (NW t) | | | | +2.3 (1.1) / +1.6 (0.7) / +4.4 (0.7) |

### Cost sensitivity (same schedule; TDEX 7.16%)

| Cost per side | 0.08% | 0.10% | 0.125% | 0.15% | **0.168%** | 0.20% | 0.25% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP net CAGR (fast) | 11.7% | 11.3% | 10.9% | 10.4% | **10.1%** | 9.6% | 8.7% |
| S5 record | 8.6% | 8.4% | 8.1% | 7.8% | **7.6%** | 7.2% | 6.6% |
| AP CAPM α t | 1.91 | 1.83 | 1.72 | 1.62 | **1.54** | 1.40 | 1.19 |

* The AP − S5 gap holds at every cost level, at 2.2 to 3.0 points a year.
* Re-running with the cost inside the decisions gives identical results, because AP never picked a cost-aware
  construction.

### Robustness of the selection

* **Bootstrap.** A stationary bootstrap (mean block 21 days, 5,000 draws) gives a Sharpe difference vs S5 of 0.16,
  95% CI −0.10 to +0.46, one-sided p 0.13. Blocks of 5 and 63 days give p 0.13 and 0.14.
  * AP vs TDEX: difference 0.19, p 0.25.
  * S5 vs TDEX: difference 0.04, p 0.46.
* **Neighbourhood of the 2026 pick (C2 IV N10) as fixed 2016-26 designs:**
  * N5 17.6%, N15 12.0%, buffer 2N 15.0%, buffer 4N 13.8%, rebalance every 10 days 12.3%.
  * All five beat S5's 7.55% (fast).
  * These fixed designs are in-sample by construction, because C2 was chosen with data through 2025. The surface
    shows stability, not out-of-sample skill.
* **PBO (CSCV, 16 blocks, the 24 configs).** 0.495 on Sharpe and 0.50 on α t over 2013-26; 0.75 on 2016-26.
  The in-sample winner's out-of-sample Sharpe degrades with slope −0.90.
* **Capacity (setalpha.risk, square-root impact, k = 1).**
  * AP trades more (15.5× vs 11.2×), so it pays about 1.5× S5's impact at any AUM.
  * At THB 50M both net about 3.0%: AP's advantage is gone.
  * With k = 0.5, AP still nets 6.5% vs S5's 5.2% at THB 50M.
  * The k = 1 model charges even tiny trades about 0.06% each, so the absolute levels are conservative.
* **Return decomposition (local factors, % a year, 2016-26).**
  * Net return 10.6 = market 5.2 + style 0.9 + specific alpha 4.5 for AP.
  * S5: 8.2 = 5.2 + 1.1 + 1.9.

## Stage 1 - factor diagnostics (2013-26, tradable top-100; descriptive, nothing selected here)

| Signal | IC 5d (NW t) | IC 20d | IC 2022-26 | D10−D1 % per 5d (t) | Lag-5 rank autocorr | Breakeven cost per side, weekly / monthly | FM slope t: raw → + styles → + sector | R² on styles + sector |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml | 0.072 (16.2) | 0.065 | 0.050 | 0.73 (8.5) | 0.55 | 0.26% / 0.41% | 8.9 → 9.9 → 10.2 | 0.54 |
| v8 | 0.059 (13.6) | 0.073 | 0.057 | 0.66 (6.9) | 0.79 | 0.35% / 0.67% | 7.7 → 8.1 → 7.4 | 0.53 |
| lowbeta | 0.039 (6.9) | 0.054 | 0.029 | 0.24 (2.3) | 0.99 | 0.27% / 0.55% | 1.6 → −0.4 → −0.4 | 0.66 |
| lowvol | 0.052 (9.7) | 0.070 | 0.052 | 0.25 (2.2) | 1.00 | 0.56% / 1.20% | 2.3 → 1.9 → 2.0 | 0.77 |
| resmom | 0.018 (2.9) | 0.032 | 0.012 | 0.22 (1.9) | 0.98 | 0.34% / 0.69% | 2.4 → 1.5 → 0.5 | 0.92 |
| hi52 | 0.040 (6.6) | 0.064 | 0.036 | 0.48 (4.1) | 0.93 | 0.25% / 0.51% | 3.8 → 1.6 → 1.4 | 0.61 |
| eps | 0.020 (4.5) | 0.022 | 0.026 | 0.17 (2.0) | 0.95 | 0.35% / 0.38% | 3.0 → 2.8 → 2.7 | 0.32 |
| divy | 0.038 (8.4) | 0.060 | 0.056 | 0.31 (3.5) | 0.99 | 0.75% / 1.17% | 3.5 → 2.8 → 2.6 | 0.44 |

Styles are size (log traded value), beta, 12-1 momentum, 60-day volatility and illiquidity (Amihud), plus sector
fixed effects. A signal's own characteristic is not used as its control.

* **H1: styles explain the literature signals but not ML, V8, EPS surprise or dividend yield.**
  * lowbeta, resmom and hi52 lose significance once styles are controlled.
    * lowbeta flips sign.
    * resmom is 0.94 rank-correlated with plain 12-1 momentum.
  * lowvol is borderline: t 1.9 with styles, 2.0 with sectors.
  * ML and V8 slopes *rise* with controls (t 8.9 → 9.9 and 7.7 → 8.1). Their neutral ICs are 0.051 and 0.038.
  * EPS surprise and dividend yield keep t 2.6-2.8.
  * The pre-registered flag (an OR rule) also flags ML and V8, because their R² on styles plus 11 sector dummies is
    0.53-0.54. That flag measures exposure, not whether the styles explain the alpha. On the slope test they are
    clearly not style beta.
* **IC decay.** All signals keep predicting beyond day 5.
  * The marginal IC over days 6-10 and 11-20 is 0.035-0.055 for ML, V8, lowvol, hi52 and dividend yield (t 6-10).
  * ML is the fastest: IC20 / IC5 is 0.92, against 1.24 for V8 and 1.35-1.83 for the literature signals.
* **ML decay over time.** ML's IC fell from 0.095 (2013-15) to 0.050 (2022-26).
  * V8 (0.06), lowvol (0.05) and dividend yield (0.036 → 0.056) held up.
  * hi52 and resmom have no IC in HMM-stressed regimes (−0.014, −0.001). ML and V8 are strongest in stressed regimes
    (0.068-0.072).
* **Costs.** ML's weekly top-decile book turns over 64% a week, so its breakeven cost is only 0.26% per side against
  the actual 0.168%.
  * The slow signals have breakevens of 0.55-1.2% per side, but their spreads are small.
  * Dividend yield has the best net weekly spread: t 2.5, 6.5% a year before construction losses.

## Stage 2 - correlation control

* **Correlation structure.**
  * ML and V8 overlap: daily-IC correlation 0.69, long-short return correlation 0.66, score correlation 0.61.
  * lowbeta and lowvol form a second block (0.71 / 0.84 / 0.74).
  * hi52 and resmom are partly linked (IC correlation 0.56).
  * Across all 8 signals, the effective number of independent bets is 5.4 by IC series, 4.8 by long-short returns and
    5.8 by cross-sectional scores (entropy measure).
  * C2's 50/50 ML+V8 blend has a diversification ratio of only 1.05-1.09. K1-K6 reach 1.37-2.03.
* **Composite comparison (weights walk-forward; 2016-26 out of sample for the weights).**

| Composite | IC5 (t) | Weekly turnover | Net top-decile spread % per week (t) | 2022-26 net spread (t) |
| --- | --- | --- | --- | --- |
| C1 ML | 0.065 (13.0) | 62% | 0.094 (1.4) | 0.020 (0.2) |
| C2 ML+V8 (S5) | 0.067 (13.6) | 48% | 0.155 (2.5) | 0.039 (0.4) |
| K1 equal 8 | 0.063 (11.1) | 18% | 0.227 (4.1) | 0.321 (4.1) |
| K2 IC-weighted | 0.065 (11.8) | 24% | 0.182 (3.5) | 0.241 (2.9) |
| K3 ICIR, shrunk to equal | 0.065 (11.7) | 21% | 0.216 (4.1) | 0.279 (3.4) |
| K4 Löwdin + IC | **0.068 (13.7)** | 42% | 0.127 (2.3) | 0.175 (2.0) |
| K5 MV on the IC covariance (Ledoit-Wolf) | **0.069 (13.6)** | 39% | 0.131 (2.4) | 0.247 (3.0) |
| K6 cluster-equal | 0.058 (10.4) | **18%** | **0.259 (4.5)** | **0.338 (4.3)** |
| WFC (walk-forward pick) | 0.069 (13.4) | 35% | 0.184 (3.5) | 0.285 (3.7) |

* **H2 partly supported.**
  * K4 and K5 raise the IC slightly.
  * The large gain is in the *cost-adjusted* top-decile spread. It comes from the diversified, slower composites
    (K1, K3, K6): 0.22-0.26% per week net against C2's 0.155%, and 0.28-0.34% vs 0.04% in 2022-26.
* **WFC choices.** The walk-forward composite picked K4 for 2014-20, C2 for 2021-22 and K6 for 2023-26.

## Stage 3 - construction (24 configs, weekly, 2016-26 diagnostic; selection used only 2013..Y−1)

| Construction (median of its configs) | Net CAGR | Gross | Sharpe | CAPM α t | Turnover | Picked |
| --- | --- | --- | --- | --- | --- | --- |
| EW top-N + buffer (S5 style) | 12.0% | 13.6% | 0.84 | 2.15 | 8× | 6 of 11 years |
| Score-tilted | 12.1% | 15.5% | 0.84 | 2.09 | 17× | 4 |
| Inverse-vol | 11.0% | 12.6% | 0.84 | 2.14 | 8× | 1 |
| HYS: cost-aware alpha hysteresis | 13.8% | 19.1% | 0.78 | 1.87 | 27× | 0 |
| MVO: Grinold α, Ledoit-Wolf, cost term, caps | 9.2% | 13.0% | 0.63 | 1.25 | 19× | 0 |

* **H3 not supported. The cost-aware constructions never won the walk-forward selection.**
  * **MVO** was the worst family on net CAGR, Sharpe and α t.
    * Size/beta bands never had to be dropped, and all 5,400 solves converged.
    * Its cost term does real work: without it (ablation A2), turnover rises from 22× to 42× and net CAGR falls from
      9.9% to 7.0% (λ = 3).
    * Diversifying (λ = 30, about 24-34 names) cost 3-5 points a year, as strategy_v3 found.
  * **HYS** had the highest hindsight return (WFC HYS N10: 15.7%, Sharpe 0.90) but deep drawdowns (C2 HYS N10:
    −55%).
    * Before 2016 its α t was poor, so it was never picked.
    * The cause is Grinold scaling, α ∝ σ·z: without a risk term, an equal-weight book drifts to volatile names.
      Held 60-day volatility is 2.4% a day vs 1.6% for EW and 2.3% for the universe.
* **Ablations (fast, net 2016-26).**

| Ablation | Net CAGR | Sharpe | CAPM α t |
| --- | --- | --- | --- |
| A1: 15% vol target | 9.4% | 0.71 | 1.50 |
| A2: MVO without cost term (λ 3 / λ 30) | 7.0% / 5.7% | 0.51 / 0.50 | 0.69 / 0.63 |
| A3: IR objective | 9.9% | 0.67 | 1.35 |
| A4: score fixed to C2 (construction effect only) | 8.8% | 0.63 | 1.17 |

## Trials

| Item | Count |
| --- | --- |
| Composites K1-K6 + WFC | 7 |
| Construction grid | 24 |
| Walk-forward meta-strategy | 1 |
| Ablations A1-A4 | 5 |
| Neighbours | 5 |
| Post-hoc C3 check (D3): 12 configs + its walk-forward | 13 |
| **Total** | **55** |

* `validity.TRIALS["alpha_portfolio_v1"] = 55`, so `N_TRIALS` = **734**.
* The Bonferroni hurdle at 734 trials is t = 3.98.

## Deviations from the protocol

* **D1 - days with no scores.** Five days (2021-12-24 to 30, the prediction-split embargo gap) have no scored names.
  * HYS and MVO keep the book unchanged on those days.
  * The EW/TILT/IV family keeps integrated_v1's behaviour (it sells if a decision falls there), so S5 replicates
    exactly.
* **D2 - bugs fixed before results were read.**
  * Composites were first stored as float32. This changed tie-breaks, so the harness did not reproduce
    integrated_v1. They are now stored as float64, and the stage-2 weights and WFC choices are unchanged.
  * Two crashes in `build` stopped it before any walk-forward result: an empty candidate set, and a tuple-indexing
    bug.
  * The G4 summary text miscounted. The gate result itself did not change.
* **D3 - post-hoc check, not in the protocol.** C3 was added under all 12 constructions to test whether excluding it
  created AP's edge. It did not change any pick. The 13 trials take the total to 55, above the "at most 45" in the
  protocol, and are counted in the DSR.
* **D4 - H1 flag.** The pre-registered rule (slope t test **or** R² > 0.5) flags ML and V8 on R² alone. Both are
  reported as flagged, with the interpretation above.
* **D5 - unused options.** The statistical factor-model covariance was not used; Ledoit-Wolf only, as frozen. No
  TDEX leg was used, so the 0.05% TDEX cost never applied: A1's remainder is held in cash.

## Caveats

* **Unequal grids.** AP is compared with S5's walk-forward record, which came from integrated_v1's 288-config grid.
  That grid included monthly rebalancing and no-buffer books; this one is weekly-only.
* **Shared caveats.** Survivorship bias, ML hyperparameters tuned on 2020-21, and the well-known 2022-26 period
  (integrated_v1) apply equally to AP and S5.
* **Static sector map.** The Yahoo sector map is today's classification. It affects only the MVO sector caps and the
  sector fixed effects in Stage 1.
* **Effect on re-running integrated_v1.** integrated_v1 computes `PRIOR_TRIALS` as
  `N_TRIALS − integrated_v1 − retrain_v1`. If it is re-run, it will now count these 55 trials as "earlier" (144
  instead of 89). That lowers its DSR slightly and mislabels the count. integrated_v1 was not modified.

## Recommendation

**Keep S5 as deployed, and register no new shadow.**

* The new process beat S5 only in 2016-21, by choosing a better signal blend than the literature book.
* In the regime that matters for live trading, 2022-26, it converged to S5's own design and did slightly worse.
* The cost-aware optimiser did not beat simple top-N books here.

Two findings are worth carrying forward as research, not as a trading change. Each should be pre-registered as its own
test and judged on future data:

1. **The slow composites.** K1, K3 and K6 have twice C2's cost-adjusted top-decile spread, including in 2022-26 (t 3-4
   vs 0.4). The construction grid did not capture this, because the portfolio objective picked C2's books on 2013..Y−1
   α t.
2. **A risk-aware hysteresis rule.** For example, Grinold α / σ², or HYS with a volatility cap, could keep HYS's high
   gross return without the volatility tilt.

## Reproduce

    cd ~/stock-markets-analytics-zoomcamp-pathorn
    for s in diag corr build c3check robust; do uv run python set-quant-alpha/experiments/alpha_portfolio_v1/alpha_portfolio_v1.py $s; done
    uv run pytest set-quant-alpha/tests -q

`diag` and `corr` take about 20 seconds, `build` about 2.5 minutes, `c3check` about 1.5 minutes and `robust` about 10
seconds. Inputs are integrated_v1's cached `output/panel.joblib`, `grid.joblib` and `wf_daily.parquet`, plus
`data/processed/dataset.parquet` and `data/raw/{prices,sectors}.parquet`.
