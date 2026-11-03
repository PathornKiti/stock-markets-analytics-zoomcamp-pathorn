# Post-hoc diagnostics (after the single test pass; not used for any selection)

## 1. Does validation rank predict test rank across the 17 strategy_v2 designs?

|  | Spearman rho (valid vs test, 17 designs) | p-value |
|---|---|---|
| Sharpe | -0.60 | 0.01 |
| IR vs TDEX | -0.41 | 0.10 |
| CAPM alpha t | -0.50 | 0.04 |
| TH alpha t | -0.43 | 0.09 |
| C4 alpha t | -0.09 | 0.74 |
| CAGR % | -0.45 | 0.07 |
| turnover 2-way x/yr | 0.91 | 0.00 |
| TH b_SMB | 0.48 | 0.05 |

## 2. Frozen pick D05 decomposed: momentum sleeve alone vs E2

|  | CAGR % | Sharpe | turnover 2-way x/yr | CAPM alpha % | TH alpha % | TH alpha t | TH b_SMB | TH t_SMB | TH b_WML | TH t_WML | avg liq rank (w) | vs EW top-100 % p.a. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| valid / diag: residual-momentum sleeve alone (4-weekly, buffer 30) | 73.58 | 1.65 | 6.32 | 59.68 | 34.20 | 2.36 | 0.06 | 0.57 | 1.04 | 15.37 | 52.65 | 43.05 |
| valid / AB3 E2: buffer 30 + 8% stop | 19.88 | 1.02 | 40.07 | 18.54 | 12.42 | 1.21 | 0.10 | 1.32 | 0.21 | 4.25 | 50.97 | -0.95 |
| test / diag: residual-momentum sleeve alone (4-weekly, buffer 30) | -14.12 | -0.48 | 6.57 | -20.07 | -18.18 | -2.21 | 0.74 | 10.45 | 0.28 | 5.29 | 51.97 | -6.71 |
| test / AB3 E2: buffer 30 + 8% stop | 1.61 | 0.18 | 26.91 | -1.49 | -1.05 | -0.20 | 0.13 | 2.54 | 0.03 | 0.98 | 37.57 | 7.67 |
