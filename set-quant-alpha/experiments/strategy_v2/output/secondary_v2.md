# Secondary comparison on `set-quant-alpha/experiments/model_v2/output/predictions_v2.parquet` (label v2)

Frozen designs re-run unchanged; no selection on this file. DSR trials = 60 (+ the ML agent's model trials, not deflated here).

|  | CAGR % | Sharpe | max DD % | turnover 2-way x/yr | cost drag % | gross CAGR % | IR vs TDEX | CAPM alpha % | CAPM alpha t | FF3 alpha % | FF3 alpha t | C4 alpha % | C4 alpha t | TH alpha % | TH alpha t | TH b_SMB | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| valid / AB0 E1: top-10 weekly | 40.49 | 1.49 | -30.27 | 63.50 | 14.58 | 55.08 | 2.07 | 34.88 | 3.23 | 17.24 | 1.61 | 19.43 | 1.52 | 28.62 | 2.78 | 0.07 | 0.72 |
| valid / AB3 E2: buffer 30 + 8% stop | 34.93 | 1.59 | -23.32 | 37.55 | 8.49 | 43.42 | 1.71 | 30.32 | 2.95 | 14.43 | 1.11 | 15.36 | 1.01 | 26.41 | 2.67 | 0.12 | 0.78 |
| valid / D05 50% E2 + 50% residual-momentum sleeve (4-weekly) | 53.68 | 1.80 | -29.81 | 20.61 | 5.31 | 58.99 | 2.26 | 44.17 | 3.29 | 26.03 | 1.78 | 19.10 | 0.96 | 29.86 | 3.02 | 0.10 | 0.85 |
| valid / P-CS: pick as 20% sleeve + TDEX core (4% TE budget) | 9.44 | 0.53 | -30.61 | 4.02 | 0.61 | 10.04 | 2.36 | 8.81 | 3.59 | -6.61 | -0.60 | 14.00 | 1.40 | 6.29 | 3.41 | 0.03 | 0.24 |
| test / AB0 E1: top-10 weekly | 0.42 | 0.12 | -44.38 | 56.15 | 8.80 | 9.21 | -0.30 | -3.30 | -0.48 | -6.15 | -0.77 | -4.07 | -0.48 | -0.74 | -0.11 | 0.31 | 0.02 |
| test / AB3 E2: buffer 30 + 8% stop | -2.10 | -0.05 | -40.16 | 30.94 | 4.72 | 2.62 | -0.56 | -5.68 | -0.89 | -10.44 | -1.22 | -8.03 | -0.82 | -3.86 | -0.62 | 0.18 | 0.01 |
| test / D05 50% E2 + 50% residual-momentum sleeve (4-weekly) | -9.23 | -0.44 | -57.56 | 17.76 | 2.53 | -6.70 | -1.09 | -14.36 | -2.27 | -16.97 | -1.73 | -16.95 | -1.53 | -12.45 | -2.07 | 0.46 | 0.00 |
| test / P-CS: pick as 20% sleeve + TDEX core (4% TE budget) | 3.77 | 0.35 | -28.24 | 3.34 | 0.46 | 4.24 | -0.95 | -2.29 | -1.87 | -4.90 | -0.61 | -2.39 | -0.28 | -1.87 | -1.61 | 0.09 | 0.07 |
