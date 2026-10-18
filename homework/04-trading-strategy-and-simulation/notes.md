# Module 04 notes - Trading Strategy and Simulation

## Lecture takeaways

- **Precision and trade count must be read together.** A higher probability threshold means higher precision and
  fewer trades, but CAGR does not follow: RF 0.55 -> 1.129, 0.60 -> 1.062, 0.65 -> 1.009.
- **CAGR here is a capital-efficiency number, not a portfolio return.** `capital = 100 * 30 * Q75(daily trades)` and
  `CAGR = ((capital + net) / capital) ** (1/4)`. Sparse, selective strategies are penalised because the reserve is
  sized by the busy days; profit per trade is the cleaner efficiency measure.
- **Fees are small** (0.2% = $0.20 per $100 trade vs $3-4 average profit per trade) but not zero: a strategy that
  trades 10x more needs a 10x edge.
- **Redefining the target changes the trade count, not the edge.** Asking for > +2.2% instead of > 0% cut
  `pred5_clf_10` from 9.0 to 7.9 trades/day at the same CAGR (1.119 -> 1.120).
- Fewer dummies helped: dropping 93 `Ticker` / `month_wom` dummies (301 -> 208 features) lifted the best RF rule from
  1.095 to 1.129 and the depth-10 tree from 1.119 to 1.127.

## Gotchas hit while doing the homework

- **The Random Forest is not reproducible across scikit-learn versions** even with `random_state=42`: Q1 is 1.1291 on
  1.6/1.7 and 1.1189 on <=1.5 and >=1.8; Q2 flips between 0.57 and 0.59. Decision trees are stable. Pin the version
  and print it in the notebook.
- **The official `simulate` drops fees for rows whose future growth is NaN** (the last ~30 days of data), because
  `gross + fees` is NaN row-wise before summing. Replicate it when matching the course numbers.
- **The Colab notebook scales `X_test` in place in the logistic-regression cell** before the threshold section calls
  `predict_proba(X_test)`; only the `df_scores` table sees scaled data, the `pred*` columns use the unscaled `X_all`.
- `pd.get_dummies` is where categoricals must be dropped: the column is `Ticker` (capital T), the task text says `ticker`.
- Data lives in the gitignored `data/raw/`; 144 MB, download once with `gdown`.

## Ideas worth reusing in the capstone project

- Report **profit per trade and hit rate** next to CAGR; CAGR alone hides that a filter halved the edge per trade.
- Test "smart" trade filters (top-K by probability, model agreement, market-regime, volatility) against the plain
  threshold before adopting them - here none of them won.
- Keep the simulation as a function of the prediction column so a new model is one extra `pred*` column.
- Judge a signal by **CAPM alpha** as well as CAGR: for the RF signals ~80% of the 30-day return is market beta
  (3.4 of 4.3 points); stop-loss variants turned alpha negative.
- Leverage test needs **equity fixed**: with the course CAGR formula capital scales with exposure, so doubling positions
  changes nothing. 2x with a Fed funds + 1% borrow cost lifted CAGR 1.100 -> 1.168 (always) / 1.136 (only while trailing
  alpha > 0). CAGR ignores risk, so the alpha gate is the safety valve; drawdown/margin still need testing.
- Compute betas with `np.cov` on rows where *both* returns are present; masking only one series silently produced NaN
  betas for half the tickers in my first scratch version.
- A trailing Kelly fraction (win rate and win/loss ratio of the last ~200 *closed* trades) is a cheap regime switch:
  it skipped ~8% of days whose trades were clearly worse. Use it as a circuit breaker; as a sizing rule it did not raise CAGR.
- Check for a calendar effect before building on it: no Monday effect here, DCA over 5 days = lump sum.
- Fundamentals (P/E, FCF, sector) are not in the course dataset; do not back-fill today's values into a backtest.
- **The target is a bigger lever than any trading rule.** Rank-based target ("top third of the day") with walk-forward
  retraining beat the baseline at every selection share; a market-direction target ("goes up") was the worst. Pick the
  target that matches what you trade (stock selection), not what is easy to label.
- **CAGR with `capital = 30 x Q75(daily $)` rewards sparse strategies.** Always check net $ and the reserve next to
  CAGR, and compare at equal capital (here: raise the selection share until the reserve matches).
- Fixed 6-month walk-forward needs a gap (45 days) between train end and refit date, or the 30-day labels leak.
- `new_df` from the course pipeline has duplicate index labels; `reset_index(drop=True)` before assigning predictions back.
- One static fit can be lucky: the walk-forward version of the same "goes up" model scored 1.043 vs 1.103.
