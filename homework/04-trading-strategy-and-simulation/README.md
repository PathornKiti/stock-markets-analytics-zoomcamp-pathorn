# Module 04 - Trading Strategy and Simulation

**Homework:** [cohorts/2026/homework4.md](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework4.md)
**Submit at:** https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw04

> Topics: Strategy design, risk management, fees, diversification, market-neutral approaches and vectorised simulation.

## How to run

From the repo root (the centralised venv is shared by every module):

```bash
make setup          # once, for the whole repo
make lab            # then open notebook.ipynb and pick the "Python (stock-markets-zoomcamp)" kernel
```

Or run the notebook without launching the UI:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace homework/04-trading-strategy-and-simulation/notebook.ipynb
```

## Helpers available

This module's work leans on the shared package (see `src/smaz/`):

- `smaz.backtest.simulate_signal`
- `smaz.backtest.compare`
- `smaz.backtest.max_drawdown`

## Files

| File | Purpose |
| --- | --- |
| `notebook.ipynb` | Working notebook -- exploration and the code behind each answer |
| `answers.md` | Final answers only, in submission order |
| `notes.md` | Lecture notes and anything worth carrying into the capstone project |

## Status

- [ ] Watched the lectures
- [ ] Notebook runs top-to-bottom without errors
- [ ] `answers.md` filled in
- [ ] Submitted on the course portal
