# Module 01 - Introduction and Data Sources

**Homework:** [cohorts/2026/homework1.md](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework1.md)
**Submit at:** https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw01

> Topics: Data-driven decision making, financial APIs (yfinance, FRED), choosing data sources, index membership and macro context.

## How to run

From the repo root (the centralised venv is shared by every module):

```bash
make setup          # once, for the whole repo
make lab            # then open notebook.ipynb and pick the "Python (stock-markets-zoomcamp)" kernel
```

Or run the notebook without launching the UI:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace homework/01-intro-and-data-sources/notebook.ipynb
```

## Helpers available

This module's work leans on the shared package (see `src/smaz/`):

- `smaz.data.load_ohlcv`
- `smaz.data.load_fred`
- `smaz.data.load_sp500_constituents`

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
