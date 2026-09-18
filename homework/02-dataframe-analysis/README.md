# Module 02 - Working with the Data (in Pandas)

**Homework:** [cohorts/2026/homework2.md](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework2.md)
**Submit at:** https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw02

> Topics: Data types, feature engineering with technical indicators, cleaning strategies and descriptive statistics.

## How to run

From the repo root (the centralised venv is shared by every module):

```bash
make setup          # once, for the whole repo
make lab            # then open notebook.ipynb and pick the "Python (stock-markets-zoomcamp)" kernel
```

Or run the notebook without launching the UI:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace homework/02-dataframe-analysis/notebook.ipynb
```

## Helpers available

This module's work leans on the shared package (see `src/smaz/`):

- `smaz.features.build_feature_set`
- `smaz.features.add_ta_indicators`
- `smaz.utils.describe_frame`

## Files

| File | Purpose |
| --- | --- |
| `notebook.ipynb` | Working notebook -- exploration and the code behind each answer |
| `answers.md` | Final answers only, in submission order |
| `notes.md` | Lecture notes and anything worth carrying into the capstone project |

## Status

- [ ] Watched the lectures
- [x] Notebook runs top-to-bottom without errors
- [x] `answers.md` filled in
- [ ] Submitted on the course portal
