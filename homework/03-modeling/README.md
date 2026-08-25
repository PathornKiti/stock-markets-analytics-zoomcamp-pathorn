# Module 03 - Analytical Modeling

**Homework:** [cohorts/2026/homework3.md](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework3.md)
**Submit at:** https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw03

> Topics: Time-series prediction, trend decomposition, regression, binary classification of price direction, optional neural networks.

## How to run

From the repo root (the centralised venv is shared by every module):

```bash
make setup          # once, for the whole repo
make lab            # then open notebook.ipynb and pick the "Python (stock-markets-zoomcamp)" kernel
```

Or run the notebook without launching the UI:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace homework/03-modeling/notebook.ipynb
```

## Helpers available

This module's work leans on the shared package (see `src/smaz/`):

- `smaz.features.make_binary_target`
- `algotrade.model.time_split`
- `algotrade.model.fit_classifier`

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
