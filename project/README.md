# Capstone Project - <your project title>

> Replace this template as you go. The headings below map 1:1 onto the
> [official evaluation criteria](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/projects/README.md),
> so a peer reviewer can find every point without hunting.
> Passing is **6 points**; the maximum is **36 + 7 bonus**.

## Problem description

<!-- Criterion 1 (4 pts). What are you predicting, for whom, and what decision does it drive?
     Go past "predict weekly stock returns" if you want the novelty point. -->

## Data sources

<!-- Criterion 2 (4 pts). List every source, the date range, the row count, and what each
     feature means. A non-yfinance/FRED source is worth a point; >1M rows is worth another. -->

| Source | What it provides | Range | Rows |
| --- | --- | --- | --- |
| Yahoo Finance (`yfinance`) | Daily OHLCV | | |
| FRED (`pandas-datareader`) | Macro series | | |
| | | | |

## Data transformations and EDA

<!-- Criterion 3 (3 pts). The combined dataframe, the engineered features, and the
     correlation / distribution analysis. -->

## Modeling

<!-- Criterion 4 (5 pts). Which models, how they were tuned, how they were validated.
     Splits must be chronological. -->

## Trading simulation

<!-- Criterion 5 (8 pts) -- the largest block. Report CAGR, Sharpe and max drawdown for
     every strategy, and put the benchmark in the same table. -->

| Strategy | CAGR | Sharpe | Max DD | Notes |
| --- | --- | --- | --- | --- |
| | | | | |
| SPY buy & hold (benchmark) | | | | |

## Automation

<!-- Criterion 6 (5 pts). How to run it, how it is scheduled, and how it loads new data. -->

## Reproducing this project

```bash
git clone https://github.com/PathornKiti/stock-markets-analytics-zoomcamp-pathorn.git
cd stock-markets-analytics-zoomcamp-pathorn
make setup                                          # builds the shared .venv from uv.lock

cp .env.example .env                                # add a free FRED API key
uv run project/scripts/run_pipeline.py --mode live  # fetch data and run end-to-end
```

Offline replay (no API keys, uses whatever is already in `project/data/`):

```bash
uv run project/scripts/run_pipeline.py --mode file
```

Or with Docker:

```bash
docker build -t algotrade -f project/Dockerfile .
docker run --rm --env-file .env algotrade
```

## Layout

```
project/
├── notebooks/            # 01_eda -> 05_report, in execution order
├── src/algotrade/        # the pipeline as an installed package
│   ├── ingest.py         # step 1: pull raw data (file | live regimes)
│   ├── transform.py      # step 2: join sources, engineer features
│   ├── model.py          # step 3: chronological splits, train, evaluate
│   └── strategy.py       # step 4: signals -> positions -> P&L
├── scripts/run_pipeline.py   # single automation entry point
├── data/                 # gitignored -- the rubric forbids committing data
└── models/               # gitignored artefacts
```

## Scoring self-check

Tick these off before submitting; each line is worth a point.

**Problem description (4)**
- [ ] Brief description in the README *(required to pass)*
- [ ] Context and a clear end goal
- [ ] Problem is novel, not the default weekly-return strategy
- [ ] Every step described well enough to reproduce

**Data sources (4)**
- [ ] Uses the lecture sources and features *(required to pass)*
- [ ] 20+ new described features
- [ ] A new data source beyond yfinance/FRED
- [ ] More than 1 million records

**Transformations + EDA (3)**
- [ ] One combined dataframe with defined feature sets *(required to pass)*
- [ ] 5+ generated features
- [ ] Correlation / exploratory analysis

**Modeling (5)**
- [ ] One lecture model *(required to pass)*
- [ ] Several models compared
- [ ] Custom decision rules for high-probability events
- [ ] Hyperparameter tuning
- [ ] An advanced model (XGBoost / regression / NN)

**Trading simulation (8)**
- [ ] Vector simulation of at least one strategy *(required to pass)*
- [ ] Two or more strategies
- [ ] Exact simulation with reinvestment and capital sizing
- [ ] CAGR, Sharpe and max drawdown reported
- [ ] Risk management in the best strategy
- [ ] A novel strategy (long-short, concurrent positions, regime-aware)
- [ ] Beats a realistic benchmark (S&P 500)
- [ ] Per-ticker/market breakdown with improvement ideas

**Automation (5)**
- [ ] Scripts exported from notebooks; a master notebook runs the workflow *(required to pass)*
- [ ] Dependency management with install instructions
- [ ] Cron-able for fresh predictions
- [ ] Dual regime (file-based and live)
- [ ] Incremental loading with external storage

**Bonus (up to 7)**
- [ ] Modular, commented code
- [ ] Broker API integration
- [ ] Dashboard or notification bot
- [ ] Containerised
- [ ] Deployed to the cloud
- [ ] Peer-reviewer discretionary bonus

**Peer review** — evaluate 3 projects (+3 points each).
