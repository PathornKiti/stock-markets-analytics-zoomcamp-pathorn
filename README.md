# Stock Markets Analytics Zoomcamp — Pathorn

Homework and capstone project for the [DataTalksClub Stock Markets Analytics Zoomcamp](https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp), **2026 cohort**.

One centralised virtual environment is shared by every module and by the project, so a helper written in week 1 is importable from week 5 without copy-paste or `sys.path` hacks.

[![CI](https://github.com/PathornKiti/stock-markets-analytics-zoomcamp-pathorn/actions/workflows/ci.yml/badge.svg)](https://github.com/PathornKiti/stock-markets-analytics-zoomcamp-pathorn/actions/workflows/ci.yml)

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/PathornKiti/stock-markets-analytics-zoomcamp-pathorn.git
cd stock-markets-analytics-zoomcamp-pathorn

make setup            # .venv + all deps + editable install + Jupyter kernel + git hooks
cp .env.example .env  # add a free FRED API key

make lab              # JupyterLab; pick the "Python (stock-markets-zoomcamp)" kernel
```

`make setup` pins Python 3.12 via `.python-version` and installs from `uv.lock`, so the environment is byte-identical on any machine — including a peer reviewer's.

## Layout

Module folders mirror the upstream course repo, so `homework/03-modeling/` lines up with the course's `03-modeling/`.

```
.
├── src/smaz/                 # shared library, importable everywhere
│   ├── config.py             # repo-root-relative paths, .env secrets
│   ├── data.py               # yfinance / FRED / Wikipedia loaders + parquet cache
│   ├── features.py           # returns, moving averages, volatility, TA indicators
│   ├── backtest.py           # CAGR, Sharpe, Sortino, max drawdown, fee-aware simulation
│   └── utils.py              # plotting defaults, frame profiling
│
├── homework/
│   ├── 01-intro-and-data-sources/
│   ├── 02-dataframe-analysis/
│   ├── 03-modeling/
│   ├── 04-trading-strategy-and-simulation/
│   └── 05-deployment-and-automation/
│       ├── README.md         # links to the question set + submission form
│       ├── notebook.ipynb    # the working notebook
│       ├── answers.md        # final answers, submission-ready
│       └── notes.md          # lecture notes
│
├── project/                  # capstone — see project/README.md for the rubric checklist
│   ├── notebooks/            # 01_eda → 05_report
│   ├── src/algotrade/        # ingest → transform → model → strategy
│   ├── scripts/run_pipeline.py
│   └── Dockerfile
│
├── data/                     # shared cache — gitignored
├── tests/
└── docs/
```

## Why one environment

Each module builds on the last: module 2's features feed module 3's models, which feed module 4's simulation, which module 5 automates. A per-week venv would mean re-implementing the same helpers five times and re-resolving the same heavy dependency tree.

So instead: one `.venv` at the repo root, one `uv.lock`, and the repo installed editable. `import smaz` resolves from any notebook in any folder.

```python
from smaz import backtest, data, features

prices = data.load_ohlcv(["AAPL", "MSFT"], start="2015-01-01")  # cached to data/cache/
df = features.build_feature_set(prices)
backtest.summary(df.query("Ticker == 'AAPL'").set_index("Date")["ret_1d"])
```

Network calls are cached to parquet on first use, so re-running a notebook top-to-bottom doesn't re-hit rate-limited APIs.

## Common commands

| Command | What it does |
| --- | --- |
| `make setup` | Full one-time environment setup |
| `make lab` | Launch JupyterLab |
| `make test` | Run pytest |
| `make lint` / `make fmt` | Ruff check / format + autofix |
| `make pipeline` | Run the capstone end-to-end offline |
| `make airflow-deps` | Install the optional Airflow extras (module 5) |
| `make clean` | Clear caches, keep `.venv` and data |

Anything else: prefix with `uv run` (e.g. `uv run python project/scripts/run_pipeline.py --mode live`) — no manual activation needed.

## Progress

| Module | Topic | Homework | Submitted |
| --- | --- | --- | --- |
| 1 | Intro and data sources | [notebook](homework/01-intro-and-data-sources/) | ☐ |
| 2 | Working with the data (pandas) | [notebook](homework/02-dataframe-analysis/) | ☐ |
| 3 | Analytical modeling | [notebook](homework/03-modeling/) | ☐ |
| 4 | Trading strategy and simulation | [notebook](homework/04-trading-strategy-and-simulation/) | ☐ |
| 5 | Deployment and automation | [notebook](homework/05-deployment-and-automation/) | ☐ |
| — | Capstone project | [project/](project/) | ☐ |

## Conventions

- **No data in git.** `data/` and `project/data/` are gitignored — the project rubric requires it.
- **No secrets in notebooks.** Keys live in `.env`; read them with `smaz.config.get_secret`.
- **Notebook outputs are stripped on commit** by an `nbstripout` pre-commit hook, keeping diffs reviewable.
- **`uv.lock` is committed** so the environment is reproducible.
- **Chronological splits only.** Random splits on time series train on the future; `algotrade.model.time_split` and the `lag` in `backtest.simulate_signal` exist to prevent that.

## Docs

- [`docs/setup.md`](docs/setup.md) — environment setup and troubleshooting
- [`docs/workflow.md`](docs/workflow.md) — the weekly homework loop
- [`project/README.md`](project/README.md) — capstone write-up and scoring checklist

## License

MIT — see [LICENSE](LICENSE).
