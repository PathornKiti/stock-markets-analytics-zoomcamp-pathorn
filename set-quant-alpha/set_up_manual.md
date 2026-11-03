# Set-up manual: from `git clone` to a running strategy

This guide takes a fresh machine to a working copy of set-quant-alpha:

* data downloaded;
* models trained;
* backtests run;
* this week's S5 stock picks and the dashboard produced.

For what the strategy does, see [strategy.md](strategy.md); for how well it did, see [result.md](result.md).

**Time needed.**
* About 10 minutes of your time, plus about 20 minutes of computer time on the first run.
* Daily updates afterwards take 1-2 minutes.

## 0. What you need

| Requirement | Notes |
| --- | --- |
| macOS or Linux | Windows works through WSL2 (Ubuntu) |
| `git` | any recent version |
| [uv](https://docs.astral.sh/uv/) | `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh \| sh`. uv installs the right Python (3.12) by itself |
| About 3 GB free disk | about 0.8 GB for the Python environment, about 1.3 GB for data and features |
| Internet | for the first data download (Yahoo Finance, FRED, Cboe, Ken French library). No API keys needed |
| *Optional:* Docker Desktop | only for the Airflow scheduler and dashboard containers (section 6) |
| *Optional:* LSEG Workspace + licence | for the LSEG research data (sections 6.1 and 7). Not needed for the strategy |

On macOS, XGBoost needs the OpenMP library. If you later see `XGBoost Library could not be loaded ... libomp`, run
`brew install libomp`.

## 1. Clone and install

```bash
git clone https://github.com/PathornKiti/stock-markets-analytics-zoomcamp-pathorn.git
cd stock-markets-analytics-zoomcamp-pathorn
make setup
```

`make setup` does the following:
* creates one virtual environment at the repo root (`.venv`) from the locked versions in `uv.lock`;
* installs the project;
* installs the pre-commit hooks;
* registers a Jupyter kernel called "smaz".

**No `make`?** Run `uv sync` instead.

**Running commands.** Every command below is prefixed with `uv run`, so you never need to activate the environment.
Run them all from the repo root (`stock-markets-analytics-zoomcamp-pathorn/`).

## 2. Optional settings (`.env`)

Everything works without this step. To change a default, copy the example file and edit it:

```bash
cp set-quant-alpha/.env.example set-quant-alpha/.env
```

| Setting | Use |
| --- | --- |
| `SETALPHA_DATA_DIR` | keep the data store outside the repo, e.g. a synced Google Drive folder |
| `LSEG_APP_KEY` | LSEG Data Library app key. With it, Airflow's `set_quant_alpha_lseg` DAG also collects LSEG data (section 6.1) |

`.env` is gitignored. Never commit keys.

## 3. First run

A fresh clone contains code only. Data, trained models and reports are not in git (no data files are committed), so
the first run downloads everything from the free sources and rebuilds it.

```bash
uv run set-quant-alpha/scripts/run_pipeline.py --mode live                 # download, build, train, backtest, report
uv run set-quant-alpha/experiments/integrated_v1/publish_deployed.py       # S5 walk-forward record for the dashboard
uv run set-quant-alpha/scripts/trial_returns.py                            # research-trial returns (overfitting checks)
uv run set-quant-alpha/scripts/run_pipeline.py --steps report              # rebuild the dashboard with both
```

The Airflow weekly DAG (section 6) runs exactly these steps.

The first command runs every pipeline step in order:

| Step | What it does | First run |
| --- | --- | --- |
| `ingest` | downloads daily prices for about 850 Thai stocks plus market, FRED, Cboe and Fama-French data | about 3 min |
| `transform` | builds the universe and the 120 inputs per stock per day | a few min |
| `train` | trains the models walk-forward and refits the production models, using the frozen design: settings from the committed `models/best_params.json`, ensemble = logistic + XGBoost (`config.ENSEMBLE`) | about 15 min |
| `simulate` | runs the backtests | a few min |
| `signals` | writes this week's picks | seconds |
| `paper` | updates the live paper-trading ledger (shadows S1-S6) | seconds |
| `report` | builds the dashboard | seconds |

The download is incremental. If it stops halfway (for example Yahoo rate-limits you), run the same command again and
it continues.


## 4. Check that it worked

```bash
open set-quant-alpha/reports/dashboard.html          # Linux: xdg-open
head set-quant-alpha/reports/latest_signals.csv      # this week's S5 picks
uv run pytest set-quant-alpha/tests                  # 24 unit tests, all should pass
```

You should see:
* the dashboard, with Performance, Factors and "Trust & risk" tabs;
* a signals file listing about 10 tickers with their scores;
* `24 passed` from the tests.

## 5. Day-to-day use

| Task | Command | When |
| --- | --- | --- |
| Daily update: new prices, signals, paper trading, dashboard | `uv run set-quant-alpha/scripts/run_pipeline.py --mode live --steps ingest transform signals paper report` | after the SET close, Mon-Fri (18:30 Bangkok is safe) |
| Weekly retrain and backtest | `uv run set-quant-alpha/scripts/run_pipeline.py --steps train simulate report` | weekends |
| Replay from local data (no network) | `uv run set-quant-alpha/scripts/run_pipeline.py --mode file` | any time |
| Redo the hyperparameter search | add `--retune` to the train command. This changes the frozen S5 design, so count it as a new trial | research only |
| Rebuild research-trial returns (for overfitting checks) | `uv run set-quant-alpha/scripts/trial_returns.py` | optional, about 20 s |

**Automate without Docker (cron).** Add this line with `crontab -e`:

```cron
30 18 * * 1-5 cd /path/to/stock-markets-analytics-zoomcamp-pathorn && .venv/bin/python set-quant-alpha/scripts/run_pipeline.py --mode live --steps ingest transform signals paper report >> set-quant-alpha/reports/cron.log 2>&1
```

## 6. Automate with Docker (Airflow + dashboard)

This path needs no local Python: the containers have their own environment. A fresh clone is enough.

```bash
cd set-quant-alpha
cp .env.example .env                                  # optional; add LSEG_APP_KEY here for section 6.1
docker compose up -d --build                          # first build takes a few minutes
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt   # admin password
```

* **Airflow** is at <http://localhost:8080>. Log in as `admin` with the password above, then follow these steps in
  order.

  | Step | DAG | What it does | Schedule afterwards |
  | --- | --- | --- | --- |
  | 1 | `set_quant_alpha_weekly` | **Trigger it once** (▶ button). On a fresh clone it downloads all Yahoo / FRED / Cboe / Fama-French data, builds the inputs, trains the models (frozen S5 design), runs the backtests, publishes the S5 track record and builds the dashboard (about 20-30 min) | Saturday 06:00 Bangkok |
  | 2 | `set_quant_alpha_daily` | **Unpause** it once step 1 is green. It adds each day's prices, writes the next-open orders, runs paper trading and refreshes the dashboard | Mon-Fri 18:30 Bangkok |
  | 3 | `set_quant_alpha_lseg` | *Optional*, see 6.1 | Mon-Fri 19:30 Bangkok |

  Unpausing the daily DAG starts the live paper-trading record. That record is append-only, so it is a one-way
  step.
* **The dashboard** is at <http://localhost:8000> and always shows the latest report. A tour of its six tabs, with screenshots, is in
  [README section 9.2](README.md#92-dashboard-tour).
* **One run without Airflow:** `docker compose run --rm pipeline --mode live`.
* **Stop:** `docker compose down`. Data, models and reports stay in the host folders, so the next start continues where it
  left off.

### 6.1 LSEG data through Airflow (optional)

1. **Add the key.** Put `LSEG_APP_KEY=<your key>` in `set-quant-alpha/.env`.
2. **Recreate the containers.** Run `docker compose up -d`. The key is read when a container starts, so after any
   change to `.env` the containers must be recreated, not just restarted.
3. **Start Workspace.** Start **LSEG Workspace** on the same computer and log in. It must be running when the DAG runs.
4. **Unpause the DAG.** In Airflow, unpause `set_quant_alpha_lseg`. Trigger it once to start straight away.

What happens when it runs:

| Situation | Result |
| --- | --- |
| No key in `.env`, or Workspace not running | the task shows **skipped** (pink), not failed, and nothing changes |
| First run with an empty LSEG store | full history pull: SET50/SET100 membership incl. delisted stocks, prices, foreign board, I/B/E/S and fundamentals (a few hours, mostly prices) |
| Later runs | incremental: new prices and foreign-board days; new index members in full; the monthly estimate and fundamental groups refreshed once a week |
| Invalid key | the task **fails**, with the LSEG error in its log |

**How the container reaches Workspace.**
* Workspace only accepts connections addressed to `localhost:9000`.
* Inside the container, `src/setalpha/lseg.py` starts a small relay on `localhost:9000` that forwards to the host
  (`host.docker.internal`), so nothing needs configuring.

**Licensing.**
* The data lands in `data/raw/lseg/`, which is gitignored.
* The daily strategy does not use it. It feeds the research experiments in `experiments/` only.

## 7. Re-running the research (optional)

Each experiment in `set-quant-alpha/experiments/<name>/` has:
* a `REPORT.md`, with the exact commands under "Reproduce";
* a frozen protocol in `output/protocol_frozen.txt`.

| Experiments | Needs |
| --- | --- |
| `integrated_v1`, `literature_v1`, `strategy_v2`-`v4`, `model_v2`, `retrain_v1`, `alpha_portfolio_v1`, `events_v1`, `dividends_v1` | the free data from step 3 |
| `survivorship_v1`, `foreign_v1`, `fundamentals_v1`, `lseg_signals_v1`, `adaptive_v1` | LSEG Workspace running on the same machine, `LSEG_APP_KEY` in `.env`, and `uv sync --group lseg`. Then `uv run --group lseg python set-quant-alpha/scripts/fetch_lseg.py members missing prices foreign fundamentals extra` |

**Rules for the LSEG experiments.**
* LSEG data is licensed. It stays in `data/raw/lseg/`, which is gitignored. Do not share
  it.
* Some experiments have a `holdout` stage that refuses to run a second time. That is intentional, because the
  out-of-sample check may only be computed once.

## 8. Troubleshooting

| Problem | Fix |
| --- | --- |
| `FileNotFoundError: No local data in .../raw` | you used `--mode file` on an empty data store: run once with `--mode live` |
| `uv: command not found` | install uv (section 0) and open a new terminal |
| `XGBoost Library could not be loaded` (macOS) | `brew install libomp` |
| Download stops or some tickers fail | Yahoo rate limit: wait a few minutes and run the same command again (it is incremental) |
| `make setup` fails at `pre-commit install` | you are not inside the cloned git folder; `cd` into it. Or skip it with `uv sync` |
| Port 8080 or 8000 already in use | stop the other service, or change the left-hand port in `set-quant-alpha/docker-compose.yml` (e.g. `"8081:8080"`) |
| Airflow password not shown | `docker compose logs airflow \| grep -i password` again after about a minute; Airflow standalone prints it once it is ready |
| `set_quant_alpha_lseg` shows skipped | expected without a key or a running Workspace; the task log says which. Add the key to `.env`, run `docker compose up -d`, start Workspace |
| `Only "127.0.0.1" and "localhost" are allowed` | a script reached Workspace from Docker without the relay: run it through `setalpha.lseg.session()` with `LSEG_PROXY_HOST` set (docker-compose sets it) |
| Results differ slightly from `result.md` | the model design is frozen, but a fresh Yahoo download differs a little from the stored history (re-adjusted prices, a changing stock screener) and grows every day. The research results in `result.md` are fixed in `experiments/*/output/` and do not change |

## 9. Where things are

| Path | Contents |
| --- | --- |
| `set-quant-alpha/src/setalpha/` | the library: ingest, transform, model, strategy, paper trading, report |
| `set-quant-alpha/scripts/run_pipeline.py` | the single entry point used by you, cron and Airflow |
| `set-quant-alpha/airflow/dags/` | daily, weekly and LSEG DAGs |
| `set-quant-alpha/data/` | data store (gitignored except `paper/`): `raw/` downloads, `processed/` features, `paper/` live ledger |
| `set-quant-alpha/models/` | trained models (gitignored) and the frozen `best_params.json` (committed) |
| `set-quant-alpha/reports/` | `dashboard.html`, `latest_signals.csv` (gitignored) |
| `set-quant-alpha/experiments/` | every research experiment, with its report and frozen protocol |
| `set-quant-alpha/docs/` | flow diagrams (`daily_flow.png`, `strategy_flow.png`) |
