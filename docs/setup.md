# Setup and troubleshooting

## Prerequisites

Only [uv](https://docs.astral.sh/uv/) is required — it manages the Python
interpreter itself, so you do **not** need pyenv, conda, or a system Python 3.12.

```bash
brew install uv                                  # macOS
curl -LsSf https://astral.sh/uv/install.sh | sh  # anywhere else
```

## First-time setup

```bash
make setup
```

That runs four things:

1. `uv sync` — downloads CPython 3.12 (per `.python-version`), creates `.venv/`,
   installs everything from `uv.lock`, and installs this repo editable so
   `smaz` and `algotrade` are importable.
2. `pre-commit install` — wires up the lint/format/nbstripout hooks.
3. `ipykernel install --name smaz` — registers the kernel Jupyter will offer.

Then add your keys:

```bash
cp .env.example .env
```

A [free FRED API key](https://fredaccount.stlouisfed.org/apikeys) is the only
one the course strictly needs. yfinance requires none.

## Running things

You never need to activate the venv — `uv run` resolves it automatically:

```bash
uv run python -c "import smaz; print(smaz.__version__)"
uv run pytest
uv run jupyter lab
```

If you prefer an activated shell:

```bash
source .venv/bin/activate
```

## Adding a dependency

```bash
uv add pandas-market-calendars      # runtime
uv add --dev mypy                   # dev tooling
uv add --group airflow apache-airflow-providers-postgres
```

`uv add` updates both `pyproject.toml` and `uv.lock`. **Commit both.**

## Troubleshooting

**Jupyter doesn't show the right kernel.**
Run `make kernel`, then restart JupyterLab. In an open notebook: *Kernel →
Change Kernel → Python (stock-markets-zoomcamp)*.

**`ModuleNotFoundError: No module named 'smaz'`.**
The notebook is on the wrong kernel (see above), or the editable install is
stale. Re-run `uv sync`.

**`import` works in the terminal but not in the notebook.**
Almost always the kernel. Confirm with:

```python
import sys

print(sys.executable)
```

It should point inside this repo's `.venv/`.

**yfinance returns empty frames or rate-limit errors.**
Yahoo throttles bursts. The loaders in `smaz.data` cache to
`data/cache/*.parquet` precisely for this — subsequent calls are offline. To
force a refresh: `data.load_ohlcv(..., refresh=True)`.

**Airflow install is slow or conflicts.**
It is deliberately excluded from the default sync. Install it only when you
reach module 5: `make airflow-deps`.

**Editing `src/smaz/` doesn't take effect in a running notebook.**
The templates start with `%autoreload 2`, which handles most edits. Adding new
top-level modules still needs a kernel restart.

**Wiping and starting over.**

```bash
rm -rf .venv && make setup
```
