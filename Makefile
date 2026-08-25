# Every target runs against the single centralised .venv that `uv sync` builds.
.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help setup kernel lab lint fmt test clean hw pipeline airflow-deps

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the shared .venv, install all deps + this repo (editable), install hooks
	uv sync
	uv run pre-commit install
	$(MAKE) kernel
	@echo ""
	@echo "Ready. Activate with:  source .venv/bin/activate   (or just prefix commands with 'uv run')"

kernel: ## Register the shared venv as a Jupyter kernel named 'smaz'
	uv run python -m ipykernel install --user --name smaz \
		--display-name "Python (stock-markets-zoomcamp)"

lab: ## Launch JupyterLab from the repo root
	uv run jupyter lab

lint: ## Ruff check
	uv run ruff check .

fmt: ## Ruff format + autofix
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run pytest
	uv run pytest

pipeline: ## Run the capstone pipeline end-to-end (offline replay)
	uv run project/scripts/run_pipeline.py --mode file

airflow-deps: ## Install the optional Airflow extras for module 5
	uv sync --group airflow

clean: ## Remove caches and build artefacts (keeps .venv and your data)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
