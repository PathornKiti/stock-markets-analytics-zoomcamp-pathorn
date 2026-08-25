"""Repo-wide paths and settings.

Paths are resolved relative to the repository root rather than to the current
working directory, so a notebook in `homework/03-modeling/` and a script in
`project/scripts/` both read and write the same cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/smaz/config.py -> src/smaz -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

HOMEWORK_DIR = REPO_ROOT / "homework"
PROJECT_DIR = REPO_ROOT / "project"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read an API key from the environment (populated from `.env`).

    Never hard-code keys in a notebook -- copy `.env.example` to `.env` instead.
    `.env` is gitignored.
    """
    return os.getenv(name, default)


FRED_API_KEY = get_secret("FRED_API_KEY")
