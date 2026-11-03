"""Project-wide settings for set-quant-alpha.

Everything that a reviewer may want to change (dates, universe filters, costs,
portfolio limits) lives here, so the modules stay free of magic numbers.

Data location: by default `set-quant-alpha/data/` (gitignored). Point the
`SETALPHA_DATA_DIR` environment variable at a Google Drive / Dropbox / NAS folder
to keep the incremental store outside the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(
    PROJECT_DIR.parent / ".env"
)  # repo-level .env, if the project lives inside the course repo


def get_secret(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


DATA_DIR = Path(os.getenv("SETALPHA_DATA_DIR", PROJECT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
for _d in (RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- history and chronological splits ------------------------------------------
START = "2010-01-01"
VALID_START = "2020-01-01"  # tune hyperparameters / thresholds on 2020-2021
TEST_START = "2022-01-01"  # walk-forward out-of-sample test from 2022 onward

# --- universe -------------------------------------------------------------------
MIN_PRICE = 1.0  # THB; penny stocks are dominated by tick size
MIN_DVOL = 20e6  # THB median daily traded value (60d) to count as liquid
TOP_N = 100  # tradable universe = top-N liquid names by traded value (a point-in-time "SET100")

# --- target ---------------------------------------------------------------------
HORIZON = 5  # trading days: signal at close t, buy open t+1, sell close t+5

# --- costs and portfolio --------------------------------------------------------
# All-in cost per side, researched 2026-11-02 (README "Trading costs"): average online commission of 30 Thai brokers
# (0.150%; most charge 0.157%) + SET 0.005% + TSD 0.001% + SEC 0.001% fees, all plus 7% VAT. Fills are at the opening
# call auction, so no extra half-spread is added. Earlier runs (research experiments) used a flat 0.15%.
COMMISSION_RATE = 0.0015
EXCHANGE_FEES = 0.00007
VAT = 0.07
FEE_PER_SIDE = (COMMISSION_RATE + EXCHANGE_FEES) * (1 + VAT)  # = 0.168% per side
INITIAL_CAPITAL = 1_000_000.0  # THB
MAX_POSITIONS = 10

# --- research features -------------------------------------------------------------
# External flow / event data never enter the production model automatically: a new feature changes the signal of
# the frozen paper-trading shadows (setalpha.paper). Promote it through a pre-registered experiment and a new shadow.
USE_FLOW_FEATURES = False
MAX_WEIGHT = 0.25  # single-name cap in the optimiser (chosen on 2020-2021 validation)
TARGET_VOL = 0.15  # annualised ex-ante portfolio volatility target
STOP_LOSS = 0.08  # per-position stop, checked on the daily low
# The production model design is frozen with shadow S5 (registered 2026-11-02): models/best_params.json is committed and
# the ensemble members are pinned, so a fresh clone or a later weekly retrain refits the same design on new data
# instead of re-selecting it. Set to None to let the validation years choose again (a new design = a new trial).
ENSEMBLE: list[str] | None = ["logit", "xgb"]
RISK_AVERSION = 1.0  # lambda in  max  a'w - lambda/2 w'Sw (chosen on validation)
CONSERVATIVE_RISK_AVERSION = 5.0  # the vol-targeted variant

# --- data sources ---------------------------------------------------------------
BENCHMARK = (
    "TDEX.BK"  # ThaiDEX SET50 ETF: an investable SET50 benchmark (Yahoo has no SET index history)
)
MACRO_YF = {  # Yahoo symbol -> column prefix
    "THB=X": "usdthb",
    "EEM": "eem",  # emerging-market ETF: proxy for global EM fund flows
    "^GSPC": "spx",
    "DX-Y.NYB": "dxy",
    "CL=F": "oil",
    "GC=F": "gold",
    "TDEX.BK": "tdex",
}
FRED_SERIES = {
    "DGS10": "us10y",
    "DTB3": "tbill3m",
    "BAA10Y": "credit_spread",  # Moody's Baa corporate yield minus 10y Treasury (risk appetite)
    "DEXTHUS": "fred_thb",  # THB per USD, noon NY buying rate
}
CBOE_SERIES = ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW"]  # Cboe's free CSV feed
