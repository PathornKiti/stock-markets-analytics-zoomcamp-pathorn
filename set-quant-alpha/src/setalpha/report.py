"""Step 6 - latest trade signals and a self-contained HTML dashboard.

`latest_signals` scores the most recent trading day and turns the scores into the deployed strategy's target book
(paper.DEPLOYED: the integrated walk-forward pick, also traded on paper as shadow S5).

`build_dashboard` writes `reports/dashboard.html`: equity curves, drawdowns, rolling
returns, HMM regimes, model quality, diagnostics, the latest signals, and the research layer:
signal tear-sheet, statistical validity, risk & capacity and GIPS-style return tables. It is plain HTML
(Plotly from a CDN), so it can be opened locally, served by the `dashboard` container in
docker compose, or published to any static host (GitHub Pages / S3) to make it live.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from setalpha import config, portfolio, strategy

META = config.MODELS_DIR / "meta.json"
SIGNALS = config.REPORTS_DIR / "latest_signals.csv"
DASHBOARD = config.REPORTS_DIR / "dashboard.html"


DEPLOYED_BACKTEST = (
    config.PROCESSED_DIR / "deployed_backtest.parquet"
)  # experiments/integrated_v1/publish_deployed.py


def load_deployed_backtest() -> dict | None:
    """Walk-forward daily returns (net, gross, tdex) and statistics of the deployed strategy, if published."""
    meta = DEPLOYED_BACKTEST.with_suffix(".json")
    if not (DEPLOYED_BACKTEST.exists() and meta.exists()):
        return None
    return {"daily": pd.read_parquet(DEPLOYED_BACKTEST), "meta": json.loads(meta.read_text())}


def save_meta(meta: dict) -> None:
    META.write_text(json.dumps(meta, indent=1, default=str))


def load_meta() -> dict:
    return json.loads(META.read_text())


# ----------------------------------------------------------------------------- signals
def latest_signals(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    features: list[str] | None = None,
    capital: float = config.INITIAL_CAPITAL,
) -> pd.DataFrame:
    """Orders for the next open from the DEPLOYED strategy (paper.DEPLOYED, the integrated walk-forward pick).

    Scores the last day with the production ensemble and the 20-day-target model, blends their ranks, and applies the
    deployed shadow's rules (top-10, 30-rank buffer, equal weight, no stop). The buffer uses the deployed paper book's
    holdings, so the list matches what that book trades; before the ledger starts it is the book from cash.
    """
    from setalpha import paper  # paper imports report: keep this import local

    meta = load_meta()
    last = df["Date"].max()
    day = df[(df["Date"] == last) & df["liquid"]].copy()
    sh = paper.deployed()
    slow = paper.slow_model(df, features or paper.slow_features(), prices)
    sc = paper.day_scores(day, slow, meta["ensemble"])
    cols = ["Ticker", "sector", "idio_vol", "beta_mkt"]
    trad = sc[sc["tradable"]].merge(day[cols], on="Ticker")
    cfg = strategy.ExactConfig(sh.name, score=sh.score, buffer=sh.buffer, stop_loss=sh.stop)
    w = strategy.target_weights(
        cfg, trad, pd.DataFrame(), meta["ic_valid"], held=paper.deployed_holdings()
    )
    if sh.tilt and len(w) > 1:
        w = paper.tilt_weights(
            list(w.index), trad.set_index("Ticker")[sh.score], config.MAX_POSITIONS
        )

    cand = trad.nlargest(2 * config.MAX_POSITIONS, sh.score).set_index("Ticker")
    cand = pd.concat(
        [cand, trad.set_index("Ticker").loc[[t for t in w.index if t not in cand.index]]]
    )
    cand["exp_alpha_5d_%"] = 100 * portfolio.alpha_from_scores(
        cand[sh.score], cand["idio_vol"], meta["ic_valid"]
    )
    closes = prices[prices["Ticker"].isin(cand.index)].pivot_table(
        index="Date", columns="Ticker", values="Close"
    )
    last_px = closes.ffill().iloc[-1].reindex(cand.index)
    cand["target_weight_%"] = 100 * w.reindex(cand.index).fillna(0.0)
    cand["last_close"] = last_px
    cand["stop_price"] = last_px * (1 - sh.stop) if sh.stop else np.nan
    cand["shares (100-lot)"] = (
        cand["target_weight_%"] / 100 * capital / cand["last_close"] / 100
    ).fillna(0).astype(int) * 100
    out = cand.reset_index()[
        [
            "Ticker",
            "sector",
            "pred_ens",
            "pred_slow",
            sh.score,
            "exp_alpha_5d_%",
            "idio_vol",
            "beta_mkt",
            "target_weight_%",
            "last_close",
            "shares (100-lot)",
            "stop_price",
        ]
    ]
    out = out.sort_values(["target_weight_%", sh.score], ascending=False)
    hmm = df.loc[df["Date"] == last, ["hmm_state", "hmm_p_stressed"]].iloc[0]
    out.insert(0, "signal_date", last.date())
    out.insert(1, "hmm_regime", hmm["hmm_state"])
    out.insert(2, "hmm_p_stressed", round(float(hmm["hmm_p_stressed"]), 3))
    out.insert(3, "strategy", sh.name)
    out.to_csv(SIGNALS, index=False)
    return out


# ----------------------------------------------------------------------------- dashboard


def build_dashboard(res: dict) -> str:
    """`res` is the dict saved by the simulate step (+ signals and the paper summary); see setalpha.dashboard."""
    from setalpha import dashboard

    page = dashboard.render(res)
    DASHBOARD.write_text(page)
    (config.REPORTS_DIR / "index.html").write_text(page)  # served at / by the dashboard container
    return str(DASHBOARD)
