"""Publish the deployed strategy's walk-forward backtest for the dashboard (data/ is mounted in the containers).

    uv run set-quant-alpha/experiments/integrated_v1/publish_deployed.py

Writes data/processed/deployed_backtest.parquet (daily net / gross returns of the exact-lot walk-forward and TDEX) and
data/processed/deployed_backtest.json (its out-of-sample statistics). Re-run after each January re-selection.
"""

import json
from pathlib import Path

import pandas as pd
from setalpha import config, paper

OUT = Path(__file__).resolve().parent / "output"

daily = pd.read_parquet(OUT / "wf_daily.parquet")
bt = daily[["wf_exact_net", "wf_exact_gross", "tdex"]].rename(
    columns={"wf_exact_net": "net", "wf_exact_gross": "gross"}
)
bt = bt.dropna(subset=["net"])
bt.to_parquet(config.PROCESSED_DIR / "deployed_backtest.parquet")

row = pd.read_csv(OUT / "wf_oos_2016_2026.csv", index_col=0).loc["WF exact lots (net)"]
pbo = pd.read_csv(OUT / "pbo.csv", index_col=0)["PBO"]
meta = {
    "name": paper.DEPLOYED,
    "description": paper.DEPLOYED_DESC,
    "source": f"experiments/integrated_v1 walk-forward, exact 100-share lots, THB 1M, {100 * config.FEE_PER_SIDE:.3f}%/side",
    "window": f"{bt.index.min().date()} to {bt.index.max().date()}",
    "trials": 671,  # trial count the DSR in wf_oos_2016_2026.csv was computed with
    "pbo_alpha_t": float(pbo.filter(like="alpha t").iloc[-1]),
    "pbo_sharpe": float(pbo.filter(like="Sharpe").iloc[0]),
    **{k: float(v) for k, v in row.dropna().items()},
}
(config.PROCESSED_DIR / "deployed_backtest.json").write_text(json.dumps(meta, indent=1))
print(
    f"published {len(bt)} days ({meta['window']}); CAGR {meta['CAGR %']:.2f}% vs TDEX {meta['TDEX CAGR %']:.2f}%"
)
