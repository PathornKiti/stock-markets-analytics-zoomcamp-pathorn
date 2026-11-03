"""Daily model-health monitor (proposed Airflow task `monitor`, runs after `paper`).

Computes, from the already-published scores and the transformed dataset:
  * rolling 60/120/250-day IC of the published score (5-day label lag), ICIR, t-stat
  * stock-level feature PSI (39 cross-sectional features) of the latest month vs the champion's training window
  * label base-rate, cross-sectional alpha dispersion (labels matured), score dispersion / persistence
Writes one row/day to METRICS (append-only), a status json, and sends alerts.  Reads production artefacts, writes none.

    uv run set-quant-alpha/experiments/retrain_v1/proposed/monitor.py [--asof YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request

import numpy as np
import pandas as pd
import policy as P
from setalpha import config, model, paper, report, transform

log = logging.getLogger("setalpha.monitor")


def stock_level_features(df: pd.DataFrame, feats: list[str]) -> list[str]:
    num = [f for f in feats if df[f].dtype != "int8"]
    liq = df[df.liquid]
    last = liq[liq.Date >= liq.Date.max() - pd.Timedelta(days=60)]
    within = last.groupby("Date")[num].std().mean()
    return [
        f for f in num if within[f] > 1e-9
    ]  # drops date-level macro/regime series (PSI = regime, not selection)


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 1000 or len(cur) < 200:
        return float("nan")
    e = np.r_[-np.inf, np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1])), np.inf]
    a = np.clip(np.histogram(ref, e)[0] / len(ref), 1e-4, None)
    b = np.clip(np.histogram(cur, e)[0] / len(cur), 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())


def compute(
    df: pd.DataFrame, feats: list[str], asof: pd.Timestamp, trained_through: pd.Timestamp
) -> dict:
    liq = df[df.liquid & (df.Date <= asof)]
    m = {"asof": asof.date().isoformat()}
    # --- IC (history = walk-forward predictions, then live published scores; see paper.ic_monitor)
    ic = paper.ic_monitor(df[df.Date <= asof])["ic"].dropna()
    lag = ic.shift(5)  # IC of day t is only known at t+5
    for w, mp in ((60, 40), (120, 80), (250, 175)):
        r = lag.rolling(w, min_periods=mp).mean().iloc[-1] if len(lag) else np.nan
        m[f"ic_{w}d"] = float(r)
    x = lag.tail(120).iloc[:: config.HORIZON].dropna()
    m["ic_120d_t_nonoverlap"] = (
        float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 3 else np.nan
    )
    m["icir_250d"] = float(lag.tail(250).mean() / lag.tail(250).std()) if len(lag) > 50 else np.nan
    # --- feature drift vs champion's training window (stock-level features only)
    stock = stock_level_features(liq, feats)
    ref = liq[liq.Date <= trained_through]
    cur = liq[liq.Date > asof - pd.Timedelta(days=31)]
    p = pd.Series(
        {f: psi(ref[f].to_numpy(dtype=float), cur[f].to_numpy(dtype=float)) for f in stock}
    )
    m.update(
        psi_stock_mean=float(p.mean()),
        psi_stock_max=float(p.max()),
        psi_stock_argmax=str(p.idxmax()),
        n_psi_watch=int((p > P.PSI_LEVELS["watch"]).sum()),
        n_psi_alert=int((p > P.PSI_LEVELS["alert"]).sum()),
        nan_share_max=float(cur[stock].isna().mean().max()),
    )
    # --- labels (matured only) and score
    lab = liq[liq[model.TARGET].notna() & (liq.Date > asof - pd.Timedelta(days=365))]
    m["label_base_rate_1y"] = float(lab[model.TARGET].mean())
    m["alpha_xs_std_1y"] = float(lab["fwd_alpha_5"].std())
    if paper.SCORES.exists():
        s = pd.read_parquet(paper.SCORES)
        s = s[s.Date <= asof]
        if len(s):
            w = s.pivot(index="Date", columns="Ticker", values="pred_ens").rank(axis=1, pct=True)
            ac = w.corrwith(w.shift(1), axis=1).dropna()
            m["rank_autocorr_1d"] = float(ac.iloc[-1]) if len(ac) else np.nan
            if "pred_xgb" in s.columns:
                m["score_xgb_std"] = float(s[s.Date == s.Date.max()]["pred_xgb"].std())
    return m


def evaluate(m: dict, days_since_fit: int) -> dict:
    alerts, level = [], "green"

    def raise_(lv, msg):
        nonlocal level
        alerts.append(f"{lv.upper()}: {msg}")
        level = "red" if lv == "red" or level == "red" else "amber"

    for k, t in P.IC_THRESHOLDS.items():
        v = m.get(k)
        if v is None or np.isnan(v):
            continue
        if t["red"] is not None and v < t["red"]:
            raise_("red", f"{k} {v:.3f} < {t['red']}")
        elif v < t["amber"] and k != "ic_60d":
            raise_("amber", f"{k} {v:.3f} < {t['amber']}")
    g = P.GUARDS
    if m["psi_stock_max"] > g["stock_feature_psi_max_monthly"]:
        raise_("amber", f"DATA GUARD feature {m['psi_stock_argmax']} PSI {m['psi_stock_max']:.1f}")
    if m["psi_stock_mean"] > g["stock_feature_psi_mean_monthly"]:
        raise_("amber", f"DATA GUARD mean stock-feature PSI {m['psi_stock_mean']:.2f}")
    if m["nan_share_max"] > g["feature_nan_share_max"]:
        raise_("red", f"DATA GUARD NaN share {m['nan_share_max']:.2f}")
    lo, hi = g["label_base_rate_trailing_year"]
    if not lo <= m["label_base_rate_1y"] <= hi:
        raise_("amber", f"label base rate {m['label_base_rate_1y']:.3f} outside [{lo},{hi}]")
    if m.get("rank_autocorr_1d", 1) < g["rank_autocorr_1d_min"]:
        raise_("amber", f"score rank autocorr {m['rank_autocorr_1d']:.2f}")
    trig = P.TRIGGERS["ic_red"]
    retrain = bool(
        m.get("ic_120d", 1) < trig["below"] and days_since_fit >= trig["min_days_since_last_fit"]
    )
    return {"status": level, "alerts": alerts, "trigger_challenger_retrain": retrain, **m}


def notify(res: dict) -> None:
    if res["status"] == "green":
        return
    msg = f"set-quant-alpha monitor {res['status'].upper()} {res['asof']}: " + "; ".join(
        res["alerts"]
    )
    log.warning(msg)
    url = config.get_secret(P.ALERT_WEBHOOK_ENV)
    if url:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": msg}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)


def run(asof: str | None = None) -> dict:
    P.HOME.mkdir(parents=True, exist_ok=True)
    df = transform.add_dummies(transform.load())
    feats = transform.feature_sets(df)["NUMERIC"] + transform.feature_sets(df)["DUMMIES"]
    meta = report.load_meta()
    champ_through = pd.Timestamp(meta["trained_through"])
    reg = P.REGISTRY / "registry.json"
    last_fit = champ_through
    if reg.exists():
        r = json.loads(reg.read_text())
        last_fit = pd.Timestamp(r["versions"][r["champion"]]["trained_through"])
    asof_ = pd.Timestamp(asof) if asof else df.Date.max()
    res = evaluate(compute(df, feats, asof_, champ_through), (asof_ - last_fit).days * 5 // 7)
    row = pd.DataFrame(
        [{k: v for k, v in res.items() if k != "alerts"} | {"alerts": " | ".join(res["alerts"])}]
    )
    old = pd.read_parquet(P.METRICS) if P.METRICS.exists() else pd.DataFrame()
    pd.concat(
        [old[old["asof"] != row["asof"].iloc[0]] if len(old) else old, row], ignore_index=True
    ).to_parquet(P.METRICS, index=False)  # idempotent per day
    P.STATUS.write_text(json.dumps(res, indent=1, default=str))
    notify(res)
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof")
    print(json.dumps(run(ap.parse_args().asof), indent=1, default=str))
