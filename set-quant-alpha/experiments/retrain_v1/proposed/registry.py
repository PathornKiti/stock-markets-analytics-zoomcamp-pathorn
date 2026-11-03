"""Model registry + champion/challenger for the ensemble (proposed; separate from models/*.joblib used by production).

registry/registry.json = {"champion": id, "challengers": [id], "versions": {id: meta}, "history": [events]}
Each version dir holds logit.joblib / xgb.joblib / meta.json (trained_through, n_train, feature hash, params, code sha).
  fit_challenger()  expanding-window refit (production hyperparameters, model._embargo), registers as challenger
  score_day()       point-in-time scores of champion + challengers, appended to CHALLENGER_SCORES (never rewritten)
  evaluate()        paired daily-IC difference challenger - champion on the same dates (shadow evidence)
  promote()/rollback()  pointer moves only; promotion is refused unless PROMOTION rules pass AND approved=True
The first champion is bootstrapped from the CURRENT production models (register_current) so nothing changes on day 1.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import policy as P
from setalpha import config, model


def _load() -> dict:
    f = P.REGISTRY / "registry.json"
    return (
        json.loads(f.read_text())
        if f.exists()
        else {"champion": None, "challengers": [], "versions": {}, "history": []}
    )


def _save(r: dict) -> None:
    P.REGISTRY.mkdir(parents=True, exist_ok=True)
    (P.REGISTRY / "registry.json").write_text(json.dumps(r, indent=1, default=str))


def _sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


def _event(r: dict, what: str, **kw) -> None:
    r["history"].append(
        {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "event": what, **kw}
    )


def fit_challenger(
    df: pd.DataFrame, feats: list[str], best: dict, asof: pd.Timestamp, reason: str
) -> str:
    lab = model.labelled(df)
    start = str(asof.date())
    train = model._embargo(lab[lab.Date < start], start)  # no look-ahead: targets end before `asof`
    vid = f"v{asof:%Y%m%d}"
    d = P.REGISTRY / vid
    d.mkdir(parents=True, exist_ok=True)
    for name in P.SCHEDULE["members"]:
        m = model.fit(name, train, feats, **best[name])
        joblib.dump({"model": m, "features": feats, "params": best[name]}, d / f"{name}.joblib")
    meta = {
        "id": vid,
        "reason": reason,
        "trained_through": str(train.Date.max().date()),
        "registered": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": int(len(train)),
        "feature_hash": hashlib.sha1("|".join(feats).encode()).hexdigest()[:12],
        "params": best,
        "members": P.SCHEDULE["members"],
        "code_sha": _sha(),
        "window": P.SCHEDULE["window"],
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    r = _load()
    r["versions"][vid] = meta
    r["challengers"] = [c for c in r["challengers"] if c != vid] + [vid]
    if r["champion"] is None:
        r["champion"] = vid
    _event(r, "challenger_fitted", id=vid, reason=reason)
    _save(r)
    return vid


def register_current() -> str:
    """Bootstrap: register the production models (read-only copy of metadata, files referenced not moved)."""
    r = _load()
    if r["champion"]:
        return r["champion"]
    meta = json.loads((config.MODELS_DIR / "meta.json").read_text())
    vid = "v-production-" + str(meta["trained_through"]).replace("-", "")
    r["versions"][vid] = {
        "id": vid,
        "reason": "bootstrap from production",
        "trained_through": meta["trained_through"],
        "members": meta["ensemble"],
        "path": str(config.MODELS_DIR),
        "code_sha": _sha(),
    }
    r["champion"] = vid
    _event(r, "bootstrap", id=vid)
    _save(r)
    return vid


def _predict(vid: str, day: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    meta = _load()["versions"][vid]
    path = P.REGISTRY / vid if "path" not in meta else config.MODELS_DIR
    out = day[["Date", "Ticker"]].copy()
    ranks = []
    for name in meta["members"]:
        art = joblib.load(path / f"{name}.joblib")
        out[f"pred_{name}"] = model.predict(art["model"], day, art["features"])
        ranks.append(out[f"pred_{name}"].rank(pct=True))
    out["score"] = sum(ranks) / len(ranks)
    return out.assign(version=vid)


def score_day(day: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Append champion + challenger scores for the latest liquid cross-section (first version of a day is kept)."""
    r = _load()
    parts = [
        _predict(v, day, feats).assign(role="champion" if v == r["champion"] else "challenger")
        for v in [r["champion"], *[c for c in r["challengers"] if c != r["champion"]]]
    ]
    new = pd.concat(parts, ignore_index=True)
    old = pd.read_parquet(P.CHALLENGER_SCORES) if P.CHALLENGER_SCORES.exists() else new.iloc[:0]
    keep = new.merge(
        old[["Date", "Ticker", "version"]],
        on=["Date", "Ticker", "version"],
        how="left",
        indicator=True,
    )
    P.CHALLENGER_SCORES.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new[(keep["_merge"] == "left_only").to_numpy()]], ignore_index=True).to_parquet(
        P.CHALLENGER_SCORES, index=False
    )
    return new


def evaluate(df: pd.DataFrame, challenger: str) -> dict:
    """Paired daily rank IC (tradable names): challenger minus champion on identical dates, 5-day non-overlapping t."""
    r = _load()
    s = pd.read_parquet(P.CHALLENGER_SCORES)
    real = df[["Date", "Ticker", "tradable", "fwd_alpha_5"]]
    ic = {}
    for v in (r["champion"], challenger):
        ic[v] = model.daily_ic(s[s.version == v].merge(real, on=["Date", "Ticker"]), "score")
    d = (ic[challenger] - ic[r["champion"]]).dropna()
    ds = d.iloc[:: config.HORIZON]
    t = float(ds.mean() / ds.std() * np.sqrt(len(ds))) if len(ds) > 3 else float("nan")
    return {
        "days": int(len(d)),
        "paired_dIC": float(d.mean()),
        "t": t,
        "champion_IC": float(ic[r["champion"]].mean()),
        "challenger_IC": float(ic[challenger].mean()),
    }


def promote(challenger: str, ev: dict, approved: bool, kind: str = "no_harm") -> bool:
    rule = P.PROMOTION
    ok = ev["days"] >= rule["min_shadow_trading_days"] and (
        ev["paired_dIC"] >= rule["no_harm"]["max_paired_dIC"]
        if kind == "no_harm"
        else ev["t"] >= rule["superiority"]["min_paired_t"]
    )
    r = _load()
    if not (ok and approved):
        _event(r, "promotion_refused", id=challenger, evidence=ev, approved=approved)
        _save(r)
        return False
    _event(r, "promoted", id=challenger, previous=r["champion"], evidence=ev)
    r["champion"] = challenger
    _save(r)
    return True


def rollback() -> str:
    """Point the champion back at the previously promoted version (artefacts are immutable, so this is instant)."""
    r = _load()
    prev = [e["previous"] for e in r["history"] if e["event"] == "promoted"]
    if not prev:
        raise RuntimeError("nothing to roll back to")
    _event(r, "rollback", id=prev[-1], from_=r["champion"])
    r["champion"] = prev[-1]
    _save(r)
    return r["champion"]
