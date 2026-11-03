#!/usr/bin/env python
"""Research pull from LSEG Workspace (desktop session) into data/raw/lseg/.

    uv sync --group lseg
    uv run python set-quant-alpha/scripts/fetch_lseg.py members prices foreign fundamentals extra   # first full pull
    uv run python set-quant-alpha/scripts/fetch_lseg.py update                                      # incremental refresh

Needs LSEG Workspace running and LSEG_APP_KEY in set-quant-alpha/.env. If either is missing the script exits with
code 99, which Airflow (set_quant_alpha_lseg DAG) shows as "skipped" instead of failed. Every stage caches to parquet
and skips work already on disk. LSEG data is licensed: data/raw/ is gitignored, so none of this is
redistributed. The daily pipeline never reads these files.

Stages
  members  SET50 / SET100 joiner-leaver events 2009 -> today, current lists, and a daily point-in-time membership
           table rebuilt by replaying the events backwards from today's list.
  missing  `prices` for the RICs Yahoo lacks only (what survivorship_v1 needs), run first because TR is slow.
  prices   daily split-adjusted OHLCV + traded value (get_history) and daily total return in % (TR.TotalReturn, dividends
           included) for every RIC ever in SET100, delisted ones included (RIC suffix ^<month letter><yy>).
  foreign  foreign-board history (<SYM>_f.BK: price, volume, value, VWAP) for every live RIC ever in SET100.
           Delisted names have no foreign-board RIC on LSEG.
  fundamentals  monthly I/B/E/S FY1 EPS mean (+ fiscal period end), number of estimates, recommendation mean,
           P/B, P/E, ROE (dated by fiscal period end) and market cap for every RIC ever in SET100.
  extra    one parquet per group in data/raw/lseg/extra/: monthly as-of I/B/E/S extras (price target, DPS, revenue,
           FY2 EPS, EPS dispersion, SmartEstimate, recommendation counts), StarMine Combined Alpha ranks, shares
           outstanding, free float, ESG, quarterly EPS actuals with report timestamp and surprise, quarterly financials
           with original announcement time, and dividend events (ex date, amount, announcement date).
  holders  semi-annual disclosed-shareholder snapshots with investor country (filings, published with a lag).
  update   what the Airflow DAG runs. On an empty store it is the full first pull (members, prices, foreign,
           fundamentals, extra; a few hours, mostly prices). Afterwards it is incremental: membership; prices and
           foreign board from 10 days before the last stored date (full re-download of a stock if a split changed
           its adjusted history); new RICs in full; and, at most once a week, the last 120 days of the monthly
           fundamentals and extra groups, merged without overwriting first-reported earnings and financials.
"""

from __future__ import annotations

import contextlib
import functools
import json
import sys
import time
import warnings

import pandas as pd
from setalpha import config, lseg

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)  # noqa: A001  (progress visible in redirected logs)
OUT = config.DATA_DIR / "raw" / "lseg"
START, END = "2009-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
INDICES = {"SET50": ".SET50", "SET100": ".SET100"}
OHLCV = {
    "OPEN_PRC": "Open",
    "HIGH_1": "High",
    "LOW_1": "Low",
    "TRDPRC_1": "Close",
    "ACVOL_UNS": "Volume",
    "TRNOVR_UNS": "Value",
}
FOREIGN = {"TRDPRC_1": "f_close", "ELG_ACVOL": "f_volume", "ELG_TNOV": "f_value", "VWAP": "f_vwap"}


def retry(f, tries: int = 3):
    for i in range(tries):
        try:
            return f()
        except Exception as e:  # LSEG returns generic errors for empty universes and timeouts alike
            if "not found" in str(e) or "No data" in str(e) or i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def members(ld) -> None:
    events, rows = [], []
    for name, ric in INDICES.items():
        ev = ld.get_data(
            [ric],
            [
                "TR.IndexJLConstituentChangeDate",
                "TR.IndexJLConstituentRIC",
                "TR.IndexJLConstituentName",
                "TR.IndexJLConstituentituentChange",
            ],
            {"SDATE": START, "EDATE": END, "IC": "B"},
        )
        ev.columns = ["index", "date", "ric", "name", "change"]
        ev["index"] = name
        events.append(ev.dropna(subset=["ric"]))
        now = set(ld.get_data([f"0#{ric}"], ["TR.CompanyName"])["Instrument"])
        # replay backwards: on an event date a Joiner was not a member before it, a Leaver was
        days = pd.bdate_range(START, END)
        cur = set(now)
        ev = ev.sort_values("date", ascending=False)
        snap = {}
        groups = {d: g for d, g in ev.groupby("date")}
        for d in days[::-1]:
            snap[d] = frozenset(cur)
            if d in groups:
                g = groups[d]
                cur -= set(g.loc[g.change == "Joiner", "ric"])
                cur |= set(g.loc[g.change == "Leaver", "ric"])
        rows += [(d, name, r) for d, s in snap.items() for r in s]
        print(
            f"{name}: {len(ev)} events, {len(now)} members today, {len(cur)} members on {days[0].date()}"
        )
    pd.concat(events).to_parquet(OUT / "index_events.parquet", index=False)
    m = pd.DataFrame(rows, columns=["Date", "index", "ric"])
    m.to_parquet(OUT / "index_members.parquet", index=False)
    print("members:", m.groupby("index").ric.nunique().to_dict(), "distinct RICs over the period")


def all_rics() -> list[str]:
    m = pd.read_parquet(OUT / "index_members.parquet", columns=["ric"])
    return sorted(m.ric.unique())


def _price_rows(ld, ric: str, start: str) -> pd.DataFrame:
    h = retry(lambda: ld.get_history(ric, fields=list(OHLCV), start=start, end=END)).rename(
        columns=OHLCV
    )
    tr = retry(
        lambda: ld.get_data(
            [ric],
            ["TR.TotalReturn.date", "TR.TotalReturn"],
            {"SDate": start, "EDate": END, "Frq": "D"},
        )
    )
    tr.columns = ["Instrument", "Date", "TotalReturn"]
    tr = tr.dropna().assign(Date=lambda d: pd.to_datetime(d.Date)).set_index("Date")["TotalReturn"]
    h.index = pd.to_datetime(h.index)
    h = h.join(tr.astype(float) / 100, how="left").reset_index().rename(columns={"index": "Date"})
    return h.assign(Ticker=ric)


def _foreign_rows(ld, ric: str, start: str) -> pd.DataFrame:
    h = retry(
        lambda: ld.get_history(
            ric.replace(".BK", "_f.BK"), fields=list(FOREIGN), start=start, end=END
        )
    )
    h = h.rename(columns=FOREIGN)
    h.index = pd.to_datetime(h.index)
    return h.reset_index().rename(columns={"index": "Date"}).assign(Ticker=ric)


def prices(ld, only_missing: bool = False) -> None:
    path = OUT / "prices.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["Ticker"])
    yp = config.RAW_DIR / "prices.parquet"
    yahoo = set(pd.read_parquet(yp, columns=["Ticker"]).Ticker) if yp.exists() else set()
    todo = [r for r in all_rics() if r not in set(have.Ticker)]
    todo.sort(key=lambda r: r in yahoo)  # names Yahoo lacks first: the survivorship fix needs them
    if only_missing:
        todo = [r for r in todo if r not in yahoo]
    print(f"prices: {len(todo)} RICs to fetch ({len(set(have.Ticker))} cached)")
    parts, failed = [have], []
    for i, ric in enumerate(todo, 1):
        try:
            parts.append(_price_rows(ld, ric, START))
        except Exception as e:
            failed.append(ric)
            print(f"  {ric}: no data ({str(e)[:60]})")
        if i % 25 == 0 or i == len(todo):
            pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
            print(f"  {i}/{len(todo)} saved")
    print(f"prices: done, {len(failed)} RICs without data: {failed}")


def foreign(ld) -> None:
    path = OUT / "foreign_board.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["Ticker"])
    none_path = OUT / "foreign_none.json"  # RICs already found to have no foreign-board line
    none = set(json.loads(none_path.read_text())) if none_path.exists() else set()
    live = [r for r in all_rics() if "^" not in r and r not in set(have.Ticker) and r not in none]
    print(f"foreign: {len(live)} RICs to fetch")
    parts, n = [have], 0
    for i, ric in enumerate(live, 1):
        try:
            parts.append(_foreign_rows(ld, ric, START))
            n += 1
        except Exception:
            none.add(ric)  # no foreign board line for this stock
        if i % 25 == 0 or i == len(live):
            pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    none_path.write_text(json.dumps(sorted(none)))
    print(f"foreign: {n} of {len(live)} stocks have a foreign-board line")


def holders(ld) -> None:
    path = OUT / "holders.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["asof"])
    rics = all_rics()
    dates = [d for d in pd.date_range("2012-06-30", END, freq="6ME") if d not in set(have.asof)]
    parts = [have]
    for d in dates:
        for k in range(0, len(rics), 25):
            chunk = rics[k : k + 25]
            try:
                h = retry(
                    lambda c=chunk, d=d: ld.get_data(
                        c,
                        [
                            "TR.InvAddrCountry",
                            "TR.InvestorType",
                            "TR.PctOfSharesOutHeld",
                            "TR.HoldingsDate",
                        ],
                        {"SDate": d.strftime("%Y-%m-%d")},
                    )
                )
                h.columns = ["ric", "country", "investor_type", "pct", "filing_date"]
                parts.append(h.dropna(subset=["pct"]).assign(asof=d))
            except Exception as e:
                print(f"  {d.date()} chunk {k}: {str(e)[:80]}")
        pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
        print(f"holders: {d.date()} saved")


FUND = {  # name -> (fields, extra parameters); each group carries its own .date so values never misalign
    "eps": (["TR.EPSMean.date", "TR.EPSMean", "TR.EPSMean.periodenddate"], {"Period": "FY1"}),
    "nest": (["TR.EPSNumberofEstimates.date", "TR.EPSNumberofEstimates"], {"Period": "FY1"}),
    "rec": (["TR.RecMean.date", "TR.RecMean"], {}),
    "pb": (["TR.PriceToBVPerShare.date", "TR.PriceToBVPerShare"], {}),
    "pe": (["TR.PE.date", "TR.PE"], {}),
    "roe": (
        ["TR.ROEPercent.date", "TR.ROEPercent"],
        {},
    ),  # dated by FISCAL PERIOD END, not availability
    "mcap": (["TR.CompanyMarketCap.date", "TR.CompanyMarketCap"], {}),
}


def fundamentals(ld, since: str | None = None) -> None:
    """Monthly as-of history 2009-06 -> today; long format (ric, field, date, value[, period_end]).

    With `since`, every group is re-fetched from that date for live RICs and merged into the stored history."""
    path = OUT / "fundamentals.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["field"])
    rics, parts = all_rics(), [have]
    if since:
        rics = [r for r in rics if "^" not in r]
    for name, (fields, extra) in FUND.items():
        if name in set(have.field) and not since:
            continue
        for k in range(0, len(rics), 20):
            chunk = rics[k : k + 20]
            try:
                d = retry(
                    lambda c=chunk, f=fields, e=extra: ld.get_data(
                        c, f, {"SDate": since or "2009-06-01", "EDate": END, "Frq": "M", **e}
                    )
                )
            except Exception as e:
                print(f"  {name} chunk {k}: {str(e)[:80]}")
                continue
            cols = ["ric", "date", "value"] + (["period_end"] if len(fields) == 3 else [])
            d.columns = cols
            parts.append(d.dropna(subset=["value"]).assign(field=name))
        if since:
            if len(parts) > 1:
                have = append_new(have, pd.concat(parts[1:], ignore_index=True))
            parts = [have]
            have.to_parquet(path, index=False)
        else:
            pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
        print(f"fundamentals: {name} saved")


EXTRA = {  # name -> (fields, column names, extra parameters); monthly as-of unless the parameters say otherwise
    "pt": (
        ["TR.PriceTargetMean.date", "TR.PriceTargetMean", "TR.PriceTargetNumIncEstimates"],
        ["pt", "pt_n"],
        {},
    ),
    "dps": (["TR.DPSMean.date", "TR.DPSMean"], ["dps"], {"Period": "FY1"}),
    "revn": (["TR.RevenueMean.date", "TR.RevenueMean"], ["revenue"], {"Period": "FY1"}),
    "eps2": (
        ["TR.EPSMean.date", "TR.EPSMean", "TR.EPSMean.periodenddate"],
        ["eps2", "period_end"],
        {"Period": "FY2"},
    ),
    "epssd": (["TR.EPSStdDev.date", "TR.EPSStdDev"], ["eps_sd"], {"Period": "FY1"}),
    "smart": (
        ["TR.EPSSmartEst.date", "TR.EPSSmartEst", "TR.EPSMean"],
        ["smart", "mean"],
        {"Period": "FY1"},
    ),
    "recn": (
        [
            "TR.NumOfStrongBuy.date",
            "TR.NumOfStrongBuy",
            "TR.NumOfBuy",
            "TR.NumOfHold",
            "TR.NumOfSell",
            "TR.NumOfStrongSell",
        ],
        ["sbuy", "buy", "hold", "sell", "ssell"],
        {},
    ),
    "ca": (
        [
            "TR.CombinedAlphaRegionRank.date",
            "TR.CombinedAlphaRegionRank",
            "TR.CombinedAlphaCountryRank",
        ],
        ["ca_region", "ca_country"],
        {},
    ),
    "shares": (["TR.SharesOutstanding.date", "TR.SharesOutstanding"], ["shares"], {}),
    "ff": (["TR.FreeFloatPct.date", "TR.FreeFloatPct"], ["ff"], {}),
    "esg": (["TR.TRESGScore.date", "TR.TRESGScore"], ["esg"], {}),  # dated by fiscal year end
    "epsq": (
        [
            "TR.EPSActValue.date",
            "TR.EPSActValue",
            "TR.EPSActReportDate",
            "TR.EPSActValue.periodenddate",
        ],
        ["eps_act", "report_date", "period_end"],
        {"Period": "FQ0"},
    ),
    "surpq": (
        ["TR.EPSActSurprise.date", "TR.EPSActSurprise"],
        ["surprise"],
        {"Period": "FQ0"},
    ),  # date = period end
    "finq": (
        [
            "TR.F.TotRevenue.date",
            "TR.F.OriginalAnnouncementDate",
            "TR.F.TotRevenue",
            "TR.F.GrossProfIndPropTot",
            "TR.F.EBIT",
            "TR.F.NetIncAfterMinIntr",
            "TR.F.NetCashFlowOp",
            "TR.F.TotAssets",
            "TR.F.ShHoldEqCom",
            "TR.F.TotLiab",
        ],
        ["announced", "rev", "gp", "ebit", "ni", "cfo", "ta", "equity", "liab"],
        {"Period": "FQ0"},
    ),
    "div": (
        ["TR.DivExDate", "TR.DivUnadjustedGross", "TR.DivAnnouncementDate", "TR.DivType"],
        ["ex_date", "amount", "announced", "type"],
        {"Frq": None},
    ),
}


DATES = {"date", "report_date", "period_end", "announced", "ex_date"}
# earnings and financials: a window refresh only adds periods never seen, so first-reported values stay (point-in-time)
FIRST_KEYS = {"epsq": ["ric", "period_end"], "surpq": ["ric", "date"], "finq": ["ric", "date"]}


def append_new(old: pd.DataFrame, new: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Stored rows are never dropped or changed; only rows not already stored are appended (and, with `keys`, only
    keys not already stored). Readers take the last row per date, so a revised value appended later wins."""
    if new.empty:
        return old
    new = new.reindex(columns=old.columns)
    for c in old.columns:  # align dtypes so identical rows compare equal
        if pd.api.types.is_datetime64_any_dtype(old[c]):
            new[c] = pd.to_datetime(new[c], errors="coerce")
        else:
            with contextlib.suppress(TypeError, ValueError):  # e.g. Int64 counts stay Int64
                new[c] = new[c].astype(old[c].dtype)
    sig = lambda d, c: pd.util.hash_pandas_object(d[c].astype(str), index=False)  # noqa: E731
    if keys:
        new = new[~sig(new, keys).isin(set(sig(old, keys))).to_numpy()]
    new = new[
        ~sig(new, list(old.columns)).isin(set(sig(old, list(old.columns)))).to_numpy()
    ].drop_duplicates()
    return pd.concat([old, new], ignore_index=True)


def extra(ld, since: str | None = None) -> None:
    """One parquet per EXTRA group. With `since`, re-fetch from that date for live RICs and merge by KEYS."""
    out = OUT / "extra"
    out.mkdir(exist_ok=True)
    rics = all_rics()
    if since:
        rics = [r for r in rics if "^" not in r]
    for name, (fields, cols, par) in EXTRA.items():
        path = out / f"{name}.parquet"
        if path.exists() and not since:
            continue
        params = {"SDate": since or "2009-06-01", "EDate": END, "Frq": "M", **par}
        params = {k: v for k, v in params.items() if v is not None}
        parts = []
        for k in range(0, len(rics), 20):
            chunk = rics[k : k + 20]
            try:
                d = retry(lambda c=chunk, f=fields, p=params: ld.get_data(c, f, p))
            except Exception as e:
                print(f"  {name} chunk {k}: {str(e)[:80]}")
                continue
            d.columns = ["ric"] + (["date"] if name != "div" else []) + cols
            parts.append(d.dropna(subset=[cols[0]]).drop_duplicates())
        if parts:
            n = pd.concat(parts, ignore_index=True)
            for c in n.columns[1:]:  # LSEG mixes '' and numbers in object columns
                if c in DATES:
                    n[c] = pd.to_datetime(
                        n[c].replace("", None), errors="coerce", utc=True
                    ).dt.tz_localize(None)
                elif c != "type":
                    n[c] = pd.to_numeric(n[c].replace("", None), errors="coerce").astype("float64")
            if path.exists():
                n = append_new(pd.read_parquet(path), n, FIRST_KEYS.get(name))
            n.to_parquet(path, index=False)
            print(f"extra: {name} saved, {len(n):,} rows, {n.ric.nunique()} RICs")


def _refresh(ld, path, rows, check: str) -> None:
    """Append recent rows for every live RIC in `path`; re-download a RIC in full if its adjusted history moved."""
    if not path.exists():
        return
    have = pd.read_parquet(path)
    have["Date"] = pd.to_datetime(have["Date"])
    have = have.drop_duplicates(["Ticker", "Date"], keep="last")
    last = have.groupby("Ticker")["Date"].max()
    parts, full, failed = [], [], 0
    for ric, d in last.items():
        if "^" in ric:
            continue  # delisted: no new data
        try:
            new = rows(ld, ric, (d - pd.Timedelta(days=10)).strftime("%Y-%m-%d")).drop_duplicates(
                "Date", keep="last"
            )
        except Exception:
            failed += 1
            continue
        old = have.loc[have.Ticker == ric].set_index("Date")[check].astype("float64")
        ov = new.set_index("Date")[check].astype("float64").reindex(old.index).dropna()
        if (
            len(ov) and ((ov / old.reindex(ov.index) - 1).abs() > 0.01).any()
        ):  # split / adjustment change
            new = rows(ld, ric, START)
            full.append(ric)
            have = have[have.Ticker != ric]
        parts.append(new)
    out = pd.concat([have] + parts, ignore_index=True).drop_duplicates(
        ["Ticker", "Date"], keep="last"
    )
    out.sort_values(["Ticker", "Date"]).to_parquet(path, index=False)
    print(
        f"{path.name}: refreshed {len(parts)} RICs to {out.Date.max().date()}; full re-download {full}; "
        f"{failed} failed"
    )


def update(ld) -> None:
    members(ld)
    prices(ld)  # RICs new to SET50 / SET100: full history
    _refresh(ld, OUT / "prices.parquet", _price_rows, "Close")
    foreign(ld)
    _refresh(ld, OUT / "foreign_board.parquet", _foreign_rows, "f_close")
    stamp = OUT / "monthly_refreshed.txt"
    first = not stamp.exists()
    fundamentals(ld)  # full history for any group not on disk yet
    extra(ld)
    age = (time.time() - stamp.stat().st_mtime) / 86400 if stamp.exists() else 1e9
    if first:
        stamp.write_text(pd.Timestamp.now().isoformat())  # groups were just pulled in full
    elif age >= 6:
        since = (pd.Timestamp.today() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        fundamentals(ld, since=since)
        extra(ld, since=since)
        stamp.write_text(pd.Timestamp.now().isoformat())
    else:
        print(f"monthly groups refreshed {age:.1f} days ago; next refresh after 6 days")


STAGES = {
    "members": members,
    "prices": prices,
    "missing": lambda ld: prices(ld, only_missing=True),
    "foreign": foreign,
    "holders": holders,
    "fundamentals": fundamentals,
    "extra": extra,
    "update": update,
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if reason := lseg.unavailable():
        print(f"LSEG skipped: {reason}")
        sys.exit(99)  # Airflow BashOperator: 99 = skipped
    with lseg.session() as ld:
        for stage in sys.argv[1:] or list(STAGES):
            STAGES[stage](ld)
