"""Factor models for performance attribution.

Two complementary versions of the Carhart (1997) four-factor model

    r_p - rf = alpha + b_MKT MKT + b_SMB SMB + b_HML HML + b_WML WML + e

1. **Fama-French Emerging Markets factors** (Kenneth French data library, monthly, USD):
   the textbook Carhart model. Thailand is part of the EM universe. Strategy returns are
   converted from THB to USD with the USD/THB rate so returns and factors are in the same
   currency, and RF is the US 1-month T-bill.

2. **Local Thai factors** built daily from our own panel (THB). This is the version that
   can see the size effect inside the Thai market:
   * MKT - the SET50 ETF (TDEX) daily return
   * SMB - equal-weight return of the smaller half of liquid stocks minus the larger half
           (size proxied by 60-day median traded value; market-cap history is not free)
   * WML - top 30% minus bottom 30% of liquid stocks by 12-1 month momentum
   Sorts use the previous day's information. There is no local HML: book values are not
   available in free data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def local_factors(df: pd.DataFrame, prices: pd.DataFrame, bench_close: pd.Series) -> pd.DataFrame:
    """Daily Thai MKT, SMB and WML factor returns (decimal)."""
    close = prices.pivot_table(index="Date", columns="Ticker", values="Close")
    ret = close.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 0.45)
    liquid = df.pivot_table(index="Date", columns="Ticker", values="liquid", aggfunc="first")
    liquid = liquid.reindex(index=ret.index, columns=ret.columns).fillna(False).astype(bool)
    size = df.pivot_table(index="Date", columns="Ticker", values="dvol60").reindex(
        index=ret.index, columns=ret.columns
    )

    size_lag = size.shift(1).where(liquid.shift(1, fill_value=False))
    med = size_lag.median(axis=1)
    smb = ret.where(size_lag.le(med, axis=0)).mean(axis=1) - ret.where(
        size_lag.gt(med, axis=0)
    ).mean(axis=1)

    mom = (close.shift(21) / close.shift(252) - 1).shift(1).where(liquid.shift(1, fill_value=False))
    hi, lo = mom.quantile(0.7, axis=1), mom.quantile(0.3, axis=1)
    wml = ret.where(mom.ge(hi, axis=0)).mean(axis=1) - ret.where(mom.le(lo, axis=0)).mean(axis=1)

    mkt = bench_close.reindex(ret.index).ffill().pct_change()
    return pd.DataFrame({"MKT": mkt, "SMB": smb, "WML": wml}).dropna()


def to_monthly_usd(daily_thb: pd.Series, usdthb: pd.Series) -> pd.Series:
    """Compound daily THB returns to calendar months and convert to USD returns."""
    m = (1 + daily_thb.dropna()).resample("ME").prod() - 1
    fx = usdthb.resample("ME").last()  # THB per 1 USD
    fx_ret = fx.reindex(m.index) / fx.shift(1).reindex(m.index)
    return ((1 + m) / fx_ret - 1).dropna()


def regress(y: pd.Series, X: pd.DataFrame, periods_per_year: int, hac_lags: int) -> dict:
    """OLS with Newey-West errors; alpha annualised in %."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    fit = sm.OLS(d["y"], sm.add_constant(d[X.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    out = {
        "alpha % p.a.": 100 * fit.params["const"] * periods_per_year,
        "alpha t": float(fit.tvalues["const"]),
    }
    for f in X.columns:
        out[f"b_{f}"] = float(fit.params[f])
        out[f"t_{f}"] = float(fit.tvalues[f])
    out["R2"] = float(fit.rsquared)
    out["obs"] = int(fit.nobs)
    return out


def carhart_em(daily_thb: pd.Series, ff: pd.DataFrame, usdthb: pd.Series) -> dict:
    """Monthly Carhart 4-factor regression on Fama-French Emerging Markets factors (USD)."""
    f = ff.set_index("Date")[["Mkt-RF", "SMB", "HML", "WML", "RF"]] / 100
    r = to_monthly_usd(daily_thb, usdthb)
    f = f.reindex(r.index)
    return regress(r - f["RF"], f[["Mkt-RF", "SMB", "HML", "WML"]], periods_per_year=12, hac_lags=3)


def thai_local(daily_thb: pd.Series, factors: pd.DataFrame) -> dict:
    """Daily regression on local Thai MKT / SMB / WML (THB, rf taken as 0)."""
    return regress(daily_thb, factors[["MKT", "SMB", "WML"]], periods_per_year=252, hac_lags=5)


def factor_table(returns: dict[str, pd.Series], fn, *args) -> pd.DataFrame:
    rows = {}
    for name, r in returns.items():
        try:
            rows[name] = fn(r, *args)
        except Exception:  # noqa: BLE001 - e.g. too few observations for a series
            rows[name] = {}
    return pd.DataFrame(rows).T.replace({np.inf: np.nan})
