"""Small conveniences used across weeks."""

from __future__ import annotations

import pandas as pd


def set_plot_defaults() -> None:
    """Consistent, readable matplotlib defaults. Call once at the top of a notebook."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (12, 5),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def describe_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A more useful `.info()`: dtype, null count/share and cardinality per column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    )


def flatten_columns(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """Flatten the MultiIndex columns yfinance returns for multi-ticker downloads."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [sep.join(str(p) for p in col if p != "") for col in out.columns]
    return out
