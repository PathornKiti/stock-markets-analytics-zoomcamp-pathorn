"""Shared helpers for the Stock Markets Analytics Zoomcamp.

Importable from every week's homework notebook and from the capstone project,
because the whole repo is installed editable into one centralised `.venv`::

    import smaz
    from smaz import data, features

Nothing here is required by the course — it exists so that code written in
module 1 does not have to be copy-pasted into module 4.
"""

from smaz import backtest, config, data, features, utils

__version__ = "0.1.0"

__all__ = ["backtest", "config", "data", "features", "utils"]
