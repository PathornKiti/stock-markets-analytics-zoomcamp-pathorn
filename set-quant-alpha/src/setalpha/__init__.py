"""set-quant-alpha: ML alpha + quantitative portfolio construction for Thai equities.

Pipeline (one module per step, shared by the notebooks, the CLI and the cron job):

    ingest     -> raw data from Yahoo (prices, screener, sectors), FRED and Cboe
    transform  -> factor model, features, targets, one combined dataset
    model      -> DecisionTree / RandomForest / Logistic / XGBoost, tuning, walk-forward
    portfolio  -> alpha scaling, Ledoit-Wolf risk model, mean-variance + vol targeting
    strategy   -> vectorised and exact (day-by-day) trading simulations
    evaluate   -> performance statistics (Sharpe, PSR/DSR, CAPM alpha) and diagnostics
    report     -> dashboard, latest signals; notify / broker are optional add-ons
"""

__version__ = "0.1.0"
