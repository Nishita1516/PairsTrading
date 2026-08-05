"""
config.py
---------
Central place for constants used across the pipeline: the NIFTY 50
ticker universe (Yahoo Finance suffix ".NS"), default date range, and
default strategy parameters. Editing this file is the easiest way to
change the scope of the project without touching pipeline code.
"""

from datetime import date, timedelta

# ---------------------------------------------------------------------
# NIFTY 50 constituents (Yahoo Finance tickers, ".NS" = NSE India)
# List is reasonably current as of 2025; swap/edit symbols if the
# index composition has changed by the time you run this.
# ---------------------------------------------------------------------
NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "ITC.NS", "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LTIM.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS",
    "SBIN.NS", "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

# ---------------------------------------------------------------------
# Default date range: last 4 years of daily data (falls inside the
# "3-5 years" range requested)
# ---------------------------------------------------------------------
DEFAULT_END_DATE = date.today()
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=4 * 365)

# ---------------------------------------------------------------------
# Default strategy / pipeline parameters
# ---------------------------------------------------------------------
DEFAULT_PARAMS = {
    "n_pca_components": 5,       # PCA components used as clustering features
    "dbscan_eps": 1.5,           # DBSCAN neighborhood radius (in PCA space)
    "dbscan_min_samples": 2,     # Minimum cluster size
    "coint_pvalue_threshold": 0.05,  # Engle-Granger p-value cutoff
    "zscore_window": 30,         # Rolling window (days) for spread z-score
    "entry_zscore": 2.0,         # Enter trade when |z| exceeds this
    "exit_zscore": 0.5,          # Exit trade when |z| falls below this
    "stop_loss_zscore": 4.0,     # Force-exit if |z| blows out this far
    "initial_capital": 1_000_000,  # For backtest sizing (INR)
    "transaction_cost_bps": 5,   # Round-trip cost per leg, in basis points
}

DATA_DIR = "data"
