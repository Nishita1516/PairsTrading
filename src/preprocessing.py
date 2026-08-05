"""
preprocessing.py
-----------------
Cleans raw price data and builds the feature matrix used for clustering
(PCA + DBSCAN in clustering.py).
"""

import numpy as np
import pandas as pd


def clean_prices(prices: pd.DataFrame, max_na_frac: float = 0.05) -> pd.DataFrame:
    """
    Drop columns with too many missing values, forward-fill small gaps,
    and drop any remaining rows with NaNs.
    """
    na_frac = prices.isna().mean()
    keep_cols = na_frac[na_frac <= max_na_frac].index
    cleaned = prices[keep_cols].ffill().dropna()
    return cleaned


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, used as the feature basis for clustering."""
    return np.log(prices / prices.shift(1)).dropna()


def build_feature_matrix(
    prices: pd.DataFrame,
    lookback: int = None,
) -> pd.DataFrame:
    """
    Build the standardized feature matrix (rows=tickers, cols=features)
    fed into PCA. Features here are simply the daily log returns time
    series per stock (transposed so each row is one stock).

    Parameters
    ----------
    prices : wide price DataFrame (date x ticker)
    lookback : int or None
        If given, only use the last `lookback` trading days.

    Returns
    -------
    pd.DataFrame  (index=ticker, columns=dates) of standardized returns
    """
    prices = clean_prices(prices)
    if lookback:
        prices = prices.tail(lookback + 1)

    rets = compute_log_returns(prices)

    # Standardize each stock's return series (zero mean, unit variance)
    # so PCA isn't dominated by high-volatility names.
    standardized = (rets - rets.mean()) / rets.std()

    # Transpose: one row per stock, one column per trading day.
    feature_matrix = standardized.T
    feature_matrix = feature_matrix.dropna(axis=1, how="any")

    return feature_matrix
