"""
signals.py
----------
Given a cointegrated pair (hedge ratio known), compute the spread,
its rolling z-score, and generate long/short/exit trading signals.
"""

import pandas as pd
import numpy as np


def compute_spread(y: pd.Series, x: pd.Series, hedge_ratio: float, intercept: float = 0.0) -> pd.Series:
    """Spread = y - hedge_ratio * x - intercept."""
    return y - hedge_ratio * x - intercept


def compute_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    """Rolling z-score of the spread."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def generate_signals(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 3.5,
) -> pd.DataFrame:
    """
    Generate discrete trading signals from a z-score series.

    Position convention:
        +1  -> long the spread (long y, short hedge_ratio*x)
        -1  -> short the spread (short y, long hedge_ratio*x)
         0  -> flat

    Logic:
        - Enter long when z < -entry_threshold
        - Enter short when z > entry_threshold
        - Exit to flat when |z| < exit_threshold
        - Force exit (stop loss) when |z| > stop_loss
        - Otherwise, hold previous position
    """
    position = 0
    positions = []

    for z in zscore:
        if pd.isna(z):
            positions.append(0)
            continue

        if position == 0:
            if z < -entry_threshold:
                position = 1
            elif z > entry_threshold:
                position = -1
        elif position == 1:
            if z > -exit_threshold or z < -stop_loss:
                position = 0
        elif position == -1:
            if z < exit_threshold or z > stop_loss:
                position = 0

        positions.append(position)

    signals = pd.DataFrame({"zscore": zscore, "position": positions}, index=zscore.index)
    signals["signal_change"] = signals["position"].diff().fillna(0) != 0
    return signals
