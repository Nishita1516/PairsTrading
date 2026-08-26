"""
backtest.py
-----------
Simulates a pairs-trading strategy given signals from signals.py and
reports standard performance metrics (cumulative return, Sharpe, max
drawdown, win rate, etc.).
"""

import numpy as np
import pandas as pd


def backtest_pair(
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: float,
    signals: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
    capital: float = 100_000.0,
    direction: str = "y_on_x",
):
    """
    Simulate P&L for a single pair given a position series.

    Parameters
    ----------
    y, x : price series for the two legs (aligned index)
    hedge_ratio : beta from cointegration
    signals : DataFrame from signals.generate_signals (has 'position', 'signal_change')
    transaction_cost_bps : round-trip cost in basis points, applied on signal changes
    capital : notional capital allocated to the pair
    direction : "y_on_x" (default) or "x_on_y" — matches cointegration.py's
        bidirectional Engle-Granger result. Determines which leg's return
        is treated as the dependent leg when computing the spread return:
            y_on_x -> (y_ret - hedge_ratio * x_ret) / gross_exposure
            x_on_y -> (x_ret - hedge_ratio * y_ret) / gross_exposure
        Getting this wrong doesn't crash — it silently computes P&L for
        the wrong economic position. Every pair result from
        cointegration.py carries a `direction` field; thread it through
        here (see backtest_multiple_pairs below for the call pattern).

    Returns
    -------
    pd.DataFrame with columns: position, spread_return, strategy_return,
                                cumulative_return, equity_curve
    dict of summary performance metrics
    """
    idx = signals.index.intersection(y.index).intersection(x.index)
    y, x, signals = y.loc[idx], x.loc[idx], signals.loc[idx]

    y_ret = y.pct_change().fillna(0)
    x_ret = x.pct_change().fillna(0)

    # Normalize pair returns by gross exposure so hedge ratios do not inflate
    # reported performance.
    gross_exposure = 1.0 + abs(hedge_ratio)
    if direction == "y_on_x":
        spread_return = (y_ret - hedge_ratio * x_ret) / gross_exposure
    elif direction == "x_on_y":
        spread_return = (x_ret - hedge_ratio * y_ret) / gross_exposure
    else:
        raise ValueError(f"Unknown direction: {direction!r} (expected 'y_on_x' or 'x_on_y')")

    position = signals["position"].shift(1).fillna(0)  # trade on next bar
    strategy_return = position * spread_return

    # Transaction costs applied whenever position changes
    executed_change = position.diff().abs().fillna(0)
    cost = executed_change * (transaction_cost_bps / 10_000.0)
    strategy_return = strategy_return - cost

    cumulative_return = (1 + strategy_return).cumprod() - 1
    equity_curve = capital * (1 + cumulative_return)

    result = pd.DataFrame(
        {
            "position": position,
            "position_change": executed_change,
            "spread_return": spread_return,
            "strategy_return": strategy_return,
            "cumulative_return": cumulative_return,
            "equity_curve": equity_curve,
        }
    )

    metrics = compute_metrics(strategy_return, equity_curve, result["position_change"])
    return result, metrics


def compute_metrics(
    strategy_return: pd.Series,
    equity_curve: pd.Series,
    position_change: pd.Series | None = None,
    periods_per_year: int = 252,
):
    """Compute standard backtest performance metrics."""
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) else 0.0

    ann_return = (1 + strategy_return.mean()) ** periods_per_year - 1
    ann_vol = strategy_return.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    wins = (strategy_return > 0).sum()
    losses = (strategy_return < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan

    num_trades = int((position_change > 0).sum()) if position_change is not None else int((strategy_return != 0).sum())

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trades": num_trades,
    }


def backtest_multiple_pairs(
    prices: pd.DataFrame,
    pair_results: list,
    zscore_window: int = 30,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 3.5,
    **kwargs,
):
    """
    Convenience wrapper: run backtest_pair for a list of cointegration
    results (as returned by cointegration.find_all_cointegrated_pairs),
    using default signal parameters. Returns a summary DataFrame.

    pair_results: list of dicts with keys 'pair', 'hedge_ratio',
    'intercept', and 'direction' (added by cointegration.py's
    bidirectional test — defaults to "y_on_x" if a result predates it).
    """
    from src.signals import compute_spread, compute_zscore, generate_signals

    summaries = []
    for res in pair_results:
        t1, t2 = res["pair"]
        if t1 not in prices.columns or t2 not in prices.columns:
            continue
        y, x = prices[t1], prices[t2]
        direction = res.get("direction", "y_on_x")
        spread = compute_spread(y, x, res["hedge_ratio"], res.get("intercept", 0.0), direction=direction)
        z = compute_zscore(spread, window=zscore_window)
        sig = generate_signals(z, entry_threshold, exit_threshold, stop_loss)
        _, metrics = backtest_pair(y, x, res["hedge_ratio"], sig, direction=direction, **kwargs)
        metrics["pair"] = f"{t1}-{t2}"
        summaries.append(metrics)

    return pd.DataFrame(summaries).set_index("pair") if summaries else pd.DataFrame()
