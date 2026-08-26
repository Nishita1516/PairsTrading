"""
src/meta_backtest.py

Applies a TRAINED meta-labeling model (from train_meta_model.py) to
LIVE signals in the dashboard, before backtesting — the piece that
actually connects the saved model back into app.py. Doesn't modify
signals.py, backtest.py, or meta_labeling.py; this is a thin
orchestration layer on top of all three plus signal_events.py.
"""

import pandas as pd

from src.backtest import backtest_pair
from src.meta_labeling import apply_meta_filter, build_features
from src.signal_events import extract_entries_from_signal
from src.signals import compute_spread, compute_zscore, generate_signals


def apply_meta_filter_to_signals(
    sig: pd.DataFrame,
    spread_z: pd.Series,
    pair,
    hedge_ratio: float,
    coint_pvalue: float,
    reg_direction: str,
    model,
    scaler,
    threshold: float = 0.55,
    lookback: int = 60,
):
    """
    Score every entry in `sig` with the trained meta-model and zero out
    the position for any trade the model doesn't clear `threshold` on —
    holding at 0 for that trade's entire would-be duration, then
    resuming normal signal behavior for the next entry.

    Returns
    -------
    filtered_sig : pd.DataFrame, same shape as `sig`, with `position`
        (and recomputed `signal_change`) replaced by the meta-filtered
        version.
    trade_log : pd.DataFrame, one row per entry event with meta_prob
        and meta_take_trade attached — for display in the dashboard.
        Empty if there were no entries, or too little history before
        the first entry to build features (lookback not yet available).
    """
    entries = extract_entries_from_signal(sig, pair, hedge_ratio, coint_pvalue, reg_direction)
    if entries.empty:
        return sig.copy(), pd.DataFrame()

    # build_features() expects a "label" column (normally added by
    # label_all_signals() during training via the triple-barrier
    # method). For LIVE signals the outcome isn't known yet, so this is
    # a placeholder — it's never read, since "label" isn't in
    # meta_labeling.FEATURE_COLS (the columns actually used for scoring).
    entries = entries.copy()
    entries["label"] = 0

    features = build_features(entries, {pair: spread_z}, lookback=lookback)
    if features.empty:
        # Not enough history before any entry to build features (e.g.
        # every entry falls within `lookback` days of the series
        # start). Can't score these — pass them through unfiltered
        # rather than silently dropping trades.
        return sig.copy(), pd.DataFrame()

    scored = apply_meta_filter(features, model, scaler, threshold=threshold)
    vetoed_dates = set(scored.loc[~scored["meta_take_trade"], "entry_date"])

    new_position = sig["position"].copy()
    if vetoed_dates:
        values = new_position.to_numpy(copy=True)
        dates = new_position.index
        i, n = 0, len(dates)
        while i < n:
            if dates[i] in vetoed_dates and values[i] != 0:
                held_value = values[i]
                j = i
                while j < n and values[j] == held_value:
                    values[j] = 0
                    j += 1
                i = j
            else:
                i += 1
        new_position = pd.Series(values, index=dates)

    filtered_sig = sig.copy()
    filtered_sig["position"] = new_position
    filtered_sig["signal_change"] = filtered_sig["position"].diff().fillna(0) != 0

    return filtered_sig, scored


def backtest_multiple_pairs_with_meta_filter(
    prices: pd.DataFrame,
    pair_results: list,
    model,
    scaler,
    zscore_window: int = 30,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 3.5,
    meta_threshold: float = 0.55,
    **kwargs,
):
    """
    Meta-filtered counterpart to backtest.backtest_multiple_pairs() —
    same signature and return shape, but every pair's signals are
    scored and filtered by the meta-model before backtesting. Use this
    in place of backtest_multiple_pairs() when the meta-model filter
    toggle is on.
    """
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

        filtered_sig, _ = apply_meta_filter_to_signals(
            sig, z, res["pair"], res["hedge_ratio"],
            res.get("eg_pvalue", res.get("adf_pvalue")), direction,
            model, scaler, threshold=meta_threshold,
        )

        _, metrics = backtest_pair(y, x, res["hedge_ratio"], filtered_sig, direction=direction, **kwargs)
        metrics["pair"] = f"{t1}-{t2}"
        summaries.append(metrics)

    return pd.DataFrame(summaries).set_index("pair") if summaries else pd.DataFrame()
