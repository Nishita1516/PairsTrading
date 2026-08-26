"""
src/signal_events.py

Bridges cointegration.py + signals.py output into the `signals_df` and
`spread_zscores` shapes meta_labeling.py expects for training. This is
the missing piece between "pairs found + signals generated" and "train
the meta-labeling classifier" — run this (via train_meta_model.py)
BEFORE calling run_meta_labeling_pipeline().

Also exposes extract_entries_from_signal(), the single-pair entry
detector shared with src/meta_backtest.py (live filtering of one
pair's signals, rather than building a training set across all pairs).

NAMING NOTE: cointegration.py's `direction` field means the REGRESSION
direction ("y_on_x" / "x_on_y"). meta_labeling.py's `direction` field
means the TRADE direction (+1 long the spread / -1 short the spread).
These are unrelated concepts that happen to share a name in two
different files. This module keeps them distinct: the regression
direction is renamed `reg_direction` here, and `direction` always means
trade direction, matching what meta_labeling.py's triple-barrier
labeling expects.
"""

import pandas as pd

from src.signals import compute_spread, compute_zscore, generate_signals


def extract_entries_from_signal(sig: pd.DataFrame, pair, hedge_ratio,
                                 coint_pvalue, reg_direction) -> pd.DataFrame:
    """
    Walk one pair's already-computed signals DataFrame (from
    signals.generate_signals) and return one row per ENTRY event
    (position moving from 0 to +-1).

    Shared by extract_signal_events() below (builds a training set
    across ALL pairs) and meta_backtest.apply_meta_filter_to_signals()
    (scores live signals for ONE pair) — keeping this logic in one
    place so the two can't silently drift apart.
    """
    events = []
    prev_position = 0
    for date, row in sig.iterrows():
        pos = row["position"]
        if prev_position == 0 and pos != 0:
            events.append({
                "entry_date": date,
                "pair": pair,
                "direction": int(pos),  # TRADE direction, +1/-1 — not reg_direction
                "z_score": row["zscore"],
                "hedge_ratio": hedge_ratio,
                "coint_pvalue": coint_pvalue,
                "reg_direction": reg_direction,
            })
        prev_position = pos
    return pd.DataFrame(events)


def extract_signal_events(
    prices: pd.DataFrame,
    pair_results: list,
    zscore_window: int = 30,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 3.5,
):
    """
    For every cointegrated pair, generate its z-score/signal series and
    extract one row per ENTRY event — these are the "trades"
    meta_labeling.py's triple-barrier method labels as win/loss.

    Parameters
    ----------
    prices : wide price DataFrame (date x ticker)
    pair_results : list[dict], output of cointegration.find_all_cointegrated_pairs()

    Returns
    -------
    signals_df : pd.DataFrame, one row per entry event, columns:
        entry_date, pair, direction (+1/-1 trade direction), z_score,
        hedge_ratio, coint_pvalue, reg_direction (kept for reference,
        not used as a training feature — see NAMING NOTE above).
    spread_zscores : dict[pair -> pd.Series]
        Full z-score series per pair, needed by meta_labeling's
        triple-barrier labeling to look forward from each entry date.
    """
    all_events = []
    spread_zscores = {}

    for res in pair_results:
        t1, t2 = res["pair"]
        if t1 not in prices.columns or t2 not in prices.columns:
            continue

        y, x = prices[t1], prices[t2]
        reg_direction = res.get("direction", "y_on_x")
        spread = compute_spread(
            y, x, res["hedge_ratio"], res.get("intercept", 0.0),
            direction=reg_direction,
        )
        z = compute_zscore(spread, window=zscore_window)
        sig = generate_signals(z, entry_threshold, exit_threshold, stop_loss)
        spread_zscores[res["pair"]] = z

        pair_events = extract_entries_from_signal(
            sig, res["pair"], res["hedge_ratio"],
            res.get("eg_pvalue", res.get("adf_pvalue")), reg_direction,
        )
        if not pair_events.empty:
            all_events.append(pair_events)

    if not all_events:
        return pd.DataFrame(), spread_zscores

    signals_df = pd.concat(all_events, ignore_index=True)
    return signals_df, spread_zscores
