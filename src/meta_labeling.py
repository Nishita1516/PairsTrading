"""
src/meta_labeling.py

Meta-labeling layer for the statistical arbitrage pipeline.

PURPOSE
-------
The primary model (PCA + DBSCAN clustering -> Engle-Granger cointegration
-> rolling z-score signal) decides WHEN a trade signal fires. This module
adds a secondary, supervised classifier that decides WHETHER to act on a
given signal. This is the "meta-labeling" approach (Lopez de Prado,
Advances in Financial Machine Learning, 2018).

It does NOT replace the primary model or its statistical rigor - it sits
downstream of signals.py and upstream of backtest.py:

    cointegration.py -> signals.py -> [meta_labeling.py] -> backtest.py

INTEGRATION ASSUMPTION
-----------------------
This module expects a `signals_df` with one row per candidate trade entry,
containing at least these columns:
    entry_date : pd.Timestamp
    pair       : tuple(ticker_a, ticker_b)
    direction  : +1 (long the spread) or -1 (short the spread)
    z_score    : float, the z-score at entry
    coint_pvalue, hedge_ratio : float, from cointegration.py

And a `spread_zscores` dict mapping pair -> pd.Series of the rolling
z-score, indexed by date, so this module can look forward (for labeling)
and backward (for features) relative to each entry_date.

Adjust the column/dict names below if your actual signals.py output
differs - the logic is what matters, not the exact interface.
"""

import warnings

import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

FEATURE_COLS = [
    "z_score_entry",
    "coint_pvalue",
    "hedge_ratio",
    "spread_vol_60d",
    "z_score_mean_60d",
    "z_score_abs_mean_60d",
    "z_score_momentum_5d",
]


# ---------------------------------------------------------------------------
# 1. Triple-barrier labeling
# ---------------------------------------------------------------------------

def triple_barrier_label(
    spread_path: pd.Series,
    entry_idx: int,
    direction: int,
    entry_z: float,
    profit_target_z: float = 0.5,
    stop_loss_z: float = 3.0,
    max_holding_days: int = 20,
) -> dict:
    """
    Apply the triple-barrier method to a single trade.

    Three exit conditions ("barriers") are checked day by day after entry:
      - profit target: the spread reverted `profit_target_z` toward its mean
      - stop loss: the spread moved `stop_loss_z` further against the trade
      - time barrier: `max_holding_days` elapsed with neither hit

    Parameters
    ----------
    spread_path : pd.Series
        Z-score of the spread, indexed by date, covering the period
        after entry_idx.
    entry_idx : int
        Positional row index in spread_path where the trade begins.
    direction : int
        +1 for long-the-spread (bet the z-score falls back toward 0),
        -1 for short-the-spread (bet the z-score rises back toward 0).
    entry_z : float
        Z-score at entry.

    Returns
    -------
    dict with keys: label (1=win, 0=loss), exit_reason, exit_date,
    holding_days, pnl_z (favorable z-score movement realized)
    """
    path = spread_path.iloc[entry_idx + 1: entry_idx + 1 + max_holding_days]
    if path.empty:
        return {
            "label": np.nan, "exit_reason": "no_data", "exit_date": None,
            "holding_days": 0, "pnl_z": 0.0,
        }

    for i, (date, z) in enumerate(path.items(), start=1):
        # "move" is positive when the spread has moved in the trade's favor
        move = direction * (entry_z - z)
        if move >= profit_target_z:
            return {
                "label": 1, "exit_reason": "profit_target", "exit_date": date,
                "holding_days": i, "pnl_z": move,
            }
        if move <= -stop_loss_z:
            return {
                "label": 0, "exit_reason": "stop_loss", "exit_date": date,
                "holding_days": i, "pnl_z": move,
            }

    # Time barrier hit before either target: label by sign of final move.
    final_move = direction * (entry_z - path.iloc[-1])
    return {
        "label": int(final_move > 0), "exit_reason": "time_barrier",
        "exit_date": path.index[-1], "holding_days": len(path), "pnl_z": final_move,
    }


def label_all_signals(
    signals_df: pd.DataFrame,
    spread_zscores: dict,
    profit_target_z: float = 0.5,
    stop_loss_z: float = 3.0,
    max_holding_days: int = 20,
) -> pd.DataFrame:
    """Apply triple-barrier labeling to every signal in signals_df."""
    records = []
    for _, row in signals_df.iterrows():
        z_series = spread_zscores.get(row["pair"])
        if z_series is None or row["entry_date"] not in z_series.index:
            continue
        entry_idx = z_series.index.get_loc(row["entry_date"])
        result = triple_barrier_label(
            z_series, entry_idx, row["direction"], row["z_score"],
            profit_target_z, stop_loss_z, max_holding_days,
        )
        records.append({**row.to_dict(), **result})

    labeled = pd.DataFrame(records)
    if labeled.empty:
        return labeled
    labeled = labeled.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)
    return labeled


# ---------------------------------------------------------------------------
# 2. Feature engineering — signal-time only, no lookahead
# ---------------------------------------------------------------------------

def build_features(
    labeled_df: pd.DataFrame,
    spread_zscores: dict,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Build features using ONLY information available at entry_date.

    Every rolling statistic below is computed on the window ENDING at
    entry_date (inclusive) — never beyond it. This is the single most
    important correctness property of this module: leaking a single
    future bar into a feature silently inflates backtest performance.
    """
    rows = []
    for _, row in labeled_df.iterrows():
        z_series = spread_zscores.get(row["pair"])
        if z_series is None or row["entry_date"] not in z_series.index:
            continue
        entry_idx = z_series.index.get_loc(row["entry_date"])
        window = z_series.iloc[max(0, entry_idx - lookback + 1): entry_idx + 1]
        if len(window) < max(10, lookback // 3):
            continue  # not enough history yet to trust these features

        rows.append({
            "entry_date": row["entry_date"],
            "pair": row["pair"],
            "z_score_entry": row["z_score"],
            "coint_pvalue": row.get("coint_pvalue", np.nan),
            "hedge_ratio": row.get("hedge_ratio", np.nan),
            "spread_vol_60d": window.std(),
            "z_score_mean_60d": window.mean(),
            "z_score_abs_mean_60d": window.abs().mean(),
            "z_score_momentum_5d": window.iloc[-1] - window.iloc[-6] if len(window) > 5 else 0.0,
            "direction": row["direction"],
            "label": row["label"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Time-based train/test split
# ---------------------------------------------------------------------------

def time_based_split(features_df: pd.DataFrame, test_frac: float = 0.3):
    """
    Sort by entry_date and hold out the LAST `test_frac` of trades as the
    test set. Never use random k-fold on trade events — nearby trades
    share market conditions, so random shuffling leaks information across
    the split and overstates performance.
    """
    df_sorted = features_df.sort_values("entry_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------

def train_meta_model(train_df: pd.DataFrame, model_type: str = "logistic"):
    """
    Train the secondary (meta) model.

    Given the likely small number of historical trades from a NIFTY 50
    universe, logistic regression is the sane default. random_forest is
    offered as a nonlinear comparison, not a default choice.
    """
    X_train = train_df[FEATURE_COLS].fillna(train_df[FEATURE_COLS].median())
    y_train = train_df["label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if model_type == "logistic":
        model = LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=RANDOM_STATE,
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    if len(y_train) < 50:
        warnings.warn(
            f"Only {len(y_train)} training trades available — meta-model "
            "results will be unstable. Treat this as illustrative, not "
            "production-grade, until more signal history is collected."
        )

    model.fit(X_train_scaled, y_train)
    return model, scaler


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, scaler, test_df: pd.DataFrame) -> dict:
    """Evaluate on the held-out (later-in-time) test trades."""
    X_test = test_df[FEATURE_COLS].fillna(test_df[FEATURE_COLS].median())
    y_test = test_df["label"]
    X_test_scaled = scaler.transform(X_test)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    preds = model.predict(X_test_scaled)

    return {
        "n_test_trades": len(y_test),
        "base_rate": float(y_test.mean()),  # win rate with NO filtering at all
        "roc_auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else np.nan,
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


# ---------------------------------------------------------------------------
# 6. Applying the filter to new / live signals
# ---------------------------------------------------------------------------

def apply_meta_filter(
    new_signals_features: pd.DataFrame, model, scaler, threshold: float = 0.55
) -> pd.DataFrame:
    """
    Score new signals with the trained meta-model. Returns the input
    frame with two extra columns: meta_prob and meta_take_trade.
    Feed only rows where meta_take_trade is True into backtest.py.
    """
    X = new_signals_features[FEATURE_COLS].fillna(new_signals_features[FEATURE_COLS].median())
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    out = new_signals_features.copy()
    out["meta_prob"] = probs
    out["meta_take_trade"] = probs >= threshold
    return out


# ---------------------------------------------------------------------------
# 7. Persistence
# ---------------------------------------------------------------------------

def save_model(model, scaler, path_prefix: str = "models/meta_model") -> None:
    directory = os.path.dirname(path_prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, f"{path_prefix}.pkl")
    joblib.dump(scaler, f"{path_prefix}_scaler.pkl")


def load_model(path_prefix: str = "models/meta_model"):
    model = joblib.load(f"{path_prefix}.pkl")
    scaler = joblib.load(f"{path_prefix}_scaler.pkl")
    return model, scaler


# ---------------------------------------------------------------------------
# 8. End-to-end orchestration
# ---------------------------------------------------------------------------

def run_meta_labeling_pipeline(
    signals_df: pd.DataFrame,
    spread_zscores: dict,
    model_type: str = "logistic",
    profit_target_z: float = 0.5,
    stop_loss_z: float = 3.0,
    max_holding_days: int = 20,
    lookback: int = 60,
    test_frac: float = 0.3,
) -> dict:
    """
    Full pipeline in one call: label -> feature engineer -> time-split ->
    train -> evaluate. Returns everything the dashboard needs to display.
    """
    labeled = label_all_signals(
        signals_df, spread_zscores, profit_target_z, stop_loss_z, max_holding_days
    )
    if labeled.empty:
        raise ValueError("No signals could be labeled — check pair keys and date alignment.")

    features = build_features(labeled, spread_zscores, lookback)
    if len(features) < 20:
        raise ValueError(
            f"Only {len(features)} labeled trades available — too few to train a "
            "meta-model reliably. Widen the date range or relax the cointegration "
            "significance level to generate more historical signals first."
        )

    train_df, test_df = time_based_split(features, test_frac)
    model, scaler = train_meta_model(train_df, model_type)
    metrics = evaluate_model(model, scaler, test_df)

    return {
        "model": model,
        "scaler": scaler,
        "labeled": labeled,
        "features": features,
        "train_df": train_df,
        "test_df": test_df,
        "metrics": metrics,
    }
