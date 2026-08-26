"""
train_meta_model.py
--------------------
Standalone training script for the meta-labeling classifier.

WHY THIS IS SEPARATE FROM app.py:
Training needs a large date range and every historical trade signal
across ALL cointegrated pairs — not just the one pair a user happens to
be inspecting live in the dashboard. Re-running this every time someone
clicks "Run Pipeline" would be slow and pointless; train once (or
periodically, e.g. whenever you re-run with updated data), save the
model, and have the dashboard load it for inference later if you wire
that in.

Usage
-----
    python train_meta_model.py

This reads the same BFSI universe and pipeline stages app.py uses, but
skips Streamlit: downloads/caches prices, runs PCA + DBSCAN, finds
cointegrated pairs, extracts every historical trade signal across all
of them, labels each with the triple-barrier method, trains the
classifier, and saves it to models/meta_model.pkl.
"""

import sys

sys.path.insert(0, ".")

from src.downloader import download_universe, BFSI_TICKERS
from src.preprocessing import build_feature_matrix
from src.clustering import run_pca, run_dbscan, get_cluster_groups, suggest_eps
from src.cointegration import find_all_cointegrated_pairs
from src.signal_events import extract_signal_events
from src.meta_labeling import run_meta_labeling_pipeline, save_model

# --- Same defaults as the dashboard sidebar; edit these directly, or
# wire up argparse if you want CLI flags instead. ---
START, END = "2019-01-01", None
N_COMPONENTS, MIN_SAMPLES = 5, 2
SIGNIFICANCE = 0.05
ZSCORE_WINDOW, ENTRY_Z, EXIT_Z, STOP_Z = 30, 2.0, 0.5, 3.5
MODEL_TYPE = "logistic"  # or "random_forest"


def find_valid_clusters(components, min_samples, max_attempts=6, growth_factor=1.6):
    """
    PCA-component scale depends entirely on the data (universe size,
    volatility, date range) — there is no single eps that works across
    runs. The dashboard handles this by showing a suggested eps and
    letting the user nudge it; this script has no slider, so it starts
    from the same suggest_eps() heuristic and automatically escalates
    if DBSCAN puts everything into noise, rather than failing silently
    on a hardcoded value.
    """
    eps = suggest_eps(components, min_samples=min_samples)
    for attempt in range(1, max_attempts + 1):
        clustered = run_dbscan(components, eps=eps, min_samples=min_samples)
        groups = get_cluster_groups(clustered)
        n_clustered = sum(len(v) for v in groups.values())
        print(f"    attempt {attempt}: eps={eps:.3f} -> {len(groups)} cluster(s), "
              f"{n_clustered}/{len(components)} tickers clustered")
        if groups:
            return groups, eps
        eps *= growth_factor
    return {}, eps


def main():
    print("Downloading price history...")
    prices = download_universe(BFSI_TICKERS, start=START, end=END)
    print(f"  {prices.shape[1]} tickers, {prices.shape[0]} trading days")

    print("Building features and running PCA + DBSCAN...")
    feature_matrix = build_feature_matrix(prices)
    components, _, _ = run_pca(feature_matrix, n_components=N_COMPONENTS)
    groups, eps_used = find_valid_clusters(components, MIN_SAMPLES)
    if not groups:
        print(f"\nDBSCAN could not form any clusters even after escalating eps "
              f"up to {eps_used:.3f}. Try lowering MIN_SAMPLES (currently "
              f"{MIN_SAMPLES}) or check that your universe has enough "
              f"correlated tickers after cleaning.")
        return
    print(f"  {len(groups)} clusters formed using eps={eps_used:.3f}")

    print("Testing pairs for cointegration (Benjamini-Hochberg corrected)...")
    pair_results = find_all_cointegrated_pairs(prices, groups, significance=SIGNIFICANCE)
    print(f"  {len(pair_results)} cointegrated pairs found")
    if not pair_results:
        print("No cointegrated pairs found — widen the date range or relax "
              "the significance level before training. Nothing to train on.")
        return

    print("Extracting historical trade signals from every pair...")
    signals_df, spread_zscores = extract_signal_events(
        prices, pair_results, ZSCORE_WINDOW, ENTRY_Z, EXIT_Z, STOP_Z
    )
    print(f"  {len(signals_df)} historical trade signals across all pairs")
    if len(signals_df) < 20:
        print("Fewer than 20 historical signals — too few to train reliably. "
              "Try a wider date range, more relaxed entry threshold, or a "
              "larger cluster (more pairs).")
        return

    print(f"Labeling trades (triple-barrier) and training the '{MODEL_TYPE}' meta-model...")
    result = run_meta_labeling_pipeline(signals_df, spread_zscores, model_type=MODEL_TYPE)

    m = result["metrics"]
    print("\n--- Meta-model evaluation (held-out, later-in-time trades) ---")
    print(f"  test trades:            {m['n_test_trades']}")
    print(f"  base rate (no filter):  {m['base_rate']:.2%}")
    print(f"  ROC-AUC:                {m['roc_auc']:.3f}")
    print("  (compare ROC-AUC to 0.5 = no better than random, and the base "
          "rate to see whether filtering actually improves the win rate.)")

    save_model(result["model"], result["scaler"], "models/meta_model")
    print("\nSaved: models/meta_model.pkl, models/meta_model_scaler.pkl")
    print("Load these later with meta_labeling.load_model('models/meta_model') "
          "to score new signals via apply_meta_filter().")


if __name__ == "__main__":
    main()
