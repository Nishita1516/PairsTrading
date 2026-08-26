"""
app.py
------
Streamlit dashboard for the PCA + DBSCAN pairs-trading pipeline.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.downloader import download_universe, BFSI_TICKERS
from src.preprocessing import build_feature_matrix
from src.clustering import run_pca, run_dbscan, get_cluster_groups, suggest_eps
from src.cointegration import find_all_cointegrated_pairs
from src.signals import compute_spread, compute_zscore, generate_signals
from src.backtest import backtest_pair, backtest_multiple_pairs
from src.meta_labeling import load_model
from src.meta_backtest import apply_meta_filter_to_signals, backtest_multiple_pairs_with_meta_filter


st.set_page_config(page_title="Pairs Trading Dashboard", layout="wide")
st.title("Statistical Arbitrage: PCA + DBSCAN Pairs Trading")
st.caption("BFSI (Banking & NBFC) universe | PCA clustering | Engle-Granger cointegration | Benjamini-Hochberg corrected | Z-score signals | Backtest")

with st.sidebar:
    st.header("Configuration")

    start_date = st.date_input("Start date", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End date", pd.to_datetime("today"))

    st.subheader("Clustering")
    n_components = st.slider("PCA components", 2, 10, 5)
    eps = st.slider("DBSCAN eps", 0.1, 5.0, 1.5, 0.1)
    min_samples = st.slider("DBSCAN min_samples", 2, 10, 2)

    st.subheader("Cointegration")
    significance = st.slider("Significance level (p-value)", 0.01, 0.10, 0.05, 0.01)

    st.subheader("Signals")
    entry_threshold = st.slider("Entry z-score", 1.0, 3.5, 2.0, 0.1)
    exit_threshold = st.slider("Exit z-score", 0.0, 1.5, 0.5, 0.1)
    stop_loss = st.slider("Stop-loss z-score", 2.5, 5.0, 3.5, 0.1)
    zscore_window = st.slider("Z-score rolling window (days)", 10, 90, 30)

    st.subheader("Data")
    force_refresh = st.checkbox(
        "Force re-download data",
        value=False,
        help="Delete cached CSV and re-download from yfinance",
    )

    st.subheader("Meta-Model Filter")
    MODEL_PATH = "models/meta_model"
    meta_model_available = os.path.exists(f"{MODEL_PATH}.pkl") and os.path.exists(f"{MODEL_PATH}_scaler.pkl")
    if meta_model_available:
        use_meta_filter = st.checkbox("Apply meta-model filter to signals", value=False)
        meta_threshold = st.slider("Meta-model probability threshold", 0.0, 1.0, 0.55, 0.05)
    else:
        use_meta_filter = False
        meta_threshold = 0.55
        st.caption(
            "No trained meta-model found. Run `python train_meta_model.py` "
            "from the project root to enable this filter."
        )

    run_button = st.button("Run Pipeline", type="primary")

# Load the meta-model once per script run (not per pair) if the filter is on.
meta_model, meta_scaler = None, None
if use_meta_filter and meta_model_available:
    try:
        meta_model, meta_scaler = load_model(MODEL_PATH)
    except Exception as e:
        st.sidebar.warning(f"Could not load meta-model: {e}")
        use_meta_filter = False

if run_button:
    with st.spinner("Downloading price data..."):
        prices = download_universe(
            BFSI_TICKERS,
            start=str(start_date),
            end=str(end_date),
            force_refresh=force_refresh,
        )

    n_returned = prices.shape[1]
    n_requested = len(BFSI_TICKERS)
    n_failed = n_requested - n_returned
    if n_failed > 0:
        st.warning(
            f"**{n_failed} ticker(s) failed to download** - only "
            f"{n_returned}/{n_requested} stocks in the universe. "
            "Fewer stocks means smaller clusters and fewer pairs. "
            "Check the terminal/logs for downloader failure lines."
        )
    st.success(f"Downloaded prices for **{n_returned}** stocks, **{prices.shape[0]}** trading days.")
    st.session_state["prices"] = prices

    with st.spinner("Building features and running PCA..."):
        feature_matrix = build_feature_matrix(prices)
        components, explained_var, _ = run_pca(feature_matrix, n_components=n_components)

    with st.spinner("Clustering with DBSCAN..."):
        clustered = run_dbscan(components, eps=eps, min_samples=min_samples)
        groups = get_cluster_groups(clustered)
    st.session_state["clustered"] = clustered
    st.session_state["groups"] = groups

    cluster_counts = clustered["cluster"].value_counts().sort_index()
    noise_count = cluster_counts.get(-1, 0)
    valid_cluster_count = int((cluster_counts.index >= 0).sum())
    if noise_count == len(clustered):
        st.error(
            f"**DBSCAN put all {noise_count} stocks into noise (cluster -1).** "
            "No valid clusters means zero pairs will be tested. "
            "Try increasing eps, for example 2.0-3.0, or decreasing min_samples to 2."
        )
    elif noise_count > len(clustered) * 0.5:
        st.warning(
            f"{noise_count}/{len(clustered)} stocks are noise (cluster -1). "
            f"Only {valid_cluster_count} cluster(s) formed. Consider raising eps."
        )
    with st.expander("Cluster label distribution (debug)", expanded=False):
        st.dataframe(cluster_counts.rename("count").rename_axis("cluster_id"))

    with st.spinner("Testing pairs for cointegration..."):
        pair_results = find_all_cointegrated_pairs(prices, groups, significance=significance)
    st.session_state["pair_results"] = pair_results

    st.success(f"Found {len(groups)} clusters and {len(pair_results)} cointegrated pairs.")

if "prices" in st.session_state:
    prices = st.session_state["prices"]
    clustered = st.session_state["clustered"]
    groups = st.session_state["groups"]
    pair_results = st.session_state["pair_results"]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Clusters", "Cointegrated Pairs", "Pair Detail & Signals", "Backtest Summary"]
    )

    with tab1:
        st.subheader("PCA Explained Variance & Clusters")
        fig, ax = plt.subplots()
        ax.scatter(clustered["PC1"], clustered["PC2"], c=clustered["cluster"], cmap="tab20")
        for ticker, row in clustered.iterrows():
            ax.annotate(ticker.replace(".NS", ""), (row["PC1"], row["PC2"]), fontsize=6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

        pc_cols = [c for c in clustered.columns if c.startswith("PC")]
        st.write(
            f"Suggested eps (heuristic): "
            f"**{suggest_eps(clustered[pc_cols], min_samples=min_samples):.3f}**"
        )
        st.dataframe(clustered)

    with tab2:
        st.subheader("Cointegrated Pairs (ranked by p-value)")
        if pair_results:
            df = pd.DataFrame(pair_results)
            df["pair_label"] = df["pair"].apply(lambda p: f"{p[0].replace('.NS', '')} / {p[1].replace('.NS', '')}")
            df["pair"] = df["pair"].apply(lambda p: f"{p[0]} / {p[1]}")

            st.markdown("**Top prominent pairs**")
            top_n = min(3, len(df))
            cols = st.columns(top_n)
            ranks = ["#1", "#2", "#3"]
            for i, (col, (_, row)) in enumerate(zip(cols, df.head(top_n).iterrows())):
                with col:
                    st.metric(ranks[i], row["pair_label"], f"EG p={row['eg_pvalue']:.4f}")
                    st.caption(
                        f"Cluster {int(row['cluster'])} | "
                        f"ADF p={row['adf_pvalue']:.4f} | "
                        f"Hedge beta={row['hedge_ratio']:.3f}"
                    )

            display_df = df[["pair_label", "cluster", "eg_pvalue", "adf_pvalue", "hedge_ratio", "intercept"]].copy()
            display_df.columns = ["Pair", "Cluster", "EG p-value", "ADF p-value", "Hedge Ratio", "Intercept"]
            display_df["Cluster"] = display_df["Cluster"].astype(int)

            styled = (
                display_df.style
                .format(
                    {
                        "EG p-value": "{:.4f}",
                        "ADF p-value": "{:.4f}",
                        "Hedge Ratio": "{:.4f}",
                        "Intercept": "{:.2f}",
                    }
                )
                .background_gradient(subset=["EG p-value"], cmap="RdYlGn_r", vmin=0, vmax=significance)
                .background_gradient(subset=["ADF p-value"], cmap="RdYlGn_r", vmin=0, vmax=significance)
                .set_properties(**{"text-align": "center"})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("No cointegrated pairs found. Try relaxing the p-value or adjusting DBSCAN.")

    with tab3:
        st.subheader("Inspect a Pair")
        if pair_results:
            pair_labels = [f"{r['pair'][0]} / {r['pair'][1]}" for r in pair_results]
            choice = st.selectbox("Select pair", pair_labels)
            res = pair_results[pair_labels.index(choice)]
            t1, t2 = res["pair"]

            y, x = prices[t1], prices[t2]
            # direction comes from cointegration.py's bidirectional Engle-Granger
            # test -- required now, since which of t1/t2 was actually the
            # regressed-on-variable can flip per pair. Defaults to the old
            # fixed "y_on_x" behavior if a result predates this field.
            reg_direction = res.get("direction", "y_on_x")
            spread = compute_spread(
                y, x, res["hedge_ratio"], res.get("intercept", 0.0),
                direction=reg_direction,
            )
            z = compute_zscore(spread, window=zscore_window)
            sig = generate_signals(z, entry_threshold, exit_threshold, stop_loss)

            if use_meta_filter and meta_model is not None:
                sig_for_backtest, trade_log = apply_meta_filter_to_signals(
                    sig, z, res["pair"], res["hedge_ratio"],
                    res.get("eg_pvalue", res.get("adf_pvalue")), reg_direction,
                    meta_model, meta_scaler, threshold=meta_threshold,
                )
                if not trade_log.empty:
                    n_total = len(trade_log)
                    n_kept = int(trade_log["meta_take_trade"].sum())
                    st.caption(
                        f"Meta-model filter: kept **{n_kept}/{n_total}** "
                        f"historical entries at threshold {meta_threshold:.2f}."
                    )
                else:
                    st.caption("Meta-model filter: not enough history before the "
                               "first entry to score it — showing unfiltered signals.")
            else:
                sig_for_backtest = sig

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(y, label=t1)
            axes[0].plot(x, label=t2)
            axes[0].legend()
            axes[0].set_title("Prices")

            axes[1].plot(spread, color="purple")
            axes[1].set_title("Spread")

            axes[2].plot(z, color="black")
            axes[2].axhline(entry_threshold, color="red", linestyle="--")
            axes[2].axhline(-entry_threshold, color="green", linestyle="--")
            axes[2].axhline(0, color="gray", linestyle=":")
            axes[2].set_title("Z-score")

            st.pyplot(fig)

            result, metrics = backtest_pair(
                y, x, res["hedge_ratio"], sig_for_backtest,
                direction=reg_direction,
            )
            metrics_label = "**Performance metrics**" + (
                " (meta-filtered)" if use_meta_filter and meta_model is not None else ""
            )
            st.write(metrics_label)
            st.json(metrics)
            st.line_chart(result["equity_curve"])
        else:
            st.info("Run the pipeline first and ensure pairs were found.")

    with tab4:
        st.subheader("Backtest Summary Across All Pairs")
        if pair_results:
            if use_meta_filter and meta_model is not None:
                summary = backtest_multiple_pairs_with_meta_filter(
                    prices,
                    pair_results,
                    meta_model,
                    meta_scaler,
                    zscore_window=zscore_window,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    stop_loss=stop_loss,
                    meta_threshold=meta_threshold,
                )
                st.caption(f"Meta-model filter active (threshold {meta_threshold:.2f}).")
            else:
                summary = backtest_multiple_pairs(
                    prices,
                    pair_results,
                    zscore_window=zscore_window,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold,
                    stop_loss=stop_loss,
                )

            if not summary.empty:
                best_idx = summary["sharpe_ratio"].idxmax()
                best = summary.loc[best_idx]
                t1_lbl, t2_lbl = best_idx.replace(".NS", "").split("-", 1)
                st.success(
                    f"Best pair: {t1_lbl} / {t2_lbl} | "
                    f"Sharpe {best['sharpe_ratio']:.3f} | "
                    f"Return {best['total_return'] * 100:.2f}% | "
                    f"Max drawdown {best['max_drawdown'] * 100:.2f}% | "
                    f"Trades {int(best['num_trades'])}"
                )

                def _sharpe_color(v):
                    if v >= 1.0:
                        return "background-color:#052e16;color:#4ade80;font-weight:700"
                    if v >= 0.5:
                        return "background-color:#1c1917;color:#facc15;font-weight:600"
                    return "background-color:#1c0505;color:#f87171"

                styled_summary = (
                    summary.style
                    .format(
                        {
                            "total_return": "{:.2%}",
                            "annualized_return": "{:.2%}",
                            "annualized_volatility": "{:.2%}",
                            "sharpe_ratio": "{:.3f}",
                            "max_drawdown": "{:.2%}",
                            "win_rate": "{:.1%}",
                            "num_trades": "{:.0f}",
                        }
                    )
                    .map(_sharpe_color, subset=["sharpe_ratio"])
                    .background_gradient(subset=["total_return"], cmap="RdYlGn", vmin=-0.3, vmax=0.3)
                    .background_gradient(subset=["max_drawdown"], cmap="RdYlGn", vmin=-0.5, vmax=0)
                    .highlight_max(subset=["win_rate"], color="#1e3a5f")
                    .highlight_max(subset=["sharpe_ratio"], color="#052e16")
                )
                st.dataframe(styled_summary, use_container_width=True)
            else:
                st.warning("Backtest produced no results.")
        else:
            st.info("No pairs to backtest.")
else:
    st.info("Configure parameters in the sidebar and click **Run Pipeline** to begin.")
