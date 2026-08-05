# Pairs Trading Dashboard — PCA + DBSCAN + Cointegration

An end-to-end statistical arbitrage research pipeline for NIFTY 50 stocks,
built as an MSc research project (3-credit) and designed to double as a
portfolio piece for quant analyst roles.

## Pipeline

```
Data Layer (yfinance)
    ↓
Feature Engineering (standardized log returns)
    ↓
PCA (dimensionality reduction)
    ↓
DBSCAN Clustering (group similar stocks)
    ↓
Engle-Granger Cointegration Test (within each cluster)
    ↓
Spread & Z-score Signal Generation
    ↓
Backtest (P&L, Sharpe, drawdown, win rate)
    ↓
Streamlit Dashboard
```

## Project Structure

```
PairsTrading/
├── data/                  # cached price CSVs (gitignore this in practice)
├── notebooks/             # exploratory analysis (optional)
├── src/
│   ├── downloader.py      # yfinance data download + caching
│   ├── preprocessing.py   # cleaning, log returns, feature matrix
│   ├── clustering.py      # PCA + DBSCAN
│   ├── cointegration.py   # Engle-Granger pair testing
│   ├── signals.py         # z-score + entry/exit/stop-loss logic
│   └── backtest.py        # P&L simulation and performance metrics
├── app.py                 # Streamlit dashboard (main entry point)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

This opens a browser window where you can:
1. Set the date range and download NIFTY 50 data.
2. Tune PCA components and DBSCAN `eps` / `min_samples`.
3. View clusters in PC1/PC2 space.
4. See ranked cointegrated pairs (Engle-Granger p-value).
5. Inspect an individual pair's spread, z-score, and trading signals.
6. Backtest a single pair or all pairs at once, with performance metrics.

## Running Components Individually (for notebooks / research)

```python
from src.downloader import download_universe, NIFTY50_TICKERS
from src.preprocessing import build_feature_matrix
from src.clustering import run_pca, run_dbscan, get_cluster_groups
from src.cointegration import find_all_cointegrated_pairs
from src.signals import compute_spread, compute_zscore, generate_signals
from src.backtest import backtest_pair

prices = download_universe(NIFTY50_TICKERS, start="2021-01-01")
features = build_feature_matrix(prices)
components, var_ratio, _ = run_pca(features, n_components=5)
clustered = run_dbscan(components, eps=1.5, min_samples=2)
groups = get_cluster_groups(clustered)
pairs = find_all_cointegrated_pairs(prices, groups, significance=0.05)

t1, t2 = pairs[0]["pair"]
spread = compute_spread(prices[t1], prices[t2], pairs[0]["hedge_ratio"])
z = compute_zscore(spread)
signals = generate_signals(z)
result, metrics = backtest_pair(prices[t1], prices[t2], pairs[0]["hedge_ratio"], signals)
print(metrics)
```

## Methodology Notes

- **Feature matrix**: each stock is represented by its standardized daily
  log-return time series. Standardizing avoids PCA being dominated by
  high-volatility names.
- **PCA**: reduces the return series to a handful of principal components
  capturing common risk factors (broadly analogous to sector/market betas).
- **DBSCAN**: clusters stocks in PCA space. Unlike k-means, it doesn't
  require specifying the number of clusters upfront and naturally labels
  outliers as noise (`cluster == -1`), which are excluded from pair search.
- **Cointegration search is restricted to within-cluster pairs.** This is
  both a computational shortcut (avoids testing all `C(50,2)` pairs) and a
  research choice: pairs from the same statistical cluster are more likely
  to share a genuine economic linkage rather than a spurious relationship.
- **Engle-Granger test**: two-step method — OLS regression to estimate the
  hedge ratio, then an ADF test on the residual spread. Both the
  `statsmodels.tsa.stattools.coint` p-value and a direct ADF p-value on the
  residuals must clear the significance threshold.
- **Signals**: rolling z-score of the spread with entry/exit/stop-loss
  thresholds — a standard, interpretable mean-reversion signal.
- **Backtest**: trades execute on the bar following a signal (avoids
  look-ahead bias), transaction costs are charged in basis points on every
  position change, and metrics include annualized return, Sharpe ratio,
  max drawdown, and win rate.

## Suggested Extensions (optional, not required for the core project)

- Kalman-filter dynamic hedge ratios instead of a static OLS beta.
- Half-life of mean reversion (Ornstein-Uhlenbeck) to size the z-score window.
- Walk-forward / rolling-window cointegration re-testing.
- Portfolio-level backtest combining multiple pairs with position sizing.
- Plotly for interactive charts if you want to go beyond Matplotlib/Streamlit's built-ins.

## Known Limitations

- NIFTY 50 constituent list in `downloader.py` should be checked against
  the current index composition before submission.
- Cointegration is tested statically over the full sample; in live trading
  relationships can break down, so out-of-sample validation matters.
- `yfinance` occasionally rate-limits bulk downloads — the downloader
  caches to `data/prices.csv` and retries politely, but very large
  universes may need `pause` increased.
