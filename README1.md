# New modules — integration guide

Two files, both written to drop straight into your existing `src/` folder:

```
your_project/
├── src/
│   ├── downloader.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── cointegration.py
│   ├── signals.py
│   ├── backtest.py
│   ├── bfsi_universe.py     <-- new
│   └── meta_labeling.py     <-- new
└── app.py
```

Just unzip this at your project root and both files land in the right place.

## 1. `bfsi_universe.py` — sector ticker universe

Replaces your fixed NIFTY 50 list with a Banking & Financial Services
(incl. NBFCs) universe: 61 tickers across 7 sub-segments → 1,830 possible
pairs.

**Wire it into `downloader.py`:**

```python
from src.bfsi_universe import get_bfsi_universe, validate_universe

tickers = get_bfsi_universe()                 # 61 .NS tickers
result = validate_universe(tickers)            # drops bad/short-history symbols
tickers = result["kept"]
print("Dropped:", result["dropped"])           # log this for your write-up
```

Run `validate_universe()` once before your first full pipeline run —
it catches recent-IPO small finance banks with too little history and
any symbol errors, instead of letting them silently break PCA later.

**No new dependencies** — only needs `pandas` and `yfinance`, which you
already have.

## 2. `meta_labeling.py` — supervised trade-filtering layer

Sits between `signals.py` and `backtest.py`. Turns your historical
signals into a labeled dataset (triple-barrier method) and trains a
classifier to predict which signals are worth acting on.

**Wire it into your dashboard / a new pipeline stage:**

```python
from src.meta_labeling import run_meta_labeling_pipeline, apply_meta_filter

result = run_meta_labeling_pipeline(
    signals_df,        # from signals.py
    spread_zscores,     # dict: pair -> rolling z-score pd.Series
    model_type="logistic",
)

print(result["metrics"])   # ROC-AUC vs. base rate, confusion matrix

# Score new/live signals and keep only high-confidence ones
filtered_signals = apply_meta_filter(
    new_signals_features, result["model"], result["scaler"], threshold=0.55
)
# feed filtered_signals[filtered_signals.meta_take_trade] into backtest.py
```

**New dependency — add to `requirements.txt`:**

```
scikit-learn
joblib
```

## Suggested next step (not included yet)

With 1,830 pairs instead of 1,225, your multiple-testing exposure goes
up — a flat 0.05 significance cutoff in `cointegration.py` would flag
~91 "significant" pairs by chance alone. Ask for the Benjamini-Hochberg
correction snippet when you're ready to wire that into
`cointegration.py`.
