# Full pipeline — integration guide

Complete `src/` + `app.py` for the BFSI pivot, with every cross-file
compatibility issue found and fixed. This is the whole project source,
not just the new pieces.

```
your_project/
├── src/
│   ├── downloader.py         <-- REPLACE (5-line patch)
│   ├── preprocessing.py      <-- unchanged, included for completeness
│   ├── clustering.py         <-- REPLACE (PCA fix, see below)
│   ├── cointegration.py      <-- REPLACE (full rewrite)
│   ├── signals.py            <-- REPLACE (direction fix)
│   ├── backtest.py           <-- REPLACE (direction fix)
│   ├── bfsi_universe.py      <-- new
│   └── meta_labeling.py      <-- new, optional, not yet wired into app.py
└── app.py                     <-- REPLACE (ticker import + direction + suggest_eps fixes)
```

Unzip at your project root. Every file except `preprocessing.py` (included
unchanged, just for a complete set) and the two new files replaces your
existing version — don't hand-merge, the interface changes are easy to
get half-right by hand.

## Changes, file by file

**`downloader.py`** — ticker source swapped from `src.config.NIFTY_50_TICKERS`
to `src.bfsi_universe.get_bfsi_universe()`. 5-line diff, nothing else touched.

**`clustering.py`** — `run_pca()` previously applied an extra `StandardScaler()`
that, after your transpose, standardized *per trading day across tickers*
rather than per ticker across time. Verified empirically: this subtracts
out the market-wide common factor before PCA runs, which contradicts
documentation §5.2's claim that PC1 "resembles broad market exposure." Now
defaults to `standardize=False` so PC1 behaves as documented; pass
`standardize=True` only if you deliberately want market-neutral PCA input
(and update §5.2 accordingly if so).

**`cointegration.py`** — full rewrite: bidirectional Engle-Granger testing
(every result now carries a `direction` field) and Benjamini-Hochberg
correction applied across the full merged pair set. `find_all_cointegrated_pairs(prices, groups, significance=significance)` —
app.py's exact call — still returns only BH-significant pairs by default,
so no call-site change was needed there.

**`signals.py`** — `compute_spread()` gained a `direction` parameter
(default `"y_on_x"`, backward compatible) — required because bidirectional
testing means the dependent/independent roles of a pair can flip.

**`backtest.py`** — two fixes, both confirmed by test to change output:
1. `backtest_multiple_pairs()` now passes `direction` into `compute_spread()`.
2. `backtest_pair()` itself gained a `direction` parameter and branches its
   internal spread-return calculation — this was the deeper bug, since it
   recomputes the spread from raw prices independently of `compute_spread()`.

**`app.py`** — five changes total:
1. Imports `BFSI_TICKERS` instead of `NIFTY50_TICKERS`.
2. Caption text reflects the BFSI universe and BH correction.
3. Tab 3's `compute_spread()` call passes `direction`.
4. Tab 3's separate `backtest_pair()` call also passes `direction`.
5. `suggest_eps()` call now uses the actual `min_samples` slider value and
   all PC columns (previously hardcoded `min_samples=2` and only PC1/PC2,
   regardless of what was actually used for DBSCAN).

## Verified end-to-end

Ran the full chain — `preprocessing → clustering → cointegration → signals
→ backtest` — on synthetic data with two economically distinct sub-groups
plus noise names, exactly matching how `app.py` orchestrates it. PCA
correctly separated the two groups along PC1 (mean +16.3 vs. -16.9),
DBSCAN clustered them correctly, cointegration found a real pair with a
`x_on_y` direction, and the backtest summary produced correct, non-crashing
output for it.

## `bfsi_universe.py` — sector ticker universe

61 tickers across 7 BFSI sub-segments → 1,830 possible pairs. Run
`validate_universe()` once before your first full pipeline run to drop any
symbol yfinance can't return enough history for.

## `meta_labeling.py` — supervised trade-filtering layer (optional)

Sits between `signals.py` and `backtest.py`, not yet wired into `app.py`.
Turns historical signals into a labeled dataset (triple-barrier method)
and trains a classifier to predict which signals are worth acting on.

**New dependency — add to `requirements.txt`:**
```
scikit-learn
joblib
```

## Loose end: `src/config.py`

Your original `downloader.py` imported `NIFTY_50_TICKERS` from
`src/config.py`. Nothing in this package imports from `config.py` anymore
— it's now orphaned unless something else in your project (a notebook,
another script) still uses it. Safe to delete if nothing else references
it, or keep it around if you want the NIFTY 50 list available for
comparison/reference later.
