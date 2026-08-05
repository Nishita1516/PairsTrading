"""
cointegration.py
-----------------
Within each cluster produced by clustering.py, test all stock pairs for
cointegration using the Engle-Granger two-step method, and rank the
statistically significant pairs.
"""

import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm


def engle_granger_test(y: pd.Series, x: pd.Series):
    """
    Run the Engle-Granger cointegration test on a pair of price series.

    Returns
    -------
    dict with keys: pvalue, tstat, hedge_ratio, spread
    """
    # Step 1: regress y on x to get the hedge ratio (beta)
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]

    spread = y - hedge_ratio * x - intercept

    # Step 2: ADF test on the residual spread
    adf_stat, adf_pvalue, *_ = adfuller(spread, autolag="AIC")

    # Also run statsmodels' built-in coint() for a cross-check p-value
    eg_tstat, eg_pvalue, _ = coint(y, x)

    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "spread": spread,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "eg_tstat": eg_tstat,
        "eg_pvalue": eg_pvalue,
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    tickers: list,
    significance: float = 0.05,
):
    """
    Test every pair within `tickers` (typically one DBSCAN cluster) for
    cointegration and return a ranked list of significant pairs.

    Parameters
    ----------
    prices : wide price DataFrame (date x ticker), full universe
    tickers : list of tickers belonging to a single cluster
    significance : p-value cutoff

    Returns
    -------
    list[dict] sorted by eg_pvalue ascending. Each dict has:
        pair, eg_pvalue, adf_pvalue, hedge_ratio, intercept

    Notes
    -----
    A pair is accepted only when **both** the Engle-Granger coint() p-value
    *and* the residual ADF p-value are below `significance`. This matches the
    two-step Engle-Granger procedure documented in the README.
    """
    results = []
    for t1, t2 in itertools.combinations(tickers, 2):
        if t1 not in prices.columns or t2 not in prices.columns:
            continue
        y = prices[t1].dropna()
        x = prices[t2].dropna()
        common_idx = y.index.intersection(x.index)
        if len(common_idx) < 60:  # need enough history for a meaningful test
            continue
        y, x = y.loc[common_idx], x.loc[common_idx]

        try:
            res = engle_granger_test(y, x)
        except Exception:
            continue

        # Both the EG coint() p-value AND the residual ADF p-value must pass.
        if res["eg_pvalue"] < significance and res["adf_pvalue"] < significance:
            results.append(
                {
                    "pair": (t1, t2),
                    "eg_pvalue": res["eg_pvalue"],
                    "adf_pvalue": res["adf_pvalue"],
                    "hedge_ratio": res["hedge_ratio"],
                    "intercept": res["intercept"],
                }
            )

    results.sort(key=lambda r: r["eg_pvalue"])
    return results


def find_all_cointegrated_pairs(prices: pd.DataFrame, cluster_groups: dict, significance: float = 0.05):
    """
    Run find_cointegrated_pairs across every cluster and merge the results.
    """
    all_results = []
    for cluster_id, tickers in cluster_groups.items():
        if len(tickers) < 2:
            continue
        pairs = find_cointegrated_pairs(prices, tickers, significance)
        for p in pairs:
            p["cluster"] = cluster_id
        all_results.extend(pairs)

    all_results.sort(key=lambda r: r["eg_pvalue"])
    return all_results
