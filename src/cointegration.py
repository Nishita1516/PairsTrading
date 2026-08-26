"""
cointegration.py
-----------------
Within each cluster produced by clustering.py, test all stock pairs for
cointegration using the Engle-Granger two-step method, and rank the
statistically significant pairs.

CHANGES FROM THE PREVIOUS VERSION
-----------------------------------
1. BIDIRECTIONAL TESTING
   Engle-Granger is asymmetric — regressing y on x and testing the
   residual is not guaranteed to give the same p-value as regressing x
   on y. engle_granger_test() now tests both directions and keeps
   whichever gives the stronger (lower) eg_pvalue, storing which
   direction was chosen so the spread can be reconstructed correctly
   downstream (see get_spread() at the bottom).

2. BENJAMINI-HOCHBERG (BH) CORRECTION, APPLIED ACROSS THE FULL RUN
   The BFSI universe (~61 tickers) can produce far more within-cluster
   candidate pairs than the old NIFTY 50 universe did. At a flat 0.05
   significance level tested pair-by-pair, you'd expect ~5% of ALL
   pairs tested to look "significant" purely by chance, regardless of
   whether any are genuinely cointegrated — this is exactly the
   multiple-testing problem already described in the project
   documentation (§2.3), just now more consequential given the larger
   pair count.

   find_cointegrated_pairs() (the per-cluster function) no longer
   filters by `significance` — it returns every pair TESTED within
   that cluster, with both p-values attached, since BH correction
   needs the full p-value distribution to compute its adaptive
   threshold correctly.

   find_all_cointegrated_pairs() (the function app.py actually calls)
   merges results from every cluster, applies BH correction across the
   FULL merged set, and by default (filter_significant=True) STILL
   returns only the pairs that pass correction — so app.py's existing
   call site, `find_all_cointegrated_pairs(prices, groups,
   significance=significance)`, keeps working exactly as before with
   no changes needed, just with a statistically sounder filter under
   the hood. Pass filter_significant=False if you want every tested
   pair back (with `bh_significant` / `raw_significant` columns) for a
   diagnostic view instead.
"""

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


# ---------------------------------------------------------------------------
# Bidirectional Engle-Granger test for a single pair
# ---------------------------------------------------------------------------

def _fit_direction(y: pd.Series, x: pd.Series) -> dict:
    """OLS regress y on x, ADF-test the residual, and cross-check with
    statsmodels' coint(). One "direction" of the bidirectional test."""
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    hedge_ratio = model.params.iloc[1]
    intercept = model.params.iloc[0]
    spread = y - hedge_ratio * x - intercept

    adf_stat, adf_pvalue, *_ = adfuller(spread, autolag="AIC")
    eg_tstat, eg_pvalue, _ = coint(y, x)

    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "eg_tstat": eg_tstat,
        "eg_pvalue": eg_pvalue,
    }


def engle_granger_test(y: pd.Series, x: pd.Series):
    """
    Bidirectional two-step Engle-Granger cointegration test on a pair of
    price series.

    Tests both "y regressed on x" and "x regressed on y", and keeps
    whichever direction gives the lower (stronger) eg_pvalue from
    statsmodels' coint() — the same metric the pipeline sorts pairs by,
    so direction selection and final ranking use a consistent test.

    Returns
    -------
    dict with keys: direction, hedge_ratio, intercept, adf_stat,
    adf_pvalue, eg_tstat, eg_pvalue.
    `direction` is "y_on_x" or "x_on_y" — pass this to get_spread()
    later so the spread is reconstructed with the correct sign/roles.
    """
    forward = _fit_direction(y, x)   # y = a + b*x + resid
    reverse = _fit_direction(x, y)   # x = a + b*y + resid

    if forward["eg_pvalue"] <= reverse["eg_pvalue"]:
        chosen = forward
        chosen["direction"] = "y_on_x"
    else:
        chosen = reverse
        chosen["direction"] = "x_on_y"

    return chosen


# ---------------------------------------------------------------------------
# Test every pair within one cluster
# ---------------------------------------------------------------------------

def find_cointegrated_pairs(
    prices: pd.DataFrame,
    tickers: list,
    significance: float = 0.05,
):
    """
    Test every pair within `tickers` (typically one DBSCAN cluster) for
    cointegration and return results for ALL tested pairs (not just the
    ones passing a flat cutoff — see the module docstring for why).

    Parameters
    ----------
    prices : wide price DataFrame (date x ticker), full universe
    tickers : list of tickers belonging to a single cluster
    significance : p-value cutoff used only to compute the diagnostic
        `raw_significant` column (old behavior) — does not filter rows.

    Returns
    -------
    list[dict] sorted by eg_pvalue ascending. Each dict has:
        pair, direction, eg_pvalue, adf_pvalue, hedge_ratio, intercept,
        raw_significant
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

        results.append({
            "pair": (t1, t2),
            "direction": res["direction"],
            "eg_pvalue": res["eg_pvalue"],
            "adf_pvalue": res["adf_pvalue"],
            "hedge_ratio": res["hedge_ratio"],
            "intercept": res["intercept"],
            "raw_significant": bool(
                res["eg_pvalue"] < significance and res["adf_pvalue"] < significance
            ),
        })

    results.sort(key=lambda r: r["eg_pvalue"])
    return results


# ---------------------------------------------------------------------------
# Benjamini-Hochberg correction
# ---------------------------------------------------------------------------

def _bh_flags(pvalues: np.ndarray, fdr: float) -> np.ndarray:
    """
    Boolean array marking which p-values pass Benjamini-Hochberg
    correction at the given false-discovery rate, aligned to the
    original (unsorted) order of `pvalues`.

    Procedure: sort ascending, p_(1) <= ... <= p_(m); find the largest
    rank k with p_(k) <= (k/m)*fdr; flag every entry at or before that
    rank. This is NOT the same as flagging each entry against its own
    threshold independently — the "largest passing rank" step is the
    part most implementations get wrong.
    """
    m = len(pvalues)
    if m == 0:
        return np.array([], dtype=bool)

    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * fdr

    below = sorted_p <= thresholds
    flags_sorted = np.zeros(m, dtype=bool)
    if below.any():
        k_max = int(np.max(np.where(below)[0]))
        flags_sorted[: k_max + 1] = True

    flags = np.zeros(m, dtype=bool)
    flags[order] = flags_sorted
    return flags


def apply_benjamini_hochberg(results: list, fdr: float = 0.05) -> list:
    """
    Apply BH correction across ALL pairs in `results`, separately to
    eg_pvalue and adf_pvalue, then require BOTH to remain significant
    after correction — preserving the original dual-check philosophy
    (§ module docstring, point 2) while controlling the false discovery
    rate for each test family rather than gating on a flat per-pair cutoff.

    Mutates and returns `results`, adding: eg_bh_significant,
    adf_bh_significant, bh_significant (the AND of both).
    """
    if not results:
        return results

    eg_pvals = np.array([r["eg_pvalue"] for r in results])
    adf_pvals = np.array([r["adf_pvalue"] for r in results])

    eg_flags = _bh_flags(eg_pvals, fdr)
    adf_flags = _bh_flags(adf_pvals, fdr)

    for r, eg_flag, adf_flag in zip(results, eg_flags, adf_flags):
        r["eg_bh_significant"] = bool(eg_flag)
        r["adf_bh_significant"] = bool(adf_flag)
        r["bh_significant"] = bool(eg_flag and adf_flag)

    return results


# ---------------------------------------------------------------------------
# Run across every cluster, then correct across the full merged set
# ---------------------------------------------------------------------------

def find_all_cointegrated_pairs(
    prices: pd.DataFrame,
    cluster_groups: dict,
    significance: float = 0.05,
    fdr: float = None,
    filter_significant: bool = True,
):
    """
    Run find_cointegrated_pairs across every cluster, merge all tested
    pairs into one list, then apply Benjamini-Hochberg correction across
    the FULL merged set — not per-cluster, since the false-discovery
    problem scales with total hypotheses tested in the run, not with
    how those hypotheses happen to be grouped.

    Parameters
    ----------
    cluster_groups : dict[cluster_id -> list[ticker]]
        Output of clustering.py's DBSCAN step, one entry per cluster
        (exclude noise / cluster -1 before calling this).
    significance : kept for backward compatibility, only used to compute
        the diagnostic `raw_significant` column — does not filter.
    fdr : the false discovery rate for the BH correction. If None
        (default), reuses `significance` as the FDR target — so
        app.py's existing "Significance level" slider now controls the
        BH false-discovery target instead of a flat per-pair cutoff,
        with no call-site change required.
    filter_significant : if True (default), returns only pairs with
        bh_significant == True — this PRESERVES the pre-filtered
        behavior app.py's call site already expects. Set False to get
        every tested pair back with full diagnostics attached (useful
        for a "pairs tested vs. pairs surviving correction" panel).

    Returns
    -------
    list[dict], sorted by eg_pvalue ascending.
    """
    if fdr is None:
        fdr = significance

    all_results = []
    for cluster_id, tickers in cluster_groups.items():
        if len(tickers) < 2:
            continue
        pairs = find_cointegrated_pairs(prices, tickers, significance)
        for p in pairs:
            p["cluster"] = cluster_id
        all_results.extend(pairs)

    all_results = apply_benjamini_hochberg(all_results, fdr=fdr)
    all_results.sort(key=lambda r: r["eg_pvalue"])

    if filter_significant:
        return [r for r in all_results if r["bh_significant"]]
    return all_results


# ---------------------------------------------------------------------------
# Reconstruct the spread for a chosen pair (feeds into signals.py)
# ---------------------------------------------------------------------------

def get_spread(prices: pd.DataFrame, pair: tuple, direction: str,
                hedge_ratio: float, intercept: float) -> pd.Series:
    """
    Reconstruct the spread series for a pair using the direction and
    hedge ratio chosen by engle_granger_test(). Needed now because
    bidirectional testing means the dependent/independent roles of
    (t1, t2) can be flipped from the pair's original order — signals.py
    should call this rather than recomputing `y - hedge_ratio*x` itself
    with a fixed assumed order.
    """
    t1, t2 = pair
    if direction == "y_on_x":
        spread = prices[t1] - hedge_ratio * prices[t2] - intercept
    elif direction == "x_on_y":
        spread = prices[t2] - hedge_ratio * prices[t1] - intercept
    else:
        raise ValueError(f"Unknown direction: {direction!r} (expected 'y_on_x' or 'x_on_y')")
    return spread.dropna()
