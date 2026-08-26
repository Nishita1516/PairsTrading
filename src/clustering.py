"""
clustering.py
--------------
Reduces the stock return feature space with PCA, then groups similar
stocks together with DBSCAN. Pairs are only tested for cointegration
*within* a cluster later on (this cuts down the O(n^2) search space and
gives economically sensible candidate pairs, since PCA+DBSCAN on returns
tends to group stocks by sector/factor exposure).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def run_pca(feature_matrix: pd.DataFrame, n_components: int = 5,
            standardize: bool = False):
    """
    Apply PCA to the (ticker x day) feature matrix.

    Parameters
    ----------
    standardize : bool, default False
        CHANGED DEFAULT — read this before flipping it back to True.

        `feature_matrix` arrives here already standardized to zero mean
        / unit variance PER TICKER, ACROSS TIME by
        preprocessing.build_feature_matrix() — exactly matching
        documentation §5.1.

        The previous version of this function unconditionally applied
        an ADDITIONAL sklearn StandardScaler() here. sklearn
        standardizes along axis 0 by default — i.e. per COLUMN, which
        after the (ticker x day) transpose means PER TRADING DAY,
        ACROSS TICKERS. That subtracts each day's cross-sectional mean
        return before PCA runs, which removes the common/market-wide
        move on that day BY CONSTRUCTION (verified empirically: the
        cross-sectional daily mean's variance goes to ~0 after this
        step). This directly contradicts documentation §5.2's claim
        that "the first component typically resembles broad market
        exposure" — with the old default, PC1 was structurally
        prevented from capturing that factor, since it was subtracted
        out before PCA ever saw the data.

        Default is now False so PC1 behaves as documented. Set True
        only if you deliberately want PCA to run on cross-sectionally
        demeaned (market-neutral) returns instead — e.g. to let later
        components isolate sector/style effects net of the market
        factor. If you do this intentionally, update §5.2 of your
        documentation to describe what PC1 actually represents instead
        of "broad market exposure".

    Returns
    -------
    components : pd.DataFrame (index=ticker, columns=PC1..PCn)
    explained_variance_ratio : np.ndarray
    pca_model : fitted sklearn PCA object
    """
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(feature_matrix.values)
    else:
        X = feature_matrix.values

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X)

    components = pd.DataFrame(
        transformed,
        index=feature_matrix.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    return components, pca.explained_variance_ratio_, pca


def run_dbscan(
    components: pd.DataFrame,
    eps: float = 1.5,
    min_samples: int = 2,
):
    """
    Cluster stocks in PCA space using DBSCAN.

    Returns
    -------
    pd.DataFrame with an added 'cluster' column (-1 = noise / no cluster).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(components.values)

    result = components.copy()
    result["cluster"] = labels
    return result


def get_cluster_groups(clustered: pd.DataFrame) -> dict:
    """
    Convert the clustered DataFrame into {cluster_id: [tickers]},
    excluding noise points (cluster == -1).

    If DBSCAN assigns every stock to noise the function returns an empty dict.
    The caller (app.py) already detects this condition and shows a descriptive
    error banner; returning an empty dict ensures find_all_cointegrated_pairs
    tests zero pairs instead of the entire universe.
    """
    groups = {}
    for cluster_id, group in clustered.groupby("cluster"):
        if cluster_id == -1:
            continue
        groups[int(cluster_id)] = list(group.index)

    # NOTE: intentionally NO fallback here.  When all stocks are noise the
    # pipeline returns zero pairs; the dashboard error banner guides the user
    # to raise eps or lower min_samples rather than silently producing
    # misleading results from the whole universe.
    return groups


def suggest_eps(components: pd.DataFrame, min_samples: int = 2) -> float:
    """
    Simple heuristic (k-distance elbow) to help pick a reasonable DBSCAN eps.
    Returns the median distance to the min_samples-th nearest neighbor.
    Useful as a starting point; inspect the dashboard's eps slider for tuning.

    Pass the SAME min_samples you intend to run DBSCAN with, and ideally
    the full PC space (all PC columns), not just PC1/PC2 — the k-distance
    elbow is only a meaningful proxy for the eps DBSCAN will actually use
    if it's computed in the same space and with the same k. See app.py's
    call site.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(components.values)
    distances, _ = nn.kneighbors(components.values)
    k_distances = np.sort(distances[:, -1])
    return float(np.median(k_distances))
