"""
downloader.py
-------------
Downloads historical OHLCV data for a universe of stocks using yfinance
and caches it locally as CSV/Parquet so repeated runs don't re-hit the API.

Usage:
    from src.downloader import download_universe, NIFTY50_TICKERS

    prices = download_universe(NIFTY50_TICKERS, start="2021-01-01", end="2026-01-01")
"""

import os
import time
import pandas as pd
import yfinance as yf

from src.config import NIFTY_50_TICKERS

# Backward-compatible name used by app.py and the README examples.
NIFTY50_TICKERS = NIFTY_50_TICKERS

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_universe(
    tickers=None,
    start="2021-01-01",
    end=None,
    field="Close",
    cache_path=None,
    force_refresh=False,
    pause=0.25,
):
    """
    Download adjusted close prices for a list of tickers and return a single
    wide DataFrame (index=date, columns=ticker).

    Parameters
    ----------
    tickers : list[str] or None
        Defaults to NIFTY50_TICKERS.
    start, end : str
        Date range (YYYY-MM-DD). end=None means "today".
    field : str
        Which OHLCV field to keep in the wide frame ('Close' is adjusted
        close when auto_adjust=True, which is the default below).
    cache_path : str or None
        If given, will read from / write to this CSV to avoid re-downloading.
    force_refresh : bool
        If True, ignores cache and re-downloads.
    pause : float
        Seconds to sleep between per-ticker downloads (politeness / rate limits).

    Returns
    -------
    pd.DataFrame
    """
    tickers = tickers or NIFTY50_TICKERS

    # --- Bug #1 fix: key cache on date range so slider changes take effect ---
    if cache_path is None:
        _start_tag = str(start).replace("-", "")
        _end_tag   = str(end   or "today").replace("-", "")
        cache_path = os.path.join(DATA_DIR, f"prices_{_start_tag}_{_end_tag}.csv")

    if os.path.exists(cache_path) and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Guard 1: all requested tickers must be present
        missing = set(tickers) - set(df.columns)
        # Guard 2: cached date range must cover the requested window
        _req_start = pd.Timestamp(start)
        _req_end   = pd.Timestamp(end) if end else pd.Timestamp.today()
        _date_ok   = (df.index.min() <= _req_start) and (df.index.max() >= _req_end - pd.Timedelta(days=5))
        if not missing and _date_ok:
            print(f"[downloader] cache hit: {cache_path}")
            return df
        else:
            print(f"[downloader] cache miss (missing tickers: {len(missing)}, date_ok: {_date_ok}) — re-downloading.")

    frames = {}
    failed = []
    for t in tickers:
        try:
            hist = yf.download(
                t, start=start, end=end, auto_adjust=True, progress=False
            )
            if hist.empty:
                failed.append(t)
                continue
            col = hist[field]
            # Newer yfinance returns a DataFrame (MultiIndex) even for a single
            # ticker; squeeze it down to a plain Series so frames stays homogeneous.
            if isinstance(col, pd.DataFrame):
                col = col.squeeze(axis=1)
            frames[t] = col
        except Exception as e:
            print(f"[downloader] failed {t}: {e}")
            failed.append(t)
        time.sleep(pause)

    if failed:
        print(f"[downloader] {len(failed)} tickers failed: {failed}")

    if not frames:
        raise RuntimeError(
            "[downloader] All tickers failed to download. "
            "Check your internet connection or ticker symbols."
        )
    prices = pd.DataFrame(frames)
    prices = prices.sort_index()
    prices = prices.dropna(axis=1, thresh=int(0.9 * len(prices)))  # drop sparse cols
    prices = prices.ffill().dropna()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices.to_csv(cache_path)

    return prices


if __name__ == "__main__":
    prices = download_universe()
    print(prices.shape)
    print(prices.tail())
