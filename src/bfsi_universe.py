"""
src/bfsi_universe.py

Ticker universe for the Banking & Financial Services (incl. NBFCs)
statistical arbitrage project. Replaces the fixed NIFTY 50 list in
downloader.py with a broader, sector-specific universe assembled from
NSE-listed banks, small finance banks, and NBFCs/housing finance
companies — not just NIFTY Bank's ~12 index constituents.

Two ways to build the universe are provided:

1. get_bfsi_universe() (default, recommended for this project)
   Returns a hand-compiled, categorized list of NSE-listed BFSI
   companies. Static lists don't drift silently and are easy to defend
   in your documentation ("N names across 7 sub-segments as of a given
   date"), but need periodic manual review — new BFSI IPOs happen a
   few times a year.

2. fetch_nse_industry_master() (optional, for a fully current list)
   Pulls NSE's official equity master file and filters by industry
   classification. More "authoritative" for a viva/interview question
   about data provenance, but NSE's site can require session headers
   to avoid being blocked. This function has NOT been network-tested
   in this environment (NSE's domain isn't reachable here) — test it
   from your own machine before relying on it.

CAVEAT: this list was compiled from web research on 2026-08-14 and
almost certainly contains at least a small error or two (a wrong
symbol, a recent delisting, a missed new IPO). validate_universe()
below is designed to catch and drop anything wrong automatically —
run it before building your return matrix, and log what got dropped
for your documentation's "known limitations" section. Also spot-check
the list yourself against nseindia.com or a data vendor before your
final submission.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Curated ticker universe, categorized by BFSI sub-segment
# ---------------------------------------------------------------------------

PUBLIC_SECTOR_BANKS = [
    "SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "BANKINDIA",
    "INDIANB", "CENTRALBK", "IOB", "UCOBANK", "MAHABANK", "PSB",
]

PRIVATE_SECTOR_BANKS = [
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
    "FEDERALBNK", "IDFCFIRSTB", "RBLBANK", "YESBANK", "BANDHANBNK",
    "CUB", "KARURVYSYA", "SOUTHBANK", "DCBBANK", "CSBBANK", "KTKBANK",
    "TMB", "J&KBANK",
]

SMALL_FINANCE_BANKS = [
    "AUBANK", "EQUITASBNK", "UJJIVANSFB", "SURYODAY", "ESAFSFB",
    "UTKARSHBNK", "CAPITALSFB",
]

DIVERSIFIED_NBFC = [
    "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "LTF",
    "M&MFIN", "SUNDARMFIN", "ABCAPITAL", "PEL", "POONAWALLA",
    "IIFL", "SBFC",
]

GOLD_LOAN_NBFC = [
    "MUTHOOTFIN", "MANAPPURAM",
]

HOUSING_FINANCE = [
    "LICHSGFIN", "PNBHOUSING", "CANFINHOME", "REPCOHOME", "AAVAS",
    "HOMEFIRST", "APTUS",
]

MICROFINANCE_NBFC = [
    "CREDITACC", "SPANDANA", "FIVESTAR",
]

# Optional broader segments — excluded by default. Insurers' and AMCs'
# business drivers (underwriting cycles, AUM-linked fees) differ enough
# from pure lenders that mixing them back in partially undoes the point
# of picking one coherent sector. Include only if you deliberately want
# a bigger, more heterogeneous universe.
INSURANCE = [
    "SBILIFE", "HDFCLIFE", "ICICIPRULI", "ICICIGI", "NIACL",
    "GICRE", "STARHEALTH", "LICI",
]

ASSET_MANAGEMENT_BROKING = [
    "HDFCAMC", "NAM-INDIA", "UTIAMC", "ANGELONE", "MOTILALOFS",
    "CDSL", "BSE", "ISEC",
]


def get_bfsi_universe(include_insurance: bool = False,
                       include_amc_broking: bool = False):
    """
    Return the BFSI ticker universe with .NS suffix, ready for yfinance.

    Default (banks + NBFCs only) keeps the universe economically
    coherent: every name here is a balance-sheet lender exposed to
    credit growth, NIMs, and asset quality cycles — the shared risk
    factor your PCA step should actually pick up on.
    """
    tickers = (
        PUBLIC_SECTOR_BANKS + PRIVATE_SECTOR_BANKS + SMALL_FINANCE_BANKS
        + DIVERSIFIED_NBFC + GOLD_LOAN_NBFC + HOUSING_FINANCE + MICROFINANCE_NBFC
    )
    if include_insurance:
        tickers = tickers + INSURANCE
    if include_amc_broking:
        tickers = tickers + ASSET_MANAGEMENT_BROKING

    # de-duplicate while preserving order, then add the .NS suffix
    seen = set()
    deduped = [t for t in tickers if not (t in seen or seen.add(t))]
    return [f"{t}.NS" for t in deduped]


# ---------------------------------------------------------------------------
# 2. Optional: pull NSE's official industry classification for verification
# ---------------------------------------------------------------------------

NSE_EQUITY_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def fetch_nse_industry_master(sector_keywords=("BANK", "FINANC", "HOUSING FINANCE")):
    """
    Pull NSE's official listed-equity master and filter to rows whose
    industry field matches any of `sector_keywords`. Use this to CROSS-
    CHECK the curated list above, or to regenerate it if the sector
    universe has changed since this was written.

    NOTE: not network-tested in this environment (NSE's domain isn't
    reachable from this sandbox). Run it from your own machine — NSE
    sometimes needs a browser-like User-Agent and/or a warmed-up
    session cookie to avoid a 403. If it fails, fall back to
    get_bfsi_universe() above, or download the CSV manually from
    nseindia.com and point pd.read_csv() at the local file instead.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    df = pd.read_csv(NSE_EQUITY_MASTER_URL, storage_options={"User-Agent": headers["User-Agent"]})
    df.columns = [c.strip() for c in df.columns]

    industry_col = next((c for c in df.columns if "INDUSTRY" in c.upper()), None)
    if industry_col is None:
        raise KeyError(
            f"Couldn't find an industry classification column. Columns found: {list(df.columns)}"
        )

    mask = df[industry_col].astype(str).str.upper().apply(
        lambda x: any(kw in x for kw in sector_keywords)
    )
    filtered = df[mask].copy()
    filtered["ticker"] = filtered["SYMBOL"].astype(str).str.strip() + ".NS"
    return filtered[["SYMBOL", "ticker", industry_col]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Validate the universe against actual price data availability
# ---------------------------------------------------------------------------

def validate_universe(tickers, min_history_days: int = 750,
                       start: str = "2020-01-01", end: str = None) -> dict:
    """
    Drop tickers that yfinance can't return sufficient history for —
    common for recent IPOs (small finance banks, some HFCs), thinly
    traded names, or a wrong/outdated symbol in the curated list above.
    Run this once before building the return matrix, and log what got
    dropped for your documentation's "known limitations" section.

    Returns {"kept": [...], "dropped": {ticker: reason, ...}}
    """
    import yfinance as yf

    kept, dropped = [], {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                dropped[ticker] = "no data returned"
            elif len(data) < min_history_days:
                dropped[ticker] = f"only {len(data)} trading days (< {min_history_days})"
            else:
                kept.append(ticker)
        except Exception as e:
            dropped[ticker] = f"download error: {e}"

    return {"kept": kept, "dropped": dropped}


if __name__ == "__main__":
    universe = get_bfsi_universe()
    n = len(universe)
    print(f"BFSI universe: {n} tickers -> {n * (n - 1) // 2} possible pairs")
    for name, group in [
        ("Public sector banks", PUBLIC_SECTOR_BANKS),
        ("Private sector banks", PRIVATE_SECTOR_BANKS),
        ("Small finance banks", SMALL_FINANCE_BANKS),
        ("Diversified NBFC", DIVERSIFIED_NBFC),
        ("Gold loan NBFC", GOLD_LOAN_NBFC),
        ("Housing finance", HOUSING_FINANCE),
        ("Microfinance NBFC", MICROFINANCE_NBFC),
    ]:
        print(f"  {name}: {len(group)}")
