"""ReconBench configuration — locked Chunk-1 constants.

All source data is public-domain SEC XBRL (data.sec.gov). No FinanceBench data
is used anywhere in the shipped artifact (locked decision 2026-06-26).
"""

# --- SEC API ---------------------------------------------------------------
# SEC requires a descriptive User-Agent; replace the email before any heavy use.
USER_AGENT = "ReconBench research (contact: priyena.career@gmail.com)"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:0>10}.json"
REQUEST_SLEEP_SEC = 0.25  # ~4 req/s, well under SEC's 10 req/s limit

# --- Companies (CIK -> name, ticker) --------------------------------------
# Large filers likely to have clean, exactly-tying balance sheets. Any company
# whose identity does not tie out *exactly* in a given period is discarded, so
# this list is just a candidate pool — correctness is enforced downstream.
COMPANIES = {
    "0000320193": ("Apple Inc.", "AAPL"),
    "0000789019": ("Microsoft Corporation", "MSFT"),
    "0001018724": ("Amazon.com, Inc.", "AMZN"),
    "0001045810": ("NVIDIA Corporation", "NVDA"),
    "0001326801": ("Meta Platforms, Inc.", "META"),
    "0000104169": ("Walmart Inc.", "WMT"),
    "0000021344": ("The Coca-Cola Company", "KO"),
    "0000077476": ("PepsiCo, Inc.", "PEP"),
    "0000200406": ("Johnson & Johnson", "JNJ"),
    "0000080424": ("The Procter & Gamble Company", "PG"),
    "0000050863": ("Intel Corporation", "INTC"),
    "0000858877": ("Cisco Systems, Inc.", "CSCO"),
    "0000796343": ("Adobe Inc.", "ADBE"),
    "0000354950": ("The Home Depot, Inc.", "HD"),
    "0000063908": ("McDonald's Corporation", "MCD"),
    "0000320187": ("NIKE, Inc.", "NKE"),
    "0000909832": ("Costco Wholesale Corporation", "COST"),
    "0000093410": ("Chevron Corporation", "CVX"),
    "0000034088": ("Exxon Mobil Corporation", "XOM"),
    "0000732717": ("AT&T Inc.", "T"),
}

# --- Accounting identities -------------------------------------------------
# Balance-sheet identity: Assets = Liabilities + total Equity.
# Equity tag varies (with/without non-controlling interest); we try both and
# keep whichever makes the identity tie EXACTLY in that period.
ASSETS_TAG = "Assets"
LIABILITIES_TAG = "Liabilities"
EQUITY_TAGS = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]

# --- Perturbation thresholds (NON_RECONCILING items) -----------------------
# A genuine, computation-only-detectable inconsistency must be:
#  - material   : discrepancy clearly above any rounding artifact
#  - plausible  : the wrong figure stays the same order of magnitude, so it
#                 cannot be spotted by eyeballing — only by doing the arithmetic
MIN_REL_DISCREPANCY = 0.005   # > 0.5% of assets
MAX_REL_DISCREPANCY = 0.15    # but <= 15%, else the gap is eyeball-obvious (not arithmetic-only)
MIN_ABS_DISCREPANCY = 1e7     # and > $10M
PLAUSIBLE_LOW = 0.30          # replacement within [0.3x, 3x] of original
PLAUSIBLE_HIGH = 3.0

RANDOM_SEED = 26
