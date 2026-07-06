"""
Equity Universe — default stock lists for walk-forward evaluation.

Organized by GICS sector. All names are US-listed, sufficient FMP coverage,
and tradeable via the Dukascopy price feed.

For meaningful cross-sectional IC computation you need >= 30 names per period.
The DEFAULT_UNIVERSE provides 61 names across 9 sectors.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-sector constituents
# ---------------------------------------------------------------------------

SECTORS: dict[str, list[str]] = {
    "technology": [
        "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
        "CRM", "ORCL", "ADBE", "INTC", "AMD", "QCOM", "AVGO", "TXN",
    ],
    "financials": [
        "JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "USB", "C", "PNC",
    ],
    "healthcare": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "MDT", "CI",
    ],
    "consumer_staples": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO",
    ],
    "consumer_discretionary": [
        "TSLA", "MCD", "HD", "NKE", "SBUX", "TGT", "LOW",
    ],
    "industrials": [
        "GE", "HON", "CAT", "BA", "UPS", "RTX",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB",
    ],
    "utilities": [
        "NEE", "DUK", "SO",
    ],
    "materials": [
        "LIN", "APD", "NEM",
    ],
}

# Flat list (61 tickers)
DEFAULT_UNIVERSE: list[str] = [t for tickers in SECTORS.values() for t in tickers]

# Smaller curated set for quick tests (covers all sectors, ~20 names)
SMALL_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL",          # Technology
    "JPM", "GS",                               # Financials
    "JNJ", "LLY", "UNH",                      # Healthcare
    "PG", "KO", "WMT",                        # Consumer Staples
    "MCD", "HD",                               # Consumer Discretionary
    "CAT", "HON",                              # Industrials
    "XOM", "CVX",                              # Energy
    "NEE",                                     # Utilities
    "LIN",                                     # Materials
]

# Company name lookup (used for display / analysis pipeline labels)
COMPANY_NAMES: dict[str, str] = {
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "NVDA":  "NVIDIA Corporation",
    "META":  "Meta Platforms Inc.",
    "GOOGL": "Alphabet Inc.",
    "AMZN":  "Amazon.com Inc.",
    "CRM":   "Salesforce Inc.",
    "ORCL":  "Oracle Corporation",
    "ADBE":  "Adobe Inc.",
    "INTC":  "Intel Corporation",
    "AMD":   "Advanced Micro Devices Inc.",
    "QCOM":  "Qualcomm Inc.",
    "AVGO":  "Broadcom Inc.",
    "TXN":   "Texas Instruments Inc.",
    "JPM":   "JPMorgan Chase & Co.",
    "BAC":   "Bank of America Corporation",
    "GS":    "The Goldman Sachs Group Inc.",
    "MS":    "Morgan Stanley",
    "WFC":   "Wells Fargo & Company",
    "BLK":   "BlackRock Inc.",
    "AXP":   "American Express Company",
    "USB":   "U.S. Bancorp",
    "C":     "Citigroup Inc.",
    "PNC":   "PNC Financial Services Group",
    "JNJ":   "Johnson & Johnson",
    "UNH":   "UnitedHealth Group Inc.",
    "PFE":   "Pfizer Inc.",
    "ABBV":  "AbbVie Inc.",
    "MRK":   "Merck & Co. Inc.",
    "LLY":   "Eli Lilly and Company",
    "BMY":   "Bristol-Myers Squibb Company",
    "AMGN":  "Amgen Inc.",
    "MDT":   "Medtronic plc",
    "CI":    "The Cigna Group",
    "PG":    "Procter & Gamble Co.",
    "KO":    "The Coca-Cola Company",
    "PEP":   "PepsiCo Inc.",
    "WMT":   "Walmart Inc.",
    "COST":  "Costco Wholesale Corporation",
    "PM":    "Philip Morris International Inc.",
    "MO":    "Altria Group Inc.",
    "MCD":   "McDonald's Corporation",
    "HD":    "The Home Depot Inc.",
    "NKE":   "NIKE Inc.",
    "SBUX":  "Starbucks Corporation",
    "TGT":   "Target Corporation",
    "LOW":   "Lowe's Companies Inc.",
    "GE":    "GE Aerospace",
    "HON":   "Honeywell International Inc.",
    "CAT":   "Caterpillar Inc.",
    "BA":    "The Boeing Company",
    "UPS":   "United Parcel Service Inc.",
    "RTX":   "RTX Corporation",
    "XOM":   "Exxon Mobil Corporation",
    "CVX":   "Chevron Corporation",
    "COP":   "ConocoPhillips",
    "SLB":   "SLB",
    "NEE":   "NextEra Energy Inc.",
    "DUK":   "Duke Energy Corporation",
    "SO":    "The Southern Company",
    "LIN":   "Linde plc",
    "APD":   "Air Products and Chemicals Inc.",
    "NEM":   "Newmont Corporation",
    "TSLA":  "Tesla Inc.",
}


def get_company_name(ticker: str) -> str:
    return COMPANY_NAMES.get(ticker.upper(), ticker.upper())


def sector_of(ticker: str) -> str | None:
    t = ticker.upper()
    for sector, tickers in SECTORS.items():
        if t in tickers:
            return sector
    return None
