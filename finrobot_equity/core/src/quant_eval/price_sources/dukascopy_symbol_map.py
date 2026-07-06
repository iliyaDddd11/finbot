"""Explicit Dukascopy instrument mapping for the first walk-forward version.

The repo supports a much wider universe, but we intentionally start with a very
small explicit map so the evaluation layer is easy to debug.
"""

from __future__ import annotations

EXPLICIT_DUKASCOPY_MAP: dict[str, str] = {
    # Crypto
    "BTCUSD": "btcusd",
    # US equities
    "MSFT": "msftususd",
    "NVDA": "nvdaususd",
    "AAPL": "aaplususd",
    "TSLA": "tslaususd",
}

SUPPORTED_WALKFORWARD_SYMBOLS = tuple(sorted(EXPLICIT_DUKASCOPY_MAP))

_ALIAS_MAP: dict[str, str] = {
    "BTC": "BTCUSD",
    "BITCOIN": "BTCUSD",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA",
    "TESLA": "TSLA",
}


def normalize_symbol(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper().replace("-", "").replace("/", "")
    return _ALIAS_MAP.get(normalized, normalized)


def get_dukascopy_instrument_id(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    instrument_id = EXPLICIT_DUKASCOPY_MAP.get(normalized)
    if instrument_id:
        return instrument_id

    supported = ", ".join(SUPPORTED_WALKFORWARD_SYMBOLS)
    raise KeyError(
        f"No Dukascopy instrument id configured for '{symbol}'. "
        f"Add it to EXPLICIT_DUKASCOPY_MAP in dukascopy_symbol_map.py. "
        f"Currently supported: {supported}."
    )
