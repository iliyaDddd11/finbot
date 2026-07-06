"""
Yahoo Finance price source — fallback for tickers not in the Dukascopy map.

Uses yfinance to download daily OHLC and returns a DataFrame in the same
schema as load_price_frame() from dukascopy_cli.py:

  timestamp (UTC, datetime64), open, high, low, close, volume,
  symbol, source_csv (path to cached parquet/CSV)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_price_frame_yahoo(
    symbol: str,
    start_date: str,
    end_date: str,
    download_dir: str,
) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo Finance, cache to CSV, return DataFrame.

    Parameters
    ----------
    symbol       : Ticker symbol, e.g. "JPM"
    start_date   : ISO date string "YYYY-MM-DD"
    end_date     : ISO date string "YYYY-MM-DD"
    download_dir : Local directory for caching

    Returns
    -------
    DataFrame with columns: timestamp (UTC), open, high, low, close, volume,
                             symbol, source_csv
    """
    import yfinance as yf

    cache_dir = Path(download_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    safe_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_file = cache_dir / f"{symbol.upper()}_{start_date}_{end_date}_d1.csv"

    if cache_file.exists():
        df = pd.read_csv(cache_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    ticker = yf.Ticker(symbol)
    raw = ticker.history(start=start_date, end=safe_end, interval="1d", auto_adjust=True)

    if raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {symbol} ({start_date} to {end_date})")

    raw = raw.reset_index()
    # yfinance Date column may be tz-aware or tz-naive depending on version
    date_col = raw.columns[0]   # "Date" or "Datetime"
    raw[date_col] = pd.to_datetime(raw[date_col], utc=True)

    df = pd.DataFrame({
        "timestamp": raw[date_col],
        "open":      raw["Open"].astype(float),
        "high":      raw["High"].astype(float),
        "low":       raw["Low"].astype(float),
        "close":     raw["Close"].astype(float),
        "volume":    raw["Volume"].astype(float),
        "symbol":    symbol.upper(),
        "source_csv": str(cache_file.resolve()),
    })

    df.to_csv(cache_file, index=False)
    return df
