from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from quant_eval.price_sources.dukascopy_cli import load_price_frame
from quant_eval.price_sources.dukascopy_symbol_map import (
    EXPLICIT_DUKASCOPY_MAP,
    normalize_symbol,
)


@dataclass
class ResearchSnapshot:
    ticker: str
    company_name: str
    as_of_date: str
    lookback_start_date: str
    dukascopy_instrument_id: str
    price_csv_path: str
    price_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_snapshot(
    ticker: str,
    company_name: str,
    as_of_date: str,
    lookback_start_date: str,
    snapshot_dir: str,
    timeframe: str = "d1",
) -> ResearchSnapshot:
    os.makedirs(snapshot_dir, exist_ok=True)
    price_df = load_price_frame(
        symbol=ticker,
        start_date=lookback_start_date,
        end_date=as_of_date,
        download_dir=snapshot_dir,
        timeframe=timeframe,
    )
    price_csv_path = str(price_df["source_csv"].iloc[0]) if not price_df.empty else ""
    # instrument ID is only available for Dukascopy-native tickers; Yahoo Finance tickers get "yahoo"
    normalized = normalize_symbol(ticker)
    instrument_id = EXPLICIT_DUKASCOPY_MAP.get(normalized, "yahoo")
    return ResearchSnapshot(
        ticker=ticker,
        company_name=company_name,
        as_of_date=as_of_date,
        lookback_start_date=lookback_start_date,
        dukascopy_instrument_id=instrument_id,
        price_csv_path=price_csv_path,
        price_rows=len(price_df),
    )
