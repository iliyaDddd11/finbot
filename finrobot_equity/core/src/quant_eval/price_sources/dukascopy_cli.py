from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .dukascopy_symbol_map import get_dukascopy_instrument_id, EXPLICIT_DUKASCOPY_MAP, normalize_symbol
from .yahoo_finance import load_price_frame_yahoo


def build_dukascopy_command(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "d1",
    output_format: str = "csv",
) -> list[str]:
    instrument_id = get_dukascopy_instrument_id(symbol)
    return [
        "npx",
        "dukascopy-node",
        "-i",
        instrument_id,
        "-from",
        start_date,
        "-to",
        end_date,
        "-t",
        timeframe,
        "-f",
        output_format,
    ]


def run_dukascopy_download(
    symbol: str,
    start_date: str,
    end_date: str,
    download_dir: str,
    timeframe: str = "d1",
    output_format: str = "csv",
    timeout_seconds: int = 600,
) -> str:
    if shutil.which("npx") is None:
        raise RuntimeError("npx is not installed or not on PATH. Install Node.js/npm on this machine first.")

    target_dir = Path(download_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    command = build_dukascopy_command(symbol, start_date, end_date, timeframe=timeframe, output_format=output_format)

    completed = subprocess.run(
        command,
        cwd=str(target_dir),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    combined = stdout + "\n" + stderr

    marker = "File saved:"
    saved_path = None
    for line in combined.splitlines():
        if marker in line:
            raw = line.split(marker, 1)[1].strip()
            # Strip trailing size annotation like " (1.38 KB)"
            if " (" in raw:
                raw = raw[: raw.rfind(" (")]
            saved_path = raw.strip()
            break

    if not saved_path:
        raise RuntimeError(
            "dukascopy-node completed but the output file path could not be parsed. "
            f"Command: {' '.join(command)}\nOutput:\n{combined}"
        )

    csv_path = Path(saved_path)
    if not csv_path.is_absolute():
        csv_path = target_dir / saved_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"dukascopy-node reported '{saved_path}', resolved to '{csv_path}', but the file does not exist.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    return str(csv_path)


def load_dukascopy_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Unexpected Dukascopy CSV schema in {csv_path}. Missing 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_price_frame(
    symbol: str,
    start_date: str,
    end_date: str,
    download_dir: str,
    timeframe: str = "d1",
) -> pd.DataFrame:
    """
    Load daily price data for `symbol`.

    Primary source: Dukascopy (for tickers in EXPLICIT_DUKASCOPY_MAP).
    Fallback:       Yahoo Finance (for all other US equity tickers).
    """
    normalized = normalize_symbol(symbol)
    use_dukascopy = normalized in EXPLICIT_DUKASCOPY_MAP

    if use_dukascopy:
        csv_path = run_dukascopy_download(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            download_dir=download_dir,
            timeframe=timeframe,
        )
        df = load_dukascopy_csv(csv_path)
        df["symbol"] = symbol.upper()
        df["source_csv"] = os.path.abspath(csv_path)
        return df

    # Yahoo Finance fallback — same output schema
    return load_price_frame_yahoo(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        download_dir=download_dir,
    )
