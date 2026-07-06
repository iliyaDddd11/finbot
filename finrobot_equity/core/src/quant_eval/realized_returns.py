from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_eval.price_sources.dukascopy_cli import load_price_frame

# Days to look back before as_of_date when downloading, so a weekend/holiday
# month-end still has a prior trading bar to anchor the entry on.
_ANCHOR_LOOKBACK_DAYS = 7


@dataclass
class RealizedOutcome:
    ticker: str
    as_of_date: str
    horizon_days: int
    entry_timestamp: str | None
    exit_timestamp: str | None
    entry_close: float | None
    exit_close: float | None
    realized_return: float | None
    benchmark_return: float | None    # SPY return over same horizon
    alpha_return: float | None        # realized_return - benchmark_return
    max_upside: float | None
    max_drawdown: float | None
    horizon_complete: bool | None     # False if the price series ended before the horizon
    price_csv_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_close_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for candidate in ["close", "bidclose", "askclose", "price"]:
        if candidate in out.columns:
            out["close_eval"] = pd.to_numeric(out[candidate], errors="coerce")
            break
    if "close_eval" not in out.columns:
        numeric_cols = [c for c in out.columns if c not in {"timestamp", "symbol", "source_csv"}]
        if not numeric_cols:
            raise ValueError("No usable close-like column found in price dataframe.")
        out["close_eval"] = pd.to_numeric(out[numeric_cols[0]], errors="coerce")
    return out.dropna(subset=["timestamp", "close_eval"]).sort_values("timestamp").reset_index(drop=True)


def _pick_anchor_row(df: pd.DataFrame, as_of_date: str) -> int | None:
    cutoff = pd.Timestamp(as_of_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    eligible = df.index[df["timestamp"] <= cutoff].tolist()
    return eligible[-1] if eligible else None


def _compute_return_from_price_df(
    df: pd.DataFrame,
    as_of_date: str,
    horizon_days: int,
) -> tuple[str | None, str | None, float | None, float | None, float | None, float | None]:
    """
    Shared logic: compute realized return + path stats from a price DataFrame.
    Returns (entry_ts, exit_ts, entry_close, exit_close, max_upside, max_drawdown,
    horizon_complete).  entry_ts is None when no return can be computed.
    """
    anchor_idx = _pick_anchor_row(df, as_of_date)
    if anchor_idx is None:
        return None, None, None, None, None, None, False

    exit_target = df.loc[anchor_idx, "timestamp"] + pd.Timedelta(days=horizon_days)
    future = df.iloc[anchor_idx + 1:].copy()
    if future.empty:
        # No forward data at all (e.g. series truncated / delisted at as_of).
        # Do NOT fabricate a 0.0 return — report it as missing.
        return None, None, None, None, None, None, False

    eligible = future.index[future["timestamp"] >= exit_target].tolist()
    if eligible:
        exit_idx = eligible[0]
        horizon_complete = True
    else:
        # The series ends before the horizon is reached: this is a partial-window
        # return (shorter than horizon_days). Flag it so callers don't treat a
        # 30-day move as if it were the full 90-day horizon.
        exit_idx = future.index[-1]
        horizon_complete = False

    entry_close = float(df.loc[anchor_idx, "close_eval"])
    exit_close  = float(df.loc[exit_idx,   "close_eval"])
    forward_slice = df.loc[anchor_idx:exit_idx]
    max_upside   = None
    max_drawdown = None
    if not forward_slice.empty and entry_close != 0:
        max_upside   = float(forward_slice["close_eval"].max() / entry_close - 1.0)
        max_drawdown = float(forward_slice["close_eval"].min() / entry_close - 1.0)

    return (
        df.loc[anchor_idx, "timestamp"].isoformat(),
        df.loc[exit_idx,   "timestamp"].isoformat(),
        entry_close,
        exit_close,
        max_upside,
        max_drawdown,
        horizon_complete,
    )


# Cache SPY returns so we download once per (as_of_date, horizon_days) pair
_SPY_CACHE: dict[tuple[str, int], float | None] = {}


def _get_benchmark_return(
    as_of_date: str,
    horizon_days: int,
    download_dir: str,
) -> float | None:
    """Download SPY over the same horizon and return its total return."""
    key = (as_of_date, horizon_days)
    if key in _SPY_CACHE:
        return _SPY_CACHE[key]

    start_date = (pd.Timestamp(as_of_date) - pd.Timedelta(days=_ANCHOR_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date = (pd.Timestamp(as_of_date) + pd.Timedelta(days=horizon_days + 7)).strftime("%Y-%m-%d")
    try:
        raw = load_price_frame(
            symbol="SPY",
            start_date=start_date,
            end_date=end_date,
            download_dir=download_dir,
            timeframe="d1",
        )
        df  = _ensure_close_column(raw)
        anchor_idx = _pick_anchor_row(df, as_of_date)
        future = df.iloc[anchor_idx + 1:].copy() if anchor_idx is not None else pd.DataFrame()
        if anchor_idx is None or future.empty:
            # Definitive "no benchmark data for this date/horizon" — safe to cache.
            _SPY_CACHE[key] = None
            return None
        exit_target = df.loc[anchor_idx, "timestamp"] + pd.Timedelta(days=horizon_days)
        eligible = future.index[future["timestamp"] >= exit_target].tolist()
        exit_idx = eligible[0] if eligible else future.index[-1]
        entry = float(df.loc[anchor_idx, "close_eval"])
        exit_ = float(df.loc[exit_idx,  "close_eval"])
        bm = float(exit_ / entry - 1.0) if entry != 0 else None
        _SPY_CACHE[key] = bm
        return bm
    except Exception:
        # Transient failure (e.g. a download hiccup): do NOT cache, so a later
        # ticker on the same date can retry instead of every alpha being nulled.
        return None


def compute_realized_outcome(
    ticker: str,
    as_of_date: str,
    horizon_days: int,
    price_download_dir: str,
    preloaded_price_df: pd.DataFrame | None = None,
    benchmark_download_dir: str | None = None,
) -> RealizedOutcome:
    # Start the download a few days BEFORE as_of_date so that a month-end
    # as_of that lands on a weekend/holiday still has a prior trading bar to
    # anchor on (otherwise the whole period is silently dropped).
    start_date = (pd.Timestamp(as_of_date) - pd.Timedelta(days=_ANCHOR_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end_date = (pd.Timestamp(as_of_date) + pd.Timedelta(days=horizon_days + 7)).strftime("%Y-%m-%d")
    if preloaded_price_df is None:
        raw_df = load_price_frame(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            download_dir=price_download_dir,
            timeframe="d1",
        )
    else:
        raw_df = preloaded_price_df.copy()

    df = _ensure_close_column(raw_df)
    entry_ts, exit_ts, entry_close, exit_close, max_upside, max_drawdown, horizon_complete = \
        _compute_return_from_price_df(df, as_of_date, horizon_days)

    if entry_ts is None:
        return RealizedOutcome(
            ticker=ticker,
            as_of_date=as_of_date,
            horizon_days=horizon_days,
            entry_timestamp=None,
            exit_timestamp=None,
            entry_close=None,
            exit_close=None,
            realized_return=None,
            benchmark_return=None,
            alpha_return=None,
            max_upside=None,
            max_drawdown=None,
            horizon_complete=False,
            price_csv_path=str(Path(price_download_dir).resolve()),
        )

    realized_return = float(exit_close / entry_close - 1.0) if entry_close != 0 else None

    # Benchmark: SPY over the same horizon
    bm_dir = benchmark_download_dir or str(Path(price_download_dir).parent / "_spy_benchmark")
    benchmark_return = _get_benchmark_return(as_of_date, horizon_days, bm_dir)
    alpha_return = (
        round(realized_return - benchmark_return, 6)
        if realized_return is not None and benchmark_return is not None
        else None
    )

    price_csv_path = ""
    if "source_csv" in df.columns and not df.empty:
        price_csv_path = str(df["source_csv"].iloc[0])

    return RealizedOutcome(
        ticker=ticker,
        as_of_date=as_of_date,
        horizon_days=horizon_days,
        entry_timestamp=entry_ts,
        exit_timestamp=exit_ts,
        entry_close=entry_close,
        exit_close=exit_close,
        realized_return=realized_return,
        benchmark_return=benchmark_return,
        alpha_return=alpha_return,
        max_upside=max_upside,
        max_drawdown=max_drawdown,
        horizon_complete=horizon_complete,
        price_csv_path=price_csv_path,
    )
