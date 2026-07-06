"""
Cross-Sectional Factor Z-Scoring

Without this, a ROIC of 52% always gets the same tanh score regardless of
whether the peer universe averages 20% or 55%.  With this, every factor is
z-scored across the live universe on each evaluation date, making all signals
relative — which is how systematic equity research works in practice.

Flow
----
1. Caller collects raw FactorResult objects for all tickers at one date.
2. `zscore_factor_bank()` normalises each factor across tickers.
3. The signal engine receives the z-scored values in place of raw scores.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_eval.factor_library import FactorResult

# Clip z-scores to this range to avoid extreme outlier dominance
_Z_CLIP = 3.0

# Minimum universe size needed to z-score a factor reliably
_MIN_UNIVERSE = 5


def zscore_factor_bank(
    ticker_raw_factors: dict[str, dict[str, FactorResult]],
) -> dict[str, dict[str, float | None]]:
    """
    Cross-sectional z-score of raw factor values across all tickers.

    Parameters
    ----------
    ticker_raw_factors : {ticker: {factor_name: FactorResult}}
        Raw factor results for all tickers at a single evaluation date.

    Returns
    -------
    {ticker: {factor_name: z_score}}
        z_score is None when the factor is unavailable for that ticker or
        when fewer than _MIN_UNIVERSE tickers have the factor available.
    """
    factor_names = list(next(iter(ticker_raw_factors.values())).keys()) if ticker_raw_factors else []
    tickers = list(ticker_raw_factors.keys())

    # Collect raw values per factor
    raw: dict[str, dict[str, float | None]] = {f: {} for f in factor_names}
    for ticker in tickers:
        for fname, fr in ticker_raw_factors[ticker].items():
            raw[fname][ticker] = fr.raw_value if fr.available and fr.raw_value is not None else None

    # Z-score each factor
    zscores: dict[str, dict[str, float | None]] = {f: {t: None for t in tickers} for f in factor_names}
    for fname in factor_names:
        vals = {t: v for t, v in raw[fname].items() if v is not None}
        if len(vals) < _MIN_UNIVERSE:
            continue  # leave all as None — not enough cross-section
        arr = np.array(list(vals.values()), dtype=float)
        mu  = arr.mean()
        std = arr.std(ddof=1)
        if std < 1e-9:
            continue  # no variation — skip
        for ticker, v in vals.items():
            z = (v - mu) / std
            # Clip and scale to [-2, +2] to match the tanh output range
            z_clipped = float(np.clip(z, -_Z_CLIP, _Z_CLIP)) * (2.0 / _Z_CLIP)
            zscores[fname][ticker] = round(z_clipped, 4)

    # Re-index as {ticker: {factor: z_score}}
    result: dict[str, dict[str, float | None]] = {t: {} for t in tickers}
    for fname in factor_names:
        for ticker in tickers:
            result[ticker][fname] = zscores[fname][ticker]

    return result


def cross_section_summary(
    ticker_raw_factors: dict[str, dict[str, FactorResult]],
) -> pd.DataFrame:
    """
    Return a DataFrame summarising the cross-sectional distribution of each
    factor (mean, std, min, max, n_available) — useful for diagnostics.
    """
    factor_names = list(next(iter(ticker_raw_factors.values())).keys()) if ticker_raw_factors else []
    rows = []
    for fname in factor_names:
        vals = [
            fr.raw_value
            for factors in ticker_raw_factors.values()
            for fn, fr in factors.items()
            if fn == fname and fr.available and fr.raw_value is not None
        ]
        if vals:
            a = np.array(vals, dtype=float)
            rows.append({
                "factor":      fname,
                "n_available": len(vals),
                "mean":        round(float(a.mean()), 4),
                "std":         round(float(a.std(ddof=1)), 4) if len(vals) > 1 else None,
                "min":         round(float(a.min()), 4),
                "max":         round(float(a.max()), 4),
            })
        else:
            rows.append({"factor": fname, "n_available": 0,
                         "mean": None, "std": None, "min": None, "max": None})
    return pd.DataFrame(rows)
