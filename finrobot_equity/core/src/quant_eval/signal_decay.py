"""
Signal Decay Analysis — Multi-Horizon IC Curves

A signal that has IC of 0.10 at 30 days but IC of 0.02 at 90 days has a
half-life of roughly 30 days.  This tells you:
  - Optimal rebalancing frequency
  - Whether you're trading momentum (decays fast) or value (decays slowly)
  - Whether the 60-day default horizon is too long or too short

This module computes realized returns at multiple horizons from the snapshot
price data, then computes IC (and t-stat) at each horizon.

Usage
-----
In score_walkforward_eval.py with --horizons 30 60 90 120:
  decay_df = compute_ic_decay(predictions_df, price_dir, [30, 60, 90, 120])
  lines = ic_decay_report_lines(decay_df)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from quant_eval.realized_returns import compute_realized_outcome


def compute_horizon_returns(
    predictions_df: pd.DataFrame,
    price_download_dir: str,
    horizons: list[int],
) -> pd.DataFrame:
    """
    For each prediction row, compute realized returns at each horizon.

    Adds columns `realized_return_{h}d` for each h in horizons.
    Downloads price data once per (ticker, date) pair and reuses.

    Parameters
    ----------
    predictions_df    : DataFrame with columns: ticker, as_of_date
    price_download_dir: Directory for price cache
    horizons          : List of horizon days, e.g. [30, 60, 90]

    Returns
    -------
    predictions_df with new `realized_return_{h}d` columns
    """
    out = predictions_df.copy()
    for h in horizons:
        col = f"realized_return_{h}d"
        if col in out.columns:
            continue
        out[col] = float("nan")

    for idx, row in out.iterrows():
        ticker     = str(row["ticker"])
        as_of_date = str(row["as_of_date"])
        for h in horizons:
            col = f"realized_return_{h}d"
            if pd.notna(out.at[idx, col]):
                continue
            try:
                ro = compute_realized_outcome(
                    ticker=ticker,
                    as_of_date=as_of_date,
                    horizon_days=h,
                    price_download_dir=str(
                        Path(price_download_dir) / f"{ticker}_{as_of_date}_{h}d"
                    ),
                )
                out.at[idx, col] = ro.realized_return
            except Exception:
                pass  # leave NaN

    return out


def compute_ic_decay(
    df: pd.DataFrame,
    horizons: list[int],
    periods_per_year: int = 6,
) -> pd.DataFrame:
    """
    Compute IC and ICIR at each horizon.

    Parameters
    ----------
    df              : Must have columns: signal, realized_return_{h}d for each h
    horizons        : List of horizon days
    periods_per_year: For ICIR annualisation

    Returns
    -------
    DataFrame: horizon_days | mean_ic | std_ic | icir | t_stat | p_value | n_obs
    """
    from quant_eval.performance_metrics import _to_numeric_signal

    signal_dir = _to_numeric_signal(df["signal"])
    rows = []

    for h in horizons:
        col = f"realized_return_{h}d"
        if col not in df.columns:
            rows.append(_null_row(h))
            continue

        ret = pd.to_numeric(df[col], errors="coerce")
        mask = signal_dir.notna() & ret.notna()
        n    = int(mask.sum())

        if n < 5:
            rows.append({**_null_row(h), "n_obs": n})
            continue

        s = signal_dir[mask].values
        r = ret[mask].values

        if np.std(s) < 1e-9 or np.std(r) < 1e-9:
            rows.append({**_null_row(h), "n_obs": n})
            continue

        ic, p = scipy_stats.spearmanr(s, r)
        ic    = float(ic) if not math.isnan(ic) else None
        t_stat = (
            float(math.sqrt(n - 2) * ic / math.sqrt(max(1e-9, 1 - ic ** 2)))
            if ic is not None and abs(ic) < 1
            else None
        )

        rows.append({
            "horizon_days": h,
            "n_obs":   n,
            "mean_ic": round(ic, 4) if ic is not None else None,
            "std_ic":  None,   # pooled IC — no per-period std
            "icir":    None,
            "t_stat":  round(t_stat, 3) if t_stat is not None else None,
            "p_value": round(float(p), 4) if not math.isnan(p) else None,
            "significant": bool(p < 0.05) if not math.isnan(p) else False,
        })

    return pd.DataFrame(rows)


def _null_row(h: int) -> dict[str, Any]:
    return {
        "horizon_days": h, "n_obs": 0, "mean_ic": None,
        "std_ic": None, "icir": None, "t_stat": None,
        "p_value": None, "significant": False,
    }


def ic_decay_report_lines(decay_df: pd.DataFrame) -> list[str]:
    """Format IC decay as a markdown table."""
    if decay_df.empty:
        return ["No IC decay data available."]

    lines = [
        "",
        "## Signal IC Decay by Horizon",
        "",
        "| Horizon | N | Mean IC | t-stat | p-value | Significant? |",
        "|---------|---|---------|--------|---------|--------------|",
    ]
    for _, row in decay_df.iterrows():
        sig = "Yes" if row.get("significant") else "No"
        lines.append(
            f"| {int(row['horizon_days'])}d "
            f"| {int(row.get('n_obs', 0))} "
            f"| {_f(row.get('mean_ic'))} "
            f"| {_f(row.get('t_stat'))} "
            f"| {_f(row.get('p_value'))} "
            f"| {sig} |"
        )
    return lines


def _f(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.4f}" if abs(float(v)) < 1 else f"{float(v):.3f}"
