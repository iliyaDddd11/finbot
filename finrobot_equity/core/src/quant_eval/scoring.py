from __future__ import annotations

from typing import Any

import pandas as pd

from quant_eval.decision_audit import summarize_patterns
from quant_eval.performance_metrics import (
    compute_full_scorecard as _quant_scorecard,
    add_bootstrap_cis,
)


SIGNAL_DIRECTION = {"long": 1, "short": -1, "neutral": 0}


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else None


def _bucket_confidence(confidence: float) -> str:
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


def enrich_scored_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal_direction"] = out["signal"].astype(str).str.lower().map(SIGNAL_DIRECTION).fillna(0)
    out["signed_return"] = out["signal_direction"] * pd.to_numeric(out["realized_return"], errors="coerce")
    out["is_right"] = (out["outcome_label"] == "right").astype(int)
    out["confidence_bucket"] = out["confidence"].apply(_bucket_confidence)
    # Calibration error only on rows with a known outcome (right/wrong)
    known_mask = out["outcome_label"].isin(["right", "wrong"])
    out["confidence_error"] = float("nan")
    out.loc[known_mask, "confidence_error"] = (
        out.loc[known_mask, "confidence"] - out.loc[known_mask, "is_right"]
    ).abs()
    out["thesis_point_count"] = out["thesis_points"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    out["risk_flag_count"] = out["risk_flags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return out


def compute_scorecard(scored_df: pd.DataFrame, periods_per_year: int = 12) -> dict[str, Any]:
    # periods_per_year defaults to 12 because the walk-forward grid is monthly
    # (date_grid.build_month_end_grid uses freq="ME").  Annualised Sharpe / ICIR
    # / Calmar are only correct when this matches the true rebalance cadence.
    if scored_df.empty:
        return {
            "num_predictions": 0,
            "right_rate": None,
            "avg_realized_return": None,
            "avg_signed_return": None,
            "avg_confidence": None,
            "confidence_calibration_error": None,
            "patterns": {"right_patterns": [], "wrong_patterns": []},
        }

    # Restrict hit_rate/right_rate to rows with known outcomes only
    known = scored_df[scored_df["outcome_label"].isin(["right", "wrong"])]
    n_known = len(known)
    n_right = int((known["outcome_label"] == "right").sum())

    metrics: dict[str, Any] = {
        "num_predictions":   int(len(scored_df)),
        "num_known":         n_known,
        "right_rate":        round(n_right / n_known, 4) if n_known else None,
        "avg_realized_return": _safe_mean(scored_df["realized_return"]),
        "avg_signed_return":   _safe_mean(scored_df["signed_return"]),
        "avg_confidence":      _safe_mean(scored_df["confidence"]),
        "confidence_calibration_error": _safe_mean(scored_df["confidence_error"]),
        "avg_max_upside":    _safe_mean(scored_df.get("max_upside", pd.Series(dtype=float))),
        "avg_max_drawdown":  _safe_mean(scored_df.get("max_drawdown", pd.Series(dtype=float))),
        "patterns":          summarize_patterns(scored_df),
    }

    # Add professional quant metrics (IC, ICIR, Sharpe, max_drawdown, Calmar)
    quant = _quant_scorecard(scored_df, periods_per_year=periods_per_year)
    metrics["sharpe_ratio"]   = quant.get("sharpe_ratio")
    metrics["max_drawdown"]   = quant.get("max_drawdown")
    metrics["calmar_ratio"]   = quant.get("calmar_ratio")
    metrics["mean_ic"]        = quant.get("mean_ic")
    metrics["icir"]           = quant.get("icir")
    metrics["ic_periods"]     = quant.get("ic_periods", 0)

    # Benchmark-relative alpha metrics
    if "alpha_return" in scored_df.columns:
        sig_dir = scored_df["signal"].astype(str).str.lower().map(
            {"long": 1, "short": -1, "neutral": 0}
        ).fillna(0)
        alpha_signed = sig_dir * pd.to_numeric(scored_df["alpha_return"], errors="coerce")
        bm_vals      = pd.to_numeric(scored_df["benchmark_return"], errors="coerce")
        metrics["avg_benchmark_return"] = float(bm_vals.dropna().mean()) if not bm_vals.dropna().empty else None
        metrics["avg_alpha_return"]     = float(alpha_signed.dropna().mean()) if not alpha_signed.dropna().empty else None
    else:
        metrics["avg_benchmark_return"] = None
        metrics["avg_alpha_return"]     = None

    # Bootstrap 95% confidence intervals
    add_bootstrap_cis(metrics, scored_df, periods_per_year=periods_per_year)

    by_signal = {}
    for signal, signal_df in scored_df.groupby(scored_df["signal"].astype(str).str.lower()):
        s_known = signal_df[signal_df["outcome_label"].isin(["right", "wrong"])]
        s_n_k   = len(s_known)
        s_n_r   = int((s_known["outcome_label"] == "right").sum())
        by_signal[signal] = {
            "count":               int(len(signal_df)),
            "right_rate":          round(s_n_r / s_n_k, 4) if s_n_k else None,
            "avg_realized_return": _safe_mean(signal_df["realized_return"]),
            "avg_signed_return":   _safe_mean(signal_df["signed_return"]),
            "avg_confidence":      _safe_mean(signal_df["confidence"]),
        }
    metrics["by_signal"] = by_signal

    by_confidence = {}
    for bucket, bucket_df in scored_df.groupby("confidence_bucket"):
        b_known = bucket_df[bucket_df["outcome_label"].isin(["right", "wrong"])]
        b_n_k   = len(b_known)
        b_n_r   = int((b_known["outcome_label"] == "right").sum())
        by_confidence[bucket] = {
            "count":               int(len(bucket_df)),
            "right_rate":          round(b_n_r / b_n_k, 4) if b_n_k else None,
            "avg_realized_return": _safe_mean(bucket_df["realized_return"]),
            "avg_signed_return":   _safe_mean(bucket_df["signed_return"]),
            "avg_confidence":      _safe_mean(bucket_df["confidence"]),
        }
    metrics["by_confidence_bucket"] = by_confidence

    by_ticker = {}
    for ticker, ticker_df in scored_df.groupby("ticker"):
        t_known = ticker_df[ticker_df["outcome_label"].isin(["right", "wrong"])]
        t_n_k   = len(t_known)
        t_n_r   = int((t_known["outcome_label"] == "right").sum())
        by_ticker[str(ticker)] = {
            "count":               int(len(ticker_df)),
            "right_rate":          round(t_n_r / t_n_k, 4) if t_n_k else None,
            "avg_realized_return": _safe_mean(ticker_df["realized_return"]),
            "avg_signed_return":   _safe_mean(ticker_df["signed_return"]),
        }
    metrics["by_ticker"] = by_ticker
    return metrics
