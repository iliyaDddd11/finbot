from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _parse_metric_value(raw) -> float | None:
    """
    Parse a financial metric cell that may be a plain float, NaN, or a
    percentage string like '5.0%' or '-2.8%'.  Always returns a decimal
    fraction (5.0% → 0.05) or None.
    """
    import math
    if raw is None:
        return None
    if isinstance(raw, float):
        return None if math.isnan(raw) else raw
    try:
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", "n/a", ""):
            return None
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except (ValueError, TypeError):
        return None


def _find_forecast_columns(indexed: "pd.DataFrame") -> list[str]:
    """
    Return the forecast year column labels present in the CSV, e.g. ["2025E", "2026E"].

    The financial analysis pipeline labels forecast years relative to the most
    recent actual data year, so the labels change every year (2025E → 2026E in
    2026 runs, etc.).  This function detects whatever labels exist rather than
    hardcoding specific years.
    """
    forecast_cols = sorted(
        [c for c in indexed.columns if str(c).endswith("E") and str(c)[:-1].isdigit()],
        key=lambda c: int(c[:-1]),
    )
    return forecast_cols


def load_analysis_metrics(analysis_dir: str) -> dict[str, float | None]:
    csv_path = Path(analysis_dir) / "financial_metrics_and_forecasts.csv"
    metrics: dict[str, float | None] = {}
    if not csv_path.exists():
        return metrics

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return metrics
    if "metrics" not in df.columns:
        return metrics

    indexed = df.set_index("metrics")
    forecast_cols = _find_forecast_columns(indexed)

    # Map generic slot names to whatever forecast year columns actually exist.
    # fy1 = nearest forecast year, fy2 = second forecast year.
    fy1 = forecast_cols[0] if len(forecast_cols) >= 1 else None
    fy2 = forecast_cols[1] if len(forecast_cols) >= 2 else None

    # Build a year-agnostic lookup: always store under generic keys so the
    # rest of the audit code doesn't need to know the actual year labels.
    # SG&A row name varies across different versions of the financial analysis pipeline
    _sga_row = next(
        (r for r in ["SG&A as % Revenue", "SG&A Margin", "SGA Margin", "SG&A % Revenue"]
         if r in indexed.index),
        "SG&A as % Revenue",   # fallback (will just be None if not found)
    )

    row_map = {
        "revenue_growth_fy1":        ("Revenue Growth",      fy1),
        "revenue_growth_fy2":        ("Revenue Growth",      fy2),
        "ebitda_margin_fy1":         ("EBITDA Margin",       fy1),
        "ebitda_margin_fy2":         ("EBITDA Margin",       fy2),
        "contribution_margin_fy1":   ("Contribution Margin", fy1),
        "sga_pct_revenue_fy1":       (_sga_row,              fy1),
    }
    # Also store under year-specific keys for backward compatibility
    if fy1:
        row_map[f"revenue_growth_{fy1.lower()}"]        = ("Revenue Growth",      fy1)
        row_map[f"ebitda_margin_{fy1.lower()}"]         = ("EBITDA Margin",       fy1)
        row_map[f"contribution_margin_{fy1.lower()}"]   = ("Contribution Margin", fy1)
        row_map[f"sga_pct_revenue_{fy1.lower()}"]       = (_sga_row,              fy1)
    if fy2:
        row_map[f"revenue_growth_{fy2.lower()}"]        = ("Revenue Growth",      fy2)
        row_map[f"ebitda_margin_{fy2.lower()}"]         = ("EBITDA Margin",       fy2)

    for out_key, (row_name, col_name) in row_map.items():
        value = None
        if col_name and row_name in indexed.index and col_name in indexed.columns:
            raw = indexed.loc[row_name, col_name]
            value = _parse_metric_value(raw)
        metrics[out_key] = value

    # Store detected forecast year labels so callers can inspect them
    metrics["_fy1_label"] = fy1  # type: ignore[assignment]
    metrics["_fy2_label"] = fy2  # type: ignore[assignment]
    return metrics


def load_text_sections(analysis_dir: str) -> dict[str, str]:
    out: dict[str, str] = {}
    path = Path(analysis_dir)
    if not path.exists():
        return out
    for txt_file in sorted(path.glob("*.txt")):
        try:
            out[txt_file.stem] = txt_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue
    return out


def classify_outcome(signal: str, realized_return: float | None, neutral_band: float = 0.02) -> str:
    # Treat NaN like a missing return: a float NaN is not None, so without this
    # guard `NaN > 0` (False) would silently label a call "wrong" and inflate
    # the right_rate denominator with spurious losses.
    if realized_return is None or pd.isna(realized_return):
        return "unknown"
    signal = str(signal or "neutral").lower()
    if signal == "long":
        return "right" if realized_return > 0 else "wrong"
    if signal == "short":
        return "right" if realized_return < 0 else "wrong"
    return "right" if abs(realized_return) <= neutral_band else "wrong"


def build_driver_flags(metrics: dict[str, float | None], confidence: float, risk_flags: list[str] | None = None) -> dict[str, Any]:
    risk_flags = risk_flags or []
    # Use generic fy1 keys (year-agnostic); fall back to legacy 2025e keys for
    # scorecards generated before this fix was applied.
    rev    = metrics.get("revenue_growth_fy1") or metrics.get("revenue_growth_2025e")
    margin = metrics.get("ebitda_margin_fy1")  or metrics.get("ebitda_margin_2025e")
    sga    = metrics.get("sga_pct_revenue_fy1") or metrics.get("sga_pct_revenue_2025e")
    return {
        "positive_growth": rev is not None and rev > 0,
        "strong_growth": rev is not None and rev >= 0.08,
        "negative_growth": rev is not None and rev < 0,
        "strong_margin": margin is not None and margin >= 0.20,
        "weak_margin": margin is not None and margin < 0.10,
        "elevated_sga": sga is not None and sga >= 0.25,
        "high_confidence": confidence >= 0.70,
        "missing_metric_flag": any("missing" in str(flag).lower() for flag in risk_flags),
    }


def build_pattern_signature(signal: str, driver_flags: dict[str, Any]) -> str:
    ordered = [signal.lower()]
    for key in sorted(driver_flags):
        if driver_flags.get(key):
            ordered.append(key)
    return "|".join(ordered)


def build_case_evidence(row: pd.Series, neutral_band: float = 0.02) -> list[str]:
    evidence: list[str] = []
    rr = row.get("realized_return")
    if pd.notna(rr):
        evidence.append(f"realized_{int(row.get('horizon_days', 0))}d_return={float(rr):+.2%}")
    for key in [
        "revenue_growth_fy1",
        "revenue_growth_fy2",
        "ebitda_margin_fy1",
        "ebitda_margin_fy2",
        "contribution_margin_fy1",
        "sga_pct_revenue_fy1",
    ]:
        value = row.get(key)
        if pd.notna(value):
            evidence.append(f"{key}={float(value):+.2%}")
    max_upside = row.get("max_upside")
    if pd.notna(max_upside):
        evidence.append(f"path_max_upside={float(max_upside):+.2%}")
    max_drawdown = row.get("max_drawdown")
    if pd.notna(max_drawdown):
        evidence.append(f"path_max_drawdown={float(max_drawdown):+.2%}")
    expected = row.get("expected_return")
    if pd.notna(expected):
        evidence.append(f"expected_return={float(expected):+.2%}")
    signal = str(row.get("signal", "neutral")).lower()
    if signal == "neutral" and pd.notna(rr) and abs(float(rr)) > neutral_band:
        evidence.append("neutral_call_missed_large_move=true")
    return evidence


def summarize_patterns(scored_df: pd.DataFrame, min_count: int = 2) -> dict[str, list[dict[str, Any]]]:
    if scored_df.empty or "pattern_signature" not in scored_df.columns:
        return {"right_patterns": [], "wrong_patterns": []}

    grouped = (
        scored_df.groupby("pattern_signature", dropna=False)
        .agg(
            n=("pattern_signature", "size"),
            right_rate=("is_right", "mean"),
            avg_realized_return=("realized_return", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_count].sort_values(["n", "right_rate"], ascending=[False, False])
    right_patterns: list[dict[str, Any]] = []
    wrong_patterns: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        pattern = {
            "pattern_signature": row["pattern_signature"],
            "count": int(row["n"]),
            "right_rate": float(row["right_rate"]) if pd.notna(row["right_rate"]) else None,
            "avg_realized_return": float(row["avg_realized_return"]) if pd.notna(row["avg_realized_return"]) else None,
            "avg_confidence": float(row["avg_confidence"]) if pd.notna(row["avg_confidence"]) else None,
        }
        if row["right_rate"] >= 0.60:
            right_patterns.append(pattern)
        if row["right_rate"] <= 0.40:
            wrong_patterns.append(pattern)
    return {
        "right_patterns": right_patterns[:10],
        "wrong_patterns": wrong_patterns[:10],
    }


def save_casepack(scored_df: pd.DataFrame, output_dir: str, top_n: int = 15) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrong = scored_df[scored_df["outcome_label"] == "wrong"].copy()
    right = scored_df[scored_df["outcome_label"] == "right"].copy()
    wrong = wrong.sort_values(["confidence", "realized_return"], ascending=[False, True]).head(top_n)
    right = right.sort_values(["confidence", "realized_return"], ascending=[False, False]).head(top_n)

    paths = {
        "wrong_cases_json": str((out_dir / "wrong_cases.json").resolve()),
        "right_cases_json": str((out_dir / "right_cases.json").resolve()),
    }
    wrong.to_json(paths["wrong_cases_json"], orient="records", indent=2)
    right.to_json(paths["right_cases_json"], orient="records", indent=2)
    return paths


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
