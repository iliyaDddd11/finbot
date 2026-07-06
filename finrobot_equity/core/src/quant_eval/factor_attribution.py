"""
Factor-Level IC Attribution with t-Tests

Answers the question: "Which of the 10 factors is actually driving the alpha?"

For each factor, compute:
  - IC  : Spearman correlation of factor score vs realized return
  - t-stat: IC / std(IC) * sqrt(N), tests H0: mean_IC = 0
  - p-value: two-tailed, from t-distribution
  - ICIR: mean_IC / std_IC * sqrt(periods_per_year)
  - Significant: p < 0.05

To run this, factor scores must be stored as individual columns in the
predictions CSV, which this module also handles via `flatten_factor_scores()`.

References
----------
Grinold & Kahn (2000) "Active Portfolio Management", Ch. 6
Fama (1970) "Efficient Capital Markets"
"""
from __future__ import annotations

import ast
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Factor score extraction
# ---------------------------------------------------------------------------

def flatten_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the `factor_scores` JSON column into individual factor columns.

    `factor_scores` is stored as a JSON string like:
      {"earnings_quality": {"score": 0.5, ...}, ...}

    Returns a copy of df with new columns: factor_<name>_score, factor_<name>_raw
    """
    if "factor_scores" not in df.columns:
        return df

    out = df.copy()

    def _parse(cell: Any) -> dict:
        if isinstance(cell, dict):
            return cell
        if isinstance(cell, str) and cell.strip():
            try:
                return json.loads(cell)
            except Exception:
                try:
                    return ast.literal_eval(cell)
                except Exception:
                    pass
        return {}

    # Collect all factor names
    factor_names: set[str] = set()
    parsed_all: list[dict] = []
    for cell in out["factor_scores"]:
        parsed = _parse(cell)
        parsed_all.append(parsed)
        factor_names.update(parsed.keys())

    for fname in sorted(factor_names):
        score_col = f"factor_{fname}_score"
        raw_col   = f"factor_{fname}_raw"
        scores, raws = [], []
        for parsed in parsed_all:
            fdata = parsed.get(fname, {})
            scores.append(fdata.get("score") if isinstance(fdata, dict) else None)
            raws.append(fdata.get("raw_value") if isinstance(fdata, dict) else None)
        out[score_col] = pd.to_numeric(scores, errors="coerce")
        out[raw_col]   = pd.to_numeric(raws,   errors="coerce")

    return out


# ---------------------------------------------------------------------------
# Factor IC attribution
# ---------------------------------------------------------------------------

def factor_ic_attribution(
    df: pd.DataFrame,
    realized_col: str = "realized_return",
    periods_per_year: int = 6,
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    Compute factor-level IC attribution.

    Parameters
    ----------
    df              : Predictions DataFrame (must have factor_scores column
                      or pre-flattened factor_<name>_score columns)
    realized_col    : Column name for realized returns
    periods_per_year: For ICIR annualisation
    min_obs         : Minimum observations needed to compute IC

    Returns
    -------
    DataFrame with one row per factor, columns:
      factor, n_obs, mean_ic, std_ic, t_stat, p_value, icir, significant
    """
    # Ensure factor columns exist
    df = flatten_factor_scores(df)

    # Identify factor score columns
    score_cols = [c for c in df.columns if c.startswith("factor_") and c.endswith("_score")]
    if not score_cols:
        return pd.DataFrame(columns=[
            "factor", "n_obs", "mean_ic", "std_ic",
            "t_stat", "p_value", "icir", "significant"
        ])

    realized = pd.to_numeric(df.get(realized_col, pd.Series(dtype=float)), errors="coerce")
    rows = []

    for col in score_cols:
        factor_name = col.replace("factor_", "").replace("_score", "")
        factor_vals = pd.to_numeric(df[col], errors="coerce")

        mask = factor_vals.notna() & realized.notna()
        n    = mask.sum()

        if n < min_obs:
            rows.append({
                "factor": factor_name, "n_obs": int(n),
                "mean_ic": None, "std_ic": None,
                "t_stat": None, "p_value": None,
                "icir": None, "significant": False,
            })
            continue

        # Per-period IC (group by date if available)
        if "as_of_date" in df.columns and df["as_of_date"].nunique() > 1:
            period_ics = []
            for _, grp in df[mask].groupby("as_of_date"):
                f_g = pd.to_numeric(grp[col], errors="coerce")
                r_g = pd.to_numeric(grp[realized_col], errors="coerce")
                m   = f_g.notna() & r_g.notna()
                if m.sum() < 3 or f_g[m].std() < 1e-9 or r_g[m].std() < 1e-9:
                    continue
                ic, _ = scipy_stats.spearmanr(f_g[m], r_g[m])
                if not math.isnan(ic):
                    period_ics.append(float(ic))

            if len(period_ics) < 2:
                # Fall back to pooled IC
                ic_pool, _ = scipy_stats.spearmanr(factor_vals[mask], realized[mask])
                mean_ic = float(ic_pool) if not math.isnan(ic_pool) else None
                std_ic  = None
                t_stat  = None
                p_val   = None
                icir    = None
            else:
                ic_arr  = np.array(period_ics)
                mean_ic = float(ic_arr.mean())
                std_ic  = float(ic_arr.std(ddof=1))
                t_stat  = float(mean_ic / std_ic * math.sqrt(len(ic_arr))) if std_ic > 1e-9 else None
                p_val   = float(2 * scipy_stats.t.sf(abs(t_stat), df=len(ic_arr) - 1)) if t_stat else None
                icir    = float(mean_ic / std_ic * math.sqrt(periods_per_year)) if std_ic > 1e-9 else None
        else:
            # Pooled IC (single period or no date column)
            ic_pool, p_pool = scipy_stats.spearmanr(factor_vals[mask], realized[mask])
            mean_ic = float(ic_pool) if not math.isnan(ic_pool) else None
            std_ic  = None
            t_stat  = float(math.sqrt(n - 2) * ic_pool / math.sqrt(1 - ic_pool ** 2)) if mean_ic and abs(ic_pool) < 1 else None
            p_val   = float(p_pool)
            icir    = None

        rows.append({
            "factor":      factor_name,
            "n_obs":       int(n),
            "mean_ic":     round(mean_ic, 4) if mean_ic is not None else None,
            "std_ic":      round(std_ic,  4) if std_ic  is not None else None,
            "t_stat":      round(t_stat,  3) if t_stat  is not None else None,
            "p_value":     round(p_val,   4) if p_val   is not None else None,
            "icir":        round(icir,    3) if icir    is not None else None,
            "significant": bool(p_val is not None and p_val < 0.05),
        })

    result = pd.DataFrame(rows)
    if not result.empty and "mean_ic" in result.columns:
        result = result.sort_values("mean_ic", ascending=False, na_position="last")
    return result.reset_index(drop=True)


def factor_attribution_report_lines(attr_df: pd.DataFrame) -> list[str]:
    """Format factor attribution as markdown table rows."""
    if attr_df.empty:
        return ["No factor attribution data available."]

    lines = [
        "| Factor | N | Mean IC | Std IC | t-stat | p-value | ICIR | Sig? |",
        "|--------|---|---------|--------|--------|---------|------|------|",
    ]
    for _, row in attr_df.iterrows():
        sig = "✓" if row.get("significant") else ""
        lines.append(
            f"| {row['factor']} | {row.get('n_obs', '')} "
            f"| {_fmt(row.get('mean_ic'))} "
            f"| {_fmt(row.get('std_ic'))} "
            f"| {_fmt(row.get('t_stat'))} "
            f"| {_fmt(row.get('p_value'))} "
            f"| {_fmt(row.get('icir'))} "
            f"| {sig} |"
        )
    return lines


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.3f}"
