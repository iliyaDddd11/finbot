"""
Point-In-Time (PIT) Financial Data Filter

The most common source of look-ahead bias in fundamental quant research
is using earnings data that had not yet been announced as of the evaluation date.

FMP (and most data vendors) store data with the fiscal period end date, NOT the
announcement/filing date.  A company with fiscal year ending Sep 30 typically
files its 10-K in late November — 60 days later.  Using September data in an
October back-test means the model "knew" numbers that were not public.

This module approximates announcement dates using standard filing lags:
  - Annual reports (10-K):     fiscal year end + 60 days
  - Quarterly reports (10-Q):  fiscal period end + 45 days

These are conservative estimates.  Real PIT databases (Compustat Point-in-Time,
FactSet Revisions) use the exact EDGAR filing timestamp.

References
----------
Hou, Xue, Zhang (2020) "Replicating Anomalies" — documents the impact of
look-ahead bias from using data before its public availability date.
"""
from __future__ import annotations

import pandas as pd

# Typical filing delays (calendar days after fiscal period end)
ANNUAL_FILING_LAG_DAYS    = 60    # 10-K / annual report
QUARTERLY_FILING_LAG_DAYS = 45    # 10-Q / quarterly report


def estimate_announcement_date(period_end: str, period: str = "annual") -> str:
    """
    Estimate the earliest date the market could have seen this data.

    Parameters
    ----------
    period_end : ISO date string of the fiscal period end
    period     : "annual" or "quarterly"

    Returns
    -------
    ISO date string of estimated announcement date
    """
    lag = ANNUAL_FILING_LAG_DAYS if period == "annual" else QUARTERLY_FILING_LAG_DAYS
    return (pd.Timestamp(period_end) + pd.Timedelta(days=lag)).strftime("%Y-%m-%d")


# Columns that record the ACTUAL public availability date → use as-is, no lag.
# (FMP spells it "fillingDate"; "filingDate"/"acceptedDate" appear in some feeds.)
_FILING_DATE_COLS = ("acceptedDate", "fillingDate", "filingDate")
# Columns that record the fiscal PERIOD END → apply an estimated filing lag.
_PERIOD_END_COLS = ("date", "Date", "calendarYear", "reportDate")


def _infer_date_column(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Find the most likely date column and its kind.

    Returns (column_name, kind) where kind is "filing" (actual availability
    date, used directly) or "period_end" (fiscal period end, needs a filing
    lag). Prefers a real filing date when present so a late filer isn't marked
    available at the estimated date. Returns (None, None) if nothing usable.
    """
    for candidate in _FILING_DATE_COLS:
        if candidate in df.columns:
            return candidate, "filing"
    for candidate in _PERIOD_END_COLS:
        if candidate in df.columns:
            return candidate, "period_end"
    return None, None


def filter_to_pit(
    df: pd.DataFrame,
    as_of_date: str,
    period: str = "annual",
    date_col: str | None = None,
) -> pd.DataFrame:
    """
    Remove rows from a financial DataFrame that were not yet publicly available
    as of `as_of_date`.

    Parameters
    ----------
    df          : Financial statement DataFrame (e.g. income_statement)
    as_of_date  : The walk-forward evaluation date
    period      : "annual" or "quarterly"
    date_col    : Column name containing the fiscal period end date.
                  Auto-detected if None.

    Returns
    -------
    Filtered DataFrame (may be empty if all data is post-announcement)
    """
    if df is None or df.empty:
        return df

    if date_col:
        col = date_col
        kind = "filing" if date_col in _FILING_DATE_COLS else "period_end"
    else:
        col, kind = _infer_date_column(df)
    if col is None:
        # Can't filter without a date column — return as-is with a warning flag
        return df

    try:
        dates = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        return df

    # If the entire column is unparseable (e.g. a "period" label column with
    # values like "FY"/"Q1"), do NOT silently drop every row — that would
    # masquerade as "all data is post-announcement" and corrupt downstream
    # signals. Leave the frame unchanged; the caller's data is the problem.
    if int(dates.notna().sum()) == 0:
        return df.copy()

    cutoff = pd.Timestamp(as_of_date)
    if kind == "filing":
        # Actual filing/acceptance date — available as soon as it is <= as_of.
        available_mask = dates <= cutoff
    else:
        # Fiscal period end + estimated filing lag.
        lag = ANNUAL_FILING_LAG_DAYS if period == "annual" else QUARTERLY_FILING_LAG_DAYS
        available_mask = (dates + pd.Timedelta(days=lag)) <= cutoff

    # NaT (unparseable) rows are treated as unavailable (conservative).
    available_mask = available_mask.fillna(False)
    return df[available_mask].copy()


def filter_financial_data_to_pit(
    financial_data: dict[str, pd.DataFrame],
    as_of_date: str,
    period: str = "annual",
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """
    Apply PIT filter to all financial statement DataFrames.

    Parameters
    ----------
    financial_data : Dict of {statement_name: DataFrame}
    as_of_date     : Walk-forward evaluation date
    period         : "annual" or "quarterly"

    Returns
    -------
    (filtered_data, rows_dropped) where rows_dropped is {statement: n_rows_removed}
    """
    filtered: dict[str, pd.DataFrame] = {}
    rows_dropped: dict[str, int] = {}

    for key, df in financial_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            filtered[key] = df
            rows_dropped[key] = 0
            continue

        original_len = len(df)
        filtered_df  = filter_to_pit(df, as_of_date, period=period)
        filtered[key] = filtered_df
        rows_dropped[key] = original_len - len(filtered_df)

    return filtered, rows_dropped


def pit_audit_line(rows_dropped: dict[str, int], as_of_date: str) -> str:
    """Return a human-readable summary of PIT filtering applied."""
    total = sum(rows_dropped.values())
    if total == 0:
        return f"PIT filter ({as_of_date}): all data available"
    details = ", ".join(f"{k}:{v}" for k, v in rows_dropped.items() if v > 0)
    return f"PIT filter ({as_of_date}): {total} row(s) removed ({details})"
