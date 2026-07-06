"""
Unit tests for quant_eval/pit_filter.py

Covers:
  - estimate_announcement_date (annual vs quarterly lag)
  - filter_to_pit (normal, PIT removal, all-available, all-filtered)
  - filter_financial_data_to_pit (multi-statement dict)
  - pit_audit_line formatting
  - Missing/unknown date columns
"""
from __future__ import annotations

import pandas as pd
import pytest

from quant_eval.pit_filter import (
    ANNUAL_FILING_LAG_DAYS,
    QUARTERLY_FILING_LAG_DAYS,
    estimate_announcement_date,
    filter_financial_data_to_pit,
    filter_to_pit,
    pit_audit_line,
)


# ---------------------------------------------------------------------------
# estimate_announcement_date
# ---------------------------------------------------------------------------

class TestEstimateAnnouncementDate:
    def test_annual_lag(self):
        result = estimate_announcement_date("2023-12-31", "annual")
        expected = (pd.Timestamp("2023-12-31") + pd.Timedelta(days=ANNUAL_FILING_LAG_DAYS)).strftime("%Y-%m-%d")
        assert result == expected

    def test_quarterly_lag(self):
        result = estimate_announcement_date("2023-09-30", "quarterly")
        expected = (pd.Timestamp("2023-09-30") + pd.Timedelta(days=QUARTERLY_FILING_LAG_DAYS)).strftime("%Y-%m-%d")
        assert result == expected

    def test_annual_60_days(self):
        # Concrete sanity: fiscal year end Dec 31 → announced Feb 29 or Mar 1
        result = estimate_announcement_date("2023-12-31", "annual")
        ts = pd.Timestamp(result)
        assert ts == pd.Timestamp("2023-12-31") + pd.Timedelta(days=60)

    def test_quarterly_45_days(self):
        result = estimate_announcement_date("2023-09-30", "quarterly")
        ts = pd.Timestamp(result)
        assert ts == pd.Timestamp("2023-09-30") + pd.Timedelta(days=45)

    def test_default_is_annual(self):
        r_explicit = estimate_announcement_date("2022-12-31", "annual")
        r_default  = estimate_announcement_date("2022-12-31")
        assert r_explicit == r_default


# ---------------------------------------------------------------------------
# filter_to_pit — basic filtering
# ---------------------------------------------------------------------------

class TestFilterToPit:
    def _make_df(self, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"date": dates, "revenue": range(len(dates))})

    def test_removes_future_row_annual(self):
        # Fiscal year end 2023-11-30 → announced ~2024-01-29 (60d)
        # If as_of_date is 2023-11-01, the row should be removed
        df = self._make_df(["2022-12-31", "2023-11-30"])
        result = filter_to_pit(df, "2023-11-01", period="annual")
        # 2022-12-31 + 60d = 2023-03-01 <= 2023-11-01 → keep
        # 2023-11-30 + 60d = 2024-01-29 > 2023-11-01 → remove
        assert len(result) == 1
        assert result.iloc[0]["date"] == "2022-12-31"

    def test_keeps_row_after_announcement(self):
        # 2022-12-31 annual → announced 2023-03-01; as_of_date = 2023-04-01 → keep
        df = self._make_df(["2022-12-31"])
        result = filter_to_pit(df, "2023-04-01", period="annual")
        assert len(result) == 1

    def test_quarterly_shorter_lag(self):
        # 2023-09-30 quarterly → announced 2023-11-14 (45d)
        # as_of_date 2023-11-01 → remove; 2023-12-01 → keep
        df = self._make_df(["2023-09-30"])
        assert len(filter_to_pit(df, "2023-11-01", period="quarterly")) == 0
        assert len(filter_to_pit(df, "2023-12-01", period="quarterly")) == 1

    def test_exact_boundary_day_is_kept(self):
        # announcement day itself should be included (<=, not <)
        df = self._make_df(["2022-12-31"])
        ann_date = estimate_announcement_date("2022-12-31", "annual")
        result = filter_to_pit(df, ann_date, period="annual")
        assert len(result) == 1

    def test_all_rows_filtered(self):
        df = self._make_df(["2024-06-30", "2024-09-30"])
        result = filter_to_pit(df, "2024-07-01", period="annual")
        assert result.empty

    def test_all_rows_kept(self):
        df = self._make_df(["2020-12-31", "2021-12-31"])
        result = filter_to_pit(df, "2025-01-01", period="annual")
        assert len(result) == 2

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["date", "revenue"])
        result = filter_to_pit(df, "2024-01-01")
        assert result.empty

    def test_none_dataframe_returns_none(self):
        result = filter_to_pit(None, "2024-01-01")
        assert result is None

    def test_no_date_column_returns_as_is(self):
        df = pd.DataFrame({"revenue": [100, 200], "profit": [10, 20]})
        result = filter_to_pit(df, "2024-01-01")
        assert len(result) == 2  # returned unchanged

    def test_explicit_date_col_parameter(self):
        df = pd.DataFrame({"reportDate": ["2022-12-31", "2024-01-01"], "val": [1, 2]})
        result = filter_to_pit(df, "2023-06-01", period="annual", date_col="reportDate")
        # 2022-12-31 + 60d = 2023-03-01 <= 2023-06-01 → keep
        # 2024-01-01 + 60d = 2024-03-01 > 2023-06-01 → remove
        assert len(result) == 1

    def test_auto_detects_calendarYear_column(self):
        df = pd.DataFrame({"calendarYear": ["2021-12-31", "2023-12-31"], "eps": [1.0, 2.0]})
        result = filter_to_pit(df, "2023-03-01", period="annual")
        # 2021-12-31 + 60d → 2022-03-01 <= 2023-03-01 → keep
        # 2023-12-31 + 60d → 2024-03-01 > 2023-03-01 → remove
        assert len(result) == 1
        assert result.iloc[0]["eps"] == 1.0

    def test_multiple_rows_correct_order_preserved(self):
        df = self._make_df(["2021-12-31", "2022-12-31", "2023-12-31"])
        result = filter_to_pit(df, "2023-06-01", period="annual")
        # 2021+60d=2022-03-01 keep, 2022+60d=2023-03-01 keep, 2023+60d=2024-03-01 remove
        assert len(result) == 2
        assert list(result["date"]) == ["2021-12-31", "2022-12-31"]


# ---------------------------------------------------------------------------
# filter_financial_data_to_pit — multi-statement dict
# ---------------------------------------------------------------------------

class TestFilterFinancialDataToPit:
    def _make_income(self, dates):
        return pd.DataFrame({"date": dates, "revenue": range(len(dates))})

    def _make_balance(self, dates):
        return pd.DataFrame({"date": dates, "assets": range(len(dates))})

    def test_filters_all_statements(self):
        fin_data = {
            "income_statement": self._make_income(["2022-12-31", "2023-12-31"]),
            "balance_sheet":    self._make_balance(["2022-12-31", "2023-12-31"]),
        }
        filtered, dropped = filter_financial_data_to_pit(fin_data, "2023-06-01", "annual")
        assert len(filtered["income_statement"]) == 1
        assert len(filtered["balance_sheet"]) == 1
        assert dropped["income_statement"] == 1
        assert dropped["balance_sheet"] == 1

    def test_empty_statement_passthrough(self):
        fin_data = {"income_statement": pd.DataFrame()}
        filtered, dropped = filter_financial_data_to_pit(fin_data, "2024-01-01")
        assert filtered["income_statement"].empty
        assert dropped["income_statement"] == 0

    def test_rows_dropped_count(self):
        fin_data = {
            "income_statement": self._make_income(["2021-12-31", "2022-12-31", "2023-12-31"]),
        }
        _, dropped = filter_financial_data_to_pit(fin_data, "2023-06-01", "annual")
        assert dropped["income_statement"] == 1  # 2023-12-31 removed


# ---------------------------------------------------------------------------
# pit_audit_line
# ---------------------------------------------------------------------------

class TestPitAuditLine:
    def test_no_drops(self):
        line = pit_audit_line({"income_statement": 0, "balance_sheet": 0}, "2024-01-01")
        assert "all data available" in line
        assert "2024-01-01" in line

    def test_with_drops(self):
        line = pit_audit_line({"income_statement": 2, "balance_sheet": 1}, "2024-01-01")
        assert "3 row(s) removed" in line
        assert "income_statement:2" in line
        assert "balance_sheet:1" in line

    def test_partial_drops(self):
        line = pit_audit_line({"income_statement": 0, "balance_sheet": 2}, "2023-06-01")
        assert "2 row(s) removed" in line
        assert "income_statement" not in line.split("(")[1]  # not in details (0 drops)
