"""
Regression tests for the Phase 1 correctness fixes (REVIEW.md #2, #9, #10,
#11, #13, #16, #17, #28; see IMPLEMENTATION_MAP.md).
"""
import numpy as np
import pandas as pd
import pytest

from quant_eval import realized_returns as rr
from quant_eval.decision_audit import classify_outcome, summarize_patterns
from quant_eval.pit_filter import filter_to_pit
from quant_eval.price_sources import dukascopy_cli
from quant_eval.price_sources.dukascopy_cli import load_dukascopy_csv, load_price_frame
from quant_eval.universe import sector_of


# ---------------------------------------------------------------------------
# #16 — NaN realized return must be "unknown", not "wrong"
# ---------------------------------------------------------------------------

class TestClassifyOutcomeNaN:
    def test_nan_is_unknown(self):
        assert classify_outcome("long", float("nan")) == "unknown"
        assert classify_outcome("short", np.nan) == "unknown"

    def test_none_is_unknown(self):
        assert classify_outcome("long", None) == "unknown"

    def test_real_values_still_classify(self):
        assert classify_outcome("long", 0.05) == "right"
        assert classify_outcome("long", -0.05) == "wrong"
        assert classify_outcome("short", -0.05) == "right"


# ---------------------------------------------------------------------------
# #13 — min_count actually gates pattern qualification
# ---------------------------------------------------------------------------

class TestPatternMinCount:
    def _df(self):
        # Pattern P appears 4x (all right); pattern Q appears 2x (all right).
        rows = []
        for _ in range(4):
            rows.append({"pattern_signature": "P", "is_right": 1, "realized_return": 0.05, "confidence": 0.7})
        for _ in range(2):
            rows.append({"pattern_signature": "Q", "is_right": 1, "realized_return": 0.05, "confidence": 0.7})
        return pd.DataFrame(rows)

    def test_higher_min_count_drops_small_patterns(self):
        sigs2 = {p["pattern_signature"] for p in summarize_patterns(self._df(), min_count=2)["right_patterns"]}
        sigs3 = {p["pattern_signature"] for p in summarize_patterns(self._df(), min_count=3)["right_patterns"]}
        assert "Q" in sigs2 and "P" in sigs2
        assert "Q" not in sigs3 and "P" in sigs3   # Q (n=2) drops at min_count=3


# ---------------------------------------------------------------------------
# #10 / #11 — PIT: prefer real filing date; don't empty on unparseable dates
# ---------------------------------------------------------------------------

class TestPitFilingDate:
    def test_filing_date_used_without_lag(self):
        # Period end 2023-12-31 (would be "available" 2024-03-01 under the 60d
        # estimate) but the actual filing was 2024-02-15. As of 2024-02-20 the
        # row IS public via the real filing date.
        df = pd.DataFrame({"date": ["2023-12-31"], "fillingDate": ["2024-02-15"], "rev": [1]})
        kept = filter_to_pit(df, "2024-02-20", period="annual")
        assert len(kept) == 1          # filing date (no lag) wins over the estimate

    def test_filing_date_excludes_future_filing(self):
        df = pd.DataFrame({"date": ["2023-12-31"], "fillingDate": ["2024-03-10"], "rev": [1]})
        kept = filter_to_pit(df, "2024-02-20", period="annual")
        assert len(kept) == 0          # not yet filed as of 2024-02-20

    def test_all_unparseable_dates_not_silently_emptied(self):
        df = pd.DataFrame({"date": ["FY", "FY"], "rev": [1, 2]})
        out = filter_to_pit(df, "2024-01-01", period="annual")
        assert len(out) == 2           # unchanged, not an empty frame


# ---------------------------------------------------------------------------
# #17 — Dukascopy timestamp parsing handles epoch-ms AND ISO-8601
# ---------------------------------------------------------------------------

class TestDukascopyTimestamp:
    def test_epoch_ms(self, tmp_path):
        p = tmp_path / "ms.csv"
        p.write_text("timestamp,close\n1706659200000,100.0\n1706745600000,101.0\n")
        df = load_dukascopy_csv(str(p))
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert df["timestamp"].iloc[0].year == 2024

    def test_iso8601(self, tmp_path):
        p = tmp_path / "iso.csv"
        p.write_text("timestamp,close\n2024-01-31T00:00:00Z,100.0\n2024-02-01T00:00:00Z,101.0\n")
        df = load_dukascopy_csv(str(p))
        assert str(df["timestamp"].dt.tz) == "UTC"
        assert df["timestamp"].iloc[0].year == 2024


# ---------------------------------------------------------------------------
# #2 — adjusted-by-default routing (no live download; patch the sources)
# ---------------------------------------------------------------------------

class TestAdjustedRouting:
    def test_prefer_adjusted_routes_mapped_ticker_to_yahoo(self, monkeypatch):
        calls = {"yahoo": 0, "duka": 0}

        def fake_yahoo(symbol, start_date, end_date, download_dir):
            calls["yahoo"] += 1
            return pd.DataFrame({"timestamp": [], "close": [], "price_adjusted": []})

        def fake_duka(*a, **k):
            calls["duka"] += 1
            return "/tmp/x.csv"

        monkeypatch.setattr(dukascopy_cli, "load_price_frame_yahoo", fake_yahoo)
        monkeypatch.setattr(dukascopy_cli, "run_dukascopy_download", fake_duka)

        # AAPL is in EXPLICIT_DUKASCOPY_MAP; with prefer_adjusted (default) it
        # must still go to the adjusted Yahoo feed.
        load_price_frame("AAPL", "2024-01-01", "2024-03-01", "/tmp", prefer_adjusted=True)
        assert calls == {"yahoo": 1, "duka": 0}

    def test_prefer_adjusted_false_uses_dukascopy_for_mapped(self, monkeypatch):
        calls = {"yahoo": 0, "duka": 0}
        monkeypatch.setattr(dukascopy_cli, "load_price_frame_yahoo",
                            lambda **k: calls.__setitem__("yahoo", calls["yahoo"] + 1) or pd.DataFrame())

        def fake_duka(*a, **k):
            calls["duka"] += 1
            return "/tmp/x.csv"
        monkeypatch.setattr(dukascopy_cli, "run_dukascopy_download", fake_duka)
        monkeypatch.setattr(dukascopy_cli, "load_dukascopy_csv",
                            lambda p: pd.DataFrame({"timestamp": pd.to_datetime([1706659200000], unit="ms", utc=True), "close": [1.0]}))

        df = load_price_frame("AAPL", "2024-01-01", "2024-03-01", "/tmp", prefer_adjusted=False)
        assert calls["duka"] == 1 and calls["yahoo"] == 0
        assert bool(df["price_adjusted"].iloc[0]) is False


# ---------------------------------------------------------------------------
# #9 — a transient benchmark error must not poison the whole date's alphas
# ---------------------------------------------------------------------------

class TestBenchmarkCache:
    def test_transient_error_not_cached(self, monkeypatch):
        rr._SPY_CACHE.clear()

        def boom(**k):
            raise RuntimeError("transient download failure")

        monkeypatch.setattr(rr, "load_price_frame", boom)
        out = rr._get_benchmark_return("2025-01-31", 60, "/tmp/spy")
        assert out is None
        assert ("2025-01-31", 60) not in rr._SPY_CACHE   # not poisoned → retryable


# ---------------------------------------------------------------------------
# #28 — universe hygiene: TSLA now has a sector
# ---------------------------------------------------------------------------

class TestUniverseHygiene:
    def test_tsla_has_sector(self):
        assert sector_of("TSLA") == "consumer_discretionary"
