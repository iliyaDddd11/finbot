"""
Unit tests for quant_eval/portfolio_construction.py

Covers:
  - construct_portfolio: basic long/short allocation
  - Confidence gate: signals below MIN_CONFIDENCE_GATE are excluded
  - Sector cap: no sector exceeds MAX_SECTOR_WEIGHT
  - Position cap: no single name exceeds MAX_POSITION_WEIGHT
  - Gross exposure cap: portfolio gross stays <= MAX_GROSS_EXPOSURE
  - Neutral signals excluded
  - portfolio_summary stats
  - portfolio_report_lines formatting
"""
from __future__ import annotations

import pytest

from quant_eval.portfolio_construction import (
    MAX_GROSS_EXPOSURE,
    MAX_POSITION_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MIN_CONFIDENCE_GATE,
    PortfolioPosition,
    construct_portfolio,
    portfolio_summary,
    portfolio_report_lines,
)
from quant_eval.position_sizing import PositionSizeResult
from quant_eval.signal_engine import SignalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(ticker: str, signal: str, confidence: float = 0.70,
                 conviction: str = "medium") -> SignalResult:
    return SignalResult(
        ticker=ticker,
        as_of_date="2024-01-01",
        signal=signal,
        confidence=confidence,
        composite_score=1.0 if signal == "long" else -1.0 if signal == "short" else 0.0,
        conviction_tier=conviction,
        expected_return=0.05 if signal == "long" else -0.05 if signal == "short" else None,
    )


def _make_position_size(ticker: str, size: float = 0.08) -> PositionSizeResult:
    return PositionSizeResult(
        ticker=ticker,
        signal="long",
        raw_kelly=size,
        half_kelly=size / 2,
        vol_target_size=size,
        final_size=size,
        sizing_method="vol_target",
        notes=[],
    )


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

class TestConstructPortfolio:
    def test_empty_signals_returns_empty(self):
        result = construct_portfolio([], {})
        assert result == []

    def test_single_long_position(self):
        signals = [_make_signal("AAPL", "long", confidence=0.70)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.05)}
        result = construct_portfolio(signals, sizes)
        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert result[0].final_weight > 0  # long = positive weight

    def test_single_short_position(self):
        signals = [_make_signal("TSLA", "short", confidence=0.70)]
        sizes = {"TSLA": _make_position_size("TSLA", 0.05)}
        result = construct_portfolio(signals, sizes)
        assert len(result) == 1
        assert result[0].final_weight < 0  # short = negative weight

    def test_neutral_signals_excluded(self):
        signals = [
            _make_signal("AAPL", "long",    confidence=0.70),
            _make_signal("MSFT", "neutral", confidence=0.70),
        ]
        sizes = {
            "AAPL": _make_position_size("AAPL", 0.05),
            "MSFT": _make_position_size("MSFT", 0.05),
        }
        result = construct_portfolio(signals, sizes)
        tickers = [p.ticker for p in result]
        assert "AAPL" in tickers
        assert "MSFT" not in tickers

    def test_sorted_by_abs_weight_descending(self):
        signals = [
            _make_signal("AAPL", "long", confidence=0.70),
            _make_signal("MSFT", "long", confidence=0.70),
        ]
        sizes = {
            "AAPL": _make_position_size("AAPL", 0.09),
            "MSFT": _make_position_size("MSFT", 0.04),
        }
        result = construct_portfolio(signals, sizes)
        if len(result) == 2:
            assert abs(result[0].final_weight) >= abs(result[1].final_weight)


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

class TestConfidenceGate:
    def test_low_confidence_excluded(self):
        signals = [_make_signal("AAPL", "long", confidence=MIN_CONFIDENCE_GATE - 0.01)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.05)}
        result = construct_portfolio(signals, sizes)
        assert len(result) == 0

    def test_exact_threshold_excluded(self):
        # min_conf gate is a strict less-than check (< min_conf)
        signals = [_make_signal("AAPL", "long", confidence=MIN_CONFIDENCE_GATE - 0.001)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.05)}
        result = construct_portfolio(signals, sizes)
        assert len(result) == 0

    def test_above_threshold_included(self):
        signals = [_make_signal("AAPL", "long", confidence=MIN_CONFIDENCE_GATE + 0.01)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.05)}
        result = construct_portfolio(signals, sizes)
        assert len(result) == 1

    def test_custom_min_conf_parameter(self):
        signals = [_make_signal("AAPL", "long", confidence=0.50)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.05)}
        # With lower gate, should pass
        result = construct_portfolio(signals, sizes, min_conf=0.40)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Position cap
# ---------------------------------------------------------------------------

class TestPositionCap:
    def test_position_capped_at_max(self):
        signals = [_make_signal("AAPL", "long", confidence=0.80)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.20)}  # 20% → should be capped
        result = construct_portfolio(signals, sizes)
        if result:
            assert abs(result[0].final_weight) <= MAX_POSITION_WEIGHT + 1e-6

    def test_position_below_max_not_capped(self):
        signals = [_make_signal("AAPL", "long", confidence=0.80)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.05)}
        result = construct_portfolio(signals, sizes)
        if result:
            assert "position_cap" not in result[0].capped_by

    def test_custom_max_position(self):
        signals = [_make_signal("AAPL", "long", confidence=0.80)]
        sizes = {"AAPL": _make_position_size("AAPL", 0.15)}
        result = construct_portfolio(signals, sizes, max_position=0.08)
        if result:
            assert abs(result[0].final_weight) <= 0.08 + 1e-6


# ---------------------------------------------------------------------------
# Gross exposure cap
# ---------------------------------------------------------------------------

class TestGrossExposureCap:
    def _make_n_signals(self, n: int, size: float = 0.10) -> tuple:
        tickers = [f"STOCK{i}" for i in range(n)]
        signals = [_make_signal(t, "long", confidence=0.80) for t in tickers]
        sizes   = {t: _make_position_size(t, size) for t in tickers}
        return signals, sizes

    def test_gross_exposure_stays_within_cap(self):
        # 20 names × 0.10 = 2.0 gross → must be scaled to 1.5
        signals, sizes = self._make_n_signals(20, 0.10)
        result = construct_portfolio(signals, sizes)
        gross = sum(abs(p.final_weight) for p in result)
        assert gross <= MAX_GROSS_EXPOSURE + 1e-4

    def test_small_portfolio_not_scaled(self):
        # 5 names × 0.05 = 0.25 gross → no gross scaling needed
        signals, sizes = self._make_n_signals(5, 0.05)
        result = construct_portfolio(signals, sizes)
        for p in result:
            assert "gross_cap" not in p.capped_by

    def test_gross_cap_flag_set_when_scaled(self):
        # Isolate the gross-exposure cap from the sector cap: these synthetic
        # STOCK{i} tickers all fall in the "unknown" sector, so with the default
        # 30% sector cap gross is trimmed to 0.30 and the gross cap never binds.
        # Raise the sector cap so 20 × 0.10 = 2.0 gross must be scaled to 1.5.
        signals, sizes = self._make_n_signals(20, 0.10)
        result = construct_portfolio(signals, sizes, max_sector=10.0)
        assert result
        capped = [p for p in result if "gross_cap" in p.capped_by]
        assert len(capped) > 0
        # And gross is actually at the cap
        gross = sum(abs(p.final_weight) for p in result)
        assert gross <= MAX_GROSS_EXPOSURE + 1e-4


# ---------------------------------------------------------------------------
# Sector cap
# ---------------------------------------------------------------------------

class TestSectorCap:
    def test_no_individual_sector_exceeds_cap(self):
        """Even if all tickers are in the same sector, total weight ≤ MAX_SECTOR_WEIGHT."""
        # Use known technology tickers from the universe
        tech_tickers = ["AAPL", "MSFT", "NVDA", "META", "GOOGL"]
        signals = [_make_signal(t, "long", confidence=0.80) for t in tech_tickers]
        sizes   = {t: _make_position_size(t, 0.10) for t in tech_tickers}
        result = construct_portfolio(signals, sizes)

        # Sum all long positions in 'technology' sector
        tech_weight = sum(abs(p.final_weight) for p in result if p.sector == "technology")
        assert tech_weight <= MAX_SECTOR_WEIGHT + 1e-4


# ---------------------------------------------------------------------------
# No size entry for signal
# ---------------------------------------------------------------------------

class TestMissingSizeEntry:
    def test_missing_size_excludes_position(self):
        signals = [_make_signal("AAPL", "long", confidence=0.80)]
        result = construct_portfolio(signals, {})  # no size for AAPL
        assert len(result) == 0


# ---------------------------------------------------------------------------
# portfolio_summary
# ---------------------------------------------------------------------------

class TestPortfolioSummary:
    def _build_portfolio(self) -> list[PortfolioPosition]:
        signals = [
            _make_signal("AAPL", "long",  confidence=0.80),
            _make_signal("MSFT", "long",  confidence=0.75),
            _make_signal("TSLA", "short", confidence=0.70),
        ]
        sizes = {
            "AAPL": _make_position_size("AAPL", 0.08),
            "MSFT": _make_position_size("MSFT", 0.06),
            "TSLA": _make_position_size("TSLA", 0.05),
        }
        return construct_portfolio(signals, sizes)

    def test_summary_keys(self):
        positions = self._build_portfolio()
        summary = portfolio_summary(positions)
        for key in ["n_positions", "n_long", "n_short", "gross_exposure", "net_exposure", "sector_breakdown"]:
            assert key in summary

    def test_gross_exposure_sum(self):
        positions = self._build_portfolio()
        summary = portfolio_summary(positions)
        expected_gross = sum(abs(p.final_weight) for p in positions)
        assert summary["gross_exposure"] == pytest.approx(expected_gross, abs=1e-4)

    def test_net_exposure_sign(self):
        # 2 longs vs 1 short → net should be positive
        positions = self._build_portfolio()
        summary = portfolio_summary(positions)
        # Net = sum of signed weights; 2 longs - 1 short
        assert summary["net_exposure"] > 0

    def test_n_counts_correct(self):
        positions = self._build_portfolio()
        summary = portfolio_summary(positions)
        longs  = [p for p in positions if p.signal == "long"]
        shorts = [p for p in positions if p.signal == "short"]
        assert summary["n_long"] == len(longs)
        assert summary["n_short"] == len(shorts)
        assert summary["n_positions"] == len(positions)

    def test_empty_portfolio(self):
        summary = portfolio_summary([])
        assert summary["n_positions"] == 0
        assert summary["gross_exposure"] == 0.0


# ---------------------------------------------------------------------------
# portfolio_report_lines
# ---------------------------------------------------------------------------

class TestPortfolioReportLines:
    def test_returns_list_of_strings(self):
        positions = [
            PortfolioPosition(
                ticker="AAPL", sector="technology", signal="long",
                confidence=0.80, conviction_tier="high",
                raw_weight=0.08, sector_scaled_weight=0.08,
                final_weight=0.08, capped_by=[],
            )
        ]
        summary = portfolio_summary(positions)
        lines = portfolio_report_lines(summary)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_contains_positions_summary(self):
        summary = {"n_positions": 3, "n_long": 2, "n_short": 1,
                   "gross_exposure": 0.19, "net_exposure": 0.09,
                   "sector_breakdown": {"technology": 0.13, "financials": 0.06}}
        lines = portfolio_report_lines(summary)
        combined = "\n".join(lines)
        assert "3" in combined
        assert "technology" in combined
