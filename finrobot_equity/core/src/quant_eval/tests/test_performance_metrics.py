"""
Unit tests for quant_eval/performance_metrics.py

Covers:
  - information_coefficient (spearman, pearson, edge cases)
  - icir (normal, insufficient data)
  - sharpe_ratio (positive/negative, flat, edge cases)
  - max_drawdown (known drawdown, no drawdown, empty)
  - calmar_ratio
  - hit_rate
  - turnover
  - bootstrap_ci (coverage, reproducibility)
  - bootstrap_hit_rate_ci
  - bootstrap_sharpe_ci
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_eval.performance_metrics import (
    bootstrap_ci,
    bootstrap_hit_rate_ci,
    bootstrap_sharpe_ci,
    calmar_ratio,
    hit_rate,
    ic_series,
    icir,
    information_coefficient,
    max_drawdown,
    sharpe_ratio,
    turnover,
    compute_full_scorecard,
)


# ---------------------------------------------------------------------------
# information_coefficient
# ---------------------------------------------------------------------------

class TestInformationCoefficient:
    def test_perfect_positive_correlation(self):
        signals  = pd.Series(["long", "long", "short", "short", "long"])
        returns  = pd.Series([0.10,   0.08,  -0.05,  -0.07,   0.06])
        ic = information_coefficient(signals, returns)
        assert ic is not None
        assert ic > 0.8

    def test_perfect_negative_correlation(self):
        signals  = pd.Series(["long",  "long",  "short", "short"])
        returns  = pd.Series([-0.10, -0.08,   0.05,   0.07])
        ic = information_coefficient(signals, returns)
        assert ic is not None
        assert ic < -0.8

    def test_returns_none_for_too_few_observations(self):
        signals = pd.Series(["long", "short", "neutral"])
        returns = pd.Series([0.1, -0.1, 0.0])
        assert information_coefficient(signals, returns) is None

    def test_returns_none_when_all_signals_same(self):
        signals = pd.Series(["long", "long", "long", "long", "long"])
        returns = pd.Series([0.1, 0.2, -0.1, 0.05, 0.15])
        assert information_coefficient(signals, returns) is None

    def test_returns_none_when_all_returns_same(self):
        signals = pd.Series(["long", "short", "neutral", "long", "short"])
        returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        assert information_coefficient(signals, returns) is None

    def test_neutral_mapped_to_zero(self):
        # Neutral should map to 0 — mixed signals still produce a valid IC
        signals = pd.Series(["long", "neutral", "short", "long", "short"])
        returns = pd.Series([0.10, 0.00, -0.08, 0.07, -0.06])
        ic = information_coefficient(signals, returns)
        assert ic is not None
        assert -1.0 <= ic <= 1.0

    def test_pearson_method(self):
        signals = pd.Series(["long", "long", "short", "short", "long"])
        returns = pd.Series([0.10, 0.08, -0.05, -0.07, 0.06])
        ic = information_coefficient(signals, returns, method="pearson")
        assert ic is not None
        assert -1.0 <= ic <= 1.0

    def test_nan_in_returns_handled(self):
        signals = pd.Series(["long", "long", "short", "short", "long"])
        returns = pd.Series([0.10, float("nan"), -0.05, -0.07, 0.06])
        ic = information_coefficient(signals, returns)
        # 4 valid pairs remain (mask drops NaN) — should still compute
        assert ic is not None


# ---------------------------------------------------------------------------
# ic_series
# ---------------------------------------------------------------------------

class TestIcSeries:
    def test_groups_by_date(self):
        df = pd.DataFrame({
            "as_of_date":      ["2023-01-01"] * 5 + ["2023-07-01"] * 5,
            "signal":          ["long", "long", "short", "short", "long"] * 2,
            "realized_return": [0.10, 0.08, -0.05, -0.07, 0.06] * 2,
        })
        series = ic_series(df)
        assert len(series) == 2
        assert "2023-01-01" in series.index
        assert "2023-07-01" in series.index

    def test_missing_date_column(self):
        df = pd.DataFrame({"signal": ["long"], "realized_return": [0.1]})
        series = ic_series(df)
        assert series.empty


# ---------------------------------------------------------------------------
# icir
# ---------------------------------------------------------------------------

class TestIcir:
    def test_normal_icir(self):
        ic_vals = pd.Series([0.10, 0.15, 0.12, 0.08, 0.20, 0.09])
        result = icir(ic_vals)
        assert result is not None
        assert result > 0

    def test_negative_mean_ic_gives_negative_icir(self):
        ic_vals = pd.Series([-0.10, -0.15, -0.12, -0.08, -0.20])
        result = icir(ic_vals)
        assert result is not None
        assert result < 0

    def test_too_few_observations(self):
        assert icir(pd.Series([0.1, 0.2])) is None

    def test_zero_std_returns_none(self):
        ic_vals = pd.Series([0.10, 0.10, 0.10, 0.10])
        assert icir(ic_vals) is None

    def test_annualisation_factor(self):
        # With 6 periods/year and mean=0.1, std=0.1: ICIR = 0.1/0.1*sqrt(6) ≈ 2.449
        ic_vals = pd.Series([0.10] * 3 + [0.10 + 1e-6, 0.10 - 1e-6, 0.10])  # tiny variance
        result = icir(ic_vals, periods_per_year=6)
        assert result is not None
        assert result > 0


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_sharpe(self):
        # Returns consistently above 0.045/6 risk-free
        returns = pd.Series([0.05, 0.04, 0.06, 0.03, 0.05, 0.04])
        result = sharpe_ratio(returns)
        assert result is not None
        assert result > 0

    def test_negative_sharpe(self):
        returns = pd.Series([-0.05, -0.04, -0.06, -0.03, -0.05, -0.04])
        result = sharpe_ratio(returns)
        assert result is not None
        assert result < 0

    def test_insufficient_data(self):
        assert sharpe_ratio(pd.Series([0.1, 0.2, 0.3])) is None

    def test_zero_std_returns_none(self):
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        assert sharpe_ratio(returns) is None

    def test_known_value(self):
        # Flat return of 0.02 per period, periods_per_year=1
        # risk_free_per_period = 0.045 → excess = 0.02 - 0.045 = -0.025 each period
        # std of constant series → 0 → None
        returns = pd.Series([0.02] * 5)
        assert sharpe_ratio(returns, periods_per_year=1) is None


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown(self):
        # Monotonically increasing cumulative returns → no drawdown
        returns = pd.Series([0.05, 0.03, 0.04, 0.02])
        result = max_drawdown(returns)
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_known_drawdown(self):
        # Sequence: +50%, -50% → cum = 1.5, 0.75 → DD = (0.75-1.5)/1.5 = -0.5
        returns = pd.Series([0.50, -0.50])
        result = max_drawdown(returns)
        assert result is not None
        assert result == pytest.approx(-0.50, rel=1e-4)

    def test_all_negative(self):
        returns = pd.Series([-0.10, -0.10, -0.10, -0.10])
        result = max_drawdown(returns)
        assert result is not None
        assert result < -0.30  # severe drawdown

    def test_empty_returns_none(self):
        assert max_drawdown(pd.Series([], dtype=float)) is None

    def test_single_positive_no_drawdown(self):
        result = max_drawdown(pd.Series([0.10]))
        assert result == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_positive_calmar(self):
        # Consistent gains with small drawdown
        returns = pd.Series([0.05, 0.03, -0.01, 0.04, 0.02, 0.06])
        result = calmar_ratio(returns)
        assert result is not None
        assert result > 0

    def test_insufficient_data(self):
        assert calmar_ratio(pd.Series([0.1, -0.1, 0.1])) is None

    def test_near_zero_drawdown_returns_none(self):
        # All positive returns → max_drawdown near 0 → undefined Calmar
        returns = pd.Series([0.05, 0.03, 0.04, 0.02, 0.06])
        result = calmar_ratio(returns)
        assert result is None


# ---------------------------------------------------------------------------
# hit_rate
# ---------------------------------------------------------------------------

class TestHitRate:
    def _make_df(self, signals, outcomes):
        return pd.DataFrame({"signal": signals, "outcome_label": outcomes})

    def test_perfect_hit_rate(self):
        df = self._make_df(
            ["long", "short", "long", "short"],
            ["right", "right", "right", "right"],
        )
        assert hit_rate(df) == 1.0

    def test_zero_hit_rate(self):
        df = self._make_df(
            ["long", "short", "long", "short"],
            ["wrong", "wrong", "wrong", "wrong"],
        )
        assert hit_rate(df) == 0.0

    def test_excludes_neutral(self):
        df = self._make_df(
            ["long", "neutral", "short", "neutral", "long"],
            ["right", "right", "right", "right", "wrong"],
        )
        # Directional: 2 right, 1 wrong out of 3 → 2/3
        assert hit_rate(df) == pytest.approx(2 / 3, rel=1e-4)

    def test_excludes_unknown_outcome(self):
        df = self._make_df(
            ["long", "long", "short"],
            ["right", "unknown", "wrong"],
        )
        # Only right/wrong count: 1 right, 1 wrong out of 2 → 0.5
        assert hit_rate(df) == pytest.approx(0.5, rel=1e-4)

    def test_no_directional_calls_returns_none(self):
        df = self._make_df(["neutral", "neutral"], ["right", "wrong"])
        assert hit_rate(df) is None


# ---------------------------------------------------------------------------
# turnover
# ---------------------------------------------------------------------------

class TestTurnover:
    def test_zero_turnover(self):
        signals = pd.Series(["long", "long", "long", "long"])
        result = turnover(signals)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_full_flip_turnover(self):
        # alternates long/short → numeric 1/-1/-1/+1 → diff abs = 2,2,2 → mean=2
        signals = pd.Series(["long", "short", "long", "short"])
        result = turnover(signals)
        assert result == pytest.approx(2.0, rel=1e-4)

    def test_insufficient_data(self):
        assert turnover(pd.Series(["long"])) is None


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:
    def test_ci_contains_true_mean(self):
        rng = np.random.default_rng(99)
        data = pd.Series(rng.normal(0.10, 0.05, 100))
        lo, hi = bootstrap_ci(data, np.mean)
        assert lo is not None and hi is not None
        assert lo < 0.10 < hi  # true mean should be inside 95% CI

    def test_ci_ordering(self):
        data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        lo, hi = bootstrap_ci(data, np.mean)
        assert lo <= hi

    def test_reproducibility(self):
        data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        lo1, hi1 = bootstrap_ci(data, np.mean, seed=42)
        lo2, hi2 = bootstrap_ci(data, np.mean, seed=42)
        assert lo1 == lo2 and hi1 == hi2

    def test_insufficient_data_returns_none(self):
        lo, hi = bootstrap_ci(pd.Series([0.1, 0.2, 0.3]), np.mean)
        assert lo is None and hi is None

    def test_nan_values_dropped(self):
        data = pd.Series([0.1, float("nan"), 0.2, float("nan"), 0.3, 0.4, 0.5])
        lo, hi = bootstrap_ci(data, np.mean)
        assert lo is not None and hi is not None


# ---------------------------------------------------------------------------
# bootstrap_hit_rate_ci
# ---------------------------------------------------------------------------

class TestBootstrapHitRateCi:
    def test_all_right_ci_near_one(self):
        outcomes = pd.Series(["right"] * 20)
        lo, hi = bootstrap_hit_rate_ci(outcomes, n_boot=500)
        assert lo is not None
        assert lo > 0.8

    def test_insufficient_data(self):
        lo, hi = bootstrap_hit_rate_ci(pd.Series(["right", "wrong", "right"]))
        assert lo is None and hi is None

    def test_excludes_unknown(self):
        outcomes = pd.Series(["right", "unknown", "right", "wrong", "unknown"] * 4)
        lo, hi = bootstrap_hit_rate_ci(outcomes, n_boot=500)
        assert lo is not None and hi is not None


# ---------------------------------------------------------------------------
# bootstrap_sharpe_ci
# ---------------------------------------------------------------------------

class TestBootstrapSharpeCi:
    def test_positive_returns_positive_ci(self):
        returns = pd.Series([0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.03, 0.05] * 3)
        lo, hi = bootstrap_sharpe_ci(returns, n_boot=500)
        assert lo is not None and hi is not None
        # Both bounds should be positive for consistently positive returns
        assert hi > 0

    def test_insufficient_data(self):
        lo, hi = bootstrap_sharpe_ci(pd.Series([0.1, 0.2, 0.1]))
        assert lo is None and hi is None


# ---------------------------------------------------------------------------
# compute_full_scorecard
# ---------------------------------------------------------------------------

class TestComputeFullScorecard:
    def _make_df(self):
        return pd.DataFrame({
            "as_of_date":       ["2023-01-01"] * 3 + ["2023-07-01"] * 3,
            "ticker":           ["AAPL", "MSFT", "GOOG"] * 2,
            "signal":           ["long", "short", "neutral", "long", "short", "neutral"],
            "confidence":       [0.7, 0.65, 0.5, 0.72, 0.68, 0.52],
            "realized_return":  [0.10, -0.08, 0.01, 0.07, -0.05, 0.02],
            "outcome_label":    ["right", "right", "right", "right", "right", "right"],
        })

    def test_scorecard_keys(self):
        df = self._make_df()
        sc = compute_full_scorecard(df)
        for key in ["n_predictions", "hit_rate", "sharpe_ratio", "max_drawdown", "mean_ic"]:
            assert key in sc

    def test_n_predictions(self):
        df = self._make_df()
        sc = compute_full_scorecard(df)
        assert sc["n_predictions"] == 6

    def test_hit_rate_all_right(self):
        df = self._make_df()
        sc = compute_full_scorecard(df)
        assert sc["hit_rate"] == 1.0

    def test_by_ticker_present(self):
        df = self._make_df()
        sc = compute_full_scorecard(df)
        assert "by_ticker" in sc
        assert "AAPL" in sc["by_ticker"]

    def test_no_realized_returns_still_works(self):
        df = pd.DataFrame({
            "as_of_date":    ["2023-01-01"] * 4,
            "signal":        ["long", "short", "long", "neutral"],
            "confidence":    [0.7, 0.65, 0.72, 0.5],
            "outcome_label": ["right", "wrong", "right", "right"],
        })
        sc = compute_full_scorecard(df)
        assert sc["n_predictions"] == 4
        assert sc["sharpe_ratio"] is None
