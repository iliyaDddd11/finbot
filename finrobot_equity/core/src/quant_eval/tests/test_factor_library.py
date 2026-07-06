"""
Unit tests for quant_eval.factor_library

Tests cover:
  - All 10 factor computations with realistic synthetic data
  - Edge cases: missing data, near-zero denominators, negative EPS, net-cash leverage
  - Score range: all scores must be in [-2, +2]
  - available=False paths
"""
from __future__ import annotations

import math
import sys
import os

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from quant_eval.factor_library import (
    FactorResult,
    _tanh_score,
    _safe_float,
    _ols_slope,
    factor_earnings_quality,
    factor_margin_quality,
    factor_leverage,
    factor_roic,
    factor_revenue_acceleration,
    factor_eps_growth,
    factor_valuation_vs_history,
    factor_price_momentum_12m1m,
    factor_price_momentum_6m,
    factor_fcf_yield,
)


# ---------------------------------------------------------------------------
# Helper data builders
# ---------------------------------------------------------------------------

def _income(n: int = 5, revenue_m: float = 10_000, ebitda_margin: float = 0.25,
            eps: float = 5.0, eps_growth: float = 0.12) -> pd.DataFrame:
    rows = []
    rev = revenue_m
    e = eps
    for i in range(n):
        rows.append({
            "date": f"202{i}-12-31",
            "revenue": rev * 1e6,
            "ebitda": rev * ebitda_margin * 1e6,
            "netIncome": rev * 0.15 * 1e6,
            "epsdiluted": e,
        })
        rev *= (1 + 0.08)
        e   *= (1 + eps_growth)
    return pd.DataFrame(rows)


def _cash_flow(cfo_m: float = 2_000, ni_m: float = 1_500) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": "2024-12-31",
        "operatingCashFlow": cfo_m * 1e6,
        "freeCashFlow":      (cfo_m - 500) * 1e6,
    }])


def _balance(net_debt_b: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": "2024-12-31",
        "netDebt": net_debt_b * 1e9,
    }])


def _key_metrics(roic: float = 0.18, ev_ebitda: float = 12.0,
                 net_debt_ebitda: float = 1.5, fcf_yield: float = 0.05,
                 ev: float = 200e9, n: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "date": f"202{i}-12-31",
            "returnOnInvestedCapital": roic * (1 + i * 0.01),
            "enterpriseValueOverEBITDA": ev_ebitda + i * 0.5,
            "netDebtToEBITDA": net_debt_ebitda,
            "freeCashFlowYield": fcf_yield,
            "enterpriseValue": ev,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _tanh_score helper
# ---------------------------------------------------------------------------

class TestTanhScore:
    def test_center_gives_zero(self):
        assert _tanh_score(1.0, center=1.0, scale=0.5) == pytest.approx(0.0, abs=1e-9)

    def test_range_bounded(self):
        for x in [-1000, -10, 0, 10, 1000]:
            score = _tanh_score(x, center=0.0, scale=1.0)
            assert -2.0 <= score <= 2.0

    def test_invert(self):
        s_pos = _tanh_score(2.0, center=0.0, scale=1.0, invert=False)
        s_inv = _tanh_score(2.0, center=0.0, scale=1.0, invert=True)
        assert s_pos > 0
        assert s_inv < 0

    def test_higher_x_gives_higher_score_without_invert(self):
        s1 = _tanh_score(0.5, center=0.0, scale=0.2)
        s2 = _tanh_score(1.5, center=0.0, scale=0.2)
        assert s2 > s1


class TestSafeFloat:
    def test_percentage_string(self):
        assert _safe_float("5.0%") == pytest.approx(0.05)

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_plain_float(self):
        assert _safe_float(0.18) == pytest.approx(0.18)


class TestOlsSlope:
    def test_rising_series(self):
        slope = _ols_slope([1, 2, 3, 4, 5])
        assert slope == pytest.approx(1.0, abs=0.01)

    def test_flat_series(self):
        slope = _ols_slope([3, 3, 3, 3])
        assert slope == pytest.approx(0.0, abs=1e-9)

    def test_too_short(self):
        assert _ols_slope([5]) is None

    def test_empty(self):
        assert _ols_slope([]) is None


# ---------------------------------------------------------------------------
# Factor 1: Earnings Quality
# ---------------------------------------------------------------------------

class TestEarningsQuality:
    def test_high_quality(self):
        cf = pd.DataFrame([{"date": "2024-12-31", "operatingCashFlow": 2e9}])
        inc = pd.DataFrame([{"date": "2024-12-31", "netIncome": 1e9}])
        r = factor_earnings_quality(cf, inc)
        assert r.available
        assert r.raw_value == pytest.approx(2.0)
        assert r.score > 0   # CFO > NI → bullish

    def test_low_quality(self):
        cf = pd.DataFrame([{"date": "2024-12-31", "operatingCashFlow": 0.5e9}])
        inc = pd.DataFrame([{"date": "2024-12-31", "netIncome": 2e9}])
        r = factor_earnings_quality(cf, inc)
        assert r.available
        assert r.score < 0   # CFO < NI → bearish

    def test_near_zero_ni(self):
        cf = pd.DataFrame([{"date": "2024-12-31", "operatingCashFlow": 1e6}])
        inc = pd.DataFrame([{"date": "2024-12-31", "netIncome": 100}])
        r = factor_earnings_quality(cf, inc)
        assert r.available
        assert r.raw_value in (0.0, 1.0)   # handled by near-zero branch

    def test_missing_data_returns_unavailable(self):
        r = factor_earnings_quality(pd.DataFrame(), pd.DataFrame())
        assert not r.available
        assert r.score is None

    def test_score_in_range(self):
        cf = _cash_flow(3000, 1000)
        inc = _income()
        r = factor_earnings_quality(cf, inc)
        if r.available:
            assert -2.0 <= r.score <= 2.0


# ---------------------------------------------------------------------------
# Factor 2: Margin Quality
# ---------------------------------------------------------------------------

class TestMarginQuality:
    def test_high_margin_bullish(self):
        inc = _income(ebitda_margin=0.40)
        r = factor_margin_quality(inc)
        assert r.available
        assert r.score > 0   # 40% margin > 20% center → positive

    def test_low_margin_bearish(self):
        inc = _income(ebitda_margin=0.05)
        r = factor_margin_quality(inc)
        assert r.available
        assert r.score < 0

    def test_missing_returns_unavailable(self):
        r = factor_margin_quality(pd.DataFrame())
        assert not r.available

    def test_score_in_range(self):
        r = factor_margin_quality(_income())
        if r.available:
            assert -2.0 <= r.score <= 2.0


# ---------------------------------------------------------------------------
# Factor 3: Leverage
# ---------------------------------------------------------------------------

class TestLeverage:
    def test_net_cash_max_bullish(self):
        km = _key_metrics(net_debt_ebitda=-1.0)
        r = factor_leverage(km, pd.DataFrame(), pd.DataFrame())
        assert r.available
        assert r.raw_value < 0
        assert r.score > 0   # net cash → bullish

    def test_high_leverage_bearish(self):
        km = _key_metrics(net_debt_ebitda=5.0)
        r = factor_leverage(km, pd.DataFrame(), pd.DataFrame())
        assert r.available
        assert r.score < 0

    def test_fallback_from_balance_sheet(self):
        # key_metrics doesn't have netDebtToEBITDA → compute from balance+income
        km = pd.DataFrame([{"date": "2024-12-31", "returnOnInvestedCapital": 0.18}])
        bal = _balance(net_debt_b=10.0)
        inc = _income(ebitda_margin=0.30)
        r = factor_leverage(km, bal, inc)
        assert r.available

    def test_missing_returns_unavailable(self):
        r = factor_leverage(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert not r.available

    def test_score_in_range(self):
        r = factor_leverage(_key_metrics(), pd.DataFrame(), pd.DataFrame())
        if r.available:
            assert -2.0 <= r.score <= 2.0


# ---------------------------------------------------------------------------
# Factor 4: ROIC
# ---------------------------------------------------------------------------

class TestRoic:
    def test_above_wacc_bullish(self):
        km = _key_metrics(roic=0.25)
        r = factor_roic(km)
        assert r.available
        assert r.score > 0   # 25% > 10% WACC → bullish

    def test_below_wacc_bearish(self):
        km = _key_metrics(roic=0.03)
        r = factor_roic(km)
        assert r.available
        assert r.score < 0

    def test_missing_returns_unavailable(self):
        r = factor_roic(pd.DataFrame())
        assert not r.available

    def test_score_in_range(self):
        r = factor_roic(_key_metrics())
        if r.available:
            assert -2.0 <= r.score <= 2.0


# ---------------------------------------------------------------------------
# Factor 5: Revenue Acceleration
# ---------------------------------------------------------------------------

class TestRevenueAcceleration:
    def test_accelerating_growth_bullish(self):
        # Growth: 5% → 8% → 12% → accelerating
        rows = [{"date": f"202{i}-12-31", "revenue": 10e9 * (1.05 ** i)} for i in range(4)]
        rows[-1]["revenue"] = rows[-2]["revenue"] * 1.20   # strong acceleration
        inc = pd.DataFrame(rows)
        r = factor_revenue_acceleration(inc)
        assert r.available
        assert r.score > 0

    def test_decelerating_growth_bearish(self):
        rows = [{"date": f"202{i}-12-31", "revenue": 10e9 * (1.20 ** i)} for i in range(4)]
        rows[-1]["revenue"] = rows[-2]["revenue"] * 1.02   # sharp deceleration
        inc = pd.DataFrame(rows)
        r = factor_revenue_acceleration(inc)
        assert r.available
        assert r.score < 0

    def test_too_few_years_unavailable(self):
        inc = _income(n=2)
        r = factor_revenue_acceleration(inc)
        assert not r.available

    def test_score_in_range(self):
        r = factor_revenue_acceleration(_income())
        if r.available:
            assert -2.0 <= r.score <= 2.0


# ---------------------------------------------------------------------------
# Factor 6: EPS Growth
# ---------------------------------------------------------------------------

class TestEpsGrowth:
    def test_strong_cagr_bullish(self):
        inc = _income(eps=3.0, eps_growth=0.25)
        r = factor_eps_growth(inc)
        assert r.available
        assert r.score > 0   # 25% CAGR > 10% hurdle

    def test_negative_eps_uses_slope(self):
        inc = pd.DataFrame([
            {"date": f"202{i}-12-31", "revenue": 10e9, "ebitda": 2e9,
             "netIncome": -1e9, "epsdiluted": -2.0 + i * 0.5}
            for i in range(5)
        ])
        r = factor_eps_growth(inc)
        assert r.available   # should not crash on negative EPS

    def test_missing_returns_unavailable(self):
        r = factor_eps_growth(pd.DataFrame())
        assert not r.available

    def test_score_in_range(self):
        r = factor_eps_growth(_income())
        if r.available:
            assert -2.0 <= r.score <= 2.0


# ---------------------------------------------------------------------------
# Factor 7: Valuation vs. Own History
# ---------------------------------------------------------------------------

class TestValuationVsHistory:
    def test_discount_to_history_bullish(self):
        km = _key_metrics(ev_ebitda=8.0)   # current cheap vs rising history
        r = factor_valuation_vs_history(km)
        assert r.available
        # current = 8x, history includes 8,8.5,9,9.5,10 → median ~9 → discount
        assert r.score > 0

    def test_premium_to_history_bearish(self):
        # Build a history where current is much more expensive
        rows = [{"date": f"202{i}-12-31", "enterpriseValueOverEBITDA": 8.0 + i}
                for i in range(5)]
        rows[-1]["enterpriseValueOverEBITDA"] = 30.0   # big premium
        km = pd.DataFrame(rows)
        r = factor_valuation_vs_history(km)
        assert r.available
        assert r.score < 0

    def test_single_period_unavailable(self):
        km = pd.DataFrame([{"date": "2024-12-31", "enterpriseValueOverEBITDA": 12.0}])
        r = factor_valuation_vs_history(km)
        assert not r.available

    def test_score_in_range(self):
        r = factor_valuation_vs_history(_key_metrics())
        if r.available:
            assert -2.0 <= r.score <= 2.0


# ---------------------------------------------------------------------------
# Factor 8 & 9: Price Momentum (requires price CSV)
# ---------------------------------------------------------------------------

class TestPriceMomentum:
    def test_no_path_returns_unavailable(self):
        r12 = factor_price_momentum_12m1m("", "2024-01-31")
        r6  = factor_price_momentum_6m("",   "2024-01-31")
        assert not r12.available
        assert not r6.available

    def test_nonexistent_path_returns_unavailable(self):
        r = factor_price_momentum_12m1m("/nonexistent/path.csv", "2024-01-31")
        assert not r.available

    def test_valid_uptrend_csv(self, tmp_path):
        """Create a synthetic price CSV with a strong uptrend."""
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = 100 * (1.001 ** np.arange(n))   # strong uptrend
        df = pd.DataFrame({"Gmt time": dates.strftime("%Y-%m-%d %H:%M:%S"), "Close": prices})
        csv_path = str(tmp_path / "price.csv")
        df.to_csv(csv_path, index=False)

        as_of = dates[-1].strftime("%Y-%m-%d")
        r12 = factor_price_momentum_12m1m(csv_path, as_of)
        r6  = factor_price_momentum_6m(csv_path, as_of)
        assert r12.available
        assert r6.available
        assert r12.score > 0   # uptrend → positive momentum
        assert r6.score  > 0
        assert -2.0 <= r12.score <= 2.0
        assert -2.0 <= r6.score  <= 2.0

    def test_valid_downtrend_csv(self, tmp_path):
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = 100 * (0.999 ** np.arange(n))   # downtrend
        df = pd.DataFrame({"Gmt time": dates.strftime("%Y-%m-%d %H:%M:%S"), "Close": prices})
        csv_path = str(tmp_path / "price.csv")
        df.to_csv(csv_path, index=False)

        as_of = dates[-1].strftime("%Y-%m-%d")
        r12 = factor_price_momentum_12m1m(csv_path, as_of)
        assert r12.available
        assert r12.score < 0   # downtrend → negative momentum


# ---------------------------------------------------------------------------
# Factor 10: FCF Yield
# ---------------------------------------------------------------------------

class TestFcfYield:
    def test_high_yield_bullish(self):
        cf = pd.DataFrame([{"date": "2024-12-31", "freeCashFlow": 10e9, "operatingCashFlow": 12e9}])
        km = pd.DataFrame([{"date": "2024-12-31", "enterpriseValue": 100e9,
                             "freeCashFlowYield": None}])
        r = factor_fcf_yield(cf, km)
        assert r.available
        assert r.score > 0   # 10% yield > 3% center → bullish

    def test_uses_direct_yield_from_key_metrics(self):
        cf = pd.DataFrame([{"date": "2024-12-31", "freeCashFlow": 1e9, "operatingCashFlow": 2e9}])
        km = pd.DataFrame([{"date": "2024-12-31", "freeCashFlowYield": 0.08,
                             "enterpriseValue": 100e9}])
        r = factor_fcf_yield(cf, km)
        assert r.available
        assert r.raw_value == pytest.approx(0.08)

    def test_zero_ev_returns_unavailable(self):
        cf = pd.DataFrame([{"date": "2024-12-31", "freeCashFlow": 1e9}])
        km = pd.DataFrame([{"date": "2024-12-31", "enterpriseValue": 0}])
        r = factor_fcf_yield(cf, km)
        assert not r.available

    def test_missing_returns_unavailable(self):
        r = factor_fcf_yield(pd.DataFrame(), pd.DataFrame())
        assert not r.available

    def test_score_in_range(self):
        cf = _cash_flow()
        km = _key_metrics()
        r = factor_fcf_yield(cf, km)
        if r.available:
            assert -2.0 <= r.score <= 2.0
