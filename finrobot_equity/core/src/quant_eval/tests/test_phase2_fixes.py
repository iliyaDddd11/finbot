"""
Regression tests for the Phase 2 methodology fixes (REVIEW.md #18, #19, #20,
#21, #22, #26 + cost-netting; see IMPLEMENTATION_MAP.md).
"""
import numpy as np
import pandas as pd
import pytest

from quant_eval.factor_library import factor_eps_growth, factor_price_momentum_12m1m
from quant_eval.performance_metrics import (
    compute_full_scorecard,
    ic_series,
    net_period_returns,
    portfolio_period_returns,
    portfolio_turnover,
    sharpe_ratio,
)
from quant_eval.portfolio_construction import construct_portfolio, portfolio_summary
from quant_eval.position_sizing import PositionSizeResult
from quant_eval.signal_engine import SignalResult


# ---------------------------------------------------------------------------
# #18 — exact skip-month 12-1 momentum = P[-22]/P[-252] - 1
# ---------------------------------------------------------------------------

def test_momentum_12m1m_is_skip_month_ratio(tmp_path):
    n = 260
    dates = pd.date_range("2023-01-02", periods=n, freq="D")
    close = [100.0] * n
    close[n - 22] = 120.0          # P[-22] = 120, P[-252] = close[8] = 100
    csv = tmp_path / "px.csv"
    pd.DataFrame({"timestamp": dates.strftime("%Y-%m-%d"), "close": close}).to_csv(csv, index=False)

    r = factor_price_momentum_12m1m(str(csv), dates[-1].strftime("%Y-%m-%d"))
    assert r.available
    assert r.raw_value == pytest.approx(0.20, abs=1e-6)   # 120/100 - 1


# ---------------------------------------------------------------------------
# #19 — EPS negative-base branch: flat EPS → neutral (not a -0.9 hurdle miss)
# ---------------------------------------------------------------------------

class TestEpsNegativeBase:
    def _income(self, eps_list):
        return pd.DataFrame([
            {"date": f"20{20+i}-12-31", "revenue": 10e9, "ebitda": 2e9,
             "netIncome": -1e9, "epsdiluted": e}
            for i, e in enumerate(eps_list)
        ])

    def test_flat_negative_eps_is_near_neutral(self):
        r = factor_eps_growth(self._income([-1.0, -1.0, -1.0, -1.0]))
        assert r.available
        assert abs(r.score) < 0.25          # flat slope → ~neutral, not strongly bearish

    def test_improving_negative_eps_is_bullish(self):
        r = factor_eps_growth(self._income([-3.0, -2.5, -2.0, -1.5]))
        assert r.available
        assert r.score > 0                  # rising EPS → positive


# ---------------------------------------------------------------------------
# #26 — IC uses the continuous composite score when present
# ---------------------------------------------------------------------------

class TestContinuousIC:
    def _df(self, with_score: bool):
        # One date, 5 names, all "long" (discrete signal has zero variance).
        rows = []
        for i in range(5):
            row = {"as_of_date": "2025-01-31", "signal": "long",
                   "realized_return": 0.01 * (i + 1)}
            if with_score:
                row["composite_score"] = 0.1 * (i + 1)   # correlated with return
            rows.append(row)
        return pd.DataFrame(rows)

    def test_continuous_score_yields_ic(self):
        ic = ic_series(self._df(with_score=True))
        assert not ic.empty and ic.iloc[0] > 0.9        # monotone → IC ~ +1

    def test_discrete_only_signal_has_no_ic(self):
        # Without a composite score, an all-long cross-section has no signal
        # variance, so IC is undefined (the coarseness #26 addresses).
        assert ic_series(self._df(with_score=False)).empty


# ---------------------------------------------------------------------------
# Cost-netting (#22 turnover) — net returns are gross minus a non-negative cost
# ---------------------------------------------------------------------------

class TestCostNetting:
    def _df(self):
        # AAA flips long/short every period (high turnover); BBB stays long.
        dates = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31"]
        rows = []
        for k, d in enumerate(dates):
            rows.append({"as_of_date": d, "ticker": "AAA",
                         "signal": "long" if k % 2 == 0 else "short",
                         "realized_return": 0.02, "outcome_label": "right", "confidence": 0.7})
            rows.append({"as_of_date": d, "ticker": "BBB", "signal": "long",
                         "realized_return": 0.01, "outcome_label": "right", "confidence": 0.7})
        return pd.DataFrame(rows)

    def test_turnover_positive_and_net_below_gross(self):
        df = self._df()
        turn = portfolio_turnover(df)
        gross = portfolio_period_returns(df)
        net = net_period_returns(df, round_trip_bps=50.0)
        assert turn.sum() > 0
        assert (net <= gross + 1e-12).all()             # cost never increases return
        assert net.reindex(gross.index).lt(gross).any()  # some period actually paid cost

    def test_scorecard_exposes_net_metrics(self):
        sc = compute_full_scorecard(self._df(), periods_per_year=12, round_trip_bps=50.0)
        for key in ("avg_turnover", "avg_net_return", "net_sharpe_ratio", "round_trip_bps"):
            assert key in sc


# ---------------------------------------------------------------------------
# #21 — scorecard Sharpe treats the signed book as self-financing (rf=0)
# ---------------------------------------------------------------------------

def test_scorecard_sharpe_uses_zero_rf():
    dates = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30"]
    df = pd.DataFrame([
        {"as_of_date": d, "ticker": "AAA", "signal": "long",
         "realized_return": 0.01 + 0.001 * i, "outcome_label": "right", "confidence": 0.7}
        for i, d in enumerate(dates)
    ])
    sc = compute_full_scorecard(df, periods_per_year=12)
    port = portfolio_period_returns(df)
    assert sc["sharpe_ratio"] == pytest.approx(sharpe_ratio(port, 12, risk_free=0.0))


# ---------------------------------------------------------------------------
# #20 — net-neutral construction balances the long and short legs
# ---------------------------------------------------------------------------

def _sig(ticker, signal, conf=0.80):
    return SignalResult(ticker=ticker, as_of_date="2024-01-01", signal=signal,
                        confidence=conf,
                        composite_score=1.0 if signal == "long" else -1.0,
                        conviction_tier="high",
                        expected_return=0.05 if signal == "long" else -0.05)


def _size(ticker, size=0.10):
    return PositionSizeResult(ticker=ticker, signal="long", raw_kelly=size,
                              half_kelly=size / 2, vol_target_size=size,
                              final_size=size, sizing_method="vol_target", notes=[])


def test_net_neutral_balances_legs():
    # 3 longs, 1 short → strongly net-long unless neutralized.
    signals = [_sig("AAPL", "long"), _sig("MSFT", "long"), _sig("NVDA", "long"),
               _sig("JPM", "short")]
    sizes = {t: _size(t) for t in ["AAPL", "MSFT", "NVDA", "JPM"]}
    plain = portfolio_summary(construct_portfolio(signals, sizes, max_sector=10.0))
    neutral = portfolio_summary(construct_portfolio(signals, sizes, max_sector=10.0, net_neutral=True))
    assert abs(plain["net_exposure"]) > 0.05
    assert abs(neutral["net_exposure"]) < 1e-3          # dollar-neutral
