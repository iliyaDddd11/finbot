"""
Regression tests for the P0 correctness fixes (see REVIEW.md / PLAN.md).

Covers:
  #1  cross-section z-scoring keeps factor direction (score, not raw metric)
  #3  scorecard annualisation uses the monthly frequency (12) by default
  #4  Sharpe / max-drawdown are per-period portfolio stats (order-independent)
  #4b max_drawdown seeds the equity curve at 1.0
  #5  realized return is None (not a fabricated 0.0) when no forward data exists
  #6  partial-horizon windows are flagged (horizon_complete=False)
  #8  transaction cost reduces net return on BOTH long and short
"""
import numpy as np
import pandas as pd
import pytest

from quant_eval.cross_section import zscore_factor_bank
from quant_eval.factor_library import FactorResult, _tanh_score
from quant_eval.performance_metrics import max_drawdown, portfolio_period_returns
from quant_eval.realized_returns import _compute_return_from_price_df
from quant_eval.scoring import compute_scorecard, enrich_scored_predictions
from quant_eval.transaction_costs import estimate_tc, net_return_after_tc


# ---------------------------------------------------------------------------
# #1 — cross-section z-scoring must preserve factor direction
# ---------------------------------------------------------------------------

class TestCrossSectionDirection:
    def _bank(self):
        """5 tickers; higher leverage / higher EV-EBITDA are bearish."""
        net_debt_ebitda = {"A": 0.0, "B": 1.0, "C": 2.0, "D": 3.0, "E": 8.0}
        ev_ebitda       = {"A": 5.0, "B": 8.0, "C": 12.0, "D": 20.0, "E": 40.0}
        bank = {}
        for t in net_debt_ebitda:
            lev = net_debt_ebitda[t]
            val = ev_ebitda[t]
            bank[t] = {
                # score encodes direction: invert=True → higher leverage = lower score
                "leverage": FactorResult("leverage", lev,
                                         _tanh_score(lev, center=2.0, scale=1.5, invert=True),
                                         True, ""),
                # valuation: cheaper-than-peers proxy via negative of the multiple
                "ev_ebitda_vs_history": FactorResult("ev_ebitda_vs_history", val,
                                                     _tanh_score(-val, center=-10.0, scale=8.0),
                                                     True, ""),
            }
        return bank

    def test_most_levered_gets_lowest_zscore(self):
        z = zscore_factor_bank(self._bank())
        # E is the most levered → must be the most bearish (lowest z), not the highest
        assert z["E"]["leverage"] < z["A"]["leverage"]
        assert z["E"]["leverage"] < 0 < z["A"]["leverage"]

    def test_most_expensive_gets_lowest_zscore(self):
        z = zscore_factor_bank(self._bank())
        # E is the most expensive → most bearish
        assert z["E"]["ev_ebitda_vs_history"] < z["A"]["ev_ebitda_vs_history"]
        assert z["E"]["ev_ebitda_vs_history"] < 0 < z["A"]["ev_ebitda_vs_history"]


# ---------------------------------------------------------------------------
# Shared scorecard fixture
# ---------------------------------------------------------------------------

def _scored_df():
    """3 dates × 3 tickers of long calls with known outcomes."""
    dates = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30",
             "2025-05-31", "2025-06-30"]
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        for t in tickers:
            rr = float(rng.normal(0.01, 0.05))
            rows.append({
                "as_of_date": d, "ticker": t, "signal": "long",
                "realized_return": rr, "confidence": 0.7,
                "outcome_label": "right" if rr > 0 else "wrong",
                "thesis_points": [], "risk_flags": [],
            })
    # Mirror the real pipeline: enrich adds signed_return / confidence buckets.
    return enrich_scored_predictions(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# #3 — annualisation frequency
# ---------------------------------------------------------------------------

class TestAnnualisation:
    def test_default_is_monthly_twelve(self):
        df = _scored_df()
        assert compute_scorecard(df)["sharpe_ratio"] == \
            compute_scorecard(df, periods_per_year=12)["sharpe_ratio"]

    def test_frequency_changes_sharpe(self):
        df = _scored_df()
        s6 = compute_scorecard(df, periods_per_year=6)["sharpe_ratio"]
        s12 = compute_scorecard(df, periods_per_year=12)["sharpe_ratio"]
        # The rebalance frequency must materially change the annualised Sharpe;
        # using 6 for a monthly grid is the bug this guards against.
        assert s6 is not None and s12 is not None
        assert s6 != pytest.approx(s12, rel=1e-3)


# ---------------------------------------------------------------------------
# #4 — per-period portfolio stats are order-independent
# ---------------------------------------------------------------------------

class TestPortfolioAggregation:
    def test_sharpe_and_drawdown_are_row_order_independent(self):
        df = _scored_df()
        base = compute_scorecard(df)
        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        perm = compute_scorecard(shuffled)
        assert base["sharpe_ratio"] == pytest.approx(perm["sharpe_ratio"], rel=1e-9)
        assert base["max_drawdown"] == pytest.approx(perm["max_drawdown"], rel=1e-9)

    def test_period_returns_one_per_date(self):
        df = _scored_df()
        port = portfolio_period_returns(df)
        assert list(port.index) == sorted(df["as_of_date"].unique())
        assert len(port) == df["as_of_date"].nunique()


# ---------------------------------------------------------------------------
# #4b — max_drawdown seeds equity at 1.0
# ---------------------------------------------------------------------------

class TestMaxDrawdownSeed:
    def test_monotonic_decline_measured_from_start(self):
        # 1 → 0.9 → 0.81 → 0.729 → 0.6561  ⇒ -34.4% peak-to-trough
        assert max_drawdown(pd.Series([-0.10] * 4)) == pytest.approx(-0.3439, abs=1e-3)


# ---------------------------------------------------------------------------
# #5 / #6 — realized return: no fabricated 0.0, partial windows flagged
# ---------------------------------------------------------------------------

def _price_df(days):
    ts = [pd.Timestamp("2025-01-31", tz="UTC") + pd.Timedelta(days=d) for d in days]
    close = [100.0 + d for d in days]
    return pd.DataFrame({"timestamp": ts, "close_eval": close}).sort_values("timestamp").reset_index(drop=True)


class TestRealizedReturnGuards:
    def test_no_forward_data_returns_none_not_zero(self):
        # Only the anchor bar exists; nothing after as_of.
        df = _price_df([0])
        out = _compute_return_from_price_df(df, "2025-01-31", horizon_days=60)
        entry_ts, _, _, _, _, _, horizon_complete = out
        assert entry_ts is None          # not a fabricated 0.0 return
        assert horizon_complete is False

    def test_partial_horizon_flagged(self):
        # Data ends 30 days out but the horizon is 60 → partial window.
        df = _price_df([0, 10, 20, 30])
        out = _compute_return_from_price_df(df, "2025-01-31", horizon_days=60)
        entry_ts, _, _, _, _, _, horizon_complete = out
        assert entry_ts is not None
        assert horizon_complete is False

    def test_full_horizon_complete(self):
        df = _price_df([0, 30, 65])
        out = _compute_return_from_price_df(df, "2025-01-31", horizon_days=60)
        entry_ts, _, _, _, _, _, horizon_complete = out
        assert entry_ts is not None
        assert horizon_complete is True


# ---------------------------------------------------------------------------
# #8 — transaction cost reduces net return on both sides
# ---------------------------------------------------------------------------

class TestTransactionCostSign:
    def test_cost_subtracted_for_long_and_short(self):
        tc = estimate_tc(liquidity_tier="large")
        tc_dec = tc.round_trip_bps / 10_000
        assert net_return_after_tc(0.05, tc, "long") == pytest.approx(0.05 - tc_dec)
        assert net_return_after_tc(0.05, tc, "short") == pytest.approx(0.05 - tc_dec)
