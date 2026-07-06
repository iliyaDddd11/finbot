"""
No-network end-to-end integration test for the quant_eval pipeline.

Exercises the real functions — compute_raw_factors -> zscore_factor_bank ->
compute_signal_from_raw -> scoring.compute_scorecard — on synthetic fundamentals
(no FMP, no price downloads). This is the guard the P0 direction/annualization
bug class needed: a unit test on a leaf function can't catch a sign inversion
that only appears once factors are combined cross-sectionally.
"""
import numpy as np
import pandas as pd

from quant_eval.cross_section import zscore_factor_bank
from quant_eval.scoring import compute_scorecard, enrich_scored_predictions
from quant_eval.signal_engine import compute_raw_factors, compute_signal_from_raw


def _fin_data(q: float) -> dict[str, pd.DataFrame]:
    """
    Synthetic 5-year financials for one ticker. q in [0, 1]: 1 = strong
    fundamentals (low leverage, high ROIC/margins/FCF, growing, cheap), 0 = weak.
    """
    years = [f"{2019 + i}-12-31" for i in range(5)]
    g = 0.02 + 0.12 * q
    rev = [100.0 * (1 + g) ** i for i in range(5)]
    margin = [(0.10 + 0.30 * q) * (1 + 0.02 * i * q) for i in range(5)]
    ebitda = [r * m for r, m in zip(rev, margin)]
    ni = [e * 0.6 for e in ebitda]
    eps = [1.0 * (1 + (0.04 + 0.22 * q)) ** i for i in range(5)]
    nd_ebitda = 0.2 + 5.8 * (1 - q)          # low for strong
    ev_mult = [20.0, 20.0, 20.0, 20.0, 25.0 - 18.0 * q]   # current cheap for strong

    income = pd.DataFrame({"date": years, "revenue": rev, "ebitda": ebitda,
                           "netIncome": ni, "epsdiluted": eps})
    balance = pd.DataFrame({"date": years, "netDebt": [nd_ebitda * ebitda[-1]] * 5})
    cash_flow = pd.DataFrame({"date": years,
                              "operatingCashFlow": [n * (1.0 + 0.5 * q) for n in ni],
                              "freeCashFlow": [e * (0.2 + 0.5 * q) for e in ebitda]})
    key_metrics = pd.DataFrame({"date": years,
                                "netDebtToEBITDA": [nd_ebitda] * 5,
                                "returnOnInvestedCapital": [0.02 + 0.30 * q] * 5,
                                "enterpriseValueOverEBITDA": ev_mult,
                                "freeCashFlowYield": [0.005 + 0.09 * q] * 5,
                                "enterpriseValue": [1000.0] * 5})
    return {"income_statement": income, "balance_sheet": balance,
            "cash_flow": cash_flow, "key_metrics": key_metrics, "ratios": pd.DataFrame()}


TICKERS = {"STRONG": 1.0, "Q75": 0.75, "MID": 0.5, "Q25": 0.25, "WEAK": 0.0}


def _signals():
    as_of = "2024-01-31"
    raw = {t: compute_raw_factors(_fin_data(q), as_of, price_csv_path="") for t, q in TICKERS.items()}
    cs = zscore_factor_bank(raw)
    return {t: compute_signal_from_raw(raw[t], as_of, ticker=t, cross_section_scores=cs[t]) for t in TICKERS}


class TestDirectionIntegrityEndToEnd:
    def test_strong_outranks_weak(self):
        sigs = _signals()
        assert sigs["STRONG"].composite_score > sigs["WEAK"].composite_score

    def test_composite_is_monotone_in_quality(self):
        sigs = _signals()
        scores = [sigs[t].composite_score for t in ["STRONG", "Q75", "MID", "Q25", "WEAK"]]
        # Strong-to-weak composites should be (weakly) decreasing.
        assert scores == sorted(scores, reverse=True)

    def test_strong_not_short_weak_not_long(self):
        sigs = _signals()
        assert sigs["STRONG"].signal in ("long", "neutral")
        assert sigs["WEAK"].signal in ("short", "neutral")


class TestScorecardEndToEnd:
    def _scored_df(self):
        # Reuse the signals across 6 monthly dates; give STRONG-ish names
        # positive realized returns and WEAK-ish names negative, so the book works.
        sigs = _signals()
        dates = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30"]
        rng = np.random.default_rng(0)
        rows = []
        for d in dates:
            for t, q in TICKERS.items():
                sr = sigs[t]
                base = (q - 0.5) * 0.06
                rr = float(base + rng.normal(0, 0.01))
                direction = 1 if sr.signal == "long" else -1 if sr.signal == "short" else 0
                outcome = "right" if direction * rr > 0 else ("wrong" if direction != 0 else "unknown")
                rows.append({
                    "as_of_date": d, "ticker": t, "signal": sr.signal,
                    "composite_score": sr.composite_score, "confidence": sr.confidence,
                    "realized_return": rr, "outcome_label": outcome,
                    "thesis_points": [], "risk_flags": [],
                })
        return enrich_scored_predictions(pd.DataFrame(rows))

    def test_scorecard_has_all_metric_families(self):
        sc = compute_scorecard(self._scored_df())
        for key in ("right_rate", "sharpe_ratio", "max_drawdown", "calmar_ratio",
                    "mean_ic", "icir", "net_sharpe_ratio", "avg_net_return",
                    "avg_turnover", "sharpe_ci_95"):
            assert key in sc

    def test_net_is_not_above_gross(self):
        sc = compute_scorecard(self._scored_df(), round_trip_bps=50.0)
        if sc["net_sharpe_ratio"] is not None and sc["sharpe_ratio"] is not None:
            # Costs can only reduce performance.
            assert sc["avg_net_return"] <= sc["avg_signed_return"] + 1e-9
