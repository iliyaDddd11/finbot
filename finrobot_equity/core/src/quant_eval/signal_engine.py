"""
Multi-Factor Signal Engine — replaces the broken 2-rule heuristic.

Architecture
------------
1. Compute up to 10 quantitative factors from FMP financial data + price CSV.
2. Score each factor on [-2, +2]; skip factors with missing data.
3. Build a weighted composite score; re-normalise weights to available factors.
4. Map composite → signal (long/short/neutral) + calibrated confidence.
5. Return full factor transparency for audit and reporting.

Integration
-----------
Called from run_walkforward_eval.py instead of _infer_signal_and_confidence().
Works standalone: pass `financial_data` dict (from FMP API) and optional
`price_csv_path` (from ResearchSnapshot).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

from quant_eval.factor_library import (
    FactorResult,
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
# Factor weights — must sum to 1.0
# Tuned to give ~equal weight to fundamental quality, growth, and price.
# ---------------------------------------------------------------------------
BASE_WEIGHTS: dict[str, float] = {
    "earnings_quality":     0.10,
    "margin_quality":       0.12,
    "leverage":             0.10,
    "roic":                 0.12,
    "revenue_acceleration": 0.10,
    "eps_growth":           0.10,
    "ev_ebitda_vs_history": 0.12,
    "price_mom_12m1m":      0.12,
    "price_mom_6m":         0.07,
    "fcf_yield":            0.05,
}

# Signal thresholds on the WEIGHTED composite score
LONG_THRESHOLD  =  0.60    # composite > 0.60 → long
SHORT_THRESHOLD = -0.60    # composite < -0.60 → short

# Minimum fraction of factors that must be available for a signal
MIN_FACTOR_COVERAGE = 0.40   # need at least 40% of weight covered


# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------

@dataclass
class FactorBreakdown:
    name: str
    raw_value: Optional[float]
    score: Optional[float]
    weight_used: float          # actual weight after re-normalisation
    available: bool
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SignalResult:
    ticker: str
    as_of_date: str
    signal: str                     # "long", "short", "neutral"
    confidence: float               # 0.0–1.0
    composite_score: float          # weighted composite
    conviction_tier: str            # "high" | "medium" | "low"
    expected_return: Optional[float]  # proxy based on composite magnitude
    factors: list[FactorBreakdown]  = field(default_factory=list)
    signal_rationale: list[str]     = field(default_factory=list)
    risk_flags: list[str]           = field(default_factory=list)
    factors_available: int          = 0
    factors_total: int              = 0
    coverage_ratio: float           = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @property
    def thesis_points(self) -> list[str]:
        return self.signal_rationale


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def compute_raw_factors(
    financial_data: dict[str, pd.DataFrame],
    as_of_date: str,
    price_csv_path: str = "",
) -> dict[str, FactorResult]:
    """
    Compute all 10 raw factors for one ticker — without combining into a signal.
    Call this for every ticker first, then cross-section z-score, then call
    `compute_signal_from_raw()` to get the final signal.
    """
    def _df(key: str) -> pd.DataFrame:
        v = financial_data.get(key)
        return v if isinstance(v, pd.DataFrame) else pd.DataFrame()

    income      = _df('income_statement')
    balance     = _df('balance_sheet')
    cash_flow   = _df('cash_flow')
    ratios      = _df('ratios')
    key_metrics = _df('key_metrics')

    return {
        "earnings_quality":     factor_earnings_quality(cash_flow, income),
        "margin_quality":       factor_margin_quality(income),
        "leverage":             factor_leverage(key_metrics, balance, income),
        "roic":                 factor_roic(key_metrics),
        "revenue_acceleration": factor_revenue_acceleration(income),
        "eps_growth":           factor_eps_growth(income),
        "ev_ebitda_vs_history": factor_valuation_vs_history(key_metrics),
        "price_mom_12m1m":      factor_price_momentum_12m1m(price_csv_path, as_of_date),
        "price_mom_6m":         factor_price_momentum_6m(price_csv_path, as_of_date),
        "fcf_yield":            factor_fcf_yield(cash_flow, key_metrics),
    }


def compute_signal_from_raw(
    raw_factors: dict[str, FactorResult],
    as_of_date: str,
    ticker: str = "",
    cross_section_scores: dict[str, float | None] | None = None,
) -> SignalResult:
    """
    Build a SignalResult from pre-computed raw factors.

    If `cross_section_scores` is provided (from cross_section.zscore_factor_bank),
    those z-scored values replace the tanh scores for the composite computation.
    This is the correct quant practice — scores are relative to the universe.

    Parameters
    ----------
    raw_factors           : Output of compute_raw_factors()
    as_of_date            : Evaluation date string
    ticker                : Ticker symbol (display only)
    cross_section_scores  : {factor_name: z_score} — optional universe-relative scores
    """
    # ------------------------------------------------------------------
    # Use cross-section z-scores if available, fall back to raw tanh scores
    # ------------------------------------------------------------------
    def _effective_score(name: str, fr: FactorResult) -> float | None:
        if cross_section_scores and name in cross_section_scores:
            cs = cross_section_scores[name]
            if cs is not None:
                return cs   # cross-section z-score in [-2, +2]
        return fr.score     # fall back to absolute tanh score

    # ------------------------------------------------------------------
    # Re-normalise weights to available factors only
    # ------------------------------------------------------------------
    available_weight = sum(
        BASE_WEIGHTS[name] for name, fr in raw_factors.items() if fr.available
    )
    coverage = available_weight

    factor_breakdowns: list[FactorBreakdown] = []
    composite = 0.0
    risk_flags: list[str] = []
    rationale_lines: list[str] = []
    used_cs = cross_section_scores is not None

    for name, fr in raw_factors.items():
        score = _effective_score(name, fr)
        if fr.available and score is not None and coverage > 0:
            effective_weight = BASE_WEIGHTS[name] / coverage
            composite += effective_weight * score
            fb = FactorBreakdown(
                name=name,
                raw_value=fr.raw_value,
                score=round(score, 4),
                weight_used=round(effective_weight, 4),
                available=True,
                rationale=fr.rationale,
            )
            contrib = effective_weight * score
            if abs(contrib) > 0.05:
                direction = "bullish" if contrib > 0 else "bearish"
                rationale_lines.append(f"{name}: {fr.rationale} [{direction}]")
        else:
            fb = FactorBreakdown(
                name=name,
                raw_value=None,
                score=None,
                weight_used=0.0,
                available=False,
                rationale=fr.rationale,
            )
            if not fr.available:
                risk_flags.append(f"factor_unavailable:{name}")

        factor_breakdowns.append(fb)

    if used_cs:
        risk_flags = [f for f in risk_flags if not f.startswith("factor_unavailable")]  # cs handles coverage

    n_available = sum(1 for fb in factor_breakdowns if fb.available)
    n_total     = len(factor_breakdowns)

    # ------------------------------------------------------------------
    # 3. Determine signal + confidence
    # ------------------------------------------------------------------
    if coverage < MIN_FACTOR_COVERAGE:
        signal     = "neutral"
        confidence = 0.45
        conviction = "low"
        risk_flags.append("insufficient_factor_coverage")
    elif composite >= LONG_THRESHOLD:
        signal     = "long"
        confidence = _map_confidence(composite, LONG_THRESHOLD, n_available, n_total)
        conviction = _map_conviction(confidence)
    elif composite <= SHORT_THRESHOLD:
        signal     = "short"
        confidence = _map_confidence(abs(composite), abs(SHORT_THRESHOLD), n_available, n_total)
        conviction = _map_conviction(confidence)
    else:
        signal     = "neutral"
        confidence = 0.45 + abs(composite) * 0.10
        conviction = "low"

    # ------------------------------------------------------------------
    # 4. Expected return proxy (naive: scale composite score → annualised return)
    # WARNING: The 0.12 coefficient is UNCALIBRATED — it is a directional proxy
    # only.  Replace with a regression coefficient derived from walk-forward
    # backtest data before using in Kelly position sizing for live trading.
    # Calibration procedure: regress realized_return ~ composite_score on the
    # scored_predictions.csv output of score_walkforward_eval.py.
    # ------------------------------------------------------------------
    if signal in ("long", "short"):
        sign = 1 if signal == "long" else -1
        expected_return = round(sign * abs(composite) * 0.12, 4)  # ~12% at full composite
    else:
        expected_return = None

    # Sort rationale lines by absolute contribution magnitude (descending)
    # Each line was appended only when |contribution| > 0.05, so extract the
    # factor name and look up the actual contribution from factor_breakdowns.
    _contrib_map: dict[str, float] = {
        fb.name: abs(fb.weight_used * (fb.score or 0.0))
        for fb in factor_breakdowns if fb.available
    }
    rationale_lines = sorted(
        rationale_lines,
        key=lambda s: _contrib_map.get(s.split(":")[0].strip(), 0.0),
        reverse=True,
    )[:8]

    return SignalResult(
        ticker=ticker,
        as_of_date=as_of_date,
        signal=signal,
        confidence=round(confidence, 4),
        composite_score=round(composite, 4),
        conviction_tier=conviction,
        expected_return=expected_return,
        factors=factor_breakdowns,
        signal_rationale=rationale_lines,
        risk_flags=risk_flags,
        factors_available=n_available,
        factors_total=n_total,
        coverage_ratio=round(coverage, 4),
    )


def compute_signal(
    financial_data: dict[str, pd.DataFrame],
    as_of_date: str,
    ticker: str = "",
    price_csv_path: str = "",
    cross_section_scores: dict[str, float | None] | None = None,
) -> SignalResult:
    """
    Convenience wrapper: compute raw factors then build signal in one call.
    For cross-sectional scoring, use compute_raw_factors() + zscore_factor_bank()
    + compute_signal_from_raw() in the walk-forward loop instead.
    """
    raw = compute_raw_factors(financial_data, as_of_date, price_csv_path)
    return compute_signal_from_raw(raw, as_of_date, ticker, cross_section_scores)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_confidence(composite_abs: float, threshold: float, n_avail: int, n_total: int) -> float:
    """
    Map composite score magnitude → confidence in [0.55, 0.90].
    Scales down when few factors are available.
    """
    # Base from composite magnitude above threshold
    excess  = composite_abs - threshold
    base    = 0.55 + min(0.30, excess * 0.35)
    # Penalise for low factor coverage
    coverage_ratio = n_avail / max(n_total, 1)
    penalty = (1 - coverage_ratio) * 0.10
    return round(max(0.50, min(0.90, base - penalty)), 4)


def _map_conviction(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"
