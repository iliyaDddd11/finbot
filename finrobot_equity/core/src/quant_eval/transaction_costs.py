"""
Transaction Cost Model — simple square-root market impact.

Based on Almgren-Chriss (2001) simplified form:
  TC ≈ spread/2 + λ * σ * sqrt(Q / ADV)

where Q = order size as fraction of portfolio, ADV = average daily volume fraction.

For research-grade use (no live execution):
  - Half-spread proxy by liquidity tier
  - Market impact scaled to typical hedge-fund order size
  - Round-trip TC returned in basis points

The model is intentionally conservative: it overstates TC to be safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Liquidity tiers  (US equities, rough proxies)
# ---------------------------------------------------------------------------
#  Tier 1: Large-cap  (market cap > $50B) — deep liquidity
#  Tier 2: Mid-cap   ($5B – $50B)
#  Tier 3: Small-cap  (< $5B)

SPREAD_BPS: dict[str, float] = {
    "large":  3.0,   # half-spread 1.5bps → round-trip 3bps
    "mid":    8.0,
    "small": 20.0,
}

IMPACT_COEFF: dict[str, float] = {
    "large":  0.05,   # λ coefficient (low impact on large-caps)
    "mid":    0.10,
    "small":  0.20,
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TCEstimate:
    one_way_bps: float       # half-round-trip (entry OR exit), in basis points
    round_trip_bps: float    # full round-trip
    annual_drag_bps: float   # round_trip × turnover_per_year
    liquidity_tier: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def estimate_tc(
    market_cap_bn: Optional[float] = None,
    daily_vol_fraction: float = 0.001,    # fraction of ADV per trade (0.1% = $100M on $100B stock)
    annual_volatility: float = 0.25,      # annualised price vol
    turnover_per_year: float = 6,         # round-trips per year (bi-monthly rebalancing)
    liquidity_tier: str = "large",
) -> TCEstimate:
    """
    Estimate round-trip transaction costs in basis points.

    Parameters
    ----------
    market_cap_bn : float, optional
        Market cap in $B.  If provided, overrides liquidity_tier.
    daily_vol_fraction : float
        Fraction of average daily volume per trade.
    annual_volatility : float
        Annualised return volatility of the asset.
    turnover_per_year : float
        Expected number of round-trips per year.
    liquidity_tier : str
        One of "large", "mid", "small".

    Returns
    -------
    TCEstimate
    """
    # Determine tier from market cap if provided
    if market_cap_bn is not None:
        if market_cap_bn >= 50:
            liquidity_tier = "large"
        elif market_cap_bn >= 5:
            liquidity_tier = "mid"
        else:
            liquidity_tier = "small"

    spread    = SPREAD_BPS.get(liquidity_tier, 8.0)
    lam       = IMPACT_COEFF.get(liquidity_tier, 0.10)
    daily_vol = annual_volatility / (252 ** 0.5)

    # Market impact (bps): λ × daily_vol × sqrt(Q/ADV) × 10000
    impact_bps = lam * daily_vol * (daily_vol_fraction ** 0.5) * 10_000

    one_way_bps    = spread / 2 + impact_bps
    round_trip_bps = 2 * one_way_bps
    annual_drag    = round_trip_bps * turnover_per_year

    return TCEstimate(
        one_way_bps=round(one_way_bps, 2),
        round_trip_bps=round(round_trip_bps, 2),
        annual_drag_bps=round(annual_drag, 2),
        liquidity_tier=liquidity_tier,
    )


def net_return_after_tc(
    gross_return: float,
    tc: TCEstimate,
    direction: str = "long",
) -> float:
    """
    Subtract round-trip TC from a gross trade return.

    Parameters
    ----------
    gross_return : float   (decimal, e.g. 0.05 for 5%)
    tc : TCEstimate
    direction : "long" or "short"

    Returns
    -------
    float : net return in decimal
    """
    tc_decimal = tc.round_trip_bps / 10_000
    sign = 1 if direction == "long" else -1
    return gross_return - sign * tc_decimal


def tc_summary_line(tc: TCEstimate) -> str:
    return (
        f"TC ({tc.liquidity_tier}-cap): {tc.round_trip_bps:.1f}bps round-trip | "
        f"annual drag {tc.annual_drag_bps:.0f}bps at {tc.round_trip_bps:.0f}bps×{int(tc.annual_drag_bps/tc.round_trip_bps)}x turnover"
        if tc.round_trip_bps > 0 else "TC: n/a"
    )
