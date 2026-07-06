"""
Position Sizing — Volatility-Target + Fractional Kelly

Two complementary methods combined:

1. Volatility-Target (Moreira & Muir, 2017):
   position_size = target_vol_per_trade / asset_vol
   Scales down positions in high-vol environments, scales up in low-vol.

2. Fractional Kelly (half-Kelly is standard in practice):
   f = 0.5 * (mu - rf) / sigma^2
   Maximises long-run geometric growth with a safety margin.

Final size = min(vol_target_size, kelly_size, max_position) * signal_direction
Then conviction tier multiplier applied: high=1.0x, medium=0.75x, low=0.50x

All sizes are expressed as fraction of total portfolio (e.g. 0.05 = 5%).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Constants (tuned for a long/short equity strategy)
# ---------------------------------------------------------------------------
TARGET_VOL_PER_TRADE = 0.015   # 1.5% portfolio vol contribution per position
MAX_POSITION         = 0.10    # Hard cap: no single name > 10% of portfolio
MIN_POSITION         = 0.005   # Below this → effectively flat (noise level)
RISK_FREE_RATE       = 0.045   # 4.5% risk-free rate (US Tbill proxy)

CONVICTION_MULTIPLIER = {
    "high":   1.00,
    "medium": 0.75,
    "low":    0.50,
}


@dataclass
class PositionSizeResult:
    ticker: str
    signal: str
    raw_kelly: float           # Full Kelly fraction (uncapped)
    half_kelly: float          # Half-Kelly (standard risk adjustment)
    vol_target_size: float     # Volatility-targeted size
    final_size: float          # Combined, capped, conviction-adjusted
    sizing_method: str         # "vol_target" | "kelly" | "min_size" | "flat"
    notes: list[str]


def _annualised_vol_from_daily(daily_vol: float) -> float:
    return daily_vol * math.sqrt(252)


def kelly_fraction(
    expected_return: float,
    asset_vol: float,
    risk_free: float = RISK_FREE_RATE,
) -> float:
    """Full Kelly fraction: f = (mu - rf) / sigma^2"""
    excess = expected_return - risk_free
    if asset_vol < 1e-6:
        return 0.0
    return excess / (asset_vol ** 2)


def vol_target_size(
    asset_vol: float,
    target_vol: float = TARGET_VOL_PER_TRADE,
) -> float:
    """Scale position so its vol contribution equals target_vol."""
    if asset_vol < 1e-6:
        return 0.0
    return target_vol / asset_vol


def compute_position_size(
    ticker: str,
    signal: str,
    confidence: float,
    composite_score: float,
    conviction_tier: str = "low",
    expected_return: Optional[float] = None,
    asset_annual_vol: Optional[float] = None,
    target_vol: float = TARGET_VOL_PER_TRADE,
    max_position: float = MAX_POSITION,
) -> PositionSizeResult:
    """
    Compute final position size combining vol-targeting and Kelly.

    Parameters
    ----------
    ticker           : Ticker symbol
    signal           : "long" | "short" | "neutral"
    confidence       : Model confidence [0, 1]
    composite_score  : Weighted factor composite
    conviction_tier  : "high" | "medium" | "low"
    expected_return  : Annualised expected return estimate (optional)
    asset_annual_vol : Annualised price volatility (optional; defaults to 30%)
    target_vol       : Target portfolio vol contribution per trade
    max_position     : Hard cap on position size

    Returns
    -------
    PositionSizeResult
    """
    notes: list[str] = []

    if signal == "neutral":
        return PositionSizeResult(
            ticker=ticker, signal=signal,
            raw_kelly=0.0, half_kelly=0.0, vol_target_size=0.0,
            final_size=0.0, sizing_method="flat", notes=["neutral signal → flat"]
        )

    # Default vol assumption if not provided
    vol = asset_annual_vol if asset_annual_vol and asset_annual_vol > 0 else 0.30
    if asset_annual_vol is None:
        notes.append("vol defaulted to 30% (no price history)")

    # --- Method 1: Volatility-target ---
    vt = min(vol_target_size(vol, target_vol), max_position)

    # --- Method 2: Fractional Kelly ---
    # Use magnitude of expected return — direction is already captured in signal
    mu_raw = expected_return if expected_return is not None else abs(composite_score) * 0.12
    mu = abs(mu_raw)  # Kelly uses |expected return|; sign is in the signal direction
    raw_k  = kelly_fraction(mu, vol)
    half_k = 0.5 * raw_k
    capped_k = min(max(half_k, 0.0), max_position)

    if raw_k <= 0:
        notes.append("Kelly negative → capped at 0")

    # --- Combine: take the more conservative of the two methods ---
    base_size = min(vt, capped_k) if capped_k > 0 else vt
    sizing_method = "vol_target" if capped_k <= 0 else ("kelly" if capped_k < vt else "vol_target")

    # --- Conviction multiplier ---
    mult = CONVICTION_MULTIPLIER.get(conviction_tier, 0.50)
    adjusted = base_size * mult

    # --- Confidence scaling (linear 0.5 → 1.0 confidence → 0% → 100% of adjusted) ---
    conf_scale = max(0.0, min(1.0, (confidence - 0.50) / 0.40))
    scaled = adjusted * conf_scale

    final = max(0.0, min(scaled, max_position))
    if final < MIN_POSITION:
        final = 0.0
        sizing_method = "flat"
        notes.append(f"below min threshold ({MIN_POSITION:.1%})")

    return PositionSizeResult(
        ticker=ticker,
        signal=signal,
        raw_kelly=round(raw_k, 4),
        half_kelly=round(half_k, 4),
        vol_target_size=round(vt, 4),
        final_size=round(final, 4),
        sizing_method=sizing_method,
        notes=notes,
    )


def position_size_summary(result: PositionSizeResult) -> str:
    direction = "+" if result.signal == "long" else "-"
    return (
        f"{result.ticker} {result.signal}: {direction}{result.final_size:.1%} "
        f"(Kelly½={result.half_kelly:.1%}, VolTgt={result.vol_target_size:.1%}, "
        f"method={result.sizing_method})"
    )
