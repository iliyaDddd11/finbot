"""
Portfolio Construction — Sector Correlation Caps + Diversification

Takes a list of SignalResults and PositionSizeResults from the signal engine
and applies portfolio-level constraints:

  1. Max single-name weight: 10% (hard cap)
  2. Max sector weight: 30% (prevents tech-only portfolio)
  3. Long/short balance: gross exposure capped at 150% (1.5x leverage)
  4. Min confidence gate: skip positions below threshold

The output is a list of PortfolioPosition objects with final_weight.

This is the layer between "signal says long" and "put 4.3% in this stock".
Without it, a model that goes long 10 tech names is just leveraged QQQ beta.

References
----------
Clarke, de Silva, Thorley (2002) "Portfolio Constraints and the Fundamental
Law of Active Management"
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from quant_eval.signal_engine import SignalResult
from quant_eval.position_sizing import PositionSizeResult
from quant_eval.universe import sector_of

# ---------------------------------------------------------------------------
# Portfolio-level limits
# ---------------------------------------------------------------------------
MAX_POSITION_WEIGHT  = 0.10    # 10% max single name
MAX_SECTOR_WEIGHT    = 0.30    # 30% max single sector (long or short)
MAX_GROSS_EXPOSURE   = 1.50    # 150% gross (sum of |weights|)
MIN_CONFIDENCE_GATE  = 0.55    # skip positions below this confidence
MIN_POSITION_TO_KEEP = 0.005   # round-trip < 50bps → drop (noise)


@dataclass
class PortfolioPosition:
    ticker: str
    sector: str | None
    signal: str               # "long" | "short"
    confidence: float
    conviction_tier: str
    raw_weight: float         # from position sizing (unsigned)
    sector_scaled_weight: float  # after sector cap scaling
    final_weight: float       # final signed weight (negative = short)
    capped_by: list[str]      # ["sector_cap", "position_cap", "gross_cap"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def construct_portfolio(
    signal_results: list[SignalResult],
    position_sizes: dict[str, PositionSizeResult],
    max_position: float = MAX_POSITION_WEIGHT,
    max_sector:   float = MAX_SECTOR_WEIGHT,
    max_gross:    float = MAX_GROSS_EXPOSURE,
    min_conf:     float = MIN_CONFIDENCE_GATE,
) -> list[PortfolioPosition]:
    """
    Build a diversified portfolio from signal + sizing inputs.

    Parameters
    ----------
    signal_results  : List of SignalResult from signal engine
    position_sizes  : {ticker: PositionSizeResult} from position_sizing
    max_position    : Max single-name weight (unsigned)
    max_sector      : Max sector weight (unsigned)
    max_gross       : Max gross exposure (sum of |weights|)
    min_conf        : Drop signals below this confidence

    Returns
    -------
    List of PortfolioPosition objects, sorted by |final_weight| descending
    """
    # --- Step 1: filter by confidence and signal direction ---
    candidates: list[dict] = []
    for sr in signal_results:
        if sr.signal == "neutral":
            continue
        if sr.confidence < min_conf:
            continue
        ps = position_sizes.get(sr.ticker)
        if ps is None or ps.final_size <= MIN_POSITION_TO_KEEP:
            continue
        candidates.append({
            "ticker":     sr.ticker,
            "sector":     sector_of(sr.ticker),
            "signal":     sr.signal,
            "confidence": sr.confidence,
            "conviction": sr.conviction_tier,
            "raw_weight": ps.final_size,
        })

    if not candidates:
        return []

    # --- Step 2: sector-level cap (long and short separately) ---
    sector_usage: dict[str, float] = {}   # {sector_direction: current_weight}
    for c in candidates:
        key = f"{c['sector'] or 'unknown'}_{c['signal']}"
        sector_usage.setdefault(key, 0.0)

    sector_scaled: dict[str, float] = {}  # ticker → sector-capped weight
    sector_accrued: dict[str, float] = {}

    # Sort by confidence descending — highest-conviction names get priority
    for c in sorted(candidates, key=lambda x: -x["confidence"]):
        t  = c["ticker"]
        sk = f"{c['sector'] or 'unknown'}_{c['signal']}"
        already = sector_accrued.get(sk, 0.0)
        cap = max_sector - already
        if cap <= 0:
            sector_scaled[t] = 0.0
            continue
        w = min(c["raw_weight"], cap)
        sector_scaled[t] = w
        sector_accrued[sk] = already + w

    # --- Step 3: position cap ---
    position_capped: dict[str, float] = {
        t: min(w, max_position)
        for t, w in sector_scaled.items()
    }

    # --- Step 4: gross exposure cap (scale all down proportionally) ---
    gross = sum(position_capped.values())
    gross_scale = min(1.0, max_gross / gross) if gross > 0 else 1.0
    final_weights: dict[str, float] = {
        t: round(w * gross_scale, 4)
        for t, w in position_capped.items()
    }

    # --- Step 5: assemble output ---
    candidate_map = {c["ticker"]: c for c in candidates}
    positions: list[PortfolioPosition] = []
    for c in candidates:
        t = c["ticker"]
        fw = final_weights.get(t, 0.0)
        if fw <= MIN_POSITION_TO_KEEP:
            continue
        capped_by = []
        if sector_scaled.get(t, c["raw_weight"]) < c["raw_weight"]:
            capped_by.append("sector_cap")
        if position_capped.get(t, sector_scaled.get(t, 0)) < sector_scaled.get(t, 0):
            capped_by.append("position_cap")
        if gross_scale < 0.999:
            capped_by.append("gross_cap")

        signed_weight = fw if c["signal"] == "long" else -fw
        positions.append(PortfolioPosition(
            ticker=t,
            sector=c["sector"],
            signal=c["signal"],
            confidence=c["confidence"],
            conviction_tier=c["conviction"],
            raw_weight=c["raw_weight"],
            sector_scaled_weight=round(sector_scaled.get(t, 0), 4),
            final_weight=signed_weight,
            capped_by=capped_by,
        ))

    return sorted(positions, key=lambda p: -abs(p.final_weight))


def portfolio_summary(positions: list[PortfolioPosition]) -> dict[str, Any]:
    """Return aggregate stats for a constructed portfolio."""
    longs  = [p for p in positions if p.signal == "long"]
    shorts = [p for p in positions if p.signal == "short"]
    gross  = sum(abs(p.final_weight) for p in positions)
    net    = sum(p.final_weight for p in positions)

    sector_breakdown: dict[str, float] = {}
    for p in positions:
        sec = p.sector or "unknown"
        sector_breakdown[sec] = sector_breakdown.get(sec, 0.0) + abs(p.final_weight)

    return {
        "n_positions":  len(positions),
        "n_long":       len(longs),
        "n_short":      len(shorts),
        "gross_exposure": round(gross, 4),
        "net_exposure":   round(net, 4),
        "sector_breakdown": {k: round(v, 4) for k, v in sorted(sector_breakdown.items(), key=lambda x: -x[1])},
        "positions": [p.to_dict() for p in positions],
    }


def portfolio_report_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"Positions: {summary['n_positions']} "
        f"({summary['n_long']} long / {summary['n_short']} short)",
        f"Gross: {summary['gross_exposure']:.1%}  Net: {summary['net_exposure']:+.1%}",
        "Sector weights:",
    ]
    for sector, w in summary.get("sector_breakdown", {}).items():
        lines.append(f"  {sector:<25} {w:.1%}")
    return lines
