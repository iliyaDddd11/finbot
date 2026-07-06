from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ResearchPrediction:
    ticker: str
    company_name: str
    as_of_date: str
    horizon_days: int
    signal: str                             # "long" | "short" | "neutral"
    confidence: float
    expected_return: Optional[float] = None
    composite_score: Optional[float] = None  # weighted multi-factor score [-2, +2]
    conviction_tier: str = "low"             # "high" | "medium" | "low"
    thesis_points: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    signal_rationale: list[str] = field(default_factory=list)   # factor-level rationale
    factor_scores: dict[str, Any] = field(default_factory=dict) # full factor breakdown
    factors_available: int = 0
    coverage_ratio: float = 0.0
    source_run_id: str = ""
    analysis_dir: str = ""
    raw_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
