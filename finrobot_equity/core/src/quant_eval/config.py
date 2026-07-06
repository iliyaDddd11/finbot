"""
Typed evaluation config for the walk-forward harness.

One object carries every parameter that must stay consistent across the predict
and score stages — most importantly the rebalance frequency, which is the value
whose hard-coded mismatch caused the annualization bug (REVIEW.md #3). Building
this once in the driver and threading it downstream makes that class of bug
structurally impossible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


# Map a pandas offset alias to the number of rebalances per year, for
# annualizing Sharpe / ICIR / Calmar.
_FREQ_PERIODS_PER_YEAR = {
    "ME": 12, "BME": 12, "M": 12,     # month end
    "W": 52,                           # weekly
    "QE": 4, "BQE": 4, "Q": 4,         # quarter end
    "2ME": 6,                          # bi-monthly
    "YE": 1, "A": 1,                   # annual
}


@dataclass
class EvalConfig:
    start: str
    end: str
    grid_freq: str = "ME"                       # month-end walk-forward grid
    horizons: list[int] = field(default_factory=lambda: [30, 60, 90])
    primary_horizon: int = 60
    round_trip_bps: float = 20.0                 # assumed transaction cost
    prefer_adjusted: bool = True                 # adjusted prices everywhere (REVIEW #2)
    min_pattern_count: int = 2
    seed: int = 42

    @property
    def periods_per_year(self) -> int:
        """Rebalances per year implied by grid_freq (defaults to monthly=12)."""
        return _FREQ_PERIODS_PER_YEAR.get(self.grid_freq, 12)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
