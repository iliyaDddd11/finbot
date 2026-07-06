from __future__ import annotations

import pandas as pd


def build_month_end_grid(start_date: str, end_date: str) -> list[str]:
    dates = pd.date_range(start=start_date, end=end_date, freq="ME")
    return [d.strftime("%Y-%m-%d") for d in dates]
