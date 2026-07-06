"""
Factor Library — individual quantitative factor computations.

Each function returns a FactorResult with:
  raw_value  : the actual computed metric
  score      : normalized [-2, +2], +2 = most bullish
  available  : whether data existed to compute this factor
  rationale  : one-line human-readable explanation

Scoring convention: tanh-based soft clipping so extreme values saturate
cleanly. Center and scale are set by economic reasoning, not data-fitting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

@dataclass
class FactorResult:
    name: str
    raw_value: Optional[float]
    score: Optional[float]       # [-2, +2]
    available: bool
    rationale: str


def _tanh_score(x: float, center: float, scale: float, invert: bool = False) -> float:
    """Maps x to [-2, 2] via 2*tanh((x - center) / scale).
    invert=True flips sign so that higher raw → more negative score.
    """
    normalized = (x - center) / (scale + 1e-12)
    if invert:
        normalized = -normalized
    return float(2.0 * math.tanh(normalized))


def _safe_float(val) -> Optional[float]:
    """Convert a value that might be a percentage-string or NaN to float."""
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip().rstrip('%')
        try:
            return float(val) / 100.0
        except ValueError:
            return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _ols_slope(y: list[float]) -> Optional[float]:
    """Simple OLS slope of y against index. Returns None if < 2 points."""
    y_clean = [v for v in y if v is not None and not math.isnan(v)]
    n = len(y_clean)
    if n < 2:
        return None
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(y_clean) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y_clean))
    den = sum((xi - x_mean) ** 2 for xi in x)
    return num / (den + 1e-12)


def _latest_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return most-recent row of a DataFrame sorted by date."""
    if df is None or df.empty:
        return None
    if 'date' in df.columns:
        df = df.sort_values('date', ascending=False)
    return df.iloc[0]


def _sorted_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return up to n most-recent rows sorted oldest-first."""
    if df is None or df.empty:
        return pd.DataFrame()
    if 'date' in df.columns:
        df = df.sort_values('date', ascending=True)
    return df.tail(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Factor 1: Earnings Quality  (CFO / |Net Income| ratio)
# ---------------------------------------------------------------------------

def factor_earnings_quality(cash_flow: pd.DataFrame, income: pd.DataFrame) -> FactorResult:
    """
    Cash flow from operations relative to net income.
    >1.0 = cash earnings exceed accrual earnings (high quality).
    Center at 1.0, scale 0.5 → score 0 when CFO == NI.
    """
    name = "earnings_quality"
    try:
        cf_row = _latest_row(cash_flow)
        ic_row = _latest_row(income)
        if cf_row is None or ic_row is None:
            return FactorResult(name, None, None, False, "missing cash flow or income data")

        cfo = _safe_float(cf_row.get('operatingCashFlow') or cf_row.get('netCashProvidedByOperatingActivities'))
        ni = _safe_float(ic_row.get('netIncome'))

        if cfo is None or ni is None:
            return FactorResult(name, None, None, False, "CFO or net income not found")

        if abs(ni) < 1e6:          # near-zero net income → ratio unstable
            ratio = 1.0 if cfo > 0 else 0.0
        else:
            ratio = cfo / abs(ni)

        score = _tanh_score(ratio, center=1.0, scale=0.5)
        direction = "above" if ratio > 1.0 else "below"
        return FactorResult(
            name, round(ratio, 3), round(score, 3), True,
            f"CFO/NI = {ratio:.2f}x ({direction} 1.0x target; +score = cash-rich earnings)"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 2: EBITDA Margin Quality (level + trend)
# ---------------------------------------------------------------------------

def factor_margin_quality(income: pd.DataFrame) -> FactorResult:
    """
    Combines current EBITDA margin level with 5-year trend.
    Level centered at 20% (broad industrial average); scale 15%.
    Trend: OLS slope in pp/yr, centered at 0, scale 0.02.
    """
    name = "margin_quality"
    try:
        rows = _sorted_rows(income)
        if rows.empty:
            return FactorResult(name, None, None, False, "no income data")

        margins = []
        for _, r in rows.iterrows():
            rev = _safe_float(r.get('revenue'))
            ebitda = _safe_float(r.get('ebitda'))
            if rev and ebitda and rev > 0:
                margins.append(ebitda / rev)

        if not margins:
            return FactorResult(name, None, None, False, "cannot compute EBITDA margins")

        current_margin = margins[-1]
        trend_slope = _ols_slope(margins) if len(margins) >= 2 else 0.0

        score_level = _tanh_score(current_margin, center=0.20, scale=0.15)
        score_trend = _tanh_score(trend_slope or 0.0, center=0.0, scale=0.02)
        combined = round(0.6 * score_level + 0.4 * score_trend, 3)

        return FactorResult(
            name, round(current_margin, 4), combined, True,
            f"EBITDA margin {current_margin:.1%}, trend {(trend_slope or 0)*100:+.2f}pp/yr"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 3: Leverage / Balance Sheet Strength (inverted)
# ---------------------------------------------------------------------------

def factor_leverage(key_metrics: pd.DataFrame, balance: pd.DataFrame, income: pd.DataFrame) -> FactorResult:
    """
    Net Debt / EBITDA. Lower = better (inverted scoring).
    Net cash (negative ratio) = max bullish. >4x = max bearish.
    Center at 2.0x, scale 1.5.
    """
    name = "leverage"
    try:
        # Try key_metrics first (has netDebtToEBITDA directly)
        km_row = _latest_row(key_metrics)
        ratio = None
        if km_row is not None:
            ratio = _safe_float(km_row.get('netDebtToEBITDA'))

        # Fallback: compute from balance sheet + income statement
        if ratio is None:
            b_row = _latest_row(balance)
            i_row = _latest_row(income)
            if b_row is not None and i_row is not None:
                net_debt = _safe_float(b_row.get('netDebt'))
                ebitda = _safe_float(i_row.get('ebitda'))
                if net_debt is not None and ebitda is not None and abs(ebitda) > 1e6:
                    ratio = net_debt / ebitda

        if ratio is None:
            return FactorResult(name, None, None, False, "net debt/EBITDA not available")

        score = _tanh_score(ratio, center=2.0, scale=1.5, invert=True)
        label = "net cash" if ratio < 0 else f"{ratio:.1f}x"
        return FactorResult(
            name, round(ratio, 2), round(score, 3), True,
            f"Net Debt/EBITDA = {label} (lower = stronger balance sheet)"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 4: Return on Invested Capital (ROIC)
# ---------------------------------------------------------------------------

def factor_roic(key_metrics: pd.DataFrame) -> FactorResult:
    """
    ROIC level + trend. WACC proxy ~10%; >10% = value creation.
    Center at 0.10, scale 0.10.
    """
    name = "roic"
    try:
        rows = _sorted_rows(key_metrics)
        if rows.empty:
            return FactorResult(name, None, None, False, "no key_metrics data")

        roics = []
        for _, r in rows.iterrows():
            v = _safe_float(r.get('returnOnInvestedCapital'))
            if v is not None:
                roics.append(v)

        if not roics:
            return FactorResult(name, None, None, False, "ROIC not in key_metrics")

        current = roics[-1]
        trend = _ols_slope(roics) if len(roics) >= 2 else 0.0
        score_level = _tanh_score(current, center=0.10, scale=0.10)
        score_trend = _tanh_score(trend or 0.0, center=0.0, scale=0.02)
        combined = round(0.7 * score_level + 0.3 * score_trend, 3)

        return FactorResult(
            name, round(current, 4), combined, True,
            f"ROIC = {current:.1%}, trend {(trend or 0)*100:+.2f}pp/yr (WACC ~10% hurdle)"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 5: Revenue Acceleration
# ---------------------------------------------------------------------------

def factor_revenue_acceleration(income: pd.DataFrame) -> FactorResult:
    """
    Change in YoY revenue growth rate (2nd derivative).
    Positive = growth accelerating = bullish.
    Center at 0, scale 0.05 (5pp acceleration).
    """
    name = "revenue_acceleration"
    try:
        rows = _sorted_rows(income)
        if len(rows) < 3:
            return FactorResult(name, None, None, False, "need ≥3 years for acceleration")

        revs = []
        for _, r in rows.iterrows():
            v = _safe_float(r.get('revenue'))
            if v is not None and v > 0:
                revs.append(v)

        if len(revs) < 3:
            return FactorResult(name, None, None, False, "insufficient revenue data")

        growth_rates = [(revs[i] - revs[i-1]) / revs[i-1] for i in range(1, len(revs))]
        latest_g = growth_rates[-1]
        prior_g  = growth_rates[-2]
        accel = latest_g - prior_g

        score = _tanh_score(accel, center=0.0, scale=0.05)
        return FactorResult(
            name, round(accel, 4), round(score, 3), True,
            f"Rev growth: {prior_g:.1%} → {latest_g:.1%} (accel {accel:+.1%})"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 6: EPS Growth CAGR
# ---------------------------------------------------------------------------

def factor_eps_growth(income: pd.DataFrame) -> FactorResult:
    """
    3-year EPS CAGR. Center at 10%, scale 10%.
    """
    name = "eps_growth"
    try:
        rows = _sorted_rows(income)
        if rows.empty:
            return FactorResult(name, None, None, False, "no income data")

        eps_vals = []
        for _, r in rows.iterrows():
            v = _safe_float(r.get('epsdiluted') or r.get('eps'))
            if v is not None and abs(v) > 0.01:
                eps_vals.append(v)

        if len(eps_vals) < 2:
            return FactorResult(name, None, None, False, "insufficient EPS history")

        n = min(len(eps_vals) - 1, 3)
        start, end = eps_vals[-(n+1)], eps_vals[-1]

        if start <= 0 or end <= 0:
            # Use simple slope instead of CAGR for negative EPS
            slope = _ols_slope(eps_vals)
            score = _tanh_score(slope or 0.0, center=0.5, scale=1.0)
            return FactorResult(name, round(slope or 0.0, 4), round(score, 3), True,
                                f"EPS slope {slope:+.2f}/yr (negative base; CAGR unreliable)")

        cagr = (end / start) ** (1 / n) - 1
        score = _tanh_score(cagr, center=0.10, scale=0.10)
        return FactorResult(
            name, round(cagr, 4), round(score, 3), True,
            f"{n}yr EPS CAGR = {cagr:.1%} (hurdle: 10%)"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 7: Valuation vs. Own History (EV/EBITDA)
# ---------------------------------------------------------------------------

def factor_valuation_vs_history(key_metrics: pd.DataFrame) -> FactorResult:
    """
    Current EV/EBITDA vs. own 5-year median.
    Discount to own history = bullish. Center at 0, scale 0.20.
    """
    name = "ev_ebitda_vs_history"
    try:
        rows = _sorted_rows(key_metrics)
        if rows.empty:
            return FactorResult(name, None, None, False, "no key_metrics data")

        multiples = []
        for _, r in rows.iterrows():
            v = _safe_float(r.get('enterpriseValueOverEBITDA') or r.get('evToEbitda'))
            if v is not None and 0 < v < 200:
                multiples.append(v)

        if len(multiples) < 2:
            return FactorResult(name, None, None, False, "insufficient EV/EBITDA history")

        current = multiples[-1]
        median  = float(np.median(multiples))
        n_yrs   = len(multiples)
        discount = (median - current) / (median + 1e-9)   # positive = cheaper than history

        score = _tanh_score(discount, center=0.0, scale=0.20)
        return FactorResult(
            name, round(current, 2), round(score, 3), True,
            f"EV/EBITDA {current:.1f}x vs. {n_yrs}yr median {median:.1f}x "
            f"({discount:+.1%} vs history)"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 8 & 9: Price Momentum  (12-1 month and 6-month)
# ---------------------------------------------------------------------------

def _load_price_series(price_csv_path: str, as_of_date: str) -> Optional[pd.Series]:
    """Load Dukascopy CSV and return Close price series indexed by date up to as_of_date."""
    try:
        df = pd.read_csv(price_csv_path)
        # Dukascopy columns: Gmt time, Open, High, Low, Close, Volume
        date_col = None
        for c in df.columns:
            if 'time' in c.lower() or 'date' in c.lower():
                date_col = c
                break
        if date_col is None:
            return None

        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        df = df[df[date_col] <= pd.Timestamp(as_of_date, tz='UTC')]

        close_col = None
        for c in df.columns:
            if c.lower() in ('close', 'bid close', 'close bid'):
                close_col = c
                break
        if close_col is None:
            # take 4th numeric column (OHLCV → index 3 = close)
            num_cols = df.select_dtypes(include='number').columns
            if len(num_cols) >= 4:
                close_col = num_cols[3]
            elif len(num_cols) >= 1:
                close_col = num_cols[0]
            else:
                return None

        series = df.set_index(date_col)[close_col].dropna()
        return series if len(series) >= 5 else None
    except Exception:
        return None


def factor_price_momentum_12m1m(price_csv_path: str, as_of_date: str) -> FactorResult:
    """
    Jegadeesh-Titman 12-1 month momentum (skip most-recent month).
    Center 0, scale 0.20.
    """
    name = "price_mom_12m1m"
    if not price_csv_path:
        return FactorResult(name, None, None, False, "no price CSV path")
    try:
        series = _load_price_series(price_csv_path, as_of_date)
        if series is None or len(series) < 22:
            return FactorResult(name, None, None, False, "insufficient price history")

        p_now  = series.iloc[-1]
        p_1m   = series.iloc[-22] if len(series) >= 22 else series.iloc[0]
        p_12m  = series.iloc[-252] if len(series) >= 252 else series.iloc[0]

        mom_12m = (p_now - p_12m) / (p_12m + 1e-12)
        mom_1m  = (p_now - p_1m)  / (p_1m  + 1e-12)
        mom     = mom_12m - mom_1m   # skip-1 momentum

        score = _tanh_score(mom, center=0.0, scale=0.20)
        return FactorResult(
            name, round(mom, 4), round(score, 3), True,
            f"12m-1m price momentum = {mom:+.1%} (raw 12m={mom_12m:+.1%}, 1m={mom_1m:+.1%})"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


def factor_price_momentum_6m(price_csv_path: str, as_of_date: str) -> FactorResult:
    """6-month price momentum. Center 0, scale 0.15."""
    name = "price_mom_6m"
    if not price_csv_path:
        return FactorResult(name, None, None, False, "no price CSV path")
    try:
        series = _load_price_series(price_csv_path, as_of_date)
        if series is None or len(series) < 22:
            return FactorResult(name, None, None, False, "insufficient price history")

        p_now = series.iloc[-1]
        p_6m  = series.iloc[-126] if len(series) >= 126 else series.iloc[0]
        mom   = (p_now - p_6m) / (p_6m + 1e-12)

        score = _tanh_score(mom, center=0.0, scale=0.15)
        return FactorResult(
            name, round(mom, 4), round(score, 3), True,
            f"6-month price momentum = {mom:+.1%}"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")


# ---------------------------------------------------------------------------
# Factor 10: Free Cash Flow Yield
# ---------------------------------------------------------------------------

def factor_fcf_yield(cash_flow: pd.DataFrame, key_metrics: pd.DataFrame) -> FactorResult:
    """
    Annualised FCF / Enterprise Value.
    Center at 3%, scale 3% → score 0 at average, +2 at 9%+.
    """
    name = "fcf_yield"
    try:
        cf_row = _latest_row(cash_flow)
        km_row = _latest_row(key_metrics)
        if cf_row is None or km_row is None:
            return FactorResult(name, None, None, False, "missing cash flow or key_metrics")

        fcf = _safe_float(cf_row.get('freeCashFlow'))
        ev  = _safe_float(km_row.get('enterpriseValue'))

        # Also try direct freeCashFlowYield from key_metrics
        direct_yield = _safe_float(km_row.get('freeCashFlowYield'))
        if direct_yield is not None and 0 < abs(direct_yield) < 1:
            score = _tanh_score(direct_yield, center=0.03, scale=0.03)
            return FactorResult(
                name, round(direct_yield, 4), round(score, 3), True,
                f"FCF yield = {direct_yield:.1%} (direct from key_metrics)"
            )

        if fcf is None or ev is None or ev <= 0:
            return FactorResult(name, None, None, False, "FCF or EV not available")

        yield_val = fcf / ev
        score = _tanh_score(yield_val, center=0.03, scale=0.03)
        return FactorResult(
            name, round(yield_val, 4), round(score, 3), True,
            f"FCF yield = {yield_val:.1%} (FCF={fcf/1e9:.1f}B / EV={ev/1e9:.1f}B)"
        )
    except Exception as e:
        return FactorResult(name, None, None, False, f"computation error: {e}")
