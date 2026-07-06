"""
Performance Metrics — professional quant evaluation statistics.

Computes:
  IC     : Information Coefficient (Spearman correlation of signal vs realised return)
  ICIR   : IC / std(IC) annualised — measures signal consistency
  Sharpe : Annualised Sharpe ratio of signed returns (strategy long/short)
  Hit Rate: Fraction of directional calls that were correct (excluding neutral/unknown)
  Max DD : Maximum peak-to-trough drawdown of cumulative signed returns
  Calmar : Annualised return / Max DD
  Turnover: Mean absolute change in signal between periods
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def information_coefficient(
    signals: pd.Series,       # "long"=1, "short"=-1, "neutral"=0
    realized_returns: pd.Series,
    method: str = "spearman",
) -> Optional[float]:
    """
    Rank-correlation of signal direction with realised return.
    Returns None if insufficient data or all-zero variance.
    """
    signal_dir = _to_numeric_signal(signals)
    returns    = pd.to_numeric(realized_returns, errors="coerce")

    mask = signal_dir.notna() & returns.notna()
    if mask.sum() < 4:
        return None

    s = signal_dir[mask].values
    r = returns[mask].values

    if np.std(s) < 1e-9 or np.std(r) < 1e-9:
        return None

    if method == "spearman":
        ic, _ = scipy_stats.spearmanr(s, r)
    else:
        ic, _ = scipy_stats.pearsonr(s, r)

    return float(ic) if not math.isnan(ic) else None


def ic_series(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-period IC: for each (as_of_date x ticker) pair.
    Returns a Series indexed by as_of_date.
    """
    if "as_of_date" not in df.columns:
        return pd.Series(dtype=float)

    results = {}
    for date, grp in df.groupby("as_of_date"):
        ic = information_coefficient(grp["signal"], grp.get("realized_return", pd.Series()))
        if ic is not None:
            results[date] = ic

    return pd.Series(results).sort_index()


def icir(ic_vals: pd.Series, periods_per_year: int = 6) -> Optional[float]:
    """
    IC Information Ratio = mean(IC) / std(IC) * sqrt(periods_per_year).
    """
    clean = ic_vals.dropna()
    if len(clean) < 3:
        return None
    mean_ic = clean.mean()
    std_ic  = clean.std()
    if std_ic < 1e-9:
        return None
    return float(mean_ic / std_ic * math.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Return-based metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    signed_returns: pd.Series,
    periods_per_year: int = 6,
    risk_free: float = 0.045,
) -> Optional[float]:
    """
    Annualised Sharpe of the signed-return stream.
    periods_per_year=6 assumes bi-monthly rebalancing.
    """
    r = pd.to_numeric(signed_returns, errors="coerce").dropna()
    if len(r) < 4:
        return None

    rf_per_period = risk_free / periods_per_year
    excess        = r - rf_per_period
    mean_e        = excess.mean()
    std_e         = excess.std(ddof=1)

    if std_e < 1e-9:
        return None

    return float(mean_e / std_e * math.sqrt(periods_per_year))


def max_drawdown(signed_returns: pd.Series) -> Optional[float]:
    """Maximum peak-to-trough drawdown of cumulative signed returns."""
    r = pd.to_numeric(signed_returns, errors="coerce").dropna()
    if r.empty:
        return None

    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())


def calmar_ratio(
    signed_returns: pd.Series,
    periods_per_year: int = 6,
) -> Optional[float]:
    """Annualised mean return / |max drawdown|."""
    r = pd.to_numeric(signed_returns, errors="coerce").dropna()
    if len(r) < 4:
        return None

    ann_return = r.mean() * periods_per_year
    mdd = max_drawdown(r)
    if mdd is None or abs(mdd) < 1e-9:
        return None

    return float(ann_return / abs(mdd))


def hit_rate(df: pd.DataFrame) -> Optional[float]:
    """
    Directional accuracy: fraction of long/short calls where
    outcome_label == 'right'.  Excludes neutral and unknown.
    """
    directed = df[df["signal"].str.lower().isin(["long", "short"])].copy()
    known    = directed[directed["outcome_label"] == "right"].copy()

    n_total = len(directed[directed["outcome_label"].isin(["right", "wrong"])])
    n_right = len(known)

    return round(n_right / n_total, 4) if n_total > 0 else None


def turnover(signals: pd.Series) -> Optional[float]:
    """Mean absolute change in numeric signal (proxy for portfolio turnover)."""
    num = _to_numeric_signal(signals).dropna()
    if len(num) < 2:
        return None
    return float(num.diff().abs().mean())


def compute_full_scorecard(
    df: pd.DataFrame,
    periods_per_year: int = 6,
) -> dict:
    """
    Full quant scorecard from a scored predictions DataFrame.
    DataFrame must have columns:
      signal, realized_return, outcome_label, confidence, as_of_date
    """
    known = df[df["outcome_label"].isin(["right", "wrong"])].copy()
    has_returns = "realized_return" in df.columns

    # Signed returns: signal direction × realized return
    if has_returns:
        sig_dir = _to_numeric_signal(df["signal"])
        signed  = sig_dir * pd.to_numeric(df["realized_return"], errors="coerce")
    else:
        signed = pd.Series(dtype=float)

    ic_s = ic_series(df) if has_returns else pd.Series(dtype=float)

    result: dict = {
        # Sample
        "n_predictions":     int(len(df)),
        "n_known":           int(len(known)),
        "n_long":            int((df["signal"].str.lower() == "long").sum()),
        "n_short":           int((df["signal"].str.lower() == "short").sum()),
        "n_neutral":         int((df["signal"].str.lower() == "neutral").sum()),
        # Direction accuracy
        "hit_rate":          hit_rate(df),
        # Return metrics
        "avg_realized_return": _safe_mean(pd.to_numeric(df.get("realized_return", pd.Series()), errors="coerce")),
        "avg_signed_return":   _safe_mean(signed),
        "sharpe_ratio":        sharpe_ratio(signed, periods_per_year),
        "max_drawdown":        max_drawdown(signed),
        "calmar_ratio":        calmar_ratio(signed, periods_per_year),
        # IC metrics
        "mean_ic":             float(ic_s.mean()) if not ic_s.empty else None,
        "icir":                icir(ic_s, periods_per_year),
        "ic_periods":          int(len(ic_s)),
        # Calibration
        "avg_confidence":      _safe_mean(pd.to_numeric(df.get("confidence", pd.Series()), errors="coerce")),
        "avg_max_upside":      _safe_mean(pd.to_numeric(df.get("max_upside", pd.Series()), errors="coerce")),
        "avg_max_drawdown":    _safe_mean(pd.to_numeric(df.get("max_drawdown", pd.Series()), errors="coerce")),
    }

    # Per-ticker breakdown
    if "ticker" in df.columns:
        by_ticker = {}
        for ticker, g in df.groupby("ticker"):
            s_dir = _to_numeric_signal(g["signal"])
            s_ret = pd.to_numeric(g.get("realized_return", pd.Series()), errors="coerce")
            sg    = s_dir * s_ret
            known_g = g[g["outcome_label"].isin(["right", "wrong"])]
            n_r = (known_g["outcome_label"] == "right").sum()
            n_k = len(known_g)
            by_ticker[str(ticker)] = {
                "n":           int(len(g)),
                "hit_rate":    round(n_r / n_k, 4) if n_k else None,
                "avg_signed":  _safe_mean(sg),
                "sharpe":      sharpe_ratio(sg, periods_per_year),
            }
        result["by_ticker"] = by_ticker

    # Per-signal breakdown (known outcomes only)
    by_signal = {}
    for sig, g in known.groupby(known["signal"].str.lower()):
        n_r = (g["outcome_label"] == "right").sum()
        by_signal[str(sig)] = {
            "n":        int(len(g)),
            "hit_rate": round(n_r / len(g), 4) if len(g) else None,
        }
    result["by_signal"] = by_signal

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SIGNAL_MAP = {"long": 1, "short": -1, "neutral": 0}


def _to_numeric_signal(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map(_SIGNAL_MAP)


def _safe_mean(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else None


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: pd.Series | np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float] | tuple[None, None]:
    """
    Non-parametric bootstrap confidence interval for any scalar statistic.

    Parameters
    ----------
    data    : 1-D array-like of observations (NaNs dropped internally)
    stat_fn : function mapping array → scalar (e.g. np.mean, np.median)
    n_boot  : number of bootstrap resamples (default 2000)
    ci      : confidence level, e.g. 0.95 for 95% CI
    seed    : random seed for reproducibility

    Returns
    -------
    (lower, upper) floats, or (None, None) if insufficient data.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 4:
        return None, None

    rng      = np.random.default_rng(seed)
    boot_stats = np.array([
        stat_fn(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha  = (1.0 - ci) / 2
    lower  = float(np.percentile(boot_stats, alpha * 100))
    upper  = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return lower, upper


def bootstrap_hit_rate_ci(
    outcome_labels: pd.Series,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float] | tuple[None, None]:
    """95% CI on hit rate restricted to right/wrong outcomes."""
    known = outcome_labels[outcome_labels.isin(["right", "wrong"])]
    if len(known) < 4:
        return None, None
    binary = (known == "right").astype(float).values
    return bootstrap_ci(binary, np.mean, n_boot=n_boot, ci=ci)


def bootstrap_sharpe_ci(
    signed_returns: pd.Series,
    periods_per_year: int = 6,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float] | tuple[None, None]:
    """95% CI on annualised Sharpe via bootstrap."""
    r = pd.to_numeric(signed_returns, errors="coerce").dropna()
    if len(r) < 4:
        return None, None

    def _sharpe(arr: np.ndarray) -> float:
        s = pd.Series(arr)
        v = sharpe_ratio(s, periods_per_year)
        return v if v is not None else float("nan")

    return bootstrap_ci(r, _sharpe, n_boot=n_boot, ci=ci)


def add_bootstrap_cis(
    scorecard: dict,
    df: pd.DataFrame,
    periods_per_year: int = 6,
    n_boot: int = 2000,
) -> dict:
    """
    Attach 95% bootstrap CIs for key metrics to an existing scorecard dict.
    Modifies in place and returns the dict.
    """
    sig_dir = _to_numeric_signal(df["signal"])
    signed  = sig_dir * pd.to_numeric(df.get("realized_return", pd.Series()), errors="coerce")

    # Hit rate CI
    hr_lo, hr_hi = bootstrap_hit_rate_ci(df.get("outcome_label", pd.Series()), n_boot=n_boot)
    scorecard["hit_rate_ci_95"] = [hr_lo, hr_hi]

    # Sharpe CI
    sh_lo, sh_hi = bootstrap_sharpe_ci(signed, periods_per_year, n_boot=n_boot)
    scorecard["sharpe_ci_95"] = [sh_lo, sh_hi]

    # Mean signed return CI
    mr_lo, mr_hi = bootstrap_ci(signed.dropna(), np.mean, n_boot=n_boot)
    scorecard["avg_signed_return_ci_95"] = [mr_lo, mr_hi]

    # Alpha CI (signed alpha = direction × alpha_return)
    if "alpha_return" in df.columns:
        alpha_signed = sig_dir * pd.to_numeric(df["alpha_return"], errors="coerce")
        al_lo, al_hi = bootstrap_ci(alpha_signed.dropna(), np.mean, n_boot=n_boot)
        scorecard["avg_alpha_ci_95"] = [al_lo, al_hi]

    return scorecard
