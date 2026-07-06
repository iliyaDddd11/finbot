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
    signal_dir = _signal_to_numeric(signals)
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

    # Prefer the continuous composite score over the discrete {-1,0,1} action —
    # a rank correlation on 3 buckets is far coarser than on the raw signal.
    signal_col = "composite_score" if "composite_score" in df.columns else "signal"

    results = {}
    for date, grp in df.groupby("as_of_date"):
        ic = information_coefficient(grp[signal_col], grp.get("realized_return", pd.Series()))
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

    # Seed the equity curve at 1.0 (the pre-first-return peak) so a monotonic
    # decline is measured from the start, not from the level after the first
    # period.  Without this, [-0.10]*4 reports -27% instead of the true -34%.
    cum = (1 + r).cumprod()
    cum = pd.concat([pd.Series([1.0]), cum], ignore_index=True)
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
    round_trip_bps: float = 20.0,
) -> dict:
    """
    Full quant scorecard from a scored predictions DataFrame.
    DataFrame must have columns:
      signal, realized_return, outcome_label, confidence, as_of_date

    round_trip_bps: assumed round-trip transaction cost used to report
    net-of-cost performance alongside the gross figures.
    """
    known = df[df["outcome_label"].isin(["right", "wrong"])].copy()
    has_returns = "realized_return" in df.columns

    # Signed returns: signal direction × realized return (one row per bet)
    if has_returns:
        sig_dir = _to_numeric_signal(df["signal"])
        signed  = sig_dir * pd.to_numeric(df["realized_return"], errors="coerce")
        # Portfolio return per rebalance date — the correct basis for Sharpe /
        # drawdown / Calmar (see portfolio_period_returns).
        port    = portfolio_period_returns(df)
        net     = net_period_returns(df, round_trip_bps)
        turn    = portfolio_turnover(df)
    else:
        signed = pd.Series(dtype=float)
        port   = pd.Series(dtype=float)
        net    = pd.Series(dtype=float)
        turn   = pd.Series(dtype=float)

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
        # A signed long/short book is self-financing, so no risk-free drag (#21).
        "sharpe_ratio":        sharpe_ratio(port, periods_per_year, risk_free=0.0),
        "max_drawdown":        max_drawdown(port),
        "calmar_ratio":        calmar_ratio(port, periods_per_year),
        # Net-of-cost performance (round_trip_bps applied to per-period turnover)
        "avg_turnover":        _safe_mean(turn),
        "avg_net_return":      _safe_mean(net),
        "net_sharpe_ratio":    sharpe_ratio(net, periods_per_year, risk_free=0.0),
        "round_trip_bps":      float(round_trip_bps),
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
            # Per-ticker Sharpe is a time series → date-sorted, one return/period.
            sg_ts = portfolio_period_returns(g)
            known_g = g[g["outcome_label"].isin(["right", "wrong"])]
            n_r = (known_g["outcome_label"] == "right").sum()
            n_k = len(known_g)
            by_ticker[str(ticker)] = {
                "n":           int(len(g)),
                "hit_rate":    round(n_r / n_k, 4) if n_k else None,
                "avg_signed":  _safe_mean(sg),
                "sharpe":      sharpe_ratio(sg_ts, periods_per_year),
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


def _signal_to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a signal column to numeric. If it is already numeric (e.g. a
    continuous composite_score), use it as-is; otherwise map the discrete
    long/short/neutral labels to +1/-1/0.
    """
    num = pd.to_numeric(series, errors="coerce")
    if len(series) and num.notna().mean() >= 0.5:
        return num
    return _to_numeric_signal(series)


def portfolio_period_returns(df: pd.DataFrame) -> pd.Series:
    """
    Collapse per-(ticker, date) signed returns into ONE portfolio return per
    rebalance date: the equal-weighted mean signed return across names on that
    date, indexed and sorted by ``as_of_date``.

    Sharpe / drawdown / Calmar must be computed on this time series, not on the
    pooled per-row returns — otherwise the "return stream" is a pile of
    independent single-name bets in arbitrary DataFrame order, which makes
    drawdown order-dependent and turns the Sharpe denominator into
    cross-sectional dispersion rather than portfolio volatility.
    """
    if "realized_return" not in df.columns:
        return pd.Series(dtype=float)
    signed = _to_numeric_signal(df["signal"]) * pd.to_numeric(df["realized_return"], errors="coerce")
    if "as_of_date" not in df.columns:
        return signed.dropna().reset_index(drop=True)
    tmp = pd.DataFrame({"as_of_date": df["as_of_date"].values, "signed": signed.values}).dropna(subset=["signed"])
    if tmp.empty:
        return pd.Series(dtype=float)
    return tmp.groupby("as_of_date")["signed"].mean().sort_index()


def portfolio_turnover(df: pd.DataFrame) -> pd.Series:
    """
    One-way turnover per rebalance date for the equal-weight signed book
    (weight_i = signal_dir_i / n_names_that_date, consistent with
    portfolio_period_returns). Turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|;
    the first date establishes the book from cash (w_{-1}=0). Indexed by date.
    """
    if "as_of_date" not in df.columns:
        return pd.Series(dtype=float)
    tmp = pd.DataFrame({
        "as_of_date": df["as_of_date"].values,
        "ticker":     df["ticker"].values if "ticker" in df.columns else range(len(df)),
        "dir":        _to_numeric_signal(df["signal"]).values,
    }).dropna(subset=["dir"])
    if tmp.empty:
        return pd.Series(dtype=float)
    dates = sorted(tmp["as_of_date"].unique())
    prev: dict = {}
    out: dict = {}
    for d in dates:
        grp = tmp[tmp["as_of_date"] == d]
        n = len(grp)
        cur = {t: float(dr) / n for t, dr in zip(grp["ticker"], grp["dir"])} if n else {}
        names = set(cur) | set(prev)
        out[d] = 0.5 * sum(abs(cur.get(t, 0.0) - prev.get(t, 0.0)) for t in names)
        prev = cur
    return pd.Series(out).sort_index()


def net_period_returns(df: pd.DataFrame, round_trip_bps: float) -> pd.Series:
    """
    Gross per-period portfolio returns minus transaction costs:
    cost_t = turnover_t * (round_trip_bps / 10_000). Returns a date-indexed
    net-return series aligned to portfolio_period_returns.
    """
    gross = portfolio_period_returns(df)
    if gross.empty:
        return gross
    turn = portfolio_turnover(df).reindex(gross.index).fillna(0.0)
    cost = turn * (round_trip_bps / 10_000.0)
    return gross - cost


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
    # Sharpe is a property of the portfolio return time series, so bootstrap the
    # per-period portfolio returns (consistent with compute_full_scorecard).
    port = portfolio_period_returns(df) if "realized_return" in df.columns else pd.Series(dtype=float)

    # Hit rate CI
    hr_lo, hr_hi = bootstrap_hit_rate_ci(df.get("outcome_label", pd.Series()), n_boot=n_boot)
    scorecard["hit_rate_ci_95"] = [hr_lo, hr_hi]

    # Sharpe CI
    sh_lo, sh_hi = bootstrap_sharpe_ci(port, periods_per_year, n_boot=n_boot)
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
