# quant_eval — walk-forward equity signal evaluation

A point-in-time, systematic factor research harness. It generates long/short/
neutral **predictions** on a monthly walk-forward grid, then **scores** them
against realized forward prices and reports a full quant scorecard.

> For the architecture, findings, and roadmap see the repo-root
> **[PLAN.md](../../../../PLAN.md)**, **[REVIEW.md](../../../../REVIEW.md)**, and
> **[IMPLEMENTATION_MAP.md](../../../../IMPLEMENTATION_MAP.md)**.

## Pipeline

```
run_walkforward_eval.py  (PREDICT)
  month-end grid → universe → per ticker:
    build_snapshot (prices)  →  FMP fundamentals  →  PIT filter
      →  compute_raw_factors (10 factors: raw metric + direction-encoded score)
  per date (≥5 tickers): zscore_factor_bank  →  compute_signal_from_raw
  → predictions.csv

score_walkforward_eval.py  (SCORE, run later)
  compute_realized_outcome (adjusted forward prices, SPY alpha, path stats)
  → enrich_scored_predictions → compute_scorecard
  → scorecard.json + decision_audit.md
```

## Design invariants

1. **No look-ahead** — fundamentals are filtered to their public-as-of date
   (`pit_filter`, preferring the real filing date); prices are clipped at `as_of`.
2. **Direction integrity** — cross-sectional scoring z-scores the *direction-
   normalized* factor score, so "better fundamental" always moves the composite
   the same way (see `cross_section.py`).
3. **Portfolio-first, net-of-cost metrics** — Sharpe / drawdown / Calmar are
   computed on the per-rebalance-date portfolio return (`portfolio_period_returns`),
   net of transaction costs (`net_period_returns`), annualized at the true
   frequency (monthly → 12).
4. **Fail loud** — missing data yields `None`, and an empty run exits non-zero
   unless `--allow-empty`.

## Quick start

```bash
cd finrobot_equity/core/src

# 1. Predict (monthly grid). --universe small = ~20 names; default = 61.
python run_walkforward_eval.py --start 2023-01-01 --end 2024-12-31 \
    --universe small --config-file ../config/config.ini

# 2. Score the predictions against realized prices.
python score_walkforward_eval.py <output-root> \
    --periods-per-year 12 --min-pattern-count 3
```

Key flags: `--periods-per-year` (annualization; default 12 for the monthly
grid), `--min-pattern-count` (pattern qualification threshold), `--allow-empty`
(don't fail on zero predictions), `--horizons` (IC-decay horizons).

## Prices & adjustment

`price_sources.load_price_frame(prefer_adjusted=True)` (the default) routes all
equities through the split/dividend-**adjusted** Yahoo feed so returns are on one
consistent basis. Raw Dukascopy is available via `prefer_adjusted=False` (verify
its adjustment convention first — see REVIEW.md #2). Frames carry a
`price_adjusted` flag.

## Config

`config.EvalConfig` bundles the parameters that must stay consistent across
stages (grid frequency → `periods_per_year`, horizons, cost, seed).

## Tests

```bash
cd finrobot_equity/core/src && python -m pytest quant_eval/tests -q
```

No network required. `tests/test_integration_pipeline.py` runs the full
predict→score path on synthetic fixtures; `tests/test_p0_correctness_fixes.py`,
`test_phase1_fixes.py`, and `test_phase2_fixes.py` pin the REVIEW.md fixes. CI
runs the suite on every PR (`.github/workflows/quant_eval-ci.yml`).
