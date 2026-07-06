# FinRobot Equity — Architecture & Roadmap (`/ultraplan`)

_Scope: the custom `finrobot_equity/` module, with primary focus on the
`quant_eval` walk-forward signal-evaluation harness. The upstream FinRobot
library is third-party and out of scope here._

> Read this together with **[REVIEW.md](REVIEW.md)** (the correctness review) and
> **[IMPLEMENTATION_MAP.md](IMPLEMENTATION_MAP.md)** (the full, phased,
> execution-ready program that operationalizes this roadmap). The roadmap below
> is prioritized by the findings in REVIEW.md.

---

## 1. What this project is

`finrobot_equity` is two systems sharing a repo:

1. **LLM equity-report generator** (`core/src/create_equity_report.py`,
   `core/src/modules/**`, `core/src/modules/equity_agents/**`) — fetches
   fundamentals from FMP, runs LLM agents per report section, and renders a
   multi-page HTML/PDF equity research report. A FastAPI app
   (`web_app/**`) wraps it with auth, an admin console, and a SQLite job store.

2. **`quant_eval` walk-forward research harness**
   (`core/src/quant_eval/**`, driven by `run_walkforward_eval.py` and
   `score_walkforward_eval.py`) — a systematic, point-in-time factor
   backtesting engine. This is the newer, correctness-critical core and the
   focus of the review. It is **not** described in the module README, so this
   document is the reference for it.

---

## 2. `quant_eval` end-to-end data flow

```
                        run_walkforward_eval.py  (PREDICT)
  ┌──────────────────────────────────────────────────────────────────────┐
  │ build_month_end_grid(start,end)        → monthly as_of dates (freq=ME) │
  │ universe.py                            → ticker set + names + sectors  │
  │  for each as_of_date:                                                  │
  │    for each ticker:                                                    │
  │      research_snapshot.build_snapshot  → price window [lookback, as_of]│
  │      run_financial_analysis_pipeline   → FMP fundamentals (FMP API)    │
  │      pit_filter.filter_financial_data  → drop rows not yet public      │  ← PIT guard
  │      factor_library.compute_raw_factors→ 10 raw FactorResults / ticker │
  │    PHASE 2 (if ≥5 tickers):                                            │
  │      cross_section.zscore_factor_bank  → cross-sectional z-scores      │  ← see REVIEW #1
  │    PHASE 3:                                                            │
  │      signal_engine.compute_composite   → long/short/neutral + conf.    │
  │      → prediction.json / predictions.csv                              │
  └──────────────────────────────────────────────────────────────────────┘
                        score_walkforward_eval.py  (SCORE, run later)
  ┌──────────────────────────────────────────────────────────────────────┐
  │ realized_returns.compute_realized_outcome → entry/exit close, return, │
  │      SPY benchmark, alpha, path max up/down (forward prices)          │  ← see REVIEW #4–6
  │ scoring.enrich_scored_predictions        → signed_return, outcome     │
  │ scoring.compute_scorecard                → hit rate, IC, Sharpe, …    │  ← see REVIEW #2,#3
  │   ├─ performance_metrics.compute_full_scorecard                       │
  │   ├─ performance_metrics.add_bootstrap_cis                            │
  │   ├─ decision_audit.summarize_patterns                               │
  │   └─ signal_decay / factor_attribution (multi-horizon IC)            │
  │ reporting.write_markdown_report          → decision_audit.md          │
  └──────────────────────────────────────────────────────────────────────┘
```

Price data comes from **Dukascopy** (`price_sources/dukascopy_cli.py`, via a
Node CLI) with a **Yahoo Finance** fallback; symbols are mapped in
`dukascopy_symbol_map.py`. Timestamps are tz-aware UTC.

---

## 3. Component inventory (`quant_eval`)

| Area | Files | Role |
|---|---|---|
| Universe & calendar | `universe.py`, `date_grid.py` | Ticker set, sectors, month-end grid |
| Data snapshot | `research_snapshot.py`, `schemas.py` | Price window capture as of a date |
| PIT guard | `pit_filter.py` | Drop fundamentals not yet public |
| Factors | `factor_library.py` (10 factors) | Raw metric + direction-encoded tanh score |
| Cross-section | `cross_section.py` | Per-date z-score across the universe |
| Signal | `signal_engine.py` | Weighted composite → long/short/neutral |
| Sizing/portfolio | `position_sizing.py`, `portfolio_construction.py` | Kelly/vol-target sizing, caps (not yet in the scored path) |
| Realized outcomes | `realized_returns.py`, `price_sources/**` | Forward returns vs entry, SPY alpha |
| Scoring | `scoring.py`, `performance_metrics.py`, `signal_decay.py`, `factor_attribution.py` | Scorecard, IC/ICIR, Sharpe, bootstrap CIs |
| Costs | `transaction_costs.py` | Illustrative TC (not netted into metrics) |
| Audit/report | `decision_audit.py`, `reporting.py` | Pattern mining, markdown scorecard |

---

## 4. Design strengths

- **PIT discipline exists and is wired in** — fundamentals are filtered to
  public-as-of-date before factors are computed (`run_walkforward_eval.py:215`).
- **Cross-sectional, point-in-time normalization** — z-scores use only the
  live universe at each date; no full-sample leakage.
- **Statistical honesty scaffolding** — bootstrap CIs, IC decay across
  horizons, per-factor attribution, decision-pattern mining.
- **Good test coverage of leaf math** — `pit_filter`, `performance_metrics`,
  `portfolio_construction`, `signal_engine`, `factor_library` all have unit
  tests (~1,700 lines).
- **Defensive data handling** — division guards, `errors="coerce"`, graceful
  per-ticker failure isolation.

---

## 5. Roadmap (prioritized by REVIEW.md)

> **Update:** P0 items 1–5 and the #8 TC-sign fix are **implemented on this
> branch** with regression tests (`quant_eval/tests/test_p0_correctness_fixes.py`).
> #2 (price adjustment) remains a follow-up pending a live data check.

### P0 — Correctness (metrics/signals are currently wrong; fix before trusting any output)
1. **Fix cross-section z-score sign/semantics** (REVIEW #1). Z-score the
   direction-normalized `score` (or carry a per-factor sign + component
   vector), not the raw metric. Today `leverage` and `valuation_vs_history`
   are **inverted** and `margin_quality`/`roic` lose their trend component on
   the default (≥5-ticker) path.
2. **Fix annualization frequency** (REVIEW #2). The grid is monthly →
   `periods_per_year` must be 12, not the hard-coded default 6. Thread the true
   frequency from the driver into `compute_scorecard`/`add_bootstrap_cis`.
3. **Aggregate to a per-period portfolio return before Sharpe/drawdown**
   (REVIEW #3). Today these are computed over pooled per-(ticker,date) rows in
   DataFrame order, making them order-dependent and conflating cross-sectional
   dispersion with time-series risk.
4. **Stop fabricating 0.0 realized returns** and **flag partial horizons**
   (REVIEW #4, #5). Return `None` when there is no post-anchor data, and mark
   returns whose window is shorter than the requested horizon.
5. **Anchor download must start before `as_of_date`** (REVIEW #6) so month-end
   dates that fall on weekends/holidays aren't silently dropped.

### P1 — Correctness (latent or narrower blast radius)
6. Fix the **short-side transaction-cost sign** in `net_return_after_tc`
   (REVIEW #7) before it is ever wired into the scored path.
7. **PIT: prefer the real filing date** (`fillingDate`/`acceptedDate`) over the
   fixed 60/45-day estimate, and **don't silently drop NaT-dated rows**
   (REVIEW #8, #9).
8. Cache benchmark **failures separately** from legitimate `None` so one
   transient SPY download error doesn't null a whole date's alphas (REVIEW #10).

### P2 — Methodology & completeness
9. **Net transaction costs into performance** (currently gross-of-cost) so
   Sharpe/returns reflect a tradeable book; wire `position_sizing` and
   `portfolio_construction` into the scored path instead of equal-weighting
   raw signals.
10. Replace the discrete {-1,0,1} signal in the IC with the **continuous
    composite score** for a sharper Information Coefficient.
11. Make **momentum exact** (skip-month ratio, not simple-return subtraction)
    and reconsider the `eps_growth` negative-base hurdle (REVIEW P2 items).
12. Enforce **long/short balance / net-neutrality** in `construct_portfolio`,
    which currently only caps gross.

### P3 — Engineering
13. Add an **end-to-end integration test** over a tiny fixture universe that
    asserts factor signs, scorecard keys, and annualization — the class of bug
    in P0 survives today precisely because only leaf functions are unit-tested.
14. Introduce a typed **config object** for horizon/frequency/rebalance so the
    frequency mismatch can't recur.
15. Document `quant_eval` in the module README (it is currently undocumented).

---

## 6. Suggested sequencing

`P0.1 → P0.2 → P0.3` unlock trustworthy signals and metrics and are the
highest ROI. `P0.4/P0.5` clean the realized-return inputs those metrics feed
on. Land `P3.13` (integration test) alongside P0 so the fixes are pinned. P1/P2
follow once the core is trustworthy.
