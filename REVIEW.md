# FinRobot `quant_eval` — Correctness Review (`/ultrareview`)

_Adversarial correctness review of the walk-forward signal-evaluation harness
(`finrobot_equity/core/src/quant_eval/**` + `run_walkforward_eval.py` /
`score_walkforward_eval.py`). Focus: look-ahead bias, return computation,
performance-metric methodology, portfolio/costs, and orchestration wiring._

Each finding is marked **CONFIRMED** (traced in source) or **PLAUSIBLE**
(depends on runtime/data not executable here). Line numbers are against the
committed source. See **[PLAN.md](PLAN.md)** for the remediation roadmap; the
`#` numbers below are referenced there.

**Headline:** the harness has PIT discipline and good scaffolding, but two
**critical** defects mean the current signals and performance numbers cannot be
trusted as-is: the cross-sectional scoring silently **inverts two factors**, and
mixing adjusted and raw price sources makes any split-spanning return wrong.

---

## CRITICAL

### #1 — Cross-section z-scoring inverts `leverage` & `valuation`, and drops the trend component of `margin_quality` & `roic` — CONFIRMED

`cross_section.py:54-55,68-72` z-scores each factor's **`raw_value`** across the
universe under a hard-coded "higher raw = more bullish" convention. But
`raw_value` is a *descriptive metric*; each factor's true direction is encoded
only in its tanh `score`. Four of the ten factors don't satisfy "higher raw is
better":

- **`factor_leverage`** (`factor_library.py:205-209`): `raw_value` = Net
  Debt/EBITDA (higher = **worse**), scored with `invert=True`. After z-scoring
  the raw ratio, the **most-levered** name gets the largest positive z and the
  biggest long tilt.
  _Failure:_ universe Net-Debt/EBITDA `[0,1,2,3,8]` → the 8× (riskiest) name is
  ranked most bullish.
- **`factor_valuation_vs_history`** (`factor_library.py:359-366`): `raw_value` =
  current EV/EBITDA (higher = **more expensive** = bearish); the intended signal
  is `discount = (median − current)/median`. Z-scoring the raw multiple makes
  the **most expensive** stock most bullish — and discards the "vs own history"
  intent entirely (it becomes a cross-sectional cheap/expensive comparison).
  _Failure:_ EV/EBITDA `[5,8,12,20,40]` → the 40× name scores most bullish.
- **`factor_margin_quality`** (`factor_library.py:162-164`) and **`factor_roic`**
  (`factor_library.py:241-243`): the tanh score blends level **and trend**
  (`0.6·level+0.4·trend`, `0.7·level+0.3·trend`), but `raw_value` stores only the
  **level**. Z-scoring drops the trend — a company with declining-but-high
  margins ranks identically to one with rising margins.

This is the **default production path**: Phase 2 z-scores whenever ≥5 tickers are
available (`run_walkforward_eval.py:243`), and the composite uses the z-scores in
place of the tanh scores (`signal_engine.py:168-173,190-193`) with no sign
correction. The module docstring calls this "how systematic equity research
works in practice," so it is intended to be on.

**Fix:** z-score the direction-normalized `score` (already in ≈[−2,2] with
correct signs), or attach a per-factor `direction` and a full component vector to
`FactorResult` and z-score those. Add a test asserting that a more-levered /
more-expensive name receives a **lower** composite.

### #2 — Adjusted vs. raw price sources are mixed; split-spanning returns are catastrophically wrong — CONFIRMED (mismatch) / PLAUSIBLE (split magnitude)

`price_sources/yahoo_finance.py:53` downloads with `auto_adjust=True`
(split/dividend-adjusted), while `price_sources/dukascopy_cli.py:118-129`
returns **raw, unadjusted** Dukascopy CSV. Routing is by symbol: the four mapped
tickers (MSFT, NVDA, AAPL, TSLA) always take the Dukascopy-raw path; SPY (the
benchmark) and everything else take the Yahoo-adjusted path.

_Failure:_ NVDA prediction `as_of=2024-05-31`, `horizon=60d`. NVDA's 10:1 split
(2024-06-10) falls inside the window. `realized_returns._compute_return_from_price_df`
reads `entry_close≈$1096` (pre-split) and `exit_close≈$117` (post-split) from the
same raw series → `realized_return ≈ −89%` vs the true ≈ +7%. TSLA's 3:1 split
(2022-08-25) does the same to any 2022 window. These feed the scorecard, IC,
Sharpe, alpha, and case studies directly.

Even with no split, adjusted (total-return) SPY vs price-only stock biases
`alpha_return` **down** by the stock's dividend yield every period.

**Verification caveat:** the split catastrophe assumes Dukascopy `d1` equity CFDs
are unadjusted — confirm against a real downloaded CSV. If they turn out to be
adjusted, this degrades from critical to the dividend-bias (medium). Either way,
**one source must not be adjusted while the other is.**

**Fix:** make both sources consistent (both total-return adjusted, ideally), and
use the same convention for the benchmark. Add a split-window regression test.

---

## HIGH

### #3 — Annualization uses `periods_per_year=6` but the grid is monthly (12) — CONFIRMED

`date_grid.build_month_end_grid` uses `freq="ME"` → **monthly** rebalancing, but
`sharpe_ratio`, `icir`, `calmar_ratio`, and `compute_full_scorecard` all default
`periods_per_year=6` ("bi-monthly"), and `scoring.compute_scorecard`
(`scoring.py:78,100`) calls them **without overriding it**. Sharpe/ICIR are
understated by `√(12/6)≈1.41×` and Calmar's annualized return by `2×`. Every
annualized number in the scorecard and its bootstrap CIs is mis-scaled.

**Fix:** thread the true rebalance frequency (12) from the driver through
`compute_scorecard` → `compute_full_scorecard`/`add_bootstrap_cis`.

### #4 — Sharpe / max-drawdown / Calmar are computed over pooled per-row returns in DataFrame order — CONFIRMED

In `compute_full_scorecard` (`performance_metrics.py:180-202`), `signed =
sig_dir * realized_return` is **one row per (ticker, date)**. It is passed
straight to `sharpe_ratio` and `max_drawdown` with no aggregation to a per-period
portfolio return and no sort:

- `max_drawdown` (`:116-125`) does `(1+r).cumprod()` in raw row order, so
  "drawdown" chains unrelated single-name returns as if sequential portfolio
  periods. **Reordering the DataFrame changes the result** → so does Calmar.
- `sharpe_ratio` (`:92-113`) takes `std(ddof=1)` over all name×date rows, so the
  denominator is cross-sectional dispersion, not portfolio volatility, and the
  `√periods_per_year` no longer matches the observation count.

**Fix:** aggregate signed returns to one portfolio return per `as_of_date`
(mean across names, long/short netted), sort by date, then compute Sharpe/DD/Calmar
on that time series.

### #5 — Insufficient forward data is reported as a fabricated `0.0` return — CONFIRMED

`realized_returns.py:66-68` → when `future` (bars after the anchor) is empty,
`exit_idx = anchor_idx`, so `entry_close == exit_close` and `realized_return =
0.0` (`:178`) with a non-None `entry_ts`. _Failure:_ a truncated/delisted series
ending at `as_of_date` yields a **fake exactly-0% horizon return** that biases IC
toward zero and pollutes the sample. Should return `None`.

### #6 — Realized return silently computed over a window shorter than the horizon — CONFIRMED

`realized_returns.py:70-71` (and the benchmark path `:125-126`): when no bar
reaches `exit_target`, `exit_idx = future.index[-1]` (last available bar). A
90-day horizon with only 30 days of trailing data stores a **30-day** return in
`realized_return_90d`, with no flag distinguishing full vs. partial windows.
This injects a time-dependent bias (later `as_of` dates have fewer trailing bars).

**Fix:** return `None` (or a `partial=True` flag) when the last bar is materially
short of `exit_target`.

### #7 — Anchor download starts at `as_of_date`, so weekend/holiday month-ends are silently dropped — CONFIRMED

`realized_returns.py:145-151` calls `load_price_frame(start_date=as_of_date, …)`,
and `_pick_anchor_row` (`:46-49`) needs a bar with `timestamp ≤ end-of(as_of)`.
Because `build_month_end_grid` emits **calendar** month-ends (`freq="ME"`), many
`as_of` dates land on weekends/holidays → the first returned bar is *after* the
cutoff → `anchor_idx=None` → an all-`None` outcome for that date. A large,
systematic fraction of evaluation periods is dropped.

**Fix:** start the download a few days **before** `as_of_date` (or snap the grid
to business month-ends, `freq="BME"`).

### #8 — Transaction costs are never netted into performance; the netting helper has a short-side sign error — CONFIRMED

`net_return_after_tc` (`transaction_costs.py:113-133`) is **never called** — the
scored path (`scoring.py`, `score_walkforward_eval.py:182`) only prints an
illustrative `tc_summary_line`. So **all reported Sharpe/returns are gross of
costs**. Worse, the unused helper is also wrong: `sign = 1 if long else -1;
gross - sign*tc` credits shorts the cost (`gross + tc`), overstating every short
round-trip by `2·tc`. _Failure:_ `gross=0.05, 20bps` → short returns `0.052`
instead of `0.048`.

**Fix:** subtract `tc` unconditionally (cost is always a cost), and net costs
into the scored returns before computing performance.

---

## MEDIUM

| # | File:line | Finding | Verdict |
|---|---|---|---|
| 9 | `realized_returns.py:130-133` | `_get_benchmark_return` caches `None` on **any** exception, keyed only on `(as_of,horizon)`. One transient SPY download error nulls every ticker's `benchmark_return`/`alpha` for that whole cross-section. | CONFIRMED |
| 10 | `pit_filter.py:52` | `_infer_date_column` prefers `date` (fiscal period-end) + a fixed 60/45-day estimate over the actual `fillingDate`/`acceptedDate` present in FMP data → real look-ahead for late filers. Use the estimate only as fallback. | CONFIRMED |
| 11 | `pit_filter.py:89,97` | `errors="coerce"` → unparseable dates become `NaT`; `(NaT+lag) ≤ cutoff` is `False`, so those rows are **silently dropped**. A statement whose only date col is `"FY"/"Q1"` → all `NaT` → empty financials returned as if legitimately pre-announcement. | CONFIRMED |
| 12 | `signal_engine.py:168-173` | When a factor's z-score is `None` (small universe / no variance), `_effective_score` falls back to the raw tanh score (~±2), mixed in the same weighted average as scaled z-scores (~±0.67 at 1σ) → composite biased toward the fallback factors. | CONFIRMED |
| 13 | `score_walkforward_eval.py:135-136`, `scoring.py:74` | `--min-pattern-count` is only used to slice the display list; `summarize_patterns` is called without it, so the qualification threshold stays hardcoded at 2. `--min-pattern-count 5` still surfaces 2-sample "patterns." | CONFIRMED |
| 14 | `run_walkforward_eval.py:337-357` | If every ticker's fetch throws (e.g. `npx`/`dukascopy-node` absent — all caught at `:193,:234`), an empty predictions.csv is written and the process **exits 0** with `num_predictions:0`. A total data outage looks like a clean run. | CONFIRMED |
| 15 | `signal_engine.py:220-221` | When any cross-section scores exist, **all** `factor_unavailable` risk flags are stripped — including for factors that genuinely have no data — hiding real missing-data flags. | CONFIRMED |
| 16 | `score_walkforward_eval.py:126`, `decision_audit.py:122-130` | Guard is `realized_return is not None`, but a float `NaN` is not `None` → `NaN>0` is `False` → a directional call gets labeled `"wrong"`, inflating the `right_rate` denominator. Triggers on `--skip-price-download` over a CSV with blank returns. | PLAUSIBLE |
| 17 | `dukascopy_cli.py:98` | `pd.to_datetime(df["timestamp"], unit="ms", utc=True)` assumes epoch-ms. If the installed `dukascopy-node` emits ISO-8601, this raises and is swallowed → all four Dukascopy tickers silently dropped. Verify against the real CSV. | PLAUSIBLE |

---

## LOW / design

| # | File:line | Finding |
|---|---|---|
| 18 | `factor_library.py:433-435` | 12-1 momentum uses `mom_12m − mom_1m` (simple-return subtraction); exact skip-month momentum is `P_{t-21}/P_{t-252} − 1`. Diverges on large moves. |
| 19 | `factor_library.py:321` | `eps_growth` negative-base branch centers slope at `0.5` — an arbitrary raw-EPS hurdle, not comparable across price levels. |
| 20 | `portfolio_construction.py:131-137` | Docstring promises long/short balance, but `construct_portfolio` only caps gross ≤150%; net neutrality is never enforced (all-long set → 150% net long). (Not yet on the scored path.) |
| 21 | `performance_metrics.py:105-106` | RF is subtracted from an already dollar-neutral signed long/short return, understating a market-neutral book's Sharpe. |
| 22 | `performance_metrics.py:159-164` | `turnover` diffs an unordered multi-ticker signal series (should be per-ticker, date-sorted). Latent — not called in the scorecard. |
| 23 | `reporting.py:162` | Prints `avg_signed=` but passes `avg_realized_return` (unsigned) → wrong sign for short-signal patterns. |
| 24 | `dukascopy_cli.py:14-35` | No `-p` flag → Dukascopy returns **bid** prices vs Yahoo trade close; small cross-source level inconsistency. |
| 25 | `decision_audit.py:126-129` | Outcomes graded on raw return, not alpha (a +1% long when SPY +5% scores `"right"`) → `right_rate` measures direction, not skill. Likely intentional. |
| 26 | `performance_metrics.py:36-52` | IC uses the discrete `{-1,0,1}` signal rather than the continuous composite score, coarsening the Information Coefficient. |
| 27 | `pit_filter.py:52` | If `date`/`Date` absent but `calendarYear` present, `to_datetime("2020")`→2020-01-01, +60d marks FY2020 data available 2020-03-01 (~a year of look-ahead). Latent behind column priority. |
| 28 | `universe.py` | `TSLA` has a `COMPANY_NAMES` entry but is absent from `SECTORS`/`DEFAULT_UNIVERSE`, so `sector_of("TSLA")` is `None`. Cosmetic. |

---

## Checked and found correct (not bugs)

- PIT filtering **is** wired in before factor computation (`run_walkforward_eval.py:215`);
  cross-section z-scoring is genuinely point-in-time (per-date, no full-sample leakage).
- Prediction ↔ realization key linkage (`ticker`/`as_of_date`/`horizon_days=60`) is consistent.
- Price loader clips at `as_of_date` (no look-ahead into the signal); timestamps are tz-aware UTC.
- `enrich_scored_predictions` short-side sign (`signed_return = −realized` for shorts) is correct.
- Division-by-zero guards, winsorization (`clip ±3 → ×2/3`), Kelly / vol-target / confidence sizing math, `estimate_tc` dimensional math, `hit_rate`, `information_coefficient`, bootstrap routines — all correct.
- Yahoo `end`-exclusivity handled via `end + 1d`; SPY memoization keys correct; currency uniformly USD.

---

## Verification status

Findings were derived by tracing the source (four independent review passes plus
direct reads of `cross_section.py`, `factor_library.py`, `performance_metrics.py`,
`pit_filter.py`, `realized_returns.py`, `scoring.py`, and the drivers). No code
was executed — items marked **PLAUSIBLE** (#2 split magnitude, #16, #17) depend
on live data/tooling and should be confirmed with a real run before remediation.
