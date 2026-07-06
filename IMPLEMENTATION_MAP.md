# FinRobot Equity — Full Implementation Map

_A complete, execution-ready program for taking `finrobot_equity` from
"promising research code with correctness defects" to a **trustworthy,
reproducible, tradeable** systematic-equity research platform._

This map operationalizes **[REVIEW.md](REVIEW.md)** (findings) and
**[PLAN.md](PLAN.md)** (roadmap). It is the single source of truth for *what to
build, in what order, and how to know it's done*.

- **Legend:** ✅ done (PR #1) · 🔜 next · ⏳ later · 🔬 needs a live data check
- **Effort:** S ≈ ½ day · M ≈ 1–2 days · L ≈ 3–5 days · XL ≈ 1–2 weeks
- Finding IDs (`#n`) reference REVIEW.md.

---

## 0. Where we are now

**Trustworthy after PR #1:** factor direction under cross-section scoring,
metric annualization, per-period Sharpe/drawdown, realized-return guards, TC
sign. Test suite: 193 green + 11 new regression tests.

**Still blocking "trust the numbers":** the adjusted-vs-raw price mismatch (#2)
can still corrupt any split-spanning return, and performance is still reported
**gross of costs** with an equal-weight (not constructed) portfolio.

**Not yet reviewed at depth:** the LLM report generator (`create_equity_report`,
`equity_agents/**`) and the FastAPI `web_app/**`. They get a dedicated review
gate (Phase 5) before any hardening work.

---

## 1. Target architecture (end state)

```
            ┌──────────────────────── config (typed, versioned) ───────────────────────┐
            │  universe · date grid · horizon · frequency · costs · data sources · seed │
            └───────────────────────────────────────────────────────────────────────────┘
 PREDICT                                                              SCORE (as-of + horizon later)
 ┌───────────────┐  ┌────────────┐  ┌───────────────┐  ┌───────────┐   ┌────────────────┐  ┌─────────────┐
 │ data adapters │→ │ PIT filter │→ │ factor library│→ │ cross-sec │→  │ realized returns│→ │  scorecard  │
 │ (FMP, prices) │  │ (filing    │  │ (raw + score) │  │ z-score of│   │ (adjusted, PIT- │  │ (net-of-cost│
 │  cached, PIT) │  │  date)     │  │               │  │  score)   │   │  safe, flagged) │  │  portfolio) │
 └───────────────┘  └────────────┘  └──────┬────────┘  └─────┬─────┘   └────────┬───────┘  └──────┬──────┘
                                           │ signal_engine ← ┘                  │                 │
                                           ▼                                    ▼                 ▼
                                    portfolio_construction  ───────────►  position sizing   reporting + audit
                                    (long/short balance, caps)            (Kelly/vol-target)  (markdown/JSON/CI)
```

**Design invariants the whole program must preserve**

1. **No look-ahead:** every input to a signal at date *t* was public at *t*.
2. **Direction integrity:** a "better" fundamental always moves the composite the
   same way, at every stage (raw → score → z-score → composite).
3. **Portfolio-first metrics:** all risk/return stats are properties of a
   date-ordered portfolio return series, net of costs.
4. **Reproducibility:** a run is a pure function of (config, data snapshot, seed).
5. **Fail loud:** missing data degrades to `None`/non-zero exit, never a
   fabricated number or a green "0 predictions" run.

---

## 2. Phase plan (milestones & exit criteria)

| Phase | Theme | Milestone / exit criteria | Items |
|---|---|---|---|
| **0** ✅ | P0 correctness | Signals correctly signed; metrics correctly annualized & order-independent | #1, #3, #4, #5, #6, #7, #8 |
| **1** 🔜 | Remaining correctness | Every realized return uses **adjusted** prices; no silent data loss; runs fail loud | #2, #9, #10, #11, #12, #13, #14, #15, #16, #17 |
| **2** ⏳ | Tradeable methodology | Metrics are **net of costs** on a **constructed** long/short book; sharper IC | cost-netting, portfolio wiring, #18, #19, #20, #21, #22, continuous-IC (#26) |
| **3** ⏳ | Reproducibility & infra | Typed config; end-to-end integration test; CI; caching; docs | config object, integration test, CI, data cache, quant_eval README |
| **4** ⏳ | Research surface | Multi-horizon, multi-universe studies; factor research tooling; dashboards | universe expansion, factor attribution UX, ablations |
| **5** ⏳ | Report gen + web app | Review gate → hardening of `create_equity_report`, `equity_agents`, `web_app` | (scoped after review) |

**Hard gates:** Phase 2 metrics are only meaningful once Phase 1 #2 lands
(adjusted prices). Phase 4 studies are only credible once Phase 3 reproducibility
lands.

---

## 3. Workstream detail

### Phase 1 — Remaining correctness (🔜)

| # | Item | Files | Approach | Tests / acceptance | Effort |
|---|---|---|---|---|---|
| **2** 🔬 | Adjusted vs raw price mismatch | `price_sources/dukascopy_cli.py`, `yahoo_finance.py`, `realized_returns.py` | **First** confirm Dukascopy `d1` equity adjustment convention against a real CSV. Then make both sources return the **same** basis (total-return adjusted preferred). Add an `adjusted: bool` contract to `load_price_frame` and assert consistency between ticker & benchmark. If Dukascopy can't be adjusted, route equities through Yahoo-adjusted and keep Dukascopy for FX only. | Split-window regression: NVDA 2024-05-31 +60d must be ≈ +7%, not −89%. Property test: benchmark and ticker use identical adjustment. | L |
| **9** | Benchmark negative caching | `realized_returns.py:_get_benchmark_return` | Cache only *successful* results; on exception, retry-once then leave uncached (distinguish "no data" `None` from "transient error"). Key includes source. | Unit: one raised exception on date D doesn't null a second ticker's alpha on D. | S |
| **10** | PIT uses estimate over real filing date | `pit_filter.py:_infer_date_column`, `filter_to_pit` | Prefer `acceptedDate`/`fillingDate` when present (use directly, no lag); fall back to `date`+lag only when absent. Add `period` inference (annual vs quarterly) per statement. | Extend `test_pit_filter.py`: late filer (real filing = end+80d) is excluded at end+60d. | M |
| **11** | Silent NaT-row drop | `pit_filter.py` | Split unparseable dates from genuinely-future rows; log/annotate dropped-unparseable count; never return an all-empty frame silently (raise or flag). | Unit: a `period="FY"` date column raises/flags instead of emptying financials. | S |
| **12** | z-score vs tanh scale mixing on fallback | `signal_engine.py:_effective_score`, `cross_section.py` | Put the tanh fallback on the same scale as z-scores (standardize both to unit-ish range), or drop factors with no cross-section value from the composite and renormalize weights. | Unit: composite of a 4-ticker universe (all fallback) matches a ≥5-ticker universe's scale. | M |
| **13** | `--min-pattern-count` not wired | `score_walkforward_eval.py`, `scoring.py`, `decision_audit.py:summarize_patterns` | Thread `min_count` into `summarize_patterns` qualification, not just display slicing. | Unit: `min_count=5` drops 2-sample patterns from `right_patterns`. | S |
| **14** | Silent empty run exits 0 | `run_walkforward_eval.py`, `score_walkforward_eval.py` | Exit non-zero (or raise) when `num_predictions==0` or all fetches failed; add a `--allow-empty` escape hatch. Summarize per-ticker failures. | Integration: all-fetch-fail run exits ≠ 0. | S |
| **15** | Risk flags stripped when any CS scores present | `signal_engine.py:220` | Only strip `factor_unavailable` for factors that actually *got* a cross-section value; keep flags for genuinely-missing factors. | Unit: a factor missing for one ticker still flags for that ticker under CS mode. | S |
| **16** | NaN realized return mislabeled "wrong" | `score_walkforward_eval.py:126`, `decision_audit.py:classify_outcome` | Treat `NaN` like `None` (→ "unknown"); guard with `pd.isna`. | Unit: a blank `realized_return` on `--skip-price-download` → "unknown", not "wrong". | S |
| **17** 🔬 | Dukascopy timestamp epoch-ms assumption | `dukascopy_cli.py:98` | Detect format (int-ms vs ISO-8601) before parse; surface parse failures instead of swallowing. Pin/verify `dukascopy-node` version. | Unit over both CSV shapes; integration asserts non-empty frame or explicit error. | S |

**Phase 1 exit:** every scored return is adjustment-correct and PIT-safe; no code
path silently produces zeros/empties; runs fail loud. Re-run the AAPL walk-forward
and diff the scorecard vs pre-fix to quantify the correction.

### Phase 2 — Tradeable methodology (⏳)

| Item | Files | Approach | Tests / acceptance | Effort |
|---|---|---|---|---|
| **Cost-netting into performance** | `performance_metrics.py`, `scoring.py`, `transaction_costs.py`, `portfolio_construction.py` | Compute per-period turnover from the constructed book, apply `estimate_tc` per name/tier, subtract from period returns → **net** Sharpe/returns alongside gross. | Golden test: net Sharpe < gross Sharpe by ≈ turnover×tc. | M |
| **Wire portfolio construction into scored path** | `run_walkforward_eval.py`, `portfolio_construction.py`, `position_sizing.py` | Replace equal-weight signed returns with a constructed long/short book (position + sector + gross caps, sizing). Persist weights per date. | Integration: scored returns == weighted book returns. | L |
| **#20 long/short balance / net-neutrality** | `portfolio_construction.py` | Enforce net exposure target (e.g. dollar-neutral) in addition to gross cap; fix the `test_gross_cap_flag_set_when_scaled` scenario or the sector/gross ordering it exposes. | Fix the pre-existing failing test; add net-exposure test. | M |
| **#26 continuous IC** | `performance_metrics.py:information_coefficient`, `ic_series` | Use the continuous composite score (not discrete {-1,0,1}) for Spearman IC; keep signed-return metrics on the discrete action. | Unit: IC on a monotone signal ≈ 1.0. | S |
| **#18 exact 12-1 momentum** | `factor_library.py:factor_price_momentum_12m1m` | Use skip-month ratio `P_{t-21}/P_{t-252} − 1`. | Unit vs a hand-computed series. | S |
| **#19 eps_growth negative-base hurdle** | `factor_library.py:factor_eps_growth` | Normalize slope by price/scale so the hurdle is comparable across names; document the convention. | Unit: two names, same % EPS trajectory, different price → same score. | S |
| **#21 rf on dollar-neutral book** | `performance_metrics.py:sharpe_ratio` | Don't subtract rf from an already self-financing long/short spread; make rf handling explicit to the book type. | Unit: dollar-neutral book Sharpe uses rf=0. | S |
| **#22 turnover ordering** | `performance_metrics.py:turnover` | Compute per-ticker, date-sorted signal change; wire into cost model. | Unit: turnover invariant to row order, correct per-name. | S |

**Phase 2 exit:** the scorecard reports **net-of-cost** performance of an actual
constructed book; IC uses the continuous signal. This is the first point the
numbers are decision-grade.

### Phase 3 — Reproducibility & infra (⏳)

| Item | Files | Approach | Tests / acceptance | Effort |
|---|---|---|---|---|
| **Typed config object** | new `quant_eval/config.py`, both drivers | One `EvalConfig` (pydantic) for universe, grid, horizon(s), `periods_per_year`, costs, sources, seed. Drivers build it once; downstream reads it (kills the frequency-mismatch class of bug). | Round-trip serialize; drivers accept `--config config.yaml`. | M |
| **End-to-end integration test** | new `quant_eval/tests/test_integration_walkforward.py` | Tiny fixture universe (≥5 synthetic tickers, canned FMP + price frames, no network) → predict → score → assert factor signs, scorecard keys, annualization, net-of-cost < gross. This is the guard the P0 bug class needed. | Deterministic scorecard on fixtures. | L |
| **Data layer + caching** | `price_sources/**`, new `data/cache.py` | Content-addressed cache of FMP + price pulls keyed by (symbol, window, source, adjust); offline replay mode for tests/CI. | CI runs the integration test with zero network. | M |
| **CI** | `.github/workflows/ci.yml` | Lint + `pytest quant_eval/tests` on PR; upload scorecard artifact for the fixture run. | Green required check on PRs. | S |
| **`quant_eval` README** | `quant_eval/README.md` | Document the pipeline, invariants, and the config/CLI. (It is currently undocumented — this map + PLAN are the interim reference.) | Docs render; commands runnable. | S |
| **#27 date-column edge** | `pit_filter.py` | Reject `calendarYear`/`period` as availability dates (year-only → nonsensical lag). | Unit from REVIEW #27. | S |
| **#28 universe hygiene** | `universe.py` | Reconcile `COMPANY_NAMES` vs `SECTORS`/`DEFAULT_UNIVERSE` (TSLA); add a consistency test. | Unit: every name has a sector; every universe member has a name. | S |

**Phase 3 exit:** a run is `f(config, snapshot, seed)`; CI proves it on fixtures
with no network; the frequency/units bug class is structurally impossible.

### Phase 4 — Research surface (⏳)

| Item | Approach | Effort |
|---|---|---|
| Multi-horizon / multi-universe studies | Parameterize grid & universe via config; batch runs; compare scorecards | M |
| Factor ablation & attribution UX | Per-factor IC/attribution already exists (`factor_attribution.py`); add leave-one-out ablation and a comparison report | M |
| Signal decay surface | Extend `signal_decay.py` reporting to a horizon×factor grid | S |
| Walk-forward stability | Rolling scorecard over time; regime splits | M |

### Phase 5 — Report generator + web app (⏳, review-gated)

1. **Review gate (prereq):** run the same adversarial pass over
   `create_equity_report.py`, `modules/**`, `equity_agents/**`, and `web_app/**`
   (auth, admin, DB, request logging) before changing them. Focus for the app:
   authz on admin routes, SQL/ORM misuse, secret handling, template injection,
   job-state races. Focus for the generator: prompt-injection from fetched
   news/filings, hallucinated numbers vs the computed financials, PDF/HTML
   rendering safety.
2. **Then** scope hardening from those findings (separate map section).

_Effort for Phase 5 is TBD pending the review gate — do not estimate blind._

---

## 4. Dependency graph (build order)

```
#2 (adjusted prices) ─┬─► Phase 2 cost-netting ─► Phase 2 portfolio wiring ─► net-of-cost scorecard
                      └─► credible realized returns
#14/#16 (fail loud)   ──► integration test (Phase 3) is meaningful
config object (P3) ───┬─► kills #3-class bugs permanently
                      └─► multi-horizon/universe studies (Phase 4)
integration test ─────► CI ─────► safe to accept external PRs / refactor
review gate (P5) ─────► report-gen & web-app hardening
```

**Critical path:** `#2 → cost-netting → portfolio wiring → net scorecard`. Everything
else parallelizes around it. `#9–#17` are independent small fixes that can land in
any order (good first-PR batch).

---

## 5. Testing strategy (per layer)

- **Unit (leaf math):** already strong (~1,700 lines). Extend per item above.
- **Property/invariant:** direction integrity (better fundamental → higher
  composite), order-independence of portfolio metrics, PIT (no future rows),
  adjustment consistency (ticker vs benchmark).
- **Golden-file:** freeze a fixture-run scorecard JSON; diff on every change so
  metric shifts are intentional and reviewed.
- **Integration:** the Phase-3 no-network end-to-end run (the missing guard).
- **Regression:** `test_p0_correctness_fixes.py` (added PR #1) grows with each phase.

---

## 6. Data & config contracts (to formalize in Phase 3)

- **`EvalConfig`** — universe, `grid_freq` ("ME"/"BME"), `start`/`end`,
  `horizons`, `periods_per_year`, `cost_model`, `price_source`, `adjust`,
  `filing_lags`, `seed`.
- **Price frame** — `timestamp` (tz-aware UTC), `close_eval`, `adjusted: bool`,
  `source`, `symbol`. `load_price_frame` guarantees the contract or raises.
- **Prediction row** — `ticker`, `as_of_date`, `horizon_days`, `signal`,
  `confidence`, `composite_score`, factor breakdown.
- **Scored row** — prediction + `realized_return`, `horizon_complete`,
  `benchmark_return`, `alpha_return`, `signed_return`, `outcome_label`,
  net-of-cost return, book weight.

---

## 7. Risk register & open questions

| Risk / question | Impact | Mitigation |
|---|---|---|
| **#2**: is Dukascopy `d1` equity adjusted? (🔬 can't verify without a live pull) | Determines whether #2 is a one-line source swap or a data re-architecture | Confirm against a real CSV before committing an approach; gate Phase 2 on it |
| Overlapping 60d windows sampled monthly | Autocorrelated returns inflate t-stats/Sharpe | Report both overlapping and non-overlapping; note in scorecard |
| FMP data licensing / rate limits for a wide universe | Blocks Phase 4 scale | Cache layer (Phase 3) + backoff; document limits |
| Short/borrow realism | Net returns optimistic for shorts | Add borrow-cost term to the cost model in Phase 2 |
| Survivorship in `universe.py` (static list) | Backtest bias | Phase 4: point-in-time universe membership |
| Report-gen hallucination vs computed numbers | Wrong figures in client-facing reports | Phase 5 review gate + number-grounding checks |

---

## 8. Suggested PR sequence

1. **PR #1 ✅** — P0 correctness (#1,#3,#4,#5,#6,#7,#8) + regression tests. *(merged/open)*
2. **PR #2** — Phase 1 "fail loud + PIT" batch: #9,#11,#13,#14,#15,#16,#27,#28 (all S, independent).
3. **PR #3** — #2 price-adjustment (after the 🔬 data check) + #17. *Critical path.*
4. **PR #4** — #10 (real filing dates), #12 (scale mixing).
5. **PR #5** — Phase 2 cost-netting + continuous IC (#26) + #18,#19,#21,#22.
6. **PR #6** — Phase 2 portfolio wiring + #20 net-neutrality.
7. **PR #7** — Phase 3 config object + integration test + CI + caching + quant_eval README.
8. **PR #8+** — Phase 4 studies; **PR (gated)** Phase 5 review + hardening.

Each PR: green `pytest quant_eval/tests`, a golden-file diff of the fixture
scorecard, and an update to the "Remediation status" block in REVIEW.md.

---

## 9. Definition of done (program level)

- All REVIEW.md CONFIRMED findings fixed; PLAUSIBLE findings confirmed or closed.
- Scorecard reports **net-of-cost** performance of a **constructed** long/short
  book, correctly annualized, on **adjusted** prices, with no look-ahead.
- A run is reproducible from `(config, snapshot, seed)` and proven by a
  no-network integration test in CI.
- `quant_eval` is documented; the report generator and web app have passed their
  own review gate and hardening.
