#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from generate_financial_analysis import run_financial_analysis_pipeline
from quant_eval.cross_section import zscore_factor_bank, cross_section_summary
from quant_eval.date_grid import build_month_end_grid
from quant_eval.pit_filter import filter_financial_data_to_pit, pit_audit_line
from quant_eval.portfolio_construction import construct_portfolio, portfolio_summary, portfolio_report_lines
from quant_eval.position_sizing import compute_position_size, position_size_summary
from quant_eval.research_snapshot import build_snapshot
from quant_eval.schemas import ResearchPrediction
from quant_eval.signal_engine import compute_raw_factors, compute_signal_from_raw, SignalResult
from quant_eval.universe import DEFAULT_UNIVERSE, SMALL_UNIVERSE, get_company_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a walk-forward evaluation over the research pipeline.")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list. Overrides --universe.")
    parser.add_argument("--universe", choices=["default", "small", "custom"], default="small",
                        help="Stock universe preset. 'default'=60 stocks, 'small'=20 stocks (fast test). "
                             "Ignored if --tickers is set. (default: small)")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2024-12-31")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--years-limit", type=int, default=5)
    parser.add_argument("--news-days-back", type=int, default=5)
    parser.add_argument("--news-limit", type=int, default=25)
    parser.add_argument("--output-root", default="./output/walkforward")
    parser.add_argument("--price-lookback-days", type=int, default=365)
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Only build frozen snapshots, no signal computation.")
    parser.add_argument("--allow-empty", action="store_true",
                        help="Exit 0 even if no predictions were produced. By default a run "
                             "that yields zero predictions (e.g. every data fetch failed) "
                             "exits non-zero so an outage isn't mistaken for a clean run.")
    return parser


def _make_analysis_args(
    ticker: str,
    company_name: str,
    as_of_date: str,
    output_dir: str,
    config_file: str | None,
    years_limit: int,
    news_days_back: int,
    news_limit: int,
) -> SimpleNamespace:
    """
    Build args namespace for the financial analysis pipeline.
    as_of_date is passed explicitly so news fetching is anchored to the
    evaluation date rather than today (prevents look-ahead bias).
    """
    return SimpleNamespace(
        company_ticker=ticker,
        company_name=company_name,
        config_file=config_file,
        years_limit=years_limit,
        output_dir=output_dir,
        output_csv_name="financial_metrics_and_forecasts.csv",
        peer_tickers=[],
        generate_text_sections=False,
        text_output_dir=None,
        news_days_back=news_days_back,
        news_limit=news_limit,
        enable_sensitivity_analysis=False,
        enable_catalyst_analysis=False,
        enable_enhanced_news=False,
        revenue_growth_2025=0.05,
        revenue_growth_2026=0.06,
        revenue_growth_2027=0.04,
        margin_improvement=0.01,
        sga_margin_improvement=-0.005,
        period="annual",
        as_of_date=as_of_date,   # anchors news to evaluation date — no look-ahead
    )


def _build_prediction(
    sr: SignalResult,
    run_id: str,
    analysis_dir: str,
    raw_summary: dict,
    company_name: str,
) -> ResearchPrediction:
    """Convert a SignalResult from the signal engine into a ResearchPrediction."""
    factor_scores: dict = {}
    for fb in sr.factors:
        factor_scores[fb.name] = {
            "raw_value":  fb.raw_value,
            "score":      fb.score,
            "weight":     fb.weight_used,
            "available":  fb.available,
            "rationale":  fb.rationale,
        }

    thesis_points = [fb.rationale for fb in sr.factors if fb.available and fb.score is not None and fb.score > 0.3]
    risk_flags    = sr.risk_flags

    return ResearchPrediction(
        ticker=sr.ticker,
        company_name=company_name,
        as_of_date=sr.as_of_date,
        horizon_days=60,
        signal=sr.signal,
        confidence=sr.confidence,
        expected_return=sr.expected_return,
        composite_score=sr.composite_score,
        conviction_tier=sr.conviction_tier,
        thesis_points=thesis_points,
        risk_flags=risk_flags,
        signal_rationale=sr.signal_rationale,
        factor_scores=factor_scores,
        factors_available=sr.factors_available,
        coverage_ratio=sr.coverage_ratio,
        source_run_id=run_id,
        analysis_dir=analysis_dir,
        raw_summary=raw_summary,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve ticker universe
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe == "default":
        tickers = DEFAULT_UNIVERSE
    else:
        tickers = SMALL_UNIVERSE

    print(f"Universe: {len(tickers)} tickers — {', '.join(tickers[:8])}{'...' if len(tickers) > 8 else ''}")

    anchor_dates = build_month_end_grid(args.start, args.end)
    if not anchor_dates:
        raise ValueError("No anchor dates were generated. Check --start and --end.")

    print(f"Anchor dates: {len(anchor_dates)} periods from {anchor_dates[0]} to {anchor_dates[-1]}")
    print(f"Total runs planned: {len(tickers) * len(anchor_dates)}\n")

    prediction_rows: list[dict] = []
    snapshot_rows: list[dict] = []

    # ======================================================================
    # OUTER LOOP: by date — enables cross-sectional z-scoring across tickers
    # ======================================================================
    for as_of_date in anchor_dates:
        print(f"\n{'='*60}")
        print(f"  Date: {as_of_date}  |  Universe: {len(tickers)} tickers")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # PHASE 1: Snapshot + pipeline for all tickers at this date
        # ------------------------------------------------------------------
        snapshots: dict[str, object]       = {}
        financial_cache: dict[str, dict]   = {}
        raw_factor_bank: dict[str, dict]   = {}

        for ticker in tickers:
            company_name = get_company_name(ticker)
            run_id       = f"{ticker}_{as_of_date}"
            run_dir      = output_root / run_id
            snapshot_dir = run_dir / "snapshot"
            analysis_dir = run_dir / "analysis"

            lookback_start = (
                pd.Timestamp(as_of_date) - pd.Timedelta(days=args.price_lookback_days)
            ).strftime("%Y-%m-%d")

            try:
                snapshot = build_snapshot(
                    ticker=ticker,
                    company_name=company_name,
                    as_of_date=as_of_date,
                    lookback_start_date=lookback_start,
                    snapshot_dir=str(snapshot_dir),
                    timeframe="d1",
                )
                snapshots[ticker] = snapshot
                snapshot_rows.append(snapshot.to_dict())
                with open(run_dir / "snapshot.json", "w", encoding="utf-8") as f:
                    json.dump(snapshot.to_dict(), f, indent=2)
            except Exception as exc:
                print(f"  [WARN] snapshot failed for {ticker}: {exc}")
                continue

            if args.skip_analysis:
                continue

            try:
                analysis_args = _make_analysis_args(
                    ticker=ticker,
                    company_name=company_name,
                    as_of_date=as_of_date,
                    output_dir=str(analysis_dir),
                    config_file=args.config_file,
                    years_limit=args.years_limit,
                    news_days_back=args.news_days_back,
                    news_limit=args.news_limit,
                )
                result = run_financial_analysis_pipeline(analysis_args)
                financial_data = result.get("financial_data") or {}

                # PIT filter — remove data not yet public as of as_of_date
                financial_data, rows_dropped = filter_financial_data_to_pit(
                    financial_data, as_of_date, period="annual"
                )
                pit_line = pit_audit_line(rows_dropped, as_of_date)
                if any(v > 0 for v in rows_dropped.values()):
                    print(f"  [PIT] {ticker}: {pit_line}")

                financial_cache[ticker] = {
                    "financial_data": financial_data,
                    "raw_summary":    result.get("summary_data", {}),
                    "analysis_dir":   str(analysis_dir),
                }

                # Compute raw factors (phase 1 — no composite yet)
                raw_factor_bank[ticker] = compute_raw_factors(
                    financial_data=financial_data,
                    as_of_date=as_of_date,
                    price_csv_path=snapshot.price_csv_path,
                )
            except Exception as exc:
                print(f"  [WARN] pipeline failed for {ticker}: {exc}")

        if args.skip_analysis:
            continue

        # ------------------------------------------------------------------
        # PHASE 2: Cross-sectional z-scoring across all tickers at this date
        # ------------------------------------------------------------------
        if len(raw_factor_bank) >= 5:
            cs_scores = zscore_factor_bank(raw_factor_bank)
            cs_df     = cross_section_summary(raw_factor_bank)
            cs_path   = output_root / f"cross_section_{as_of_date}.csv"
            cs_df.to_csv(cs_path, index=False)
            print(f"  Cross-section z-scored: {len(raw_factor_bank)} tickers")
        else:
            cs_scores = {}
            print(f"  [INFO] < 5 tickers with factors — skipping cross-section z-score")

        # ------------------------------------------------------------------
        # PHASE 3: Compute composite signals with z-scored factors
        # ------------------------------------------------------------------
        date_signals: list[SignalResult] = []

        for ticker in raw_factor_bank:
            snap       = snapshots.get(ticker)
            cache      = financial_cache.get(ticker, {})
            run_id     = f"{ticker}_{as_of_date}"
            run_dir    = output_root / run_id
            company_name = get_company_name(ticker)

            try:
                sr: SignalResult = compute_signal_from_raw(
                    raw_factors=raw_factor_bank[ticker],
                    as_of_date=as_of_date,
                    ticker=ticker,
                    cross_section_scores=cs_scores.get(ticker),
                )
            except Exception as exc:
                print(f"  [WARN] signal engine failed for {run_id}: {exc}")
                sr = SignalResult(
                    ticker=ticker, as_of_date=as_of_date,
                    signal="neutral", confidence=0.5,
                    composite_score=0.0, conviction_tier="low",
                    expected_return=None, factors=[],
                    signal_rationale=[str(exc)],
                    risk_flags=["signal_engine_error"],
                    factors_available=0, factors_total=10, coverage_ratio=0.0,
                )

            date_signals.append(sr)

            prediction = _build_prediction(
                sr=sr,
                run_id=run_id,
                analysis_dir=cache.get("analysis_dir", ""),
                raw_summary=cache.get("raw_summary", {}),
                company_name=company_name,
            )
            prediction_rows.append(prediction.to_dict())
            with open(run_dir / "prediction.json", "w", encoding="utf-8") as f:
                json.dump(prediction.to_dict(), f, indent=2)

            cov_pct = f"{sr.coverage_ratio:.0%}" if sr.coverage_ratio else "0%"
            cs_tag  = "[CS]" if cs_scores.get(ticker) else "    "
            print(
                f"  {cs_tag} {run_id:<32}  signal={sr.signal:<8}  "
                f"composite={sr.composite_score:+.3f}  conf={sr.confidence:.2f}  "
                f"conviction={sr.conviction_tier:<6}  "
                f"factors={sr.factors_available}/{sr.factors_total} ({cov_pct})"
            )

        # ------------------------------------------------------------------
        # PHASE 4: Portfolio construction with sector caps + position sizing
        # ------------------------------------------------------------------
        if date_signals:
            position_sizes = {}
            for sr in date_signals:
                ps = compute_position_size(
                    ticker=sr.ticker,
                    signal=sr.signal,
                    confidence=sr.confidence,
                    composite_score=sr.composite_score,
                    conviction_tier=sr.conviction_tier,
                    expected_return=sr.expected_return,
                )
                position_sizes[sr.ticker] = ps

            portfolio = construct_portfolio(date_signals, position_sizes)
            port_summary = portfolio_summary(portfolio)
            port_path = output_root / f"portfolio_{as_of_date}.json"
            with open(port_path, "w") as f:
                json.dump(port_summary, f, indent=2)

            print(f"\n  Portfolio ({as_of_date}):")
            for line in portfolio_report_lines(port_summary):
                print(f"    {line}")

    snapshots_df = pd.DataFrame(snapshot_rows)
    snapshots_path = output_root / "snapshots.csv"
    snapshots_df.to_csv(snapshots_path, index=False)

    predictions_path = output_root / "predictions.csv"
    if prediction_rows:
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    else:
        pd.DataFrame(columns=list(ResearchPrediction.__dataclass_fields__.keys())).to_csv(
            predictions_path, index=False
        )

    summary = {
        "tickers":        tickers,
        "start":          args.start,
        "end":            args.end,
        "anchor_dates":   anchor_dates,
        "num_snapshots":  len(snapshot_rows),
        "num_predictions": len(prediction_rows),
        "skip_analysis":  bool(args.skip_analysis),
        "output_root":    str(output_root.resolve()),
        "snapshot_file":  str(snapshots_path.resolve()),
        "prediction_file": str(predictions_path.resolve()),
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    # Fail loud: a run that produced no predictions (short of an explicit
    # snapshot-only run) usually means every fetch failed — don't let that
    # look like a successful "0 predictions" run.
    if not args.skip_analysis and len(prediction_rows) == 0 and not args.allow_empty:
        print(
            "[ERROR] No predictions were produced. If this is expected, re-run with "
            "--allow-empty; otherwise check the [WARN] lines above for data-fetch failures.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
