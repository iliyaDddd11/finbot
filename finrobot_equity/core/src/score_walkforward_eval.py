#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_eval.decision_audit import (
    build_case_evidence,
    build_driver_flags,
    build_pattern_signature,
    classify_outcome,
    load_analysis_metrics,
    load_text_sections,
    save_casepack,
)
from quant_eval.factor_attribution import (
    flatten_factor_scores,
    factor_ic_attribution,
    factor_attribution_report_lines,
)
from quant_eval.realized_returns import compute_realized_outcome
from quant_eval.reporting import write_markdown_report
from quant_eval.scoring import compute_scorecard, enrich_scored_predictions
from quant_eval.signal_decay import compute_horizon_returns, compute_ic_decay, ic_decay_report_lines
from quant_eval.transaction_costs import estimate_tc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score walk-forward predictions and audit decision patterns with hard evidence.")
    parser.add_argument("--walkforward-root", required=True, help="Output root of run_walkforward_eval.py")
    parser.add_argument("--predictions-file", default=None, help="Optional explicit predictions.csv path")
    parser.add_argument("--output-dir", default=None, help="Directory for scored outputs. Default: <walkforward-root>/scorecard")
    parser.add_argument("--neutral-band", type=float, default=0.02, help="Neutral calls inside this realized return band count as right.")
    parser.add_argument("--min-pattern-count", type=int, default=2)
    parser.add_argument("--periods-per-year", type=int, default=12,
                        help="Rebalance frequency for annualising Sharpe/ICIR/Calmar. "
                             "Default 12 (the walk-forward grid is monthly).")
    parser.add_argument("--skip-price-download", action="store_true", help="Only score rows that already have realized_return columns.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 90],
                        help="Horizon days for IC decay analysis (default: 30 60 90)")
    return parser


def _parse_list_like(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else [text]
        except Exception:
            return [text]
    return [value]


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)
    if "thesis_points" in df.columns:
        df["thesis_points"] = df["thesis_points"].apply(_parse_list_like)
    if "risk_flags" in df.columns:
        df["risk_flags"] = df["risk_flags"].apply(_parse_list_like)
    return df


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    walkforward_root = Path(args.walkforward_root)
    predictions_path = Path(args.predictions_file) if args.predictions_file else walkforward_root / "predictions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else walkforward_root / "scorecard"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = _load_predictions(predictions_path)
    scored_rows: list[dict[str, Any]] = []

    for _, row in predictions_df.iterrows():
        row_dict = row.to_dict()
        analysis_dir = str(row_dict.get("analysis_dir", "") or "")
        metrics = load_analysis_metrics(analysis_dir)
        text_sections = load_text_sections(analysis_dir)

        realized = {
            "realized_return":        row_dict.get("realized_return"),
            "benchmark_return":       row_dict.get("benchmark_return"),
            "alpha_return":           row_dict.get("alpha_return"),
            "entry_timestamp":        row_dict.get("entry_timestamp"),
            "exit_timestamp":         row_dict.get("exit_timestamp"),
            "entry_close":            row_dict.get("entry_close"),
            "exit_close":             row_dict.get("exit_close"),
            "max_upside":             row_dict.get("max_upside"),
            "max_drawdown":           row_dict.get("max_drawdown"),
            "realized_price_csv_path": row_dict.get("realized_price_csv_path"),
        }
        if not args.skip_price_download and pd.isna(pd.to_numeric(pd.Series([realized["realized_return"]]), errors="coerce").iloc[0]):
            realized_outcome = compute_realized_outcome(
                ticker=str(row_dict["ticker"]),
                as_of_date=str(row_dict["as_of_date"]),
                horizon_days=int(row_dict.get("horizon_days", 60)),
                price_download_dir=str(output_dir / "prices" / f"{row_dict['ticker']}_{row_dict['as_of_date']}_{int(row_dict.get('horizon_days', 60))}d"),
            )
            realized = {
                "realized_return":        realized_outcome.realized_return,
                "benchmark_return":       realized_outcome.benchmark_return,
                "alpha_return":           realized_outcome.alpha_return,
                "entry_timestamp":        realized_outcome.entry_timestamp,
                "exit_timestamp":         realized_outcome.exit_timestamp,
                "entry_close":            realized_outcome.entry_close,
                "exit_close":             realized_outcome.exit_close,
                "max_upside":             realized_outcome.max_upside,
                "max_drawdown":           realized_outcome.max_drawdown,
                "realized_price_csv_path": realized_outcome.price_csv_path,
            }

        combined = {**row_dict, **metrics, **realized}
        combined["text_sections"] = text_sections
        combined["driver_flags"] = build_driver_flags(metrics, float(row_dict.get("confidence", 0.0) or 0.0), _parse_list_like(row_dict.get("risk_flags")))
        combined["pattern_signature"] = build_pattern_signature(str(row_dict.get("signal", "neutral")), combined["driver_flags"])
        combined["outcome_label"] = classify_outcome(str(row_dict.get("signal", "neutral")), pd.to_numeric(pd.Series([combined.get("realized_return")]), errors="coerce").iloc[0] if combined.get("realized_return") is not None else None, neutral_band=args.neutral_band)
        combined["case_evidence"] = build_case_evidence(pd.Series(combined), neutral_band=args.neutral_band)
        scored_rows.append(combined)

    scored_df = pd.DataFrame(scored_rows)
    scored_df = enrich_scored_predictions(scored_df)

    scorecard = compute_scorecard(scored_df, periods_per_year=args.periods_per_year)
    scorecard["patterns"] = scorecard.get("patterns", {})
    scorecard["patterns"]["right_patterns"] = scorecard["patterns"].get("right_patterns", [])[: args.min_pattern_count * 5]
    scorecard["patterns"]["wrong_patterns"] = scorecard["patterns"].get("wrong_patterns", [])[: args.min_pattern_count * 5]

    scored_csv = output_dir / "scored_predictions.csv"
    export_df = scored_df.copy()
    for col in ["thesis_points", "risk_flags", "driver_flags", "case_evidence"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
    if "text_sections" in export_df.columns:
        export_df["text_sections"] = export_df["text_sections"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
    export_df.to_csv(scored_csv, index=False)

    # ------------------------------------------------------------------
    # Factor IC attribution
    # ------------------------------------------------------------------
    attr_df = factor_ic_attribution(flatten_factor_scores(scored_df))
    if not attr_df.empty:
        attr_df.to_csv(output_dir / "factor_attribution.csv", index=False)
        print("\nFactor IC Attribution:")
        for line in factor_attribution_report_lines(attr_df):
            print(" ", line)

    # ------------------------------------------------------------------
    # Multi-horizon signal IC decay
    # ------------------------------------------------------------------
    horizons = args.horizons
    if not args.skip_price_download and len(horizons) > 1:
        print(f"\nComputing multi-horizon returns ({horizons})...")
        scored_df = compute_horizon_returns(
            scored_df,
            price_download_dir=str(output_dir / "prices"),
            horizons=horizons,
        )
        decay_df = compute_ic_decay(scored_df, horizons)
        decay_df.to_csv(output_dir / "ic_decay.csv", index=False)
        for line in ic_decay_report_lines(decay_df):
            print(line)
    else:
        decay_df = pd.DataFrame()

    scorecard_path = output_dir / "scorecard.json"
    scorecard_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    write_markdown_report(scorecard, scored_df, str(output_dir / "decision_audit.md"),
                          attr_df=attr_df, decay_df=decay_df)
    casepack_paths = save_casepack(scored_df, str(output_dir))

    # TC-adjusted return estimate (large-cap baseline, 6 round-trips/year)
    tc = estimate_tc(liquidity_tier="large")
    avg_signed = scorecard.get("avg_signed_return")
    tc_adj_return = (avg_signed - tc.round_trip_bps / 10_000) if avg_signed is not None else None

    summary = {
        "walkforward_root":       str(walkforward_root.resolve()),
        "predictions_file":       str(predictions_path.resolve()),
        "output_dir":             str(output_dir.resolve()),
        "scored_predictions_csv": str(scored_csv.resolve()),
        "scorecard_json":         str(scorecard_path.resolve()),
        "decision_audit_md":      str((output_dir / "decision_audit.md").resolve()),
        **casepack_paths,
        "num_predictions":        int(len(scored_df)),
        "num_known":              scorecard.get("num_known"),
        "right_rate":             scorecard.get("right_rate"),
        "sharpe_ratio":           scorecard.get("sharpe_ratio"),
        "max_drawdown":           scorecard.get("max_drawdown"),
        "calmar_ratio":           scorecard.get("calmar_ratio"),
        "mean_ic":                scorecard.get("mean_ic"),
        "icir":                   scorecard.get("icir"),
        "avg_signed_return":      avg_signed,
        "avg_benchmark_return":   scorecard.get("avg_benchmark_return"),
        "avg_alpha_return":       scorecard.get("avg_alpha_return"),
        "hit_rate_ci_95":         scorecard.get("hit_rate_ci_95"),
        "sharpe_ci_95":           scorecard.get("sharpe_ci_95"),
        "avg_alpha_ci_95":        scorecard.get("avg_alpha_ci_95"),
        "tc_round_trip_bps":      tc.round_trip_bps,
        "avg_signed_after_tc":    round(tc_adj_return, 6) if tc_adj_return is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
