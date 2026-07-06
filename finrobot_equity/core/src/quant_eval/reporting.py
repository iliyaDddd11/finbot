from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_eval.factor_attribution import factor_attribution_report_lines
from quant_eval.signal_decay import ic_decay_report_lines
from quant_eval.transaction_costs import estimate_tc, tc_summary_line


def _fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):+.2%}"


def _fmt_float(value: float | None, precision: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):.{precision}f}"


def _fmt_signed(value: float | None, precision: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{float(value):+.{precision}f}"


def write_markdown_report(
    scorecard: dict[str, Any],
    scored_df: pd.DataFrame,
    output_path: str,
    attr_df: pd.DataFrame | None = None,
    decay_df: pd.DataFrame | None = None,
) -> None:
    wrong_df = scored_df[scored_df["outcome_label"] == "wrong"].sort_values(
        ["confidence", "realized_return"], ascending=[False, True]
    ).head(10)
    right_df = scored_df[scored_df["outcome_label"] == "right"].sort_values(
        ["confidence", "realized_return"], ascending=[False, False]
    ).head(10)

    lines: list[str] = []
    lines.append("# Walk-Forward Score and Decision Audit")
    lines.append("")

    # -----------------------------------------------------------------------
    # Scorecard — main summary
    # -----------------------------------------------------------------------
    lines.append("## Scorecard")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    def _ci_str(ci: list | None) -> str:
        if not ci or ci[0] is None:
            return "n/a"
        return f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]"

    lines.append(f"| Predictions (total) | {scorecard.get('num_predictions', 0)} |")
    lines.append(f"| Known outcomes | {scorecard.get('num_known', 'n/a')} |")
    hr   = scorecard.get('right_rate')
    hr_ci = scorecard.get('hit_rate_ci_95')
    lines.append(f"| Hit rate (long/short only) | {_fmt_pct(hr)} &nbsp; 95% CI {_ci_str(hr_ci)} |")
    lines.append(f"| Avg realized return | {_fmt_pct(scorecard.get('avg_realized_return'))} |")
    sr   = scorecard.get('avg_signed_return')
    sr_ci = scorecard.get('avg_signed_return_ci_95')
    lines.append(f"| Avg signed return | {_fmt_pct(sr)} &nbsp; 95% CI {_ci_str(sr_ci)} |")
    lines.append(f"| Avg benchmark return (SPY) | {_fmt_pct(scorecard.get('avg_benchmark_return'))} |")
    al   = scorecard.get('avg_alpha_return')
    al_ci = scorecard.get('avg_alpha_ci_95')
    lines.append(f"| Avg alpha (vs SPY) | {_fmt_pct(al)} &nbsp; 95% CI {_ci_str(al_ci)} |")
    sh   = scorecard.get('sharpe_ratio')
    sh_ci = scorecard.get('sharpe_ci_95')
    lines.append(f"| Sharpe ratio (annualised) | {_fmt_signed(sh)} &nbsp; 95% CI [{_fmt_float(sh_ci[0] if sh_ci else None)}, {_fmt_float(sh_ci[1] if sh_ci else None)}] |")
    lines.append(f"| Max drawdown | {_fmt_pct(scorecard.get('max_drawdown'))} |")
    lines.append(f"| Calmar ratio | {_fmt_signed(scorecard.get('calmar_ratio'))} |")
    nsh = scorecard.get('net_sharpe_ratio')
    rtb = scorecard.get('round_trip_bps')
    lines.append(f"| Net Sharpe (after {_fmt_float(rtb)}bps) | {_fmt_signed(nsh)} |")
    lines.append(f"| Avg net return / period | {_fmt_pct(scorecard.get('avg_net_return'))} |")
    lines.append(f"| Avg turnover / period | {_fmt_pct(scorecard.get('avg_turnover'))} |")
    lines.append(f"| Mean IC | {_fmt_float(scorecard.get('mean_ic'))} |")
    lines.append(f"| ICIR | {_fmt_float(scorecard.get('icir'))} |")
    lines.append(f"| IC periods | {scorecard.get('ic_periods', 0)} |")
    lines.append(f"| Avg confidence | {_fmt_pct(scorecard.get('avg_confidence'))} |")
    lines.append(f"| Calibration error | {_fmt_float(scorecard.get('confidence_calibration_error'))} |")
    lines.append("")

    # -----------------------------------------------------------------------
    # Transaction cost estimate (large-cap baseline)
    # -----------------------------------------------------------------------
    lines.append("## Transaction Cost Estimate")
    lines.append("")
    tc_large = estimate_tc(liquidity_tier="large")
    tc_mid   = estimate_tc(liquidity_tier="mid")
    tc_small = estimate_tc(liquidity_tier="small")
    lines.append(f"- Large-cap: {tc_summary_line(tc_large)}")
    lines.append(f"- Mid-cap:   {tc_summary_line(tc_mid)}")
    lines.append(f"- Small-cap: {tc_summary_line(tc_small)}")
    lines.append("")

    # -----------------------------------------------------------------------
    # Per-signal breakdown
    # -----------------------------------------------------------------------
    by_signal = scorecard.get("by_signal", {})
    if by_signal:
        lines.append("## Signal Breakdown")
        lines.append("")
        lines.append("| Signal | N | Hit Rate | Avg Realized | Avg Signed |")
        lines.append("|--------|---|----------|--------------|------------|")
        for sig, stats in sorted(by_signal.items()):
            lines.append(
                f"| {sig} | {stats.get('count', 0)} | "
                f"{_fmt_pct(stats.get('right_rate'))} | "
                f"{_fmt_pct(stats.get('avg_realized_return'))} | "
                f"{_fmt_pct(stats.get('avg_signed_return'))} |"
            )
        lines.append("")

    # -----------------------------------------------------------------------
    # Per-ticker breakdown
    # -----------------------------------------------------------------------
    by_ticker = scorecard.get("by_ticker", {})
    if by_ticker:
        lines.append("## Per-Ticker Breakdown")
        lines.append("")
        lines.append("| Ticker | N | Hit Rate | Avg Realized | Avg Signed |")
        lines.append("|--------|---|----------|--------------|------------|")
        for ticker, stats in sorted(by_ticker.items()):
            lines.append(
                f"| {ticker} | {stats.get('count', 0)} | "
                f"{_fmt_pct(stats.get('right_rate'))} | "
                f"{_fmt_pct(stats.get('avg_realized_return'))} | "
                f"{_fmt_pct(stats.get('avg_signed_return'))} |"
            )
        lines.append("")

    # -----------------------------------------------------------------------
    # Factor attribution (if factor_scores columns exist)
    # -----------------------------------------------------------------------
    factor_cols = [c for c in scored_df.columns if c.startswith("factor_") or c in (
        "earnings_quality", "margin_quality", "leverage", "roic",
        "revenue_acceleration", "eps_growth", "ev_ebitda_vs_history",
        "price_mom_12m1m", "price_mom_6m", "fcf_yield",
    )]
    if factor_cols:
        lines.append("## Factor Attribution")
        lines.append("")
        lines.append("Average factor score (all predictions):")
        lines.append("")
        for fc in factor_cols:
            avg = pd.to_numeric(scored_df[fc], errors="coerce").mean()
            lines.append(f"- {fc}: {_fmt_signed(avg)}")
        lines.append("")

    # -----------------------------------------------------------------------
    # Patterns
    # -----------------------------------------------------------------------
    lines.append("## Repeated Patterns That Worked")
    lines.append("")
    for item in scorecard.get("patterns", {}).get("right_patterns", [])[:10]:
        lines.append(
            f"- `{item['pattern_signature']}` → count={item['count']}, "
            f"hit={_fmt_pct(item['right_rate'])}, "
            f"avg_realized={_fmt_pct(item['avg_realized_return'])}, "
            f"conf={_fmt_pct(item['avg_confidence'])}"
        )
    if not scorecard.get("patterns", {}).get("right_patterns"):
        lines.append("- none with enough repeated evidence yet")
    lines.append("")

    lines.append("## Repeated Patterns That Failed")
    lines.append("")
    for item in scorecard.get("patterns", {}).get("wrong_patterns", [])[:10]:
        lines.append(
            f"- `{item['pattern_signature']}` → count={item['count']}, "
            f"hit={_fmt_pct(item['right_rate'])}, "
            f"avg_return={_fmt_pct(item['avg_realized_return'])}, "
            f"conf={_fmt_pct(item['avg_confidence'])}"
        )
    if not scorecard.get("patterns", {}).get("wrong_patterns"):
        lines.append("- none with enough repeated evidence yet")
    lines.append("")

    # -----------------------------------------------------------------------
    # Case studies
    # -----------------------------------------------------------------------
    lines.append("## High-Confidence Wrong Calls")
    lines.append("")
    for _, row in wrong_df.iterrows():
        evidence = ", ".join(row.get("case_evidence", [])[:8]) if isinstance(row.get("case_evidence"), list) else ""
        lines.append(
            f"- **{row['ticker']}** {row['as_of_date']} "
            f"signal={row['signal']} conf={_fmt_pct(row.get('confidence'))} "
            f"realized={_fmt_pct(row.get('realized_return'))}  "
            f"_{evidence}_"
        )
    if wrong_df.empty:
        lines.append("- none")
    lines.append("")

    lines.append("## High-Confidence Right Calls")
    lines.append("")
    for _, row in right_df.iterrows():
        evidence = ", ".join(row.get("case_evidence", [])[:8]) if isinstance(row.get("case_evidence"), list) else ""
        lines.append(
            f"- **{row['ticker']}** {row['as_of_date']} "
            f"signal={row['signal']} conf={_fmt_pct(row.get('confidence'))} "
            f"realized={_fmt_pct(row.get('realized_return'))}  "
            f"_{evidence}_"
        )
    if right_df.empty:
        lines.append("- none")
    lines.append("")

    # -----------------------------------------------------------------------
    # Factor IC attribution table
    # -----------------------------------------------------------------------
    if attr_df is not None and not attr_df.empty:
        lines.append("## Factor IC Attribution")
        lines.append("")
        lines.extend(factor_attribution_report_lines(attr_df))
        lines.append("")

    # -----------------------------------------------------------------------
    # Multi-horizon IC decay
    # -----------------------------------------------------------------------
    if decay_df is not None and not decay_df.empty:
        lines.extend(ic_decay_report_lines(decay_df))
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
