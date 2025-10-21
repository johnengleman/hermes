#!/usr/bin/env python3
"""
Post-run diagnostics for Hermes optimization results.

Usage:
    python post_run_checks.py
    python post_run_checks.py --pattern "hermes_stage2_BTC.csv"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_PATTERNS = (
    "hermes_stage1_*.csv",
    "hermes_stage2_*.csv",
)


def find_result_files(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(Path(".").glob(pattern)))
    return files


def summarize_result_frame(df: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    
    summary["rows"] = len(df)
    if len(df) == 0:
        return summary
    
    summary["mean_test_composite"] = float(df.get("test_composite", pd.Series(dtype=float)).mean())
    summary["mean_test_sortino"] = float(df.get("test_sortino", pd.Series(dtype=float)).mean())
    summary["mean_test_calmar"] = float(df.get("test_calmar", pd.Series(dtype=float)).mean())
    
    train_sortino = df.get("train_sortino")
    test_sortino = df.get("test_sortino")
    if train_sortino is not None and test_sortino is not None:
        ratio = (test_sortino.replace(0, np.nan) / train_sortino.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        summary["median_sortino_ratio"] = float(ratio.median(skipna=True))
        summary["severe_degradation"] = int((ratio < 0.30).sum(skipna=True))
        summary["lucky_windows"] = int((ratio > 1.50).sum(skipna=True))
    else:
        summary["median_sortino_ratio"] = math.nan
        summary["severe_degradation"] = 0
        summary["lucky_windows"] = 0
    
    max_dd = df.get("test_max_dd")
    if max_dd is not None:
        summary["drawdown_gt_60pct"] = int((max_dd > 0.60).sum())
    
    if "bootstrap_return_p05" in df.columns:
        summary["bootstrap_return_p05"] = float(df["bootstrap_return_p05"].mean())
        summary["bootstrap_max_dd_p95"] = float(df.get("bootstrap_max_dd_p95", pd.Series(dtype=float)).mean())
    
    return summary


def print_summary(path: Path, summary: dict[str, float]) -> None:
    print(f"\n📄 {path}")
    if summary.get("rows", 0) == 0:
        print("  (empty)")
        return
    
    print(f"  Windows analysed: {int(summary['rows'])}")
    if not math.isnan(summary.get("mean_test_composite", math.nan)):
        print(f"  Avg test composite: {summary['mean_test_composite']:.2f}")
    if not math.isnan(summary.get("mean_test_sortino", math.nan)):
        print(f"  Avg test Sortino: {summary['mean_test_sortino']:.2f}")
    if not math.isnan(summary.get("mean_test_calmar", math.nan)):
        print(f"  Avg test Calmar: {summary['mean_test_calmar']:.2f}")
    if not math.isnan(summary.get("median_sortino_ratio", math.nan)):
        print(f"  Median test/train Sortino ratio: {summary['median_sortino_ratio']:.2f}")
        print(f"  Severe degradation (<0.3): {summary['severe_degradation']}")
        print(f"  Likely luck (>1.5): {summary['lucky_windows']}")
    if "drawdown_gt_60pct" in summary:
        print(f"  Drawdowns >60%: {summary['drawdown_gt_60pct']}")
    if "bootstrap_return_p05" in summary:
        print(f"  Bootstrap return p05: {summary['bootstrap_return_p05']:.2f}")
        print(f"  Bootstrap max DD p95: {summary['bootstrap_max_dd_p95']:.2f}")


def list_flagged_windows(df: pd.DataFrame) -> None:
    if df.empty:
        return
    
    if "test_sortino" in df.columns and "train_sortino" in df.columns:
        ratio = (df["test_sortino"].replace(0, np.nan) / df["train_sortino"].replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        flagged = df.loc[ratio < 0.30]
        if not flagged.empty:
            print("  ⚠️  Windows with severe degradation:")
            cols = ["asset", "bull_period", "window_id", "train_sortino", "test_sortino"]
            cols = [c for c in cols if c in flagged.columns]
            print(flagged[cols].head(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes post-run verification.")
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        default=list(DEFAULT_PATTERNS),
        help="Glob pattern(s) for result CSV files (default: %(default)s)",
    )
    args = parser.parse_args()
    
    result_files = find_result_files(args.patterns)
    if not result_files:
        print("✗ No result files matched the provided patterns.")
        return
    
    for path in result_files:
        try:
            df = pd.read_csv(path)
        except Exception as err:
            print(f"\n✗ Failed to read {path}: {err}")
            continue
        summary = summarize_result_frame(df)
        print_summary(path, summary)
        list_flagged_windows(df)


if __name__ == "__main__":
    main()
