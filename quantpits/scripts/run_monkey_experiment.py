#!/usr/bin/env python
"""Run the reproducible CSI300 monkey portfolio Monte Carlo experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantpits.experiments.monkey import REGIME_METRIC_SUFFIXES, REGIME_NAMES


RESULT_FIELDS = [
    "run_id", "status", "signal_seed", "strategy_seed", "final_nav",
    "total_return", "annualized_return", "annualized_volatility", "sharpe",
    "max_drawdown", "calmar", "benchmark_total_return", "benchmark_cagr",
    "excess_total_return", "excess_cagr", "total_turnover", "total_cost",
    "annualized_alpha", "benchmark_beta", "alpha_t_stat", "alpha_p_value",
    "hac_alpha_t_stat", "hac_alpha_p_value", "information_ratio",
    "recent_start_date", "recent_end_date", "recent_start_nav", "recent_final_nav",
    "recent_return", "recent_annualized_return", "recent_benchmark_return",
    "recent_excess_return", "recent_annualized_volatility", "recent_sharpe",
    "recent_max_drawdown", "recent_calmar", "recent_total_turnover",
    "recent_total_cost", "prior_return", "prior_benchmark_return",
    "prior_excess_return",
    "initial_holdings", "initial_holding_count", "min_holding_count",
    "max_holding_count", "rebalance_periods", "full_three_by_three_periods",
    "full_three_by_three_rate", "mean_sells", "mean_buys",
    "execution_day_rule", "execution_periods", "first_execution_date",
    "last_execution_date", "first_valuation_date", "last_valuation_date",
    "order_timestamp_mismatches", "regime_labeled_weeks", "regime_unlabeled_weeks",
] + [
    f"{regime}_{suffix}"
    for regime in REGIME_NAMES
    for suffix in REGIME_METRIC_SUFFIXES
] + ["error", "traceback"]
CURVE_FIELDS = [
    "run_id", "datetime", "execution_datetime", "signal_datetime",
    "trend_regime", "vol_regime", "nav",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Monkey YAML; defaults to workspace config/monkey_config.yaml")
    parser.add_argument("--runs", type=int, help="Override number of Monte Carlo trials")
    parser.add_argument("--workers", type=int, help="Override spawned worker count")
    parser.add_argument("--chunk-size", type=int, help="Override trials per submitted worker task")
    parser.add_argument("--master-seed", type=int, help="Override master random seed")
    parser.add_argument("--run-dir", help="Explicit output directory")
    parser.add_argument("--resume", action="store_true", help="Continue an existing matching run directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without backtesting")
    return parser.parse_args()


def _load_config(args: argparse.Namespace):
    from quantpits.experiments.monkey import MonkeyExperimentConfig
    from quantpits.utils import env

    config_path = Path(args.config) if args.config else Path(env.ROOT_DIR) / "config" / "monkey_config.yaml"
    strategy_path = Path(env.ROOT_DIR) / "config" / "strategy_config.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    strategy_config = yaml.safe_load(strategy_path.read_text(encoding="utf-8"))
    config = MonkeyExperimentConfig.from_mapping(raw, strategy_config=strategy_config)
    overrides: dict[str, Any] = {}
    for arg_name, field_name in (
        ("runs", "runs"),
        ("workers", "workers"),
        ("chunk_size", "chunk_size"),
        ("master_seed", "master_seed"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if overrides:
        config = replace(config, **overrides)
        config.validate()
    return config, config_path.resolve()


def _default_run_dir(config, workspace_root: Path) -> Path:
    name = (
        f"monkey_{config.start_time}_{config.end_time}"
        f"_seed{config.master_seed}_n{config.runs}"
    )
    return workspace_root / config.output_root / name


def _read_completed(results_path: Path) -> set[int]:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(results_path, usecols=["run_id", "status"])
    frame = frame[frame["status"].eq("ok")]
    return set(pd.to_numeric(frame["run_id"], errors="coerce").dropna().astype(int))


def _prepare_resume_checkpoint(results_path: Path, failed_attempts_path: Path) -> None:
    """Keep successful trial IDs unique and archive failures for retry."""

    if not results_path.exists() or results_path.stat().st_size == 0:
        return
    frame = pd.read_csv(results_path)
    failed = frame[~frame["status"].eq("ok")].copy()
    successful = frame[frame["status"].eq("ok")].drop_duplicates("run_id", keep="last")
    if failed.empty and len(successful) == len(frame):
        return
    if not failed.empty:
        failed.to_csv(
            failed_attempts_path,
            mode="a",
            header=not failed_attempts_path.exists(),
            index=False,
            encoding="utf-8-sig",
        )
    temporary = results_path.with_suffix(".resume.tmp.csv")
    successful.reindex(columns=RESULT_FIELDS).to_csv(
        temporary, index=False, encoding="utf-8"
    )
    os.replace(temporary, results_path)
    print(
        f"Resume cleanup: retained {len(successful)} successful trials; "
        f"archived {len(failed)} failed attempts.",
        flush=True,
    )


def _append_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        return
    existing_fields: list[str] = []
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle))
    if existing_fields and existing_fields != fields:
        raise ValueError(f"checkpoint schema mismatch in {path}")
    normalized = [{key: row.get(key, "") for key in fields} for row in rows]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not existing_fields:
            writer.writeheader()
        writer.writerows(normalized)
        handle.flush()


def main() -> int:
    args = _parse_args()
    from quantpits.experiments.monkey import execute_trials, write_analysis_artifacts
    from quantpits.utils import env

    config, config_path = _load_config(args)
    workspace_root = Path(env.ROOT_DIR).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(config, workspace_root)
    results_path = run_dir / "results.csv"
    curves_path = run_dir / "sample_curves.csv"
    metadata_path = run_dir / "run_metadata.json"

    plan = {
        "workspace": str(workspace_root),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "period": [config.start_time, config.end_time],
        "runs": config.runs,
        "workers": config.workers,
        "holdings": config.holdings,
        "weekly_replacements": config.weekly_replacements,
        "master_seed": config.master_seed,
        "python_hash_seed": config.python_hash_seed,
        "execution_day": config.execution_day,
        "execution_price": config.strategy_config["backtest"]["exchange_kwargs"].get(
            "deal_price", "close"
        ),
        "fingerprint": config.fingerprint(),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    # Must be inherited by spawned Python interpreters before they initialize.
    # The strategy independently sorts stock IDs, covering the workers=1 path.
    os.environ["PYTHONHASHSEED"] = str(config.python_hash_seed)

    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"run directory already contains data; pass --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config.fingerprint():
            raise ValueError("resume fingerprint mismatch: experiment semantics changed")
    else:
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace_root.parent.parent,
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_commit = "unknown"
        try:
            import cupy

            cupy_version = cupy.__version__
        except ImportError:
            cupy_version = None
        import numpy
        import qlib

        metadata = {
            **plan,
            "status": "running",
            "started_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "python": sys.version,
            "pyqlib": qlib.__version__,
            "numpy": numpy.__version__,
            "cupy": cupy_version,
            "qlib_data_dir": os.environ.get("QLIB_DATA_DIR"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "execution_semantics": {
                "signal": "previous weekly interval",
                "execution_day": config.execution_day,
                "execution_price": plan["execution_price"],
                "tradability_check": "exact execution day",
                "valuation_label": "last actual trading day of weekly interval",
            },
            "git_commit": git_commit,
            "experiment_config": config.semantic_payload(),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.resume:
        _prepare_resume_checkpoint(results_path, run_dir / "failed_attempts.csv")
    completed = _read_completed(results_path)
    pending = [run_id for run_id in range(config.runs) if run_id not in completed]
    print(f"Completed: {len(completed)}; pending: {len(pending)}", flush=True)
    started = time.perf_counter()
    newly_completed = 0

    def checkpoint(batch: list[dict[str, Any]]) -> None:
        nonlocal newly_completed
        summaries = [item["summary"] for item in batch]
        curves = [point for item in batch for point in (item.get("curve") or [])]
        _append_rows(results_path, summaries, RESULT_FIELDS)
        _append_rows(curves_path, curves, CURVE_FIELDS)
        newly_completed += len(summaries)
        elapsed = max(time.perf_counter() - started, 1e-9)
        rate = newly_completed / elapsed
        remaining = len(pending) - newly_completed
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"Checkpoint: {len(completed) + newly_completed}/{config.runs} "
            f"({rate:.2f} trials/s, ETA {eta/60:.1f} min)",
            flush=True,
        )

    try:
        execute_trials(config, pending, on_batch=checkpoint)
    except KeyboardInterrupt:
        print("Interrupted; completed checkpoints can be resumed with --resume.", flush=True)
        return 130

    summary = write_analysis_artifacts(results_path, run_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "complete" if summary["failed_runs"] == 0 else "complete_with_errors",
            "finished_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "successful_runs": summary["successful_runs"],
            "failed_runs": summary["failed_runs"],
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Experiment complete: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
