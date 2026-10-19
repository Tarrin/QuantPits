#!/usr/bin/env python
"""Audit IC, RankIC, and random-stream properties of a monkey experiment."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew


_CELLS: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
_SIGNAL_LENGTH = 0


def _initialize_audit_worker(
    cells: list[tuple[np.ndarray, np.ndarray, np.ndarray]], signal_length: int
) -> None:
    global _CELLS, _SIGNAL_LENGTH
    _CELLS = cells
    _SIGNAL_LENGTH = signal_length


def _correlation_with_unit_target(values: np.ndarray, target: np.ndarray) -> float:
    centered = values - values.mean()
    norm = float(np.linalg.norm(centered))
    return float(centered @ target / norm) if norm > 0 else math.nan


def _audit_trial_chunk(rows: list[tuple[int, int]]) -> dict[str, Any]:
    if _CELLS is None:
        raise RuntimeError("audit worker was not initialized")

    summaries: list[dict[str, float | int]] = []
    all_ic: list[np.ndarray] = []
    all_rank_ic: list[np.ndarray] = []
    for run_id, signal_seed in rows:
        scores = np.random.default_rng(signal_seed).random(_SIGNAL_LENGTH)
        ic_values = np.empty(len(_CELLS), dtype=np.float64)
        rank_ic_values = np.empty(len(_CELLS), dtype=np.float64)
        for cell_no, (positions, return_unit, return_rank_unit) in enumerate(_CELLS):
            cross_section = scores[positions]
            ic_values[cell_no] = _correlation_with_unit_target(
                cross_section, return_unit
            )
            order = np.argsort(cross_section, kind="quicksort")
            ranks = np.empty(len(order), dtype=np.float64)
            ranks[order] = np.arange(len(order), dtype=np.float64)
            rank_ic_values[cell_no] = _correlation_with_unit_target(
                ranks, return_rank_unit
            )

        ic_std = float(np.std(ic_values, ddof=1))
        rank_ic_std = float(np.std(rank_ic_values, ddof=1))
        summary: dict[str, float | int] = {
            "run_id": run_id,
            "signal_seed": signal_seed,
            "mean_ic": float(np.mean(ic_values)),
            "ic_std": ic_std,
            "icir": float(np.mean(ic_values) / ic_std),
            "annualized_icir": float(np.mean(ic_values) / ic_std * math.sqrt(52)),
            "mean_rank_ic": float(np.mean(rank_ic_values)),
            "rank_ic_std": rank_ic_std,
            "rank_icir": float(np.mean(rank_ic_values) / rank_ic_std),
            "annualized_rank_icir": float(
                np.mean(rank_ic_values) / rank_ic_std * math.sqrt(52)
            ),
            "positive_ic_rate": float(np.mean(ic_values > 0)),
            "positive_rank_ic_rate": float(np.mean(rank_ic_values > 0)),
        }
        for horizon in (25, 50, len(_CELLS)):
            summary[f"mean_ic_t{horizon}"] = float(np.mean(ic_values[:horizon]))
            summary[f"mean_rank_ic_t{horizon}"] = float(
                np.mean(rank_ic_values[:horizon])
            )
        summaries.append(summary)
        all_ic.append(ic_values)
        all_rank_ic.append(rank_ic_values)

    return {
        "summaries": summaries,
        "ic": np.concatenate(all_ic),
        "rank_ic": np.concatenate(all_rank_ic),
    }


def _unit_center(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64) - float(np.mean(values))
    norm = float(np.linalg.norm(centered))
    if norm <= 0:
        raise ValueError("constant cross-sectional target")
    return centered / norm


def _prepare_cells(
    start_time: str, end_time: str
) -> tuple[
    pd.MultiIndex,
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    tuple[np.ndarray, np.ndarray],
    list[int],
]:
    from quantpits.utils import env

    env.init_qlib()
    from qlib.backtest.utils import TradeCalendarManager
    from qlib.data import D

    signal_start = (pd.Timestamp(start_time) - timedelta(days=31)).strftime("%Y-%m-%d")
    features = D.features(
        D.instruments(market="csi300"),
        ["$close"],
        start_time=signal_start,
        end_time=end_time,
        freq="day",
    )
    close = pd.to_numeric(features.iloc[:, 0], errors="coerce")
    eligible = close[close.notna() & np.isfinite(close) & (close > 0)].sort_index()
    eligible_index = eligible.index
    index_frame = pd.DataFrame(
        {
            "instrument": eligible_index.get_level_values("instrument").astype(str),
            "datetime": pd.DatetimeIndex(eligible_index.get_level_values("datetime")),
            "position": np.arange(len(eligible_index), dtype=np.int64),
            "spot": eligible.to_numpy(dtype=np.float64),
        }
    )

    all_codes = sorted(index_frame["instrument"].unique())
    price_frame = D.features(
        all_codes,
        ["$close"],
        start_time=signal_start,
        end_time=(pd.Timestamp(end_time) + timedelta(days=7)).strftime("%Y-%m-%d"),
        freq="day",
    )
    prices = pd.to_numeric(price_frame.iloc[:, 0], errors="coerce")
    price_wide = prices.unstack("instrument").sort_index()
    price_wide.index = pd.DatetimeIndex(price_wide.index)

    calendar = TradeCalendarManager(
        freq="week", start_time=start_time, end_time=end_time
    )
    cells: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cell_sizes: list[int] = []
    first_unfiltered: tuple[np.ndarray, np.ndarray] | None = None
    for step in range(calendar.get_trade_len()):
        pred_start, pred_end = calendar.get_step_time(step, shift=1)
        trade_start, trade_end = calendar.get_step_time(step, shift=0)
        within_prediction = index_frame[
            index_frame["datetime"].between(pred_start, pred_end)
        ]
        latest = (
            within_prediction.sort_values("datetime")
            .groupby("instrument", sort=False)
            .tail(1)
            .set_index("instrument")
        )
        if step == 0:
            first_unfiltered = (
                latest["position"].to_numpy(dtype=np.int64),
                latest.index.to_numpy(dtype=str),
            )

        future_window = price_wide.loc[
            (price_wide.index >= trade_start) & (price_wide.index <= trade_end)
        ]
        future_close = future_window.ffill().iloc[-1].rename("future")
        aligned = latest.join(future_close, how="inner")
        aligned["forward_return"] = aligned["future"] / aligned["spot"] - 1.0
        aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["forward_return"]
        )
        if len(aligned) < 20:
            raise RuntimeError(f"only {len(aligned)} valid IC observations at step {step}")

        returns = aligned["forward_return"].to_numpy(dtype=np.float64)
        return_ranks = pd.Series(returns).rank(method="average").to_numpy(dtype=np.float64)
        cells.append(
            (
                aligned["position"].to_numpy(dtype=np.int64),
                _unit_center(returns),
                _unit_center(return_ranks),
            )
        )
        cell_sizes.append(len(aligned))

    if first_unfiltered is None:
        raise RuntimeError("no weekly IC cells were constructed")
    return eligible_index, cells, first_unfiltered, cell_sizes


def _distribution(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "positive_rate": float(np.mean(array > 0)),
        "skew": float(skew(array, bias=False)),
        "p05_plus_p95": float(np.quantile(array, 0.05) + np.quantile(array, 0.95)),
    }


def _chunked(values: list[tuple[int, int]], size: int) -> list[list[tuple[int, int]]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-runs", type=int, help="Optional deterministic prefix for validation")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    results = pd.read_csv(run_dir / "results.csv")
    results = results[results["status"] == "ok"].sort_values("run_id")
    if args.max_runs is not None:
        results = results.head(args.max_runs)
    start_time, end_time = metadata["period"]

    eligible_index, cells, first_unfiltered, cell_sizes = _prepare_cells(
        start_time, end_time
    )
    first_positions, first_instruments = first_unfiltered
    first_seed = int(results.iloc[0]["signal_seed"])
    first_scores = np.random.default_rng(first_seed).random(len(eligible_index))
    top_order = np.argsort(first_scores[first_positions])[-metadata["holdings"] :]
    reconstructed = sorted(first_instruments[top_order].tolist())
    actual = sorted(str(results.iloc[0]["initial_holdings"]).split("|"))

    trial_rows = [
        (int(row.run_id), int(row.signal_seed))
        for row in results[["run_id", "signal_seed"]].itertuples(index=False)
    ]
    batches = _chunked(trial_rows, args.chunk_size)
    summaries: list[dict[str, float | int]] = []
    ic_blocks: list[np.ndarray] = []
    rank_ic_blocks: list[np.ndarray] = []
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_initialize_audit_worker,
        initargs=(cells, len(eligible_index)),
    ) as pool:
        futures = [pool.submit(_audit_trial_chunk, batch) for batch in batches]
        completed = 0
        for future in as_completed(futures):
            block = future.result()
            summaries.extend(block["summaries"])
            ic_blocks.append(block["ic"])
            rank_ic_blocks.append(block["rank_ic"])
            completed += len(block["summaries"])
            if completed % 500 == 0 or completed == len(trial_rows):
                print(f"IC checkpoint: {completed}/{len(trial_rows)}", flush=True)

    trial_metrics = pd.DataFrame(summaries).sort_values("run_id")
    trial_metrics.to_csv(run_dir / "ic_audit_by_trial.csv", index=False)
    all_ic = np.concatenate(ic_blocks)
    all_rank_ic = np.concatenate(rank_ic_blocks)
    horizon_scaling: dict[str, Any] = {}
    for metric in ("mean_ic", "mean_rank_ic"):
        horizon_scaling[metric] = {}
        for horizon in (25, 50, len(cells)):
            values = trial_metrics[f"{metric}_t{horizon}"]
            horizon_scaling[metric][str(horizon)] = {
                "std": float(values.std(ddof=1)),
                "std_times_sqrt_t": float(values.std(ddof=1) * math.sqrt(horizon)),
            }

    audit = {
        "definition": {
            "signal": "last random score in the prior Qlib weekly decision interval",
            "label": "last close in the following trade interval / signal-date close - 1",
            "ic": "cross-sectional Pearson correlation",
            "rank_ic": "cross-sectional Spearman correlation",
            "icir": "mean weekly IC / sample standard deviation of weekly IC",
            "annualized_icir": "ICIR * sqrt(52)",
        },
        "runs": int(len(trial_metrics)),
        "weekly_cells": int(len(cells)),
        "cell_size": {
            "min": int(min(cell_sizes)),
            "mean": float(np.mean(cell_sizes)),
            "max": int(max(cell_sizes)),
        },
        "run0_initial_topk_reconstruction": {
            "matches": reconstructed == actual,
            "overlap": len(set(reconstructed) & set(actual)),
        },
        "weekly_ic": _distribution(all_ic),
        "weekly_rank_ic": _distribution(all_rank_ic),
        "trial_mean_ic": _distribution(trial_metrics["mean_ic"]),
        "trial_mean_rank_ic": _distribution(trial_metrics["mean_rank_ic"]),
        "trial_icir": _distribution(trial_metrics["icir"]),
        "trial_rank_icir": _distribution(trial_metrics["rank_icir"]),
        "trial_annualized_icir": _distribution(trial_metrics["annualized_icir"]),
        "trial_annualized_rank_icir": _distribution(
            trial_metrics["annualized_rank_icir"]
        ),
        "horizon_scaling": horizon_scaling,
    }
    (run_dir / "ic_audit_summary.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
