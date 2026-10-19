#!/usr/bin/env python
"""Generate actual-trading-day rolling two-year monkey window configs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main() -> int:
    args = _args()
    source_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    cfg = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    experiment = cfg["experiment"]
    simulation = cfg["simulation"]
    portfolio = cfg["portfolio"]

    from quantpits.utils import env

    env.init_qlib()
    from qlib.data import D

    analysis_start = pd.Timestamp(experiment["analysis_start"])
    analysis_end = pd.Timestamp(experiment["analysis_end"])
    window_months = int(experiment.get("window_months", 24))
    step_months = int(experiment.get("step_months", 3))
    calendar = pd.DatetimeIndex(pd.to_datetime(D.calendar(
        start_time=analysis_start.strftime("%Y-%m-%d"),
        end_time=analysis_end.strftime("%Y-%m-%d"), freq="day",
    ))).normalize().sort_values()
    if calendar.empty:
        raise RuntimeError("Qlib returned no trading calendar for the requested period")

    workspace_root = Path(env.ROOT_DIR).resolve()
    experiment_root = source_path.parent.parent
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    # This directory is owned by the plan generator. Remove only its previous
    # deterministic window configs so changed anchors cannot leave stale files.
    for existing in config_dir.glob("W*.yaml"):
        existing.unlink()
    rows: list[dict[str, object]] = []
    index = 0
    while True:
        nominal_start = analysis_start + pd.DateOffset(months=index * step_months)
        nominal_end = nominal_start + pd.DateOffset(months=window_months) - pd.Timedelta(days=1)
        if nominal_end > analysis_end:
            break
        start_pos = int(calendar.searchsorted(nominal_start, side="left"))
        end_pos = int(calendar.searchsorted(nominal_end, side="right")) - 1
        if start_pos >= len(calendar) or end_pos < start_pos:
            raise RuntimeError(f"no trading dates for window anchored at {nominal_start.date()}")
        start = calendar[start_pos]
        end = calendar[end_pos]
        window_id = f"W{index + 1:02d}"
        label = f"{window_id}_{start:%Y%m%d}_{end:%Y%m%d}"
        window_config = {
            "experiment": {
                "name": f"monkey_rolling_{window_id.lower()}",
                "market": experiment.get("market", "csi300"),
                "benchmark": experiment.get("benchmark", "SH000300"),
                "start_time": start.strftime("%Y-%m-%d"),
                "end_time": end.strftime("%Y-%m-%d"),
            },
            "portfolio": portfolio,
            "simulation": {
                "runs": int(simulation.get("scan_runs", 2000)),
                "master_seed": int(simulation.get("master_seed", 20260829)),
                "workers": int(simulation.get("workers", 8)),
                "chunk_size": int(simulation.get("chunk_size", 10)),
                "frequency": "week",
                "execution_day": "first_trade_day",
                "python_hash_seed": int(simulation.get("python_hash_seed", 0)),
                "signal_lookback_days": int(simulation.get("signal_lookback_days", 31)),
                "sample_curve_runs": int(simulation.get("sample_curve_runs", 20)),
            },
            "analysis": {
                "hac_lags": int(cfg.get("analysis", {}).get("hac_lags", 4)),
                "regime_labels_path": cfg["analysis"]["regime_labels_path"],
            },
            "output": {"root": "output/rolling_regime"},
        }
        config_path = config_dir / f"{label}.yaml"
        config_path.write_text(
            yaml.safe_dump(window_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        is_anchor = index % max(1, window_months // step_months) == 0
        rows.append({
            "window_id": window_id,
            "label": label,
            "nominal_start": nominal_start.date().isoformat(),
            "start_time": start.date().isoformat(),
            "end_time": end.date().isoformat(),
            "is_anchor": int(is_anchor),
            "config_path": _relative(config_path, workspace_root),
            "scan_run_dir": _relative(experiment_root / "output" / "scan_n2000" / label, workspace_root),
            "deep_run_dir": _relative(experiment_root / "output" / "deep_n10000" / label, workspace_root),
        })
        index += 1

    manifest_path = output_dir / "window_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "window_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected = output_dir / "selected_windows.txt"
    if not selected.exists():
        anchor_ids = [str(row["window_id"]) for row in rows if row["is_anchor"]]
        selected.write_text(
            "# 扫描报告生成后可修改此文件；每行一个窗口 ID。\n" + "\n".join(anchor_ids) + "\n",
            encoding="utf-8",
        )
    print(f"Rolling plan: {manifest_path} ({len(rows)} windows)")
    print("Anchors: " + ", ".join(str(row["window_id"]) for row in rows if row["is_anchor"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
