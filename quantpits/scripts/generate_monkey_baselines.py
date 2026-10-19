#!/usr/bin/env python
"""Generate resumable matched cap/equal-weight controls for rolling windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    from quantpits.experiments.monkey import MonkeyExperimentConfig
    from quantpits.experiments.monkey_baselines import run_equal_weight_baseline
    from quantpits.utils import env

    workspace = Path(env.ROOT_DIR).resolve()
    manifest = pd.read_csv(Path(args.manifest).resolve())
    strategy = yaml.safe_load((workspace / "config" / "strategy_config.yaml").read_text(encoding="utf-8"))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    weekly_parts: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        window_dir = output / str(row.window_id)
        summary_path = window_dir / "summary.json"
        weekly_path = window_dir / "weekly.csv"
        if summary_path.exists() and weekly_path.exists() and not args.force:
            print(f"SKIP complete baseline {row.window_id}", flush=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            weekly = pd.read_csv(weekly_path)
        else:
            config_path = workspace / str(row.config_path)
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config = MonkeyExperimentConfig.from_mapping(raw, strategy_config=strategy)
            print(f"RUN equal-weight baseline {row.window_id}: {config.start_time} -> {config.end_time}", flush=True)
            summary, weekly = run_equal_weight_baseline(config)
            summary = {"window_id": str(row.window_id), **summary, "status": "complete"}
            weekly.insert(0, "window_id", str(row.window_id))
            window_dir.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            weekly.to_csv(weekly_path, index=False, encoding="utf-8-sig")
        rows.append(summary)
        weekly_parts.append(weekly)
    combined = pd.DataFrame(rows).sort_values("window_id")
    combined.to_csv(output / "baseline_window_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(weekly_parts, ignore_index=True).to_csv(
        output / "baseline_weekly.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "windows": int(len(combined)),
        "all_complete": bool(combined["status"].eq("complete").all()),
        "semantics": {
            "universe": "dynamic CSI300 eligible at previous signal interval",
            "weights": "equal weight, rebalanced weekly",
            "risk_degree": "same as monkey config (0.95)",
            "execution": "next interval first actual trading-day close",
            "tradability": "exact execution day",
            "costs": "same Qlib exchange configuration as monkey",
        },
    }
    (output / "baseline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Baseline summary: {output / 'baseline_window_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
