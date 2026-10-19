#!/usr/bin/env python
"""Run/resume a selected phase from a generated monkey rolling manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--phase", choices=("anchors", "remaining", "all", "deep"), required=True)
    parser.add_argument("--selected-file")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected_ids(path: Path) -> set[str]:
    return {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _complete(run_dir: Path, runs: int) -> bool:
    metadata = run_dir / "run_metadata.json"
    if not metadata.exists():
        return False
    data = json.loads(metadata.read_text(encoding="utf-8"))
    return data.get("status") == "complete" and int(data.get("successful_runs", -1)) == runs


def main() -> int:
    args = _args()
    from quantpits.utils import env

    workspace = Path(env.ROOT_DIR).resolve()
    frame = pd.read_csv(Path(args.manifest).resolve())
    if args.phase == "anchors":
        frame = frame[frame["is_anchor"].astype(int).eq(1)]
    elif args.phase == "remaining":
        frame = frame[frame["is_anchor"].astype(int).eq(0)]
    elif args.phase == "deep":
        if not args.selected_file:
            raise ValueError("--selected-file is required for the deep phase")
        wanted = _selected_ids(Path(args.selected_file).resolve())
        unknown = wanted.difference(frame["window_id"].astype(str))
        if unknown:
            raise ValueError(f"unknown selected window IDs: {sorted(unknown)}")
        frame = frame[frame["window_id"].astype(str).isin(wanted)]

    default_runs = 10000 if args.phase == "deep" else 2000
    runs = int(args.runs or default_runs)
    run_column = "deep_run_dir" if args.phase == "deep" else "scan_run_dir"
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        config_path = workspace / str(row.config_path)
        run_dir = workspace / str(getattr(row, run_column))
        if _complete(run_dir, runs):
            print(f"SKIP complete {row.window_id}: {run_dir}")
            continue
        command = [
            sys.executable, "-m", "quantpits.scripts.run_monkey_experiment",
            "--config", str(config_path), "--runs", str(runs), "--run-dir", str(run_dir),
        ]
        if args.workers:
            command += ["--workers", str(args.workers)]
        if run_dir.exists() and any(run_dir.iterdir()):
            command.append("--resume")
        print("RUN " + subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=workspace.parent.parent)
        if completed.returncode != 0:
            failures.append(str(row.window_id))
            print(f"FAILED {row.window_id}; continuing with later windows", flush=True)
    if failures:
        print("Failed windows: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
