#!/usr/bin/env python
"""Compare one or more model backtests with a completed monkey baseline."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


METRICS = ("total_return", "excess_total_return", "annualized_alpha", "information_ratio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey-run-dir", required=True)
    parser.add_argument(
        "--model-results",
        required=True,
        help="CSV with model_name, total_return, annualized_alpha and optional comparison metrics",
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to <monkey-run-dir>/model_alpha_comparison",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def compare(monkey: pd.DataFrame, models: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"model_name", "total_return", "annualized_alpha"}
    missing = required - set(models.columns)
    if missing:
        raise ValueError(f"model results are missing columns: {sorted(missing)}")

    monkey = monkey[monkey["status"] == "ok"].copy()
    if monkey.empty:
        raise ValueError("monkey baseline contains no successful trials")
    thresholds: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        if metric not in monkey:
            continue
        values = pd.to_numeric(monkey[metric], errors="coerce").dropna()
        thresholds[metric] = {
            "p50": float(values.quantile(0.50)),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
        }

    rows: list[dict[str, Any]] = []
    for _, model in models.iterrows():
        row = model.to_dict()
        for metric, cuts in thresholds.items():
            if metric not in model or pd.isna(model[metric]):
                continue
            value = float(model[metric])
            baseline = pd.to_numeric(monkey[metric], errors="coerce").dropna()
            row[f"{metric}_monkey_percentile"] = float((baseline <= value).mean())
            row[f"{metric}_beats_p75"] = bool(value > cuts["p75"])
            row[f"{metric}_beats_p95"] = bool(value > cuts["p95"])
        alpha_p = pd.to_numeric(pd.Series([model.get("alpha_p_value")]), errors="coerce").iloc[0]
        row["alpha_significant_5pct"] = bool(pd.notna(alpha_p) and alpha_p < 0.05)
        row["strict_random_baseline_pass"] = bool(
            row.get("total_return_beats_p95", False)
            and row.get("annualized_alpha_beats_p95", False)
            and row["alpha_significant_5pct"]
        )
        rows.append(row)

    compared = pd.DataFrame(rows)
    summary = {
        "monkey_trials": int(len(monkey)),
        "thresholds": thresholds,
        "models": compared.to_dict(orient="records"),
        "strict_rule": (
            "total return > monkey P95 AND annualized alpha > monkey P95 "
            "AND alpha p-value < 0.05"
        ),
    }
    return compared, summary


def _pct(value: Any) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 100:.2f}%"


def write_html(compared: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    thresholds = summary["thresholds"]
    threshold_rows = "".join(
        f"<tr><th>{html.escape(metric)}</th><td>{_pct(values['p50'])}</td>"
        f"<td>{_pct(values['p75'])}</td><td>{_pct(values['p95'])}</td></tr>"
        for metric, values in thresholds.items()
    )
    model_rows = []
    for _, row in compared.iterrows():
        model_rows.append(
            "<tr>"
            f"<th>{html.escape(str(row['model_name']))}</th>"
            f"<td>{_pct(row['total_return'])}</td>"
            f"<td>{_pct(row.get('total_return_monkey_percentile'))}</td>"
            f"<td>{_pct(row['annualized_alpha'])}</td>"
            f"<td>{_pct(row.get('annualized_alpha_monkey_percentile'))}</td>"
            f"<td>{_pct(row.get('alpha_p_value'))}</td>"
            f"<td>{'通过' if row['strict_random_baseline_pass'] else '未通过'}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>模型与猴群Alpha比较</title><style>
body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#eef2f7;color:#172033;margin:0}}
main{{max-width:1200px;margin:auto;background:white;min-height:100vh;padding:36px}}
h1{{margin-top:0}} table{{width:100%;border-collapse:collapse;margin:18px 0 32px}}
th,td{{padding:10px 12px;border:1px solid #dbe3ee;text-align:right}}th:first-child{{text-align:left}}
thead th{{background:#f4f7fb}} .note{{padding:14px 18px;border-left:4px solid #2563eb;background:#f4f7fb}}
</style></head><body><main><h1>模型与猴群Alpha比较</h1>
<div class="note">严格判定：模型全期收益超过猴群P95、年化Alpha超过猴群P95，且Alpha p值小于0.05。所有模型必须使用相同股票池、成本、成交时点和窗口。</div>
<h2>猴群门槛（n={summary['monkey_trials']:,}）</h2>
<table><thead><tr><th>指标</th><th>P50</th><th>P75</th><th>P95</th></tr></thead><tbody>{threshold_rows}</tbody></table>
<h2>模型结果</h2><table><thead><tr><th>模型</th><th>累计收益</th><th>收益百分位</th><th>年化Alpha</th><th>Alpha百分位</th><th>Alpha p值</th><th>严格判定</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.monkey_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "model_alpha_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkey = pd.read_csv(run_dir / "results.csv")
    models = pd.read_csv(Path(args.model_results).resolve())
    compared, summary = compare(monkey, models)
    compared.to_csv(output_dir / "model_alpha_comparison.csv", index=False)
    (output_dir / "model_alpha_comparison.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_html(compared, summary, output_dir / "model_alpha_comparison.html")
    print(json.dumps({"output_dir": str(output_dir), "models": len(compared)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
