#!/usr/bin/env python
"""Compare per-window model metrics with rolling monkey P75/P95 gates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkey-summary", required=True)
    parser.add_argument("--model-results", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    monkey = pd.read_csv(Path(args.monkey_summary).resolve())
    model = pd.read_csv(Path(args.model_results).resolve())
    required = {"window_id", "total_return", "annualized_alpha"}
    missing = required.difference(model.columns)
    if missing:
        raise ValueError(f"model result CSV is missing columns: {sorted(missing)}")
    merged = model.merge(monkey, on="window_id", how="inner", validate="one_to_one", suffixes=("_model", "_monkey"))
    if merged.empty:
        raise ValueError("model results and monkey summary have no matching window_id")
    for metric in ("total_return", "annualized_alpha"):
        for gate in ("p75", "p95"):
            merged[f"beats_monkey_{metric}_{gate}"] = (
                pd.to_numeric(merged[metric], errors="coerce")
                > pd.to_numeric(merged[f"{metric}_{gate}"], errors="coerce")
            )
    if "hac_alpha_p_value" in merged:
        merged["model_hac_significant_positive"] = (
            (pd.to_numeric(merged["hac_alpha_p_value"], errors="coerce") < 0.05)
            & (pd.to_numeric(merged["annualized_alpha"], errors="coerce") > 0)
        )
    else:
        merged["model_hac_significant_positive"] = False
    merged["passes_strict_alpha_gate"] = (
        merged["beats_monkey_annualized_alpha_p95"]
        & merged["model_hac_significant_positive"]
    )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "model_vs_rolling_monkey.csv"
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    x = merged["window_id"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=merged["annualized_alpha"], name="模型年化Alpha", marker_color="#1565c0"))
    fig.add_trace(go.Scatter(x=x, y=merged["annualized_alpha_p75"], name="猴群P75", line=dict(color="#ef6c00")))
    fig.add_trace(go.Scatter(x=x, y=merged["annualized_alpha_p95"], name="猴群P95", line=dict(color="#c62828")))
    fig.update_layout(template="plotly_white", height=560, yaxis_tickformat=".1%", hovermode="x unified")
    rates = {
        "alpha_p75": float(merged["beats_monkey_annualized_alpha_p75"].mean()),
        "alpha_p95": float(merged["beats_monkey_annualized_alpha_p95"].mean()),
        "strict": float(merged["passes_strict_alpha_gate"].mean()),
    }
    report = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>模型与猴群滚动门槛比较</title><style>body{{font-family:'Microsoft YaHei',sans-serif;max-width:1300px;margin:30px auto;color:#172033}}.cards{{display:flex;gap:16px}}.card{{flex:1;padding:18px;background:#eef4ff;border-radius:12px}}b{{font-size:26px;color:#0d47a1}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:right}}</style></head><body><h1>模型与滚动猴群门槛比较</h1><div class=\"cards\"><div class=\"card\">超过 Alpha P75<br><b>{rates['alpha_p75']:.1%}</b></div><div class=\"card\">超过 Alpha P95<br><b>{rates['alpha_p95']:.1%}</b></div><div class=\"card\">严格门槛<br><b>{rates['strict']:.1%}</b><br>超过P95且HAC p&lt;0.05</div></div>{fig.to_html(full_html=False,include_plotlyjs=True,config={'displaylogo':False})}<h2>逐窗口结果</h2>{merged.to_html(index=False,border=0)}</body></html>"""
    report_path = output / "model_vs_rolling_monkey.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"Comparison: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
