#!/usr/bin/env python
"""Aggregate rolling monkey windows and build an interactive HTML report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantpits.experiments.monkey import REGIME_NAMES


METRICS = (
    "total_return", "annualized_return", "benchmark_total_return",
    "excess_total_return", "annualized_alpha", "benchmark_beta",
    "information_ratio", "max_drawdown", "total_cost", "total_turnover",
    "hac_alpha_t_stat", "hac_alpha_p_value",
)
QUANTILES = {"p05": 0.05, "p25": 0.25, "median": 0.50, "p75": 0.75, "p95": 0.95}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tier", choices=("scan", "deep"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baselines", help="Matched baseline_window_summary.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def _bootstrap_interval(values: pd.Series, quantile: float, samples: int, seed: int) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(array) or samples <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        estimates[index] = np.quantile(rng.choice(array, size=len(array), replace=True), quantile)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(number) else f"{number:.2%}"


def _build_report(summary: pd.DataFrame, regimes: pd.DataFrame, output: Path, tier: str) -> None:
    labels = summary["window_id"] + "<br>" + summary["start_time"] + " → " + summary["end_time"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("两年总收益分布", "年化 Alpha 随窗口变化", "胜率与显著性", "回撤与交易成本"),
        vertical_spacing=0.16, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Scatter(x=labels, y=summary["total_return_p95"], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=summary["total_return_p05"], fill="tonexty", fillcolor="rgba(33,150,243,.18)", line=dict(width=0), name="P05–P95"), row=1, col=1)
    fig.add_trace(go.Scatter(x=labels, y=summary["total_return_median"], name="猴群中位数", line=dict(color="#1565c0", width=3)), row=1, col=1)
    cap_weight = summary["cap_weight_total_return"] if "cap_weight_total_return" in summary else summary["benchmark_total_return_median"]
    fig.add_trace(go.Scatter(x=labels, y=cap_weight, name="沪深300市值加权", line=dict(color="#ef6c00", dash="dash")), row=1, col=1)
    if "equal_weight_gross_total_return" in summary:
        fig.add_trace(go.Scatter(x=labels, y=summary["equal_weight_gross_total_return"], name="动态等权毛收益", line=dict(color="#00897b", dash="dot")), row=1, col=1)
    if "equal_weight_net_total_return" in summary:
        fig.add_trace(go.Scatter(x=labels, y=summary["equal_weight_net_total_return"], name="动态等权净收益", line=dict(color="#00695c", width=3)), row=1, col=1)
    for statistic, color in (("p95", "#c62828"), ("p75", "#ef6c00"), ("median", "#1565c0")):
        fig.add_trace(go.Scatter(x=labels, y=summary[f"annualized_alpha_{statistic}"], name=f"Alpha {statistic.upper()}", line=dict(color=color)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#777", row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=summary["beat_benchmark_probability"], name="跑赢基准概率", marker_color="#2e7d32"), row=2, col=1)
    if "beat_equal_weight_net_probability" in summary:
        fig.add_trace(go.Scatter(x=labels, y=summary["beat_equal_weight_net_probability"], name="跑赢等权净值概率", line=dict(color="#00897b", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=labels, y=summary["hac_significant_positive_probability"], name="HAC显著正Alpha概率", line=dict(color="#ad1457", width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=labels, y=summary["max_drawdown_median"], name="中位最大回撤", line=dict(color="#6a1b9a")), row=2, col=2)
    fig.add_trace(go.Scatter(x=labels, y=summary["total_cost_median"], name="中位总成本", line=dict(color="#5d4037"), yaxis="y4"), row=2, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=2)
    fig.update_layout(height=850, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=-0.14), margin=dict(l=50, r=30, t=70, b=140))
    chart_html = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})

    regime_html = "<p>当前层没有可用的市场状态数据。</p>"
    if not regimes.empty:
        pivot = regimes.pivot(index="regime", columns="window_id", values="annualized_alpha_median").reindex(REGIME_NAMES)
        heat = go.Figure(go.Heatmap(
            z=pivot.to_numpy(), x=pivot.columns, y=pivot.index,
            colorscale="RdBu", zmid=0, colorbar=dict(tickformat=".1%", title="年化Alpha"),
            hovertemplate="%{y} / %{x}<br>中位年化Alpha %{z:.2%}<extra></extra>",
        ))
        heat.update_layout(height=420, template="plotly_white", margin=dict(l=90, r=40, t=30, b=50))
        regime_html = heat.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})

    table_columns = [
        "window_id", "start_time", "end_time", "successful_runs",
        "benchmark_total_return_median", "total_return_p05", "total_return_median", "total_return_p95",
        "annualized_alpha_p75", "annualized_alpha_p95", "beat_benchmark_probability",
        "hac_significant_positive_probability",
    ]
    if "equal_weight_net_total_return" in summary:
        insert_at = table_columns.index("total_return_p05")
        table_columns[insert_at:insert_at] = [
            "equal_weight_gross_total_return", "equal_weight_net_total_return",
            "monkey_median_minus_equal_weight_net", "beat_equal_weight_net_probability",
        ]
    display = summary[table_columns].copy()
    for column in display.columns:
        if column not in {"window_id", "start_time", "end_time", "successful_runs"}:
            display[column] = display[column].map(_format_percent)
    base_headers = ["窗口", "开始", "结束", "有效Trial", "市值加权", "猴群P05", "猴群中位", "猴群P95", "Alpha P75", "Alpha P95", "跑赢概率", "HAC显著正Alpha"]
    if "equal_weight_net_total_return" in summary:
        base_headers[5:5] = ["等权毛收益", "等权净收益", "猴群中位−等权净值", "跑赢等权概率"]
    display.columns = base_headers

    market_column = "cap_weight_total_return" if "cap_weight_total_return" in summary else "benchmark_total_return_median"
    best = summary.loc[summary[market_column].idxmax()]
    worst = summary.loc[summary[market_column].idxmin()]
    generated = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M")
    document = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>猴群滚动窗口与市场状态报告</title><style>
body{{font-family:Inter,'Microsoft YaHei',sans-serif;background:#f4f7fb;color:#172033;margin:0}}main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin-bottom:4px}}.sub{{color:#607086}}.cards{{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:14px;margin:24px 0}}.card,section{{background:white;border-radius:14px;box-shadow:0 4px 18px #20305012;padding:18px}}.card b{{display:block;font-size:26px;color:#0d47a1;margin-top:8px}}section{{margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #e8edf4;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:#eef3fa}}.scroll{{overflow:auto;max-height:550px}}code{{background:#eef3fa;padding:2px 5px;border-radius:4px}}.warn{{border-left:5px solid #ef6c00}}@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>猴群滚动两年窗口与牛熊切片报告</h1><div class=\"sub\">层级：{html.escape(tier)} ｜ 生成时间：{generated} ｜ 所有状态均由成交前已知数据定义</div>
<div class=\"cards\"><div class=\"card\">已完成窗口<b>{len(summary)}</b></div><div class=\"card\">有效Trial<b>{int(summary['successful_runs'].sum()):,}</b></div><div class=\"card\">最强市场窗口<b>{html.escape(str(best['window_id']))}</b><span>{_format_percent(best[market_column])}</span></div><div class=\"card\">最弱市场窗口<b>{html.escape(str(worst['window_id']))}</b><span>{_format_percent(worst[market_column])}</span></div></div>
<section><h2>跨窗口分布</h2>{chart_html}</section><section><h2>市场状态下的猴群 Alpha</h2><p>颜色是各状态子样本内、跨 Trial 的中位年化 Alpha。单状态周数较少时只用于探索，显著性应查看 HAC 结果。</p>{regime_html}</section>
<section><h2>窗口明细</h2><div class=\"scroll\">{display.to_html(index=False, border=0, escape=True)}</div></section>
<section class=\"warn\"><h2>如何用于模型门槛</h2><p>先检查猴群中位数是否接近采用同一成交、不可交易、95%仓位和费用语义的动态等权净收益；两者差异衡量有限持仓、部分换手、权重漂移与收益分布偏斜。之后模型再与猴群 Alpha P75/P95 对账，并检查 HAC 显著性。</p><p>季度步长的两年窗口高度重叠，不能把 {len(summary)} 个窗口当作 {len(summary)} 个独立样本；W01/W09/W17 等不重叠锚点用于独立复核。扫描层用于找问题和挑重点窗口，深跑层才用于稳定估计尾部分位数。</p></section>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")


def main() -> int:
    args = _args()
    from quantpits.utils import env

    workspace = Path(env.ROOT_DIR).resolve()
    manifest = pd.read_csv(Path(args.manifest).resolve())
    run_column = "scan_run_dir" if args.tier == "scan" else "deep_run_dir"
    summary_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for window_index, row in manifest.iterrows():
        results_path = workspace / str(row[run_column]) / "results.csv"
        if not results_path.exists():
            missing.append(str(row["window_id"]))
            continue
        frame = pd.read_csv(results_path)
        successful = frame[frame["status"].eq("ok")].copy()
        if successful.empty:
            missing.append(str(row["window_id"]))
            continue
        item: dict[str, object] = {
            "window_id": row["window_id"], "start_time": row["start_time"], "end_time": row["end_time"],
            "is_anchor": int(row["is_anchor"]), "successful_runs": int(len(successful)),
            "failed_runs": int(len(frame) - len(successful)),
        }
        for metric in METRICS:
            if metric not in successful:
                continue
            values = pd.to_numeric(successful[metric], errors="coerce").dropna()
            if values.empty:
                continue
            item[f"{metric}_mean"] = float(values.mean())
            for name, quantile in QUANTILES.items():
                item[f"{metric}_{name}"] = float(values.quantile(quantile))
        item["beat_benchmark_probability"] = float((pd.to_numeric(successful["excess_total_return"], errors="coerce") > 0).mean())
        hac_t = pd.to_numeric(successful.get("hac_alpha_t_stat"), errors="coerce")
        hac_p = pd.to_numeric(successful.get("hac_alpha_p_value"), errors="coerce")
        alpha = pd.to_numeric(successful["annualized_alpha"], errors="coerce")
        item["hac_significant_positive_probability"] = float(((hac_t > 0) & (hac_p < 0.05) & (alpha > 0)).mean())
        for metric in ("total_return", "annualized_alpha"):
            values = pd.to_numeric(successful[metric], errors="coerce")
            for name, quantile in (("p75", 0.75), ("p95", 0.95)):
                low, high = _bootstrap_interval(values, quantile, args.bootstrap_samples, seed=1000 + window_index * 20 + int(quantile * 100))
                item[f"{metric}_{name}_ci_low"] = low
                item[f"{metric}_{name}_ci_high"] = high
        summary_rows.append(item)

        for regime in REGIME_NAMES:
            return_col = f"{regime}_return"
            if return_col not in successful:
                continue
            regime_item: dict[str, object] = {"window_id": row["window_id"], "regime": regime}
            for metric in ("weeks", "return", "benchmark_return", "excess_return", "annualized_alpha", "hac_alpha_t_stat", "hac_alpha_p_value", "information_ratio", "cost", "turnover"):
                column = f"{regime}_{metric}"
                values = pd.to_numeric(successful[column], errors="coerce").dropna()
                if not values.empty:
                    regime_item[f"{metric}_median"] = float(values.median())
                    regime_item[f"{metric}_p75"] = float(values.quantile(0.75))
                    regime_item[f"{metric}_p95"] = float(values.quantile(0.95))
            regime_rows.append(regime_item)

    if not summary_rows:
        raise RuntimeError(f"no completed {args.tier} results were found")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    regimes = pd.DataFrame(regime_rows)
    if args.baselines:
        baselines = pd.read_csv(Path(args.baselines).resolve())
        required = {
            "window_id", "cap_weight_total_return",
            "equal_weight_gross_total_return", "equal_weight_net_total_return",
        }
        missing_columns = required.difference(baselines.columns)
        if missing_columns:
            raise ValueError(f"baseline summary is missing columns: {sorted(missing_columns)}")
        missing_windows = set(summary["window_id"]).difference(baselines["window_id"])
        if missing_windows:
            raise ValueError(f"baseline summary is missing windows: {sorted(missing_windows)}")
        summary = summary.merge(baselines, on="window_id", how="left", validate="one_to_one", suffixes=("", "_baseline"))
        summary["monkey_median_minus_equal_weight_net"] = (
            summary["total_return_median"] - summary["equal_weight_net_total_return"]
        )
        summary["monkey_mean_minus_equal_weight_net"] = (
            summary["total_return_mean"] - summary["equal_weight_net_total_return"]
        )
        beat_probabilities: list[float] = []
        equal_weight_percentiles: list[float] = []
        manifest_by_window = manifest.set_index("window_id")
        for baseline_row in summary.itertuples(index=False):
            manifest_row = manifest_by_window.loc[baseline_row.window_id]
            results_path = workspace / str(manifest_row[run_column]) / "results.csv"
            trial_frame = pd.read_csv(results_path, usecols=["status", "total_return"])
            trial_returns = pd.to_numeric(
                trial_frame.loc[trial_frame["status"].eq("ok"), "total_return"],
                errors="coerce",
            ).dropna()
            equal_weight_net = float(baseline_row.equal_weight_net_total_return)
            beat_probabilities.append(float((trial_returns > equal_weight_net).mean()))
            equal_weight_percentiles.append(float((trial_returns <= equal_weight_net).mean()))
        summary["beat_equal_weight_net_probability"] = beat_probabilities
        summary["equal_weight_net_monkey_percentile"] = equal_weight_percentiles
    summary.to_csv(output_dir / "rolling_window_summary.csv", index=False, encoding="utf-8-sig")
    regimes.to_csv(output_dir / "regime_summary.csv", index=False, encoding="utf-8-sig")
    audit = {"tier": args.tier, "completed_windows": len(summary), "missing_windows": missing, "bootstrap_samples": args.bootstrap_samples}
    (output_dir / "aggregation_metadata.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_dir / "rolling_regime_report.html"
    _build_report(summary, regimes, report_path, args.tier)
    print(f"Rolling report: {report_path}")
    if missing:
        print("Not aggregated (not completed in this tier): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
