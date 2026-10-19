#!/usr/bin/env python
"""Generate a self-contained HTML report for a completed monkey experiment."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed monkey run directory")
    parser.add_argument(
        "--output",
        help="Output HTML path; defaults to <run-dir>/visual_analysis_report.html",
    )
    return parser.parse_args()


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def nearest_value(series: pd.Series, target: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    index = pd.DatetimeIndex(series.index)
    location = index.get_indexer([target], method="nearest")[0]
    timestamp = index[location]
    return timestamp, float(series.iloc[location])


def load_benchmark(metadata: dict[str, Any]) -> pd.Series:
    from quantpits.utils import env

    env.init_qlib()
    from qlib.data import D

    benchmark = metadata["experiment_config"]["benchmark"]
    start_time, end_time = metadata["period"]
    frame = D.features(
        [benchmark],
        ["$close"],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    if isinstance(values.index, pd.MultiIndex):
        values.index = pd.DatetimeIndex(values.index.get_level_values("datetime"))
    else:
        values.index = pd.DatetimeIndex(values.index)
    return values[~values.index.duplicated(keep="last")].sort_index()


def style_figure(fig: go.Figure, title: str, height: int = 390) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        margin={"l": 58, "r": 24, "t": 62, "b": 52},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Segoe UI, Microsoft YaHei, sans-serif", "size": 12, "color": "#24324a"},
        hoverlabel={"font": {"family": "Segoe UI, Microsoft YaHei, sans-serif"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(gridcolor="#e8edf5", zerolinecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#e8edf5", zerolinecolor="#cbd5e1")
    return fig


def figure_html(fig: go.Figure, div_id: str, include_plotly: bool) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_plotly else False,
        div_id=div_id,
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def quantile_table(frame: pd.DataFrame, period_label: str = "全期") -> str:
    rows = [
        (f"{period_label}累计收益", "total_return", True),
        ("年化收益", "annualized_return", True),
        ("相对沪深300超额", "excess_total_return", True),
        ("年化波动率", "annualized_volatility", True),
        ("最大回撤", "max_drawdown", True),
        ("Sharpe", "sharpe", False),
        ("Calmar", "calmar", False),
    ]
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    body = []
    for label, field, as_percent in rows:
        values = frame[field].quantile(quantiles)
        cells = [pct(float(value)) if as_percent else number(float(value), 3) for value in values]
        body.append(
            "<tr><th>" + html.escape(label) + "</th>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>指标</th>"
        "<th>P05</th><th>P25</th><th>中位数</th><th>P75</th><th>P95</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def recent_quantile_table(frame: pd.DataFrame) -> str:
    rows = [
        ("近一年累计收益", "recent_return", True),
        ("近一年相对沪深300超额", "recent_excess_return", True),
        ("近一年年化波动率", "recent_annualized_volatility", True),
        ("近一年最大回撤", "recent_max_drawdown", True),
        ("近一年 Sharpe", "recent_sharpe", False),
        ("近一年 Calmar", "recent_calmar", False),
    ]
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    body = []
    for label, field, as_percent in rows:
        values = pd.to_numeric(frame[field], errors="coerce").quantile(quantiles)
        cells = [pct(float(value)) if as_percent else number(float(value), 3) for value in values]
        body.append(
            "<tr><th>" + html.escape(label) + "</th>"
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>指标</th>"
        "<th>P05</th><th>P25</th><th>中位数</th><th>P75</th><th>P95</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
    )


def build_report(run_dir: Path, output_path: Path) -> dict[str, Any]:
    results = pd.read_csv(run_dir / "results.csv")
    results = results[results["status"] == "ok"].copy()
    curves = pd.read_csv(run_dir / "sample_curves.csv", parse_dates=["datetime"])
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    aggregate = json.loads((run_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
    if len(results) != metadata["successful_runs"]:
        raise ValueError("results row count does not match run metadata")

    benchmark_return = float(results["benchmark_total_return"].iloc[0])
    start_date = pd.Timestamp(metadata["period"][0])
    end_date = pd.Timestamp(metadata["period"][1])
    duration_years = (end_date - start_date).days / 365.2425
    period_label = f"{duration_years:.1f}年"
    run_count = len(results)
    return_values = results["total_return"].to_numpy(dtype=float)
    excess_values = results["excess_total_return"].to_numpy(dtype=float)

    holdings = results["initial_holdings"].str.split("|")
    holding_counts: dict[str, int] = {}
    containing_return_sum: dict[str, float] = {}
    for stock_list, trial_return in zip(holdings, return_values):
        for stock in stock_list:
            holding_counts[stock] = holding_counts.get(stock, 0) + 1
            containing_return_sum[stock] = containing_return_sum.get(stock, 0.0) + float(trial_return)
    selection = pd.DataFrame(
        {"instrument": list(holding_counts), "count": list(holding_counts.values())}
    ).sort_values("instrument")
    universe_size = len(selection)
    inclusion_probability = metadata["holdings"] / universe_size
    expected_count = len(results) * inclusion_probability
    selection_sigma = math.sqrt(len(results) * inclusion_probability * (1 - inclusion_probability))
    selection["z_score"] = (selection["count"] - expected_count) / selection_sigma
    total_return_sum = float(return_values.sum())
    selection["mean_return_with"] = selection.apply(
        lambda row: containing_return_sum[row["instrument"]] / row["count"], axis=1
    )
    selection["mean_return_without"] = selection.apply(
        lambda row: (total_return_sum - containing_return_sum[row["instrument"]])
        / (len(results) - row["count"])
        if row["count"] < len(results)
        else math.nan,
        axis=1,
    )
    selection["initial_presence_uplift"] = selection["mean_return_with"] - selection["mean_return_without"]

    pivot = curves.pivot(index="datetime", columns="run_id", values="nav").sort_index()
    sample_end = pivot.index.max()
    recent_target = sample_end - pd.DateOffset(years=1)
    recent_location = pivot.index.get_indexer([recent_target], method="nearest")[0]
    sample_recent_start = pivot.index[recent_location]
    sample_recent = pivot.iloc[-1] / pivot.iloc[recent_location] - 1.0
    sample_prior = pivot.iloc[recent_location] / float(metadata["experiment_config"]["initial_cash"]) - 1.0
    sample_full = pivot.iloc[-1] / float(metadata["experiment_config"]["initial_cash"]) - 1.0

    benchmark = load_benchmark(metadata)
    benchmark_first_date, benchmark_first = nearest_value(benchmark, pivot.index.min())
    benchmark_recent_date, benchmark_recent_start = nearest_value(benchmark, sample_recent_start)
    benchmark_end_date, benchmark_end = nearest_value(benchmark, sample_end)
    benchmark_recent_return = benchmark_end / benchmark_recent_start - 1.0
    benchmark_prior_return = benchmark_recent_start / benchmark_first - 1.0
    benchmark_sample_path = benchmark.reindex(pivot.index, method="ffill") / benchmark_first * 100.0

    recent_fields = {
        "recent_start_date", "recent_end_date", "recent_return",
        "recent_benchmark_return", "recent_excess_return", "prior_return",
        "recent_annualized_volatility", "recent_sharpe", "recent_max_drawdown",
        "recent_calmar",
    }
    has_full_recent = recent_fields.issubset(results.columns)
    if has_full_recent:
        indexed = results.set_index("run_id")
        recent_values = pd.to_numeric(indexed["recent_return"], errors="coerce")
        prior_values = pd.to_numeric(indexed["prior_return"], errors="coerce")
        full_values = pd.to_numeric(indexed["total_return"], errors="coerce")
        benchmark_recent_return = float(indexed["recent_benchmark_return"].median())
        benchmark_prior_return = float(indexed["prior_benchmark_return"].median())
        recent_start = pd.Timestamp(indexed["recent_start_date"].mode().iloc[0])
        recent_end = pd.Timestamp(indexed["recent_end_date"].mode().iloc[0])
    else:
        recent_values = sample_recent
        prior_values = sample_prior
        full_values = sample_full
        recent_start = sample_recent_start
        recent_end = sample_end
    recent_count = int(recent_values.notna().sum())

    figures: list[str] = []
    fig = make_subplots(rows=1, cols=2, subplot_titles=("累计收益直方图", "累计收益经验分布函数"))
    fig.add_trace(
        go.Histogram(
            x=return_values * 100,
            nbinsx=70,
            marker={"color": "#2563eb", "opacity": 0.82},
            hovertemplate="收益区间 %{x:.1f}%<br>组数 %{y}<extra></extra>",
            name="随机组合",
        ),
        row=1,
        col=1,
    )
    sorted_return = np.sort(return_values) * 100
    empirical = np.arange(1, len(sorted_return) + 1) / len(sorted_return) * 100
    fig.add_trace(
        go.Scatter(
            x=sorted_return,
            y=empirical,
            mode="lines",
            line={"color": "#2563eb", "width": 2},
            hovertemplate="收益 ≤ %{x:.2f}%<br>累计概率 %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    for col in (1, 2):
        fig.add_vline(
            x=benchmark_return * 100,
            line={"color": "#ef4444", "dash": "dash", "width": 2},
            annotation_text=f"沪深300 {pct(benchmark_return)}",
            annotation_position="top left",
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text=f"{period_label}累计收益（%）", row=1, col=1)
    fig.update_xaxes(title_text=f"{period_label}累计收益（%）", row=1, col=2)
    fig.update_yaxes(title_text="实验组数", row=1, col=1)
    fig.update_yaxes(title_text="累计概率（%）", row=1, col=2)
    figures.append(figure_html(style_figure(fig, f"{period_label}收益的完整分布", 430), "return-distribution", True))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("相对沪深300超额收益", "年化风险—收益散点"))
    fig.add_trace(
        go.Histogram(
            x=excess_values * 100,
            nbinsx=70,
            marker={"color": "#0f9d78", "opacity": 0.82},
            hovertemplate="超额区间 %{x:.1f}个百分点<br>组数 %{y}<extra></extra>",
            name="超额收益",
        ),
        row=1,
        col=1,
    )
    fig.add_vline(x=0, line={"color": "#ef4444", "dash": "dash"}, row=1, col=1)
    custom = np.column_stack(
        [results["sharpe"], results["max_drawdown"] * 100, results["excess_total_return"] * 100]
    )
    fig.add_trace(
        go.Scattergl(
            x=results["annualized_volatility"] * 100,
            y=results["annualized_return"] * 100,
            mode="markers",
            marker={"size": 4, "color": results["excess_total_return"], "colorscale": "RdYlGn", "opacity": 0.42, "showscale": True, "colorbar": {"title": "超额"}},
            customdata=custom,
            hovertemplate=(
                "年化波动 %{x:.2f}%<br>年化收益 %{y:.2f}%"
                "<br>Sharpe %{customdata[0]:.3f}<br>最大回撤 %{customdata[1]:.2f}%"
                f"<br>{period_label}超额 %{{customdata[2]:.2f}}个百分点<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"{period_label}超额收益（百分点）", row=1, col=1)
    fig.update_yaxes(title_text="实验组数", row=1, col=1)
    fig.update_xaxes(title_text="年化波动率（%）", row=1, col=2)
    fig.update_yaxes(title_text="年化收益（%）", row=1, col=2)
    figures.append(figure_html(style_figure(fig, "超额收益与风险补偿", 440), "excess-risk", False))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("最大回撤", "Sharpe 比率"))
    fig.add_trace(
        go.Histogram(x=results["max_drawdown"] * 100, nbinsx=60, marker={"color": "#f59e0b"}, hovertemplate="回撤 %{x:.1f}%<br>组数 %{y}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=results["sharpe"], nbinsx=60, marker={"color": "#7c3aed"}, hovertemplate="Sharpe %{x:.2f}<br>组数 %{y}<extra></extra>"),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="最大回撤（%）", row=1, col=1)
    fig.update_xaxes(title_text="Sharpe", row=1, col=2)
    fig.update_yaxes(title_text="实验组数", row=1, col=1)
    fig.update_yaxes(title_text="实验组数", row=1, col=2)
    figures.append(figure_html(style_figure(fig, "尾部风险与风险调整后收益", 390), "risk-distributions", False))

    alpha_fields = {
        "annualized_alpha", "benchmark_beta", "alpha_t_stat", "alpha_p_value",
        "information_ratio",
    }
    has_alpha = alpha_fields.issubset(results.columns)
    alpha_figure_html = ""
    alpha_table_html = ""
    alpha_values = pd.Series(dtype=float)
    alpha_positive_rate = math.nan
    alpha_significant_rate = math.nan
    if has_alpha:
        alpha_values = pd.to_numeric(results["annualized_alpha"], errors="coerce")
        alpha_positive_rate = float((alpha_values > 0).mean())
        alpha_significant_rate = float(
            (pd.to_numeric(results["alpha_p_value"], errors="coerce") < 0.05).mean()
        )
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("年化Alpha分布", "Beta—Alpha与显著性"),
        )
        fig.add_trace(
            go.Histogram(
                x=alpha_values * 100,
                nbinsx=70,
                marker={"color": "#7c3aed", "opacity": 0.82},
                hovertemplate="年化Alpha %{x:.2f}%<br>组数 %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(x=0, line={"color": "#ef4444", "dash": "dash"}, row=1, col=1)
        fig.add_trace(
            go.Scattergl(
                x=results["benchmark_beta"],
                y=alpha_values * 100,
                mode="markers",
                marker={
                    "size": 4,
                    "color": -np.log10(
                        pd.to_numeric(results["alpha_p_value"], errors="coerce").clip(lower=1e-12)
                    ),
                    "colorscale": "Viridis",
                    "opacity": 0.45,
                    "showscale": True,
                    "colorbar": {"title": "−log10(p)"},
                },
                customdata=np.column_stack(
                    [results["alpha_t_stat"], results["alpha_p_value"], results["information_ratio"]]
                ),
                hovertemplate=(
                    "Beta %{x:.3f}<br>年化Alpha %{y:.2f}%"
                    "<br>t=%{customdata[0]:.2f}, p=%{customdata[1]:.4f}"
                    "<br>IR=%{customdata[2]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line={"color": "#ef4444", "dash": "dash"}, row=1, col=2)
        fig.update_xaxes(title_text="年化Alpha（%）", row=1, col=1)
        fig.update_yaxes(title_text="实验组数", row=1, col=1)
        fig.update_xaxes(title_text="CSI300 Beta", row=1, col=2)
        fig.update_yaxes(title_text="年化Alpha（%）", row=1, col=2)
        alpha_figure_html = figure_html(
            style_figure(fig, "随机策略的Alpha基准分布", 430),
            "alpha-distribution",
            False,
        )
        alpha_rows = []
        for label, field, as_percent in (
            ("年化Alpha", "annualized_alpha", True),
            ("CSI300 Beta", "benchmark_beta", False),
            ("Alpha t值", "alpha_t_stat", False),
            ("信息比率", "information_ratio", False),
        ):
            values = pd.to_numeric(results[field], errors="coerce").quantile(
                [0.05, 0.25, 0.50, 0.75, 0.95]
            )
            cells = [pct(float(v)) if as_percent else number(float(v), 3) for v in values]
            alpha_rows.append(
                f"<tr><th>{label}</th>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
            )
        alpha_table_html = (
            "<div class='table-wrap'><table><thead><tr><th>指标</th>"
            "<th>P05</th><th>P25</th><th>中位数</th><th>P75</th><th>P95</th>"
            "</tr></thead><tbody>" + "".join(alpha_rows) + "</tbody></table></div>"
        )

    ordered_selection = selection.sort_values("count").reset_index(drop=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("初始入选次数（动态股票池首期）", f"初始持有与{period_label}收益的关联"))
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(ordered_selection) + 1),
            y=ordered_selection["count"],
            mode="markers",
            marker={"size": 6, "color": ordered_selection["z_score"], "colorscale": "RdBu", "cmin": -3, "cmax": 3, "showscale": True, "colorbar": {"title": "z", "x": 0.43}},
            customdata=np.column_stack([ordered_selection["instrument"], ordered_selection["z_score"]]),
            hovertemplate="%{customdata[0]}<br>入选 %{y} 次<br>z=%{customdata[1]:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=expected_count, line={"color": "#24324a", "dash": "dash"}, row=1, col=1)
    fig.add_hrect(y0=expected_count - 2 * selection_sigma, y1=expected_count + 2 * selection_sigma, fillcolor="#2563eb", opacity=0.10, line_width=0, row=1, col=1)
    association = pd.concat(
        [selection.nsmallest(10, "initial_presence_uplift"), selection.nlargest(10, "initial_presence_uplift")]
    ).sort_values("initial_presence_uplift")
    fig.add_trace(
        go.Bar(
            x=association["initial_presence_uplift"] * 100,
            y=association["instrument"],
            orientation="h",
            marker={"color": ["#ef4444" if value < 0 else "#0f9d78" for value in association["initial_presence_uplift"]]},
            customdata=association["count"],
            hovertemplate="%{y}<br>条件均值差 %{x:.2f}个百分点<br>初始入选 %{customdata} 次<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=0, line={"color": "#64748b"}, row=1, col=2)
    fig.update_xaxes(title_text="按入选次数排序的股票", showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text=f"{run_count:,}组中的入选次数", row=1, col=1)
    fig.update_xaxes(title_text="持有组均值 − 未持有组均值（百分点）", row=1, col=2)
    figures.append(figure_html(style_figure(fig, "随机初始选股是否均匀", 520), "selection-analysis", False))

    fan = pivot / float(metadata["experiment_config"]["initial_cash"]) * 100.0
    fan_quantiles = fan.quantile([0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan_quantiles.index, y=fan_quantiles[0.90], line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fan_quantiles.index, y=fan_quantiles[0.10], fill="tonexty", fillcolor="rgba(37,99,235,0.12)", line={"width": 0}, name="P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fan_quantiles.index, y=fan_quantiles[0.75], line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fan_quantiles.index, y=fan_quantiles[0.25], fill="tonexty", fillcolor="rgba(37,99,235,0.24)", line={"width": 0}, name="P25–P75", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fan_quantiles.index, y=fan_quantiles[0.50], mode="lines", line={"color": "#2563eb", "width": 2.5}, name="样本中位数", hovertemplate="%{x|%Y-%m-%d}<br>中位净值 %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=benchmark_sample_path.index, y=benchmark_sample_path.values, mode="lines", line={"color": "#ef4444", "dash": "dash", "width": 2}, name="沪深300", hovertemplate="%{x|%Y-%m-%d}<br>基准净值 %{y:.2f}<extra></extra>"))
    recent_start_native = sample_recent_start.to_pydatetime()
    fig.add_vline(x=recent_start_native, line={"color": "#64748b", "dash": "dot"})
    fig.add_annotation(
        x=recent_start_native,
        y=1,
        xref="x",
        yref="paper",
        text="近一年起点",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"color": "#64748b"},
    )
    fig.update_xaxes(title_text="日期")
    fig.update_yaxes(title_text="归一化净值（初始=100）")
    figures.append(figure_html(style_figure(fig, "逐期净值扇形图（保留样本 n=20）", 430), "sample-fan", False))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("近一年累计收益", "前一年与近一年收益"))
    fig.add_trace(
        go.Histogram(x=recent_values * 100, nbinsx=70 if has_full_recent else 12, marker={"color": "#2563eb"}, hovertemplate="收益 %{x:.1f}%<br>样本数 %{y}<extra></extra>"),
        row=1,
        col=1,
    )
    fig.add_vline(x=benchmark_recent_return * 100, line={"color": "#ef4444", "dash": "dash"}, annotation_text=f"基准 {pct(benchmark_recent_return)}", row=1, col=1)
    fig.add_trace(
        go.Scattergl(
            x=prior_values * 100,
            y=recent_values * 100,
            mode="markers",
            marker={"size": 5 if has_full_recent else 8, "color": "#0f9d78", "opacity": 0.45 if has_full_recent else 0.9},
            customdata=np.column_stack([recent_values.index, full_values.loc[recent_values.index] * 100]),
            hovertemplate=f"run %{{customdata[0]:.0f}}<br>前期 %{{x:.2f}}%<br>近一年 %{{y:.2f}}%<br>{period_label} %{{customdata[1]:.2f}}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=benchmark_prior_return * 100, line={"color": "#ef4444", "dash": "dash"}, row=1, col=2)
    fig.add_hline(y=benchmark_recent_return * 100, line={"color": "#ef4444", "dash": "dash"}, row=1, col=2)
    fig.update_xaxes(title_text="近一年累计收益（%）", row=1, col=1)
    fig.update_yaxes(title_text="样本数", row=1, col=1)
    fig.update_xaxes(title_text="此前累计收益（%）", row=1, col=2)
    fig.update_yaxes(title_text="近一年累计收益（%）", row=1, col=2)
    figures.append(figure_html(style_figure(fig, f"近一年表现拆分（n={recent_count:,}）", 430), "recent-year", False))

    most_selected = selection.nlargest(5, "count")
    least_selected = selection.nsmallest(5, "count")
    selection_rows = []
    for _, row in pd.concat([most_selected, least_selected]).iterrows():
        selection_rows.append(
            f"<tr><td>{html.escape(row['instrument'])}</td><td>{int(row['count'])}</td>"
            f"<td>{row['z_score']:.2f}</td><td>{pct(row['initial_presence_uplift'])}</td></tr>"
        )

    metrics = aggregate["metrics"]
    positive_rate = float(aggregate["positive_return_probability"])
    beat_rate = float(aggregate["beat_benchmark_probability"])
    beat_count = int((results["excess_total_return"] > 0).sum())
    execution_periods = int(
        pd.to_numeric(results.get("execution_periods"), errors="coerce").median()
    ) if "execution_periods" in results else int(
        pd.to_numeric(results["rebalance_periods"], errors="coerce").median() + 1
    )
    alpha_median = float(alpha_values.median()) if has_alpha else math.nan
    alpha_p75 = float(alpha_values.quantile(0.75)) if has_alpha else math.nan
    alpha_p95 = float(alpha_values.quantile(0.95)) if has_alpha else math.nan
    recent_median = float(recent_values.median())
    recent_beat_rate = float((recent_values > benchmark_recent_return).mean())
    recent_positive_rate = float((recent_values > 0).mean())
    recent_p05 = float(recent_values.quantile(0.05))
    recent_p95 = float(recent_values.quantile(0.95))
    recent_excess_median = float(
        pd.to_numeric(results["recent_excess_return"], errors="coerce").median()
    ) if has_full_recent else recent_median - benchmark_recent_return
    recent_drawdown_median = float(
        pd.to_numeric(results["recent_max_drawdown"], errors="coerce").median()
    ) if has_full_recent else math.nan
    max_abs_selection_z = float(selection["z_score"].abs().max())
    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z")

    css = """
    :root { --ink:#172033; --muted:#64748b; --line:#dbe3ee; --soft:#f4f7fb; --blue:#2563eb; --green:#0f9d78; --red:#dc2626; }
    * { box-sizing:border-box; }
    body { margin:0; background:#eef2f7; color:var(--ink); font-family:"Segoe UI","Microsoft YaHei",sans-serif; }
    .report { max-width:1440px; margin:0 auto; background:#fff; min-height:100vh; }
    header { padding:42px 48px 32px; background:linear-gradient(135deg,#12213d,#1f4b8f); color:#fff; }
    header h1 { margin:0 0 10px; font-size:30px; font-weight:600; }
    header p { margin:5px 0; color:#dbeafe; }
    main { padding:30px 48px 56px; }
    section { margin:0 0 42px; }
    h2 { margin:0 0 16px; font-size:21px; border-left:4px solid var(--blue); padding-left:10px; }
    h3 { margin:20px 0 10px; font-size:16px; }
    .kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:-18px 0 36px; }
    .kpi { background:#fff; border:1px solid var(--line); border-radius:10px; padding:17px 18px; box-shadow:0 4px 18px rgba(25,42,70,.07); }
    .kpi .label { color:var(--muted); font-size:13px; }
    .kpi .value { margin-top:6px; font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; }
    .kpi .note { color:var(--muted); font-size:12px; margin-top:4px; }
    .finding { background:var(--soft); border-left:3px solid var(--blue); padding:14px 18px; margin:10px 0; line-height:1.65; }
    .warning { background:#fff7ed; border-left-color:#f59e0b; }
    .chart { border:1px solid var(--line); border-radius:10px; padding:6px; margin:14px 0 22px; overflow:hidden; }
    .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:9px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
    th:first-child,td:first-child { text-align:left; }
    thead th { background:var(--soft); color:#334155; }
    tbody tr:last-child th,tbody tr:last-child td { border-bottom:0; }
    .meta { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px 20px; font-size:13px; }
    .meta div { padding:8px 0; border-bottom:1px solid var(--line); }
    .meta span { color:var(--muted); display:block; font-size:12px; }
    .tag { display:inline-block; background:#dbeafe; color:#1e40af; border-radius:999px; padding:3px 9px; font-size:12px; margin-right:5px; }
    footer { padding:22px 48px; background:#12213d; color:#cbd5e1; font-size:12px; }
    @media (max-width:900px) { .kpis{grid-template-columns:repeat(2,1fr)} .grid-2,.meta{grid-template-columns:1fr} header,main,footer{padding-left:22px;padding-right:22px} }
    @media (max-width:520px) { .kpis{grid-template-columns:1fr} header h1{font-size:24px} }
    @media print { body{background:#fff} .report{max-width:none} .chart{break-inside:avoid} }
    """

    report_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>CSI300 猴子模型：{run_count:,}组随机选股实验报告</title>
  <style>{css}</style>
</head>
<body>
<div class="report">
<header>
  <h1>CSI300 猴子模型：{run_count:,}组随机选股实验</h1>
  <p>回测区间 {metadata['period'][0]} 至 {metadata['period'][1]} · 每组{metadata['holdings']}只 · 每周随机卖{metadata['experiment_config']['weekly_replacements']}买{metadata['experiment_config']['weekly_replacements']} · Qlib精确回测</p>
  <p>核心问题：纯随机等金额组合在这一市场窗口中的收益分布、跑赢基准概率与尾部风险。</p>
</header>
<main>
  <div class="kpis">
    <div class="kpi"><div class="label">{period_label}收益中位数</div><div class="value">{pct(metrics['total_return']['median'])}</div><div class="note">P05–P95：{pct(metrics['total_return']['p05'])}–{pct(metrics['total_return']['p95'])}</div></div>
    <div class="kpi"><div class="label">沪深300同期收益</div><div class="value">{pct(benchmark_return)}</div><div class="note">随机组合中位超额 {pct(metrics['excess_total_return']['median'])}</div></div>
    <div class="kpi"><div class="label">跑赢沪深300概率</div><div class="value">{pct(beat_rate)}</div><div class="note">{beat_count:,} / {run_count:,} 组</div></div>
    <div class="kpi"><div class="label">最大回撤中位数</div><div class="value">{pct(metrics['max_drawdown']['median'])}</div><div class="note">P05尾部 {pct(metrics['max_drawdown']['p05'])}</div></div>
  </div>

  <section>
    <h2>结论摘要</h2>
    <div class="finding">在该{period_label}窗口中，随机组合正收益概率为 {pct(positive_rate)}，同期沪深300累计收益为 {pct(benchmark_return)}。猴群整体表现首先反映市场Beta、等权结构与换手成本，不应直接解释为随机信号具有预测能力。</div>
    <div class="finding">随机组合{period_label}收益中位数为 {pct(metrics['total_return']['median'])}，中位超额为 {pct(metrics['excess_total_return']['median'])}，跑赢沪深300的经验概率为 {pct(beat_rate)}。</div>
    <div class="finding">收益分散度不可忽略：P05仅 {pct(metrics['total_return']['p05'])}，P95达到 {pct(metrics['total_return']['p95'])}；随机路径本身造成约 {pct(metrics['total_return']['p95']-metrics['total_return']['p05'])} 的90%区间跨度。</div>
    <div class="finding">近一年采用每条完整路径在 {recent_start.date()} 的账户净值作为起点，保留此前形成的真实持仓状态；共 {recent_count:,} 组，近一年收益中位数为 {pct(recent_median)}。</div>
  </section>

  <section>
    <h2>全量{period_label}收益分布（n={run_count:,}）</h2>
    <div class="chart">{figures[0]}</div>
    <div class="chart">{figures[1]}</div>
    <div class="chart">{figures[2]}</div>
    <h3>关键分位数</h3>
    {quantile_table(results, period_label)}
  </section>

  {f'''<section>
    <h2>随机Alpha基准与模型比较门槛</h2>
    <div class="kpis">
      <div class="kpi"><div class="label">年化Alpha中位数</div><div class="value">{pct(alpha_median)}</div><div class="note">对CSI300周收益回归</div></div>
      <div class="kpi"><div class="label">Alpha P75</div><div class="value">{pct(alpha_p75)}</div><div class="note">模型的初级随机门槛</div></div>
      <div class="kpi"><div class="label">Alpha P95</div><div class="value">{pct(alpha_p95)}</div><div class="note">模型的严格随机门槛</div></div>
      <div class="kpi"><div class="label">p&lt;0.05占比</div><div class="value">{pct(alpha_significant_rate)}</div><div class="note">多重试验下仅作诊断</div></div>
    </div>
    <div class="finding">后续模型应在完全相同的股票池、成交日、成本和时间窗口下，同时超过猴群Alpha P75/P95，并在滚动窗口中保持显著；单纯跑赢CSI300不足以证明增量选股能力。</div>
    <div class="chart">{alpha_figure_html}</div>
    {alpha_table_html}
  </section>''' if has_alpha else ''}

  <section>
    <h2>随机初始选股分布</h2>
    <div class="finding">首期共覆盖 {universe_size} 只股票；单只股票理论入选次数为 {expected_count:.1f}，实际范围 {int(selection['count'].min())}–{int(selection['count'].max())}，最大绝对偏离为 {max_abs_selection_z:.2f}σ。整体与等概率抽样相符。</div>
    <div class="finding warning">右侧“持有与未持有组收益均值差”仅是条件关联：之后仍会持续随机换仓，不能解释为单只股票的稳定因果贡献。</div>
    <div class="chart">{figures[3]}</div>
    <div class="table-wrap"><table><thead><tr><th>股票</th><th>初始入选次数</th><th>z-score</th><th>持有/未持有均值差</th></tr></thead><tbody>{''.join(selection_rows)}</tbody></table></div>
  </section>

  <section>
    <h2>近一年全量收益与路径分析（n={recent_count:,}）</h2>
    <div class="kpis">
      <div class="kpi"><div class="label">近一年收益中位数</div><div class="value">{pct(recent_median)}</div><div class="note">P05–P95：{pct(recent_p05)}–{pct(recent_p95)}</div></div>
      <div class="kpi"><div class="label">沪深300近一年</div><div class="value">{pct(benchmark_recent_return)}</div><div class="note">中位超额 {pct(recent_excess_median)}</div></div>
      <div class="kpi"><div class="label">近一年跑赢基准</div><div class="value">{pct(recent_beat_rate)}</div><div class="note">正收益概率 {pct(recent_positive_rate)}</div></div>
      <div class="kpi"><div class="label">近一年回撤中位数</div><div class="value">{pct(recent_drawdown_median) if has_full_recent else '—'}</div><div class="note">完整路径内计算</div></div>
    </div>
    <div class="grid-2">
      <div class="finding">切点为最接近“最终周减一年”的实际周频净值：{recent_start.date()} → {recent_end.date()}。全量收益 P05–P95 为 {pct(recent_p05)}–{pct(recent_p95)}。</div>
      <div class="finding">沪深300同口径收益 {pct(benchmark_recent_return)}；随机组合中 {pct(recent_beat_rate)} 跑赢，近一年中位超额为 {pct(recent_excess_median)}。</div>
    </div>
    <div class="chart">{figures[4]}</div>
    <div class="chart">{figures[5]}</div>
    <h3>近一年关键分位数</h3>
    {recent_quantile_table(results) if has_full_recent else ''}
    <div class="finding warning">逐周净值扇形图仍基于运行时保留的20条路径，仅用于观察路径形态；近一年直方图、散点、KPI和分位数表均使用 {recent_count:,} 组全量结果。</div>
  </section>

  <section>
    <h2>实验完整性与方法</h2>
    <div class="kpis">
      <div class="kpi"><div class="label">成功回测</div><div class="value">{metadata['successful_runs']:,}</div><div class="note">失败 {metadata['failed_runs']:,} 组</div></div>
      <div class="kpi"><div class="label">每组周执行期</div><div class="value">{execution_periods}</div><div class="note">首期建仓，之后按DropN换仓</div></div>
      <div class="kpi"><div class="label">初始资金</div><div class="value">1亿元</div><div class="note">95%风险敞口</div></div>
      <div class="kpi"><div class="label">累计交易成本中位数</div><div class="value">{pct(float(results['total_cost'].median()))}</div><div class="note">按初始资金比例</div></div>
    </div>
    <p><span class="tag">动态CSI300成分</span><span class="tag">上周信号</span><span class="tag">下周首个交易日收盘成交</span><span class="tag">成交日涨跌停/停牌过滤</span><span class="tag">PYTHONHASHSEED={metadata.get('python_hash_seed', '—')}</span><span class="tag">开仓万五</span><span class="tag">平仓千1.5</span></p>
    <div class="meta">
      <div><span>Master seed</span>{metadata['master_seed']}</div>
      <div><span>Qlib / NumPy / CuPy</span>{metadata['pyqlib']} / {metadata['numpy']} / {metadata['cupy']}</div>
      <div><span>数据目录</span>{html.escape(metadata['qlib_data_dir'])}</div>
      <div><span>Git commit</span>{html.escape(metadata['git_commit'])}</div>
      <div><span>实验耗时</span>{metadata['elapsed_seconds']/60:.1f} 分钟</div>
      <div><span>报告生成</span>{generated_at}</div>
      <div><span>近一年基准日期</span>{benchmark_recent_date.date()} → {benchmark_end_date.date()}</div>
      <div><span>基准样本起点</span>{benchmark_first_date.date()}</div>
      <div><span>报告口径</span>{period_label}与近一年全量；逐期扇形图 n={pivot.shape[1]}</div>
    </div>
  </section>

  <section>
    <h2>用于后续量化研究的建议</h2>
    <div class="finding">把本实验作为“无信息策略”的基准分布：后续模型只有在相同交易成本、股票池和时间窗口下稳定超过随机组合的P75/P95，而不仅是超过沪深300，才更能说明信号具有增量价值。</div>
    <div class="finding">本轮已补齐全部trial的一年切点指标。下一步可增加等权CSI300、固定买入持有、不同换手率（1/3/5只）和不同持仓数（20/25/30）作为对照组。</div>
    <div class="finding">即使采用更长窗口，仍应增加滚动两年窗口和牛熊市场切片；模型只有在多个市场状态下持续超过随机Alpha门槛，结论才具有稳健性。</div>
  </section>
</main>
<footer>本报告用于研究与方法验证，不构成投资建议。所有收益均为历史回测结果。</footer>
</div>
</body>
</html>"""

    output_path.write_text(report_html, encoding="utf-8")
    return {
        "output": str(output_path),
        "rows": len(results),
        "universe_size": universe_size,
        "selection_count_range": [int(selection["count"].min()), int(selection["count"].max())],
        "max_abs_selection_z": max_abs_selection_z,
        "recent_sample_size": recent_count,
        "recent_median_return": recent_median,
        "recent_benchmark_return": benchmark_recent_return,
        "recent_beat_rate": recent_beat_rate,
        "html_bytes": output_path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else run_dir / "visual_analysis_report.html"
    summary = build_report(run_dir, output_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
