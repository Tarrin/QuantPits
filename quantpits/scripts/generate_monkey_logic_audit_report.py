#!/usr/bin/env python
"""Generate a self-contained HTML report from the monkey logic audit pack."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _figure_html(fig: go.Figure, div_id: str, include_plotly: bool) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_plotly else False,
        div_id=div_id,
        config={"displaylogo": False, "responsive": True},
    )


def _style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        margin={"l": 62, "r": 30, "t": 68, "b": 58},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={"family": "Segoe UI, Microsoft YaHei, sans-serif", "color": "#24324a"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"font": {"family": "Segoe UI, Microsoft YaHei, sans-serif"}},
    )
    fig.update_xaxes(gridcolor="#e8edf5", zerolinecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#e8edf5", zerolinecolor="#cbd5e1")
    return fig


def _step_table(summary: dict[str, Any]) -> str:
    rows = [
        ("动态 universe", "dynamic_universe", "01_dynamic_universe.md"),
        ("决策/成交时点", "execution_timing", "02_execution_timing.md"),
        ("DropN语义", "dropn", "03_dropn_semantics.md"),
        ("交易成本", "costs", "04_costs.md"),
        ("不可交易处理", "tradability", "05_tradability.md"),
        ("年化算法", "annualization", "06_annualization.md"),
        ("市场归因", "market_attribution", "07_market_attribution.md"),
        ("Trial复现", "reproducibility", "08_reproducibility.md"),
    ]
    labels = {
        "pass": "通过",
        "warning": "需改进",
        "issue": "问题",
        "pass_with_convention_note": "通过/口径说明",
        "market_driven": "市场驱动",
    }
    body = []
    for label, key, evidence in rows:
        step = summary["steps"][key]
        status = summary["severity"].get(key, "warning")
        body.append(
            f"<tr><th>{html.escape(label)}</th>"
            f"<td><span class='status {status}'>{labels.get(status, status)}</span></td>"
            f"<td>{html.escape(step['verdict'])}</td>"
            f"<td><code>{evidence}</code></td></tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>审计层</th><th>状态</th>"
        "<th>结论</th><th>证据文件</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def build_report(audit_dir: Path, output_path: Path) -> dict[str, Any]:
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    timing = pd.read_csv(audit_dir / "01_02_universe_timing_weekly.csv", parse_dates=[
        "report_index", "last_signal_date", "actual_close_execution_date"
    ])
    tradability = pd.read_csv(audit_dir / "05_tradability_weekly.csv")
    controls = pd.read_csv(audit_dir / "07_market_controls.csv")
    control_trials = pd.read_csv(audit_dir / "07_market_control_trials.csv")
    dynamic_weekly = pd.read_csv(
        audit_dir / "07_dynamic_equal_weight_weekly.csv", parse_dates=["start_date", "end_date"]
    )
    run_dir = Path(summary["run_dir"])
    results = pd.read_csv(run_dir / "results.csv")
    ic_trials = pd.read_csv(run_dir / "ic_audit_by_trial.csv")

    market = summary["steps"]["market_attribution"]
    universe = summary["steps"]["dynamic_universe"]
    timing_step = summary["steps"]["execution_timing"]
    dropn = summary["steps"]["dropn"]
    costs = summary["steps"]["costs"]
    trade = summary["steps"]["tradability"]
    annual = summary["steps"]["annualization"]
    repro = summary["steps"]["reproducibility"]
    ic = summary["ic_audit"]

    figures: list[str] = []
    label_map = {
        "CSI300_cap_weight": "沪深300市值加权",
        "CSI300_initial_static_equal_weight": "初始CSI300等权持有",
        "CSI300_dynamic_weekly_equal_weight": "动态CSI300周等权",
        "fixed_random_25_mean": "固定随机25均值",
        "fixed_random_25_median": "固定随机25中位数",
        "weekly_monkey_25_mean": "周换仓猴群均值",
        "weekly_monkey_25_median": "周换仓猴群中位数",
    }
    controls["label"] = controls["control"].map(label_map)
    colors = ["#ef4444", "#7c3aed", "#0f9d78", "#60a5fa", "#2563eb", "#f59e0b", "#d97706"]
    fig = go.Figure(
        go.Bar(
            x=controls["label"],
            y=controls["return"] * 100,
            marker={"color": colors},
            text=[f"{value:.2f}%" for value in controls["return"] * 100],
            textposition="outside",
            hovertemplate="%{x}<br>累计收益 %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="两年累计收益（%）")
    fig.update_xaxes(tickangle=-18)
    figures.append(_figure_html(_style(fig, "高收益从哪里来：市场与等权对照", 460), "market-controls", True))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("完整两年分布", "第一年与近一年"))
    fig.add_trace(
        go.Histogram(
            x=control_trials["monkey_total_return"] * 100,
            nbinsx=65,
            marker={"color": "#2563eb", "opacity": 0.62},
            name="每周3换3",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=control_trials["fixed_initial_25_return"] * 100,
            nbinsx=65,
            marker={"color": "#7c3aed", "opacity": 0.50},
            name="固定25只持有",
        ), row=1, col=1,
    )
    periods = ["第一年", "近一年"]
    fig.add_trace(
        go.Bar(
            x=periods,
            y=[market["prior_period"]["benchmark"] * 100, market["recent_period"]["benchmark"] * 100],
            name="沪深300",
            marker={"color": "#ef4444"},
        ), row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=periods,
            y=[market["prior_period"]["monkey_median"] * 100, market["recent_period"]["monkey_median"] * 100],
            name="猴群中位数",
            marker={"color": "#2563eb"},
        ), row=1, col=2,
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="累计收益（%）", row=1, col=1)
    fig.update_yaxes(title_text="trial数", row=1, col=1)
    fig.update_yaxes(title_text="区间收益（%）", row=1, col=2)
    figures.append(_figure_html(_style(fig, "随机换仓没有制造高收益：上涨集中在第一年", 440), "return-attribution", False))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("候选股票池大小", "成分切换周的并集扩张"))
    fig.add_trace(
        go.Scatter(
            x=timing["last_signal_date"], y=timing["candidate_count"], mode="lines+markers",
            line={"color": "#2563eb"}, marker={"size": 4}, name="周频候选数",
            hovertemplate="%{x|%Y-%m-%d}<br>候选 %{y}只<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timing["last_signal_date"], y=timing["membership_at_signal_close"], mode="lines",
            line={"color": "#0f9d78", "dash": "dash"}, name="周末正式成员",
        ), row=1, col=1,
    )
    surplus = timing[timing["candidate_surplus_vs_signal_close"] > 0]
    fig.add_trace(
        go.Bar(
            x=surplus["last_signal_date"], y=surplus["candidate_surplus_vs_signal_close"],
            marker={"color": "#f59e0b"}, name="旧成分额外数量",
            hovertemplate="%{x|%Y-%m-%d}<br>额外 %{y}只<extra></extra>",
        ), row=1, col=2,
    )
    fig.update_yaxes(title_text="股票数量", row=1, col=1)
    fig.update_yaxes(title_text="额外旧成分数量", row=1, col=2)
    figures.append(_figure_html(_style(fig, "动态股票池：点时成员正确，但周频采样会合并周内新旧成分", 410), "universe-audit", False))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("累计成本率", "换手率与收益"))
    fig.add_trace(
        go.Histogram(x=results["total_cost"] * 100, nbinsx=60, marker={"color": "#dc2626"}, name="累计成本率"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=results["total_turnover"] * 100,
            y=results["total_return"] * 100,
            mode="markers",
            marker={"size": 4, "opacity": 0.35, "color": "#2563eb"},
            customdata=results["run_id"],
            hovertemplate="run %{customdata}<br>累计换手 %{x:.1f}%<br>收益 %{y:.2f}%<extra></extra>",
            showlegend=False,
        ), row=1, col=2,
    )
    fig.update_xaxes(title_text="两年累计cost rate（%）", row=1, col=1)
    fig.update_yaxes(title_text="trial数", row=1, col=1)
    fig.update_xaxes(title_text="两年累计turnover rate（%）", row=1, col=2)
    fig.update_yaxes(title_text="两年累计收益（%）", row=1, col=2)
    figures.append(_figure_html(_style(fig, "成本已经扣除：它只会压低收益", 420), "cost-audit", False))

    mismatch_labels = ["整周买入阻断", "整周卖出阻断", "周末买入受限但获准", "周末卖出受限但获准", "使用较早有效价"]
    mismatch_values = [
        trade["weekly_buy_blocked"], trade["weekly_sell_blocked"],
        trade["last_close_buy_blocked_but_weekly_allowed"],
        trade["last_close_sell_blocked_but_weekly_allowed"], trade["stale_last_valid_close"],
    ]
    fig = go.Figure(
        go.Bar(
            x=mismatch_labels, y=mismatch_values,
            marker={"color": ["#64748b", "#64748b", "#f59e0b", "#f59e0b", "#dc2626"]},
            text=[f"{value:,}" for value in mismatch_values], textposition="outside",
        )
    )
    fig.update_yaxes(title_text="候选股票-周数量")
    figures.append(_figure_html(_style(fig, "不可交易处理：整周判断与周末收盘成交存在语义错位", 390), "tradability-audit", False))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("每条trial的Mean IC", "每条trial的ICIR"))
    fig.add_trace(
        go.Histogram(x=ic_trials["mean_ic"], nbinsx=65, marker={"color": "#0f9d78"}, name="Mean IC"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=ic_trials["icir"], nbinsx=65, marker={"color": "#7c3aed"}, name="ICIR"),
        row=1, col=2,
    )
    fig.add_vline(x=0, line={"color": "#ef4444", "dash": "dash"}, row=1, col=1)
    fig.add_vline(x=0, line={"color": "#ef4444", "dash": "dash"}, row=1, col=2)
    fig.update_xaxes(title_text="Mean IC", row=1, col=1)
    fig.update_xaxes(title_text="ICIR", row=1, col=2)
    fig.update_yaxes(title_text="trial数", row=1, col=1)
    fig.update_yaxes(title_text="trial数", row=1, col=2)
    figures.append(_figure_html(_style(fig, "随机信号没有预测力：IC和ICIR以0为中心", 400), "ic-audit", False))

    repro_full = repro["full_run_comparison"]
    repro_unfixed = repro["single_process_without_hash_seed"]
    repro_fixed = repro["single_process_with_hash_seed_zero"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("相同种子的路径差异", "总体分布仍稳定"))
    fig.add_trace(
        go.Bar(
            x=["完整运行未固定hash", "单进程未固定hash", "单进程hash=0"],
            y=[
                repro_full["different_return_runs"] / repro_full["matched_runs"] * 100,
                repro_unfixed["different_return_runs"] / repro_unfixed["matched_runs"] * 100,
                repro_fixed["different_return_runs"] / repro_fixed["matched_runs"] * 100,
            ],
            marker={"color": ["#dc2626", "#f59e0b", "#0f9d78"]},
            text=["54.79%", "60.00%", "0.00%"], textposition="outside", showlegend=False,
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=["旧完整运行均值", "新完整运行均值", "旧中位数", "新中位数"],
            y=[
                repro_full["left_return_mean"] * 100, repro_full["right_return_mean"] * 100,
                repro_full["left_return_median"] * 100, repro_full["right_return_median"] * 100,
            ],
            marker={"color": ["#64748b", "#2563eb", "#64748b", "#2563eb"]},
            textposition="outside", texttemplate="%{y:.2f}%", showlegend=False,
        ), row=1, col=2,
    )
    fig.update_yaxes(title_text="收益不同的trial比例（%）", row=1, col=1)
    fig.update_yaxes(title_text="两年收益（%）", row=1, col=2)
    figures.append(_figure_html(_style(fig, "复现缺陷：影响路径映射，不改变总体结论", 410), "repro-audit", False))

    first = timing_step["first_step"]
    last = timing_step["last_step"]
    generated_at = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z")
    css = """
    :root { --ink:#172033; --muted:#64748b; --line:#dbe3ee; --soft:#f4f7fb; --blue:#2563eb; --green:#0f9d78; --red:#dc2626; --amber:#d97706; }
    * { box-sizing:border-box; }
    body { margin:0; background:#edf2f7; color:var(--ink); font-family:"Segoe UI","Microsoft YaHei",sans-serif; }
    .report { max-width:1440px; margin:0 auto; background:#fff; min-height:100vh; }
    header { padding:42px 48px 34px; background:linear-gradient(135deg,#12213d,#1f4b8f); color:#fff; }
    header h1 { margin:0 0 10px; font-size:30px; font-weight:600; }
    header p { margin:6px 0; color:#dbeafe; }
    main { padding:30px 48px 58px; }
    section { margin:0 0 42px; }
    h2 { margin:0 0 16px; font-size:21px; border-left:4px solid var(--blue); padding-left:10px; }
    h3 { margin:22px 0 10px; font-size:16px; }
    .kpis { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:-18px 0 34px; }
    .kpi { border:1px solid var(--line); border-radius:10px; padding:17px 18px; box-shadow:0 4px 18px rgba(25,42,70,.07); }
    .kpi .label { color:var(--muted); font-size:13px; }
    .kpi .value { margin-top:6px; font-size:26px; font-weight:600; font-variant-numeric:tabular-nums; }
    .kpi .note { color:var(--muted); font-size:12px; margin-top:4px; }
    .finding { background:var(--soft); border-left:3px solid var(--blue); padding:14px 18px; margin:10px 0; line-height:1.65; }
    .finding.warn { background:#fff7ed; border-left-color:#f59e0b; }
    .finding.issue { background:#fef2f2; border-left-color:#dc2626; }
    .chart { border:1px solid var(--line); border-radius:10px; padding:6px; margin:14px 0 22px; overflow:hidden; }
    .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:9px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    thead th { background:var(--soft); color:#334155; }
    tbody tr:last-child th,tbody tr:last-child td { border-bottom:0; }
    .status { display:inline-block; padding:2px 8px; border-radius:999px; white-space:nowrap; }
    .status.pass,.status.market_driven { background:#dcfce7; color:#166534; }
    .status.warning,.status.pass_with_convention_note { background:#fef3c7; color:#92400e; }
    .status.issue { background:#fee2e2; color:#991b1b; }
    code { color:#334155; }
    footer { padding:22px 48px; background:#12213d; color:#cbd5e1; font-size:12px; }
    @media (max-width:900px) { .kpis{grid-template-columns:repeat(2,1fr)} .grid-2{grid-template-columns:1fr} header,main,footer{padding-left:22px;padding-right:22px} }
    @media (max-width:520px) { .kpis{grid-template-columns:1fr} header h1{font-size:24px} }
    @media print { body{background:#fff} .report{max-width:none} .chart{break-inside:avoid} }
    """

    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CSI300猴子实验：六层逻辑审计与高收益归因</title><style>{css}</style></head>
<body><div class="report"><header>
<h1>CSI300猴子实验：六层逻辑审计与高收益归因</h1>
<p>动态 universe → 调仓执行时点 → DropN语义 → 成本 → 不可交易处理 → 年化算法</p>
<p>审计对象：10,000条路径 · {html.escape(str(summary['run_dir']))}</p>
</header><main>
<div class="kpis">
  <div class="kpi"><div class="label">沪深300同期</div><div class="value">{_pct(market['benchmark_cap_weight_return'])}</div><div class="note">市场自身显著上涨</div></div>
  <div class="kpi"><div class="label">猴群两年中位数</div><div class="value">{_pct(market['monkey_total_return']['median'])}</div><div class="note">中位简单超额 {_pct(market['monkey_excess_return']['median'])}</div></div>
  <div class="kpi"><div class="label">固定随机25只中位数</div><div class="value">{_pct(market['fixed_random_25_return']['median'])}</div><div class="note">不换仓反而略高</div></div>
  <div class="kpi"><div class="label">动态CSI300等权</div><div class="value">{_pct(market['dynamic_weekly_csi300_equal_weight_return'])}</div><div class="note">粗对照，不含持续调仓费</div></div>
</div>

<section><h2>最终判断</h2>
  <div class="finding"><strong>高收益主要是真的市场收益，不是随机信号产生了alpha。</strong> 沪深300本身上涨 {_pct(market['benchmark_cap_weight_return'])}；等权CSI300与固定随机25只也得到约44%–47%的收益，而猴群IC、RankIC和ICIR均以0为中心。</div>
  <div class="finding">猴群每周换仓中位数 {_pct(market['monkey_total_return']['median'])}，低于固定随机25只中位数 {_pct(market['fixed_random_25_return']['median'])}。随机换仓和成本没有制造收益，整体上反而形成拖累。</div>
  <div class="finding warn">实验分布结论仍有参考价值，但执行层并非严格的“周末可交易、周末收盘成交”：原子周频会查询整周可交易状态，并用该周最后有效close。</div>
  <div class="finding issue">trial级复现存在确定性缺陷：未固定PYTHONHASHSEED时，相同种子的两次完整实验有 {repro_full['different_return_runs']:,} 条路径不同。固定hash后探针完全一致。</div>
  <div class="chart">{figures[0]}</div><div class="chart">{figures[1]}</div>
</section>

<section><h2>八项对账总表</h2>{_step_table(summary)}</section>

<section><h2>01 动态 universe</h2>
  <div class="grid-2"><div class="finding">累计历史成分 {universe['unique_historical_constituents']} 只，首尾均为300只，首尾重合 {universe['initial_final_overlap']} 只。候选股票均属于对应预测周的历史成分，并非拿期末成分回填全历史。</div>
  <div class="finding warn">成分切换周会把周内新旧成分合并：{universe['weeks_with_union_surplus']}周存在额外旧成分，最多 {universe['max_union_surplus']} 只，候选池最高达到 {int(universe['candidate_count']['max'])} 只。</div></div>
  <div class="chart">{figures[2]}</div>
</section>

<section><h2>02 调仓决策与成交时点</h2>
  <div class="finding">第一步：信号最后日期 {first['last_signal_date'][:10]} → 执行区间 {first['execution_interval_start'][:10]} 至 {first['execution_interval_end'][:10]} → 实际成交close日期 {first['actual_close_execution_date'][:10]}。最后一步实际估值日为 {last['actual_close_execution_date'][:10]}。</div>
  <div class="finding warn">HTML净值曲线显示的周一日期实际是trade_start索引，数值反映该周最后有效收盘价。新买入股票在该周末成交，从下一周才产生持仓收益。</div>
  <div class="finding warn">这不是随机信号前视，但only_tradable在生成决策时读取整个未来执行周。建议改成周频决策＋日频内层执行，或明确“周末生成信号、下一交易日成交”。</div>
</section>

<section><h2>03 DropN语义</h2>
  <div class="finding">TopK={dropn['topk']}、DropN={dropn['n_drop']}。共 {dropn['transitions']:,} 次持仓转换，全部严格卖3买3，持仓始终25只。</div>
  <div class="finding">Qlib的random buy语义是：先按随机信号选Top25候选，再从未持有候选中随机补足；不是直接从全部CSI300均匀抽3只。由于分数为IID随机数，边际选股概率仍近似均匀。</div>
</section>

<section><h2>04 成本</h2>
  <div class="grid-2"><div class="finding">开仓 {_pct(costs['open_cost'])}、平仓 {_pct(costs['close_cost'])}、最低5元；impact cost为0，未设成交量限制。</div>
  <div class="finding">两年累计cost rate中位数 {_pct(costs['total_cost_rate']['median'])}，近一年 {_pct(costs['recent_cost_rate']['median'])}。最终收益直接由扣费后的account净值计算。</div></div>
  <div class="chart">{figures[3]}</div>
</section>

<section><h2>05 不可交易处理</h2>
  <div class="finding warn">在 {trade['candidate_stock_weeks']:,} 个候选股票-周中，周末买入受限但整周规则仍允许 {trade['last_close_buy_blocked_but_weekly_allowed']:,} 个（{_pct(trade['buy_close_mismatch_rate'])}），卖出对应 {trade['last_close_sell_blocked_but_weekly_allowed']:,} 个（{_pct(trade['sell_close_mismatch_rate'])}）。</div>
  <div class="finding warn">另有 {trade['stale_last_valid_close']} 个候选股票-周使用早于周末的最后有效close。方向不一定系统性抬高收益，但与“周末收盘真实可成交”假设不一致。</div>
  <div class="chart">{figures[4]}</div>
</section>

<section><h2>06 年化算法</h2>
  <div class="finding">累计收益和CAGR复算最大误差 {annual['cagr_max_absolute_recalculation_error']:.3e}，几何年化正确。两年中位累计收益 {_pct(annual['median_total_return'])} 对应年化 {_pct(annual['median_annualized_return'])}。</div>
  <div class="finding warn">波动率与Sharpe使用√52；Qlib默认周频常数为50，因此当前数值比Qlib默认口径高约 {_pct(annual['sqrt_scaler_difference'])}。这只影响风险指标缩放，不会改变累计收益。</div>
</section>

<section><h2>07 随机信号与市场归因</h2>
  <div class="grid-2"><div class="finding">100万个周截面IC均值 {ic['weekly_ic']['mean']:.6f}、RankIC均值 {ic['weekly_rank_ic']['mean']:.6f}；正值比例均约50%，没有预测力。</div>
  <div class="finding">第一年猴群中位数 {_pct(market['prior_period']['monkey_median'])}、沪深300 {_pct(market['prior_period']['benchmark'])}；近一年分别 {_pct(market['recent_period']['monkey_median'])}/{_pct(market['recent_period']['benchmark'])}。高收益集中在第一年强行情。</div></div>
  <div class="chart">{figures[5]}</div>
</section>

<section><h2>08 Trial复现</h2>
  <div class="finding issue">Qlib <code>Position.get_stock_list()</code> 使用 <code>list(set(...))</code>。不同Python进程的hash顺序不同，使相同随机数可能选择到不同股票。</div>
  <div class="finding">未固定hash：两次完整运行 {repro_full['different_return_runs']:,}/{repro_full['matched_runs']:,} 条收益不同；但均值仅 {_pct(repro_full['left_return_mean'])} vs {_pct(repro_full['right_return_mean'])}，KS检验p={repro_full['distribution_ks_pvalue']:.6f}。固定 <code>PYTHONHASHSEED=0</code> 后10条A/B逐值一致。</div>
  <div class="chart">{figures[6]}</div>
</section>

<section><h2>修正优先级</h2>
  <div class="finding issue"><strong>P0：确定性。</strong> 在启动Python前设置并记录 <code>PYTHONHASHSEED=0</code>；同时在策略随机抽样前显式排序当前持仓列表。</div>
  <div class="finding warn"><strong>P1：执行时点。</strong> 使用周频决策＋日频内层执行，或规定周末生成决策、下一交易日开盘/收盘成交，避免整周执行区间读取。</div>
  <div class="finding warn"><strong>P1：可交易性。</strong> 在实际成交bar判断停牌与涨跌停，不使用“整周全部受限才阻断”的聚合语义。</div>
  <div class="finding"><strong>P2：对照实验。</strong> 正式报告固定加入市值加权CSI300、动态等权CSI300、固定随机25只、零成本/有成本四组对照。</div>
  <div class="finding"><strong>P2：股票池快照。</strong> 在每个决策时点只取该时点正式成员，避免成分切换周形成新旧并集。</div>
</section>

<section><h2>证据索引</h2>
  <div class="finding">本目录保存8个步骤的Markdown与JSON、周频股票池/时点CSV、不可交易统计CSV、动态等权序列、10,000组固定持有对照和总审计JSON。报告生成时间：{generated_at}。</div>
</section>
</main><footer>本报告用于回测逻辑审计与研究方法验证，不构成投资建议。</footer></div></body></html>"""
    output_path.write_text(report, encoding="utf-8")
    return {
        "output": str(output_path),
        "html_bytes": output_path.stat().st_size,
        "figures": len(figures),
        "market_verdict": market["verdict"],
        "reproducibility_issue_runs": repro_full["different_return_runs"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else audit_dir / "monkey_logic_audit_report.html"
    result = build_report(audit_dir, output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
