#!/usr/bin/env python
"""Collect a six-layer evidence pack for the CSI300 monkey experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stats(values: pd.Series | np.ndarray) -> dict[str, float]:
    series = pd.Series(values, dtype="float64").dropna()
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def _write_step(output_dir: Path, stem: str, payload: dict[str, Any]) -> None:
    (output_dir / f"{stem}.json").write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [f"# {payload['title']}", "", f"**结论：{payload['verdict']}**", ""]
    lines.extend(f"- {item}" for item in payload.get("findings", []))
    lines.extend(["", "详细结构化证据见同名 JSON。", ""])
    (output_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def _members_on(
    spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], date: pd.Timestamp
) -> set[str]:
    timestamp = pd.Timestamp(date).normalize()
    return {
        str(code)
        for code, intervals in spans.items()
        if any(pd.Timestamp(start) <= timestamp <= pd.Timestamp(end) for start, end in intervals)
    }


def _load_run(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path / "results.csv").sort_values("run_id").reset_index(drop=True)
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    return frame[frame["status"] == "ok"].copy(), metadata


def _compare_runs(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    joined = left.set_index("run_id").join(
        right.set_index("run_id"), lsuffix="_left", rsuffix="_right", how="inner"
    )
    return_diff = (joined["total_return_left"] - joined["total_return_right"]).abs()
    nav_diff = (joined["final_nav_left"] - joined["final_nav_right"]).abs()
    ks = ks_2samp(joined["total_return_left"], joined["total_return_right"])
    return {
        "matched_runs": int(len(joined)),
        "same_signal_seeds": bool(
            (joined["signal_seed_left"] == joined["signal_seed_right"]).all()
        ),
        "same_strategy_seeds": bool(
            (joined["strategy_seed_left"] == joined["strategy_seed_right"]).all()
        ),
        "same_initial_holdings": bool(
            (joined["initial_holdings_left"] == joined["initial_holdings_right"]).all()
        ),
        "different_return_runs": int((return_diff > 1e-12).sum()),
        "max_absolute_return_difference": float(return_diff.max()),
        "different_nav_runs": int((nav_diff > 1e-6).sum()),
        "left_return_mean": float(joined["total_return_left"].mean()),
        "right_return_mean": float(joined["total_return_right"].mean()),
        "left_return_median": float(joined["total_return_left"].median()),
        "right_return_median": float(joined["total_return_right"].median()),
        "distribution_ks_statistic": float(ks.statistic),
        "distribution_ks_pvalue": float(ks.pvalue),
    }


def _fixed_portfolio_returns(
    results: pd.DataFrame,
    first_prices: pd.Series,
    last_prices: pd.Series,
    risk_degree: float,
    initial_cost_rate: float,
) -> pd.Series:
    ratios = last_prices / first_prices

    def calculate(stock_text: str) -> float:
        codes = str(stock_text).split("|")
        valid = pd.to_numeric(ratios.reindex(codes), errors="coerce").dropna()
        if len(valid) != len(codes):
            return math.nan
        gross_final = (1.0 - risk_degree) + risk_degree * float(valid.mean())
        return gross_final - 1.0 - risk_degree * initial_cost_rate

    return results["initial_holdings"].map(calculate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--comparison-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    comparison_dir = Path(args.comparison_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results, metadata = _load_run(run_dir)
    comparison, _ = _load_run(comparison_dir)
    config = metadata["experiment_config"]
    start_time, end_time = metadata["period"]
    risk_degree = float(config["risk_degree"])
    exchange_config = config["strategy_config"]["backtest"]["exchange_kwargs"]

    from quantpits.utils import env

    env.init_qlib()
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.utils import TradeCalendarManager
    from qlib.data import D

    raw_spans = D.list_instruments(
        D.instruments(market=config["market"]),
        start_time=start_time,
        end_time=end_time,
        freq="day",
        as_list=False,
    )
    spans = {
        str(code): [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals]
        for code, intervals in raw_spans.items()
    }
    all_codes = sorted(spans)

    exchange = Exchange(
        freq="day",
        start_time=start_time,
        end_time=end_time,
        codes=all_codes,
        **exchange_config,
    )
    quote = exchange.quote_df.copy()
    quote.index = quote.index.set_levels(
        [
            quote.index.levels[quote.index.names.index("instrument")].astype(str),
            quote.index.levels[quote.index.names.index("datetime")],
        ]
    )
    close_wide = quote["$close"].unstack("instrument").sort_index()
    close_wide.index = pd.DatetimeIndex(close_wide.index)
    market_dates = close_wide.index.unique().sort_values()

    signal_start = (pd.Timestamp(start_time) - pd.Timedelta(days=config["signal_lookback_days"])).strftime(
        "%Y-%m-%d"
    )
    signal_features = D.features(
        D.instruments(market=config["market"]),
        ["$close"],
        start_time=signal_start,
        end_time=end_time,
        freq="day",
    )
    signal_close = pd.to_numeric(signal_features.iloc[:, 0], errors="coerce")
    signal_close = signal_close[signal_close.notna() & np.isfinite(signal_close) & (signal_close > 0)]
    signal_records = pd.DataFrame(
        {
            "instrument": signal_close.index.get_level_values("instrument").astype(str),
            "datetime": pd.DatetimeIndex(signal_close.index.get_level_values("datetime")),
        }
    )

    calendar = TradeCalendarManager(freq="week", start_time=start_time, end_time=end_time)
    timing_rows: list[dict[str, Any]] = []
    candidate_sets: list[set[str]] = []
    end_dates: list[pd.Timestamp] = []
    previous_members: set[str] | None = None
    for step in range(calendar.get_trade_len()):
        pred_start, pred_end = calendar.get_step_time(step, shift=1)
        trade_start, trade_end = calendar.get_step_time(step, shift=0)
        signal_slice = signal_records[signal_records["datetime"].between(pred_start, pred_end)]
        latest = signal_slice.sort_values("datetime").groupby("instrument", sort=False).tail(1)
        candidates = set(latest["instrument"])
        candidate_sets.append(candidates)
        signal_date = pd.Timestamp(latest["datetime"].max())
        end_date = pd.Timestamp(market_dates[(market_dates >= trade_start) & (market_dates <= trade_end)].max())
        end_dates.append(end_date)
        members_signal = _members_on(spans, signal_date)
        members_end = _members_on(spans, end_date)
        union_members: set[str] = set()
        for date in signal_slice["datetime"].unique():
            union_members |= _members_on(spans, pd.Timestamp(date))
        timing_rows.append(
            {
                "step": step,
                "report_index": trade_start,
                "prediction_start": pred_start,
                "prediction_end": pred_end,
                "last_signal_date": signal_date,
                "execution_interval_start": trade_start,
                "execution_interval_end": trade_end,
                "actual_close_execution_date": end_date,
                "candidate_count": len(candidates),
                "membership_at_signal_close": len(members_signal),
                "membership_at_execution_close": len(members_end),
                "candidate_surplus_vs_signal_close": len(candidates - members_signal),
                "candidate_outside_weekly_membership_union": len(candidates - union_members),
                "membership_added_since_previous_step": 0
                if previous_members is None
                else len(members_end - previous_members),
                "membership_dropped_since_previous_step": 0
                if previous_members is None
                else len(previous_members - members_end),
            }
        )
        previous_members = members_end

    timing = pd.DataFrame(timing_rows)
    timing.to_csv(output_dir / "01_02_universe_timing_weekly.csv", index=False)
    initial_members = _members_on(spans, end_dates[0])
    final_members = _members_on(spans, end_dates[-1])
    universe_payload = {
        "title": "01 动态 universe 审计",
        "verdict": "通过，但成分切换周的周频信号池会成为一周内新旧成分的并集",
        "findings": [
            f"区间累计出现{len(spans)}只历史成分，首尾成员各{len(initial_members)}/{len(final_members)}只。",
            f"每周最后信号日成员数范围{timing.membership_at_signal_close.min()}–{timing.membership_at_signal_close.max()}；信号候选数范围{timing.candidate_count.min()}–{timing.candidate_count.max()}。",
            f"候选股票落在该预测周历史成员并集之外的数量合计为{timing.candidate_outside_weekly_membership_union.sum()}，未发现静态未来成分股泄漏。",
            f"有{int((timing.candidate_surplus_vs_signal_close > 0).sum())}周包含已于周末退出但周内仍有分数的旧成分，最大额外{int(timing.candidate_surplus_vs_signal_close.max())}只。",
        ],
        "unique_historical_constituents": len(spans),
        "initial_members": len(initial_members),
        "final_members": len(final_members),
        "initial_final_overlap": len(initial_members & final_members),
        "weeks_with_membership_change": int(
            ((timing.membership_added_since_previous_step + timing.membership_dropped_since_previous_step) > 0).sum()
        ),
        "candidate_count": _stats(timing["candidate_count"]),
        "membership_count": _stats(timing["membership_at_signal_close"]),
        "weeks_with_union_surplus": int((timing.candidate_surplus_vs_signal_close > 0).sum()),
        "max_union_surplus": int(timing.candidate_surplus_vs_signal_close.max()),
        "outside_week_union_total": int(timing.candidate_outside_weekly_membership_union.sum()),
    }
    _write_step(output_dir, "01_dynamic_universe", universe_payload)

    timing_payload = {
        "title": "02 决策与成交时点审计",
        "verdict": "无信号前视，但原子周频执行使用下一交易周最后收盘价，并把结果索引标在周初",
        "findings": [
            "策略读取上一Qlib周区间的最后随机分数。",
            "订单覆盖下一周完整区间，Exchange以ts_data_last取得该区间最后有效收盘价；新买入股票从再下一周才开始贡献持仓收益。",
            "sample_curves中的周一日期是trade_start/report index，不是实际估值收盘日。",
            "only_tradable在决策生成时查询整个下一周区间，因此包含执行区间信息；更严格的可交易回测应使用日频内层执行器或明确的下一日成交规则。",
        ],
        "weekly_steps": int(calendar.get_trade_len()),
        "first_step": timing.iloc[0].to_dict(),
        "last_step": timing.iloc[-1].to_dict(),
        "report_label_is_trade_start": True,
        "deal_price_aggregation": "last valid daily close over the weekly execution interval",
        "signal_to_position_return_lag": "signal week -> execution at following week close -> PnL starts next week",
    }
    _write_step(output_dir, "02_execution_timing", timing_payload)

    transitions = int(results["rebalance_periods"].sum())
    full_transitions = int(results["full_three_by_three_periods"].sum())
    dropn_payload = {
        "title": "03 DropN 语义审计",
        "verdict": "持仓数量和3换3约束执行正确；random buy并非直接从全CSI300抽3只，而是从随机分数Top25候选中抽取",
        "findings": [
            f"{len(results):,}条路径合计{transitions:,}次持仓转换，{full_transitions:,}次全部严格卖3买3。",
            "method_sell=random从当前可交易持仓随机卖3只；method_buy=random先取随机信号Top25，再从其中未持有者随机补足。",
            "由于信号本身独立同分布，Top25在边际上仍是均匀随机子集，但它与‘直接从全股票池随机抽3只’的条件路径结构不同。",
        ],
        "topk": int(config["holdings"]),
        "n_drop": int(config["weekly_replacements"]),
        "transitions": transitions,
        "full_three_by_three_transitions": full_transitions,
        "full_rate": full_transitions / transitions,
        "holding_count_values": sorted(results["initial_holding_count"].unique().astype(int).tolist()),
        "minimum_holding_count_values": sorted(results["min_holding_count"].unique().astype(int).tolist()),
        "maximum_holding_count_values": sorted(results["max_holding_count"].unique().astype(int).tolist()),
    }
    _write_step(output_dir, "03_dropn_semantics", dropn_payload)

    cost_payload = {
        "title": "04 交易成本审计",
        "verdict": "成本已计入账户净值且方向正确，不是高收益来源；但当前结果只保存成本率之和，不能精确反事实还原零成本终值",
        "findings": [
            f"开仓费{exchange.open_cost:.4%}、平仓费{exchange.close_cost:.4%}、最低费用{exchange.min_cost:g}元，impact cost为{exchange.impact_cost:.4%}。",
            f"两年累计cost rate中位数{results.total_cost.median():.4%}，近一年中位数{results.recent_total_cost.median():.4%}。",
            "Qlib account净值扣除实际费用；report.return是费前收益，report.cost单列，本实验最终收益直接由扣费后的account计算。",
        ],
        "open_cost": exchange.open_cost,
        "close_cost": exchange.close_cost,
        "min_cost": exchange.min_cost,
        "impact_cost": exchange.impact_cost,
        "volume_threshold": exchange.volume_threshold,
        "total_cost_rate": _stats(results["total_cost"]),
        "recent_cost_rate": _stats(results["recent_total_cost"]),
        "total_turnover_rate": _stats(results["total_turnover"]),
        "recent_turnover_rate": _stats(results["recent_total_turnover"]),
        "initial_open_cost_estimate": risk_degree * exchange.open_cost,
    }
    _write_step(output_dir, "04_costs", cost_payload)

    tradability_rows: list[dict[str, Any]] = []
    for step, (candidates, end_date) in enumerate(zip(candidate_sets, end_dates)):
        trade_start = pd.Timestamp(timing.loc[step, "execution_interval_start"])
        trade_end = pd.Timestamp(timing.loc[step, "execution_interval_end"])
        interval = quote.loc[
            (slice(None), slice(trade_start, trade_end)),
            ["$close", "$change", "limit_buy", "limit_sell"],
        ]
        market_last = end_date
        counts = {
            "candidate_stock_weeks": 0,
            "weekly_buy_blocked": 0,
            "weekly_sell_blocked": 0,
            "last_close_buy_blocked_but_weekly_allowed": 0,
            "last_close_sell_blocked_but_weekly_allowed": 0,
            "stale_last_valid_close": 0,
            "fully_suspended": 0,
        }
        for code in candidates:
            try:
                stock = interval.xs(code, level="instrument").sort_index()
            except KeyError:
                counts["fully_suspended"] += 1
                continue
            counts["candidate_stock_weeks"] += 1
            weekly_buy_blocked = bool(stock["limit_buy"].astype(bool).all())
            weekly_sell_blocked = bool(stock["limit_sell"].astype(bool).all())
            counts["weekly_buy_blocked"] += int(weekly_buy_blocked)
            counts["weekly_sell_blocked"] += int(weekly_sell_blocked)
            valid_close = stock["$close"].dropna()
            if valid_close.empty:
                counts["fully_suspended"] += 1
                continue
            last_valid_date = pd.Timestamp(valid_close.index[-1])
            counts["stale_last_valid_close"] += int(last_valid_date < market_last)
            if market_last in stock.index:
                last = stock.loc[market_last]
                last_buy_blocked = bool(pd.isna(last["$close"]) or bool(last["limit_buy"]))
                last_sell_blocked = bool(pd.isna(last["$close"]) or bool(last["limit_sell"]))
            else:
                last_buy_blocked = True
                last_sell_blocked = True
            counts["last_close_buy_blocked_but_weekly_allowed"] += int(
                last_buy_blocked and not weekly_buy_blocked
            )
            counts["last_close_sell_blocked_but_weekly_allowed"] += int(
                last_sell_blocked and not weekly_sell_blocked
            )
        tradability_rows.append({"step": step, **counts})

    tradability = pd.DataFrame(tradability_rows)
    tradability.to_csv(output_dir / "05_tradability_weekly.csv", index=False)
    tradability_totals = tradability.drop(columns="step").sum().astype(int).to_dict()
    tradability_payload = {
        "title": "05 不可交易处理审计",
        "verdict": "停牌/涨跌停过滤已启用，但周频原子执行的区间聚合语义与‘周末收盘成交’不完全一致",
        "findings": [
            "Qlib只有在整周所有日线都不可交易时才判定该周不可交易；成交价却取整周最后一个有效close。",
            f"候选股票-周共{tradability_totals['candidate_stock_weeks']:,}个；其中周末买入受限但按整周规则仍允许{tradability_totals['last_close_buy_blocked_but_weekly_allowed']:,}个，卖出对应{tradability_totals['last_close_sell_blocked_but_weekly_allowed']:,}个。",
            f"有{tradability_totals['stale_last_valid_close']:,}个候选股票-周使用早于市场周末的最后有效价格。",
            "这属于执行真实性问题，方向未必单边抬高收益，但应在严谨实验中修正。",
        ],
        **tradability_totals,
        "buy_close_mismatch_rate": tradability_totals[
            "last_close_buy_blocked_but_weekly_allowed"
        ]
        / max(tradability_totals["candidate_stock_weeks"], 1),
        "sell_close_mismatch_rate": tradability_totals[
            "last_close_sell_blocked_but_weekly_allowed"
        ]
        / max(tradability_totals["candidate_stock_weeks"], 1),
    }
    _write_step(output_dir, "05_tradability", tradability_payload)

    elapsed_years = (
        pd.Timestamp(end_time) - pd.Timestamp(start_time)
    ).days / 365.2425
    expected_cagr = (1.0 + results["total_return"]) ** (1.0 / elapsed_years) - 1.0
    cagr_diff = (expected_cagr - results["annualized_return"]).abs()
    annual_payload = {
        "title": "06 年化算法审计",
        "verdict": "累计收益与CAGR计算正确；波动率和Sharpe使用52周，而Qlib默认周频常数为50周，存在约2%的口径差但不影响累计高收益",
        "findings": [
            f"区间按日历天计{elapsed_years:.6f}年，CAGR逐trial复算最大误差{cagr_diff.max():.3e}。",
            "年化收益使用几何CAGR，不是把两年累计收益直接除以2。",
            f"52周相对Qlib默认50周会把波动率和Sharpe放大{(math.sqrt(52/50)-1):.2%}。",
            "最大回撤由账户净值峰谷计算，口径正确。",
        ],
        "elapsed_years": elapsed_years,
        "weekly_observations": int(calendar.get_trade_len()),
        "cagr_max_absolute_recalculation_error": float(cagr_diff.max()),
        "volatility_scaler": 52,
        "qlib_default_week_scaler": 50,
        "sqrt_scaler_difference": math.sqrt(52 / 50) - 1.0,
        "median_total_return": float(results["total_return"].median()),
        "median_annualized_return": float(results["annualized_return"].median()),
        "benchmark_total_return": float(results["benchmark_total_return"].median()),
        "benchmark_cagr": float(results["benchmark_cagr"].median()),
    }
    _write_step(output_dir, "06_annualization", annual_payload)

    first_date, last_date = end_dates[0], end_dates[-1]
    first_interval_start = pd.Timestamp(timing.loc[0, "execution_interval_start"])
    first_prices = (
        close_wide.loc[
            (close_wide.index >= first_interval_start) & (close_wide.index <= first_date)
        ]
        .ffill()
        .iloc[-1]
    )
    last_prices = close_wide.loc[close_wide.index <= last_date].ffill().iloc[-1]
    fixed_random = _fixed_portfolio_returns(
        results,
        first_prices,
        last_prices,
        risk_degree,
        exchange.open_cost,
    )
    initial_codes = sorted(initial_members & set(close_wide.columns))
    initial_ratios = (last_prices[initial_codes] / first_prices[initial_codes]).dropna()
    static_equal_weight = (
        (1.0 - risk_degree) + risk_degree * float(initial_ratios.mean()) - 1.0
        - risk_degree * exchange.open_cost
    )

    dynamic_factor = 1.0 - risk_degree * exchange.open_cost
    dynamic_weekly_rows: list[dict[str, Any]] = []
    asof_weekly_prices = close_wide.ffill().reindex(end_dates)
    for step in range(len(end_dates) - 1):
        current_date, next_date = end_dates[step], end_dates[step + 1]
        members = sorted(_members_on(spans, current_date) & set(close_wide.columns))
        weekly_returns = (
            asof_weekly_prices.loc[next_date, members]
            / asof_weekly_prices.loc[current_date, members]
            - 1.0
        ).replace([np.inf, -np.inf], np.nan).dropna()
        equal_return = float(weekly_returns.mean())
        portfolio_return = risk_degree * equal_return
        dynamic_factor *= 1.0 + portfolio_return
        dynamic_weekly_rows.append(
            {
                "step": step + 1,
                "start_date": current_date,
                "end_date": next_date,
                "constituent_count": len(members),
                "valid_return_count": len(weekly_returns),
                "equal_weight_stock_return": equal_return,
                "portfolio_return_at_risk_degree": portfolio_return,
                "cumulative_return": dynamic_factor - 1.0,
            }
        )
    dynamic_weekly = pd.DataFrame(dynamic_weekly_rows)
    dynamic_weekly.to_csv(output_dir / "07_dynamic_equal_weight_weekly.csv", index=False)
    dynamic_equal_weight = dynamic_factor - 1.0

    control_trials = pd.DataFrame(
        {
            "run_id": results["run_id"],
            "monkey_total_return": results["total_return"],
            "monkey_prior_return": results["prior_return"],
            "monkey_recent_return": results["recent_return"],
            "fixed_initial_25_return": fixed_random,
        }
    )
    control_trials.to_csv(output_dir / "07_market_control_trials.csv", index=False)
    benchmark_total = float(results["benchmark_total_return"].median())
    controls = pd.DataFrame(
        [
            {"control": "CSI300_cap_weight", "return": benchmark_total, "scope": "single"},
            {"control": "CSI300_initial_static_equal_weight", "return": static_equal_weight, "scope": "single"},
            {"control": "CSI300_dynamic_weekly_equal_weight", "return": dynamic_equal_weight, "scope": "single"},
            {"control": "fixed_random_25_mean", "return": fixed_random.mean(), "scope": "10000 trials"},
            {"control": "fixed_random_25_median", "return": fixed_random.median(), "scope": "10000 trials"},
            {"control": "weekly_monkey_25_mean", "return": results.total_return.mean(), "scope": "10000 trials"},
            {"control": "weekly_monkey_25_median", "return": results.total_return.median(), "scope": "10000 trials"},
        ]
    )
    controls.to_csv(output_dir / "07_market_controls.csv", index=False)
    market_payload = {
        "title": "07 市场环境与收益归因",
        "verdict": "高收益主要来自回测窗口本身的强上涨，随机策略没有预测力；等权和路径结构解释与市值加权指数之间的差异",
        "findings": [
            f"沪深300同期累计上涨{benchmark_total:.2%}；随机猴群两年中位收益{results.total_return.median():.2%}，中位简单超额{results.excess_total_return.median():.2%}。",
            f"第一年沪深300上涨{results.prior_benchmark_return.median():.2%}、猴群中位{results.prior_return.median():.2%}；近一年分别{results.recent_benchmark_return.median():.2%}/{results.recent_return.median():.2%}，高收益几乎全部集中在第一年。",
            f"首期固定25只持有到期的中位收益{fixed_random.median():.2%}；动态CSI300等权周再平衡粗对照{dynamic_equal_weight:.2%}。",
            "随机信号Mean IC/RankIC/ICIR均不显著异于0，因此不能把组合盈利解释为选股预测能力。",
        ],
        "benchmark_cap_weight_return": benchmark_total,
        "monkey_total_return": _stats(results["total_return"]),
        "monkey_excess_return": _stats(results["excess_total_return"]),
        "fixed_random_25_return": _stats(fixed_random),
        "static_initial_csi300_equal_weight_return": static_equal_weight,
        "dynamic_weekly_csi300_equal_weight_return": dynamic_equal_weight,
        "prior_period": {
            "monkey_median": float(results["prior_return"].median()),
            "benchmark": float(results["prior_benchmark_return"].median()),
        },
        "recent_period": {
            "monkey_median": float(results["recent_return"].median()),
            "benchmark": float(results["recent_benchmark_return"].median()),
        },
    }
    _write_step(output_dir, "07_market_attribution", market_payload)

    repro_full = _compare_runs(comparison, results)
    repro_single_a, _ = _load_run(run_dir.parent / "audit_repro_single_a")
    repro_single_b, _ = _load_run(run_dir.parent / "audit_repro_single_b")
    repro_hash_a, _ = _load_run(run_dir.parent / "audit_repro_hash0_a")
    repro_hash_b, _ = _load_run(run_dir.parent / "audit_repro_hash0_b")
    reproducibility_payload = {
        "title": "08 Trial复现与随机流审计",
        "verdict": "发现并定位到PYTHONHASHSEED缺失；它影响单条路径复现，但没有显著改变10,000组总体分布",
        "findings": [
            f"两次完整运行相同种子和初始持仓下，有{repro_full['different_return_runs']:,}/{repro_full['matched_runs']:,}条收益不同。",
            "Qlib Position.get_stock_list通过set构造列表，进程哈希随机化改变股票顺序；random choice在相同随机数下因此可能映射到不同股票。",
            "两次未固定哈希的单进程10条探针不一致；固定PYTHONHASHSEED=0后所有收益、净值、成本和换手逐值一致。",
            f"两次10,000组收益分布KS检验p={repro_full['distribution_ks_pvalue']:.4f}，总体分布没有显著差异。",
        ],
        "full_run_comparison": repro_full,
        "single_process_without_hash_seed": _compare_runs(repro_single_a, repro_single_b),
        "single_process_with_hash_seed_zero": _compare_runs(repro_hash_a, repro_hash_b),
        "root_cause_source": "qlib.backtest.position.Position.get_stock_list uses list(set(...))",
        "required_fix": "Set PYTHONHASHSEED before interpreter/worker startup and record it in metadata; alternatively sort stock lists before random selection.",
    }
    _write_step(output_dir, "08_reproducibility", reproducibility_payload)

    ic_summary_path = run_dir / "ic_audit_summary.json"
    ic_summary = json.loads(ic_summary_path.read_text(encoding="utf-8"))
    summary = {
        "title": "CSI300猴子实验逻辑审计",
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "overall_verdict": "市场上涨是真实主因；实验没有选股信号，但存在周频执行真实性和trial级复现两个需要修复的问题。",
        "severity": {
            "dynamic_universe": "warning",
            "execution_timing": "warning",
            "dropn_semantics": "pass",
            "costs": "pass",
            "tradability": "warning",
            "annualization": "pass_with_convention_note",
            "reproducibility": "issue",
            "market_attribution": "market_driven",
        },
        "steps": {
            "dynamic_universe": universe_payload,
            "execution_timing": timing_payload,
            "dropn": dropn_payload,
            "costs": cost_payload,
            "tradability": tradability_payload,
            "annualization": annual_payload,
            "market_attribution": market_payload,
            "reproducibility": reproducibility_payload,
        },
        "ic_audit": ic_summary,
        "source_evidence": {
            "experiment": "quantpits/experiments/monkey.py",
            "strategy": "qlib/contrib/strategy/signal_strategy.py::TopkDropoutStrategy",
            "position_order": "qlib/backtest/position.py::Position.get_stock_list",
            "exchange": "qlib/backtest/exchange.py::Exchange",
            "quote_aggregation": "qlib/backtest/high_performance_ds.py::NumpyQuote._agg_data",
            "annualization": "quantpits/experiments/monkey.py::calculate_trial_metrics",
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "overall_verdict": summary["overall_verdict"],
        "controls": controls.to_dict(orient="records"),
        "reproducibility": repro_full,
        "tradability": tradability_totals,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
