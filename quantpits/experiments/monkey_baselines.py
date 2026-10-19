"""Matched CSI300 equal-weight controls for the monkey experiment."""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantpits.experiments.monkey import (
    MonkeyExperimentConfig,
    _disable_qlib_progress_bar,
    first_trade_day,
    last_trade_day,
)


def run_equal_weight_baseline(config: MonkeyExperimentConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    """Backtest dynamic CSI300 equal weight under monkey execution semantics.

    The portfolio targets equal weights across the previous signal interval's
    eligible CSI300 constituents, invests ``risk_degree`` of account value,
    and rebalances on the following interval's first actual trading-day close.
    Qlib applies the same limit/suspension checks and transaction-cost config
    used by the monkey trials.
    """

    from quantpits.utils import env

    env.init_qlib()
    from qlib.backtest import backtest_loop
    from qlib.backtest.account import Account
    from qlib.backtest.decision import OrderDir, TradeDecisionWO
    from qlib.backtest.exchange import Exchange
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.backtest.utils import CommonInfrastructure
    from qlib.contrib.strategy import WeightStrategyBase
    from qlib.data import D
    from quantpits.utils.backtest_utils import extract_report_df

    signal_start = (
        pd.Timestamp(config.start_time) - pd.Timedelta(days=config.signal_lookback_days)
    ).strftime("%Y-%m-%d")
    features = D.features(
        D.instruments(market=config.market), ["$close"],
        start_time=signal_start, end_time=config.end_time, freq="day",
    )
    close = pd.to_numeric(features.iloc[:, 0], errors="coerce")
    eligible = close[close.notna() & np.isfinite(close) & (close > 0)].sort_index()
    if eligible.empty:
        raise RuntimeError("no eligible CSI300 observations for equal-weight baseline")
    signal = pd.Series(1.0, index=eligible.index, name="score")
    calendar = pd.DatetimeIndex(pd.to_datetime(D.calendar(
        start_time=signal_start, end_time=config.end_time, freq="day",
    ))).normalize().sort_values()

    backtest_config = config.strategy_config["backtest"]
    exchange_kwargs = copy.deepcopy(backtest_config["exchange_kwargs"])
    exchange_freq = exchange_kwargs.pop("freq", "day")
    class InventorySafeExchange(Exchange):
        """Floor adjusted lots without Qlib's absolute +0.1 epsilon.

        The upstream epsilon can cross a lot boundary when the adjusted factor
        is small, producing a fractional sell slightly larger than inventory.
        """

        def is_stock_tradable(self, stock_id, start_time, end_time, direction=None):
            if not super().is_stock_tradable(
                stock_id, start_time, end_time, direction=direction
            ):
                return False
            close_value = self.get_close(stock_id, start_time, end_time)
            return bool(
                close_value is not None
                and np.isscalar(close_value)
                and np.isfinite(close_value)
                and float(close_value) > 0
            )

        def round_amount_by_trade_unit(self, deal_amount, factor=None):
            if not np.isfinite(deal_amount):
                return 0.0
            if (
                self.trade_unit is not None
                and factor is not None
                and np.isfinite(factor)
                and float(factor) > 0
            ):
                units = math.floor(
                    max(float(deal_amount), 0.0) * float(factor)
                    / float(self.trade_unit)
                    + 1e-12
                )
                return units * float(self.trade_unit) / float(factor)
            return deal_amount

    exchange = InventorySafeExchange(
        freq=exchange_freq,
        start_time=config.start_time,
        end_time=config.end_time,
        codes=sorted(eligible.index.get_level_values("instrument").unique()),
        **exchange_kwargs,
    )

    class ExactDayEqualWeightStrategy(WeightStrategyBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.execution_log: list[dict[str, Any]] = []

        def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
            if isinstance(score, pd.DataFrame):
                score = score.iloc[:, 0]
            codes = sorted(str(code) for code in score.dropna().index.unique())
            if not codes:
                return {}
            weight = 1.0 / len(codes)
            return {code: weight for code in codes}

        def generate_trade_decision(self, execute_result=None):
            trade_step = self.trade_calendar.get_trade_step()
            trade_start, trade_end = self.trade_calendar.get_step_time(trade_step)
            pred_start, pred_end = self.trade_calendar.get_step_time(trade_step, shift=1)
            execution_time = first_trade_day(calendar, trade_start, trade_end)
            valuation_time = last_trade_day(
                calendar, trade_start,
                min(pd.Timestamp(trade_end), pd.Timestamp(config.end_time)),
            )
            signal_time = last_trade_day(
                calendar, pred_start,
                min(pd.Timestamp(pred_end), execution_time - pd.Timedelta(days=1)),
            )
            score = self.signal.get_signal(start_time=pred_start, end_time=pred_end)
            if score is None:
                return TradeDecisionWO([], self)
            current = copy.deepcopy(self.trade_position)
            target = self.generate_target_weight_position(
                score, current, execution_time, execution_time
            )
            tradable_codes = [
                code for code in target
                if self.trade_exchange.is_stock_tradable(
                    code, start_time=execution_time, end_time=execution_time
                )
            ]
            target = (
                {code: 1.0 / len(tradable_codes) for code in tradable_codes}
                if tradable_codes
                else {}
            )
            current_amounts = current.get_stock_amount_dict()
            target_amounts: dict[str, float] = {}
            # Frozen positions cannot be sold on this execution date. Retain
            # their exact inventory and value them using the account's last
            # known close instead of asking Qlib for a missing deal price.
            for code, amount in current_amounts.items():
                if not self.trade_exchange.is_stock_tradable(
                    code, execution_time, execution_time
                ):
                    target_amounts[str(code)] = float(amount)

            risk_value = (
                self.get_risk_degree(trade_step)
                * float(current.calculate_value())
                / (1.0 + max(self.trade_exchange.open_cost, self.trade_exchange.close_cost))
            )
            target_value = risk_value / len(target) if target else 0.0
            for code in target:
                price = self.trade_exchange.get_deal_price(
                    code, execution_time, execution_time, direction=OrderDir.BUY
                )
                if price is not None and np.isfinite(price) and price > 0:
                    target_amounts[code] = float(target_value // float(price))
            orders = self.trade_exchange.generate_order_for_target_amount_position(
                target_position=target_amounts,
                current_position=current_amounts,
                start_time=execution_time,
                end_time=execution_time,
            )
            # Qlib's target-amount rounding can exceed a fractional adjusted
            # holding by less than one share when factors change.  A real sell
            # cannot exceed inventory, so cap it to the exact current amount.
            for order in orders:
                if order.direction == order.SELL:
                    available = float(current.get_stock_amount(order.stock_id))
                    order.amount = min(float(order.amount), available)
                    factor = self.trade_exchange.get_factor(
                        order.stock_id, execution_time, execution_time
                    )
                    if order.stock_id not in target:
                        # Exact liquidation makes Qlib skip lot rounding.
                        order.amount = available
                    elif (
                        factor is not None
                        and np.isfinite(factor)
                        and float(factor) > 0
                        and self.trade_exchange.trade_unit
                    ):
                        # Qlib rounds a target delta once when creating the
                        # order and again when dealing it.  Leave one adjusted
                        # lot of headroom on partial sells so the second pass
                        # cannot exceed inventory because of its +0.1 epsilon.
                        lot = float(self.trade_exchange.trade_unit) / float(factor)
                        order.amount = min(order.amount, max(0.0, available - lot))
            self.execution_log.append({
                "signal_time": signal_time,
                "execution_time": execution_time,
                "valuation_time": valuation_time,
                "constituent_count": len(target),
                "order_count": len(orders),
                "timestamp_mismatches": sum(
                    order.start_time != execution_time or order.end_time != execution_time
                    for order in orders
                ),
            })
            return TradeDecisionWO(orders, self)

    strategy = ExactDayEqualWeightStrategy(signal=signal, risk_degree=config.risk_degree)
    account = Account(
        init_cash=config.initial_cash,
        freq=config.frequency,
        benchmark_config={
            "benchmark": config.benchmark,
            "start_time": config.start_time,
            "end_time": config.end_time,
        },
    )
    infrastructure = CommonInfrastructure(trade_account=account, trade_exchange=exchange)
    strategy.reset_common_infra(infrastructure)
    executor = SimulatorExecutor(
        time_per_step=config.frequency,
        generate_portfolio_metrics=True,
        verbose=False,
        common_infra=infrastructure,
    )
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        with _disable_qlib_progress_bar():
            raw_metrics, _ = backtest_loop(
                start_time=config.start_time,
                end_time=config.end_time,
                trade_strategy=strategy,
                trade_executor=executor,
            )
    report = extract_report_df(raw_metrics).copy()
    log = strategy.execution_log
    if len(report) != len(log):
        raise RuntimeError(f"equal-weight report/log length mismatch: {len(report)} != {len(log)}")
    report.index = pd.DatetimeIndex([item["valuation_time"] for item in log])

    net_nav = pd.to_numeric(report["account"], errors="coerce")
    gross_return = pd.to_numeric(report["return"], errors="coerce").fillna(0.0)
    benchmark_return = pd.to_numeric(report["bench"], errors="coerce").fillna(0.0)
    gross_nav = config.initial_cash * (1.0 + gross_return).cumprod()
    benchmark_nav = config.initial_cash * (1.0 + benchmark_return).cumprod()
    elapsed_years = max(
        (pd.Timestamp(config.end_time) - pd.Timestamp(config.start_time)).days / 365.2425,
        1.0 / 52.0,
    )
    net_total = float(net_nav.iloc[-1] / config.initial_cash - 1.0)
    gross_total = float(gross_nav.iloc[-1] / config.initial_cash - 1.0)
    benchmark_total = float(benchmark_nav.iloc[-1] / config.initial_cash - 1.0)
    summary = {
        "start_time": config.start_time,
        "end_time": config.end_time,
        "weeks": int(len(report)),
        "cap_weight_total_return": benchmark_total,
        "cap_weight_cagr": (1.0 + benchmark_total) ** (1.0 / elapsed_years) - 1.0,
        "equal_weight_gross_total_return": gross_total,
        "equal_weight_gross_cagr": (1.0 + gross_total) ** (1.0 / elapsed_years) - 1.0,
        "equal_weight_net_total_return": net_total,
        "equal_weight_net_cagr": (1.0 + net_total) ** (1.0 / elapsed_years) - 1.0,
        "equal_weight_cost_drag": gross_total - net_total,
        "equal_weight_total_cost": float(pd.to_numeric(report["cost"], errors="coerce").fillna(0.0).sum()),
        "equal_weight_total_turnover": float(pd.to_numeric(report["turnover"], errors="coerce").fillna(0.0).sum()),
        "mean_constituent_count": float(np.mean([item["constituent_count"] for item in log])),
        "min_constituent_count": int(min(item["constituent_count"] for item in log)),
        "max_constituent_count": int(max(item["constituent_count"] for item in log)),
        "order_timestamp_mismatches": int(sum(item["timestamp_mismatches"] for item in log)),
    }
    weekly = pd.DataFrame({
        "valuation_date": report.index,
        "signal_date": [item["signal_time"] for item in log],
        "execution_date": [item["execution_time"] for item in log],
        "constituent_count": [item["constituent_count"] for item in log],
        "order_count": [item["order_count"] for item in log],
        "equal_weight_net_nav": net_nav.to_numpy(dtype=float),
        "equal_weight_gross_nav": gross_nav.to_numpy(dtype=float),
        "cap_weight_nav": benchmark_nav.to_numpy(dtype=float),
        "equal_weight_gross_return": gross_return.to_numpy(dtype=float),
        "cap_weight_return": benchmark_return.to_numpy(dtype=float),
        "cost": pd.to_numeric(report["cost"], errors="coerce").to_numpy(dtype=float),
        "turnover": pd.to_numeric(report["turnover"], errors="coerce").to_numpy(dtype=float),
    })
    weekly["equal_weight_net_return"] = weekly["equal_weight_net_nav"].pct_change()
    weekly.loc[0, "equal_weight_net_return"] = net_nav.iloc[0] / config.initial_cash - 1.0
    return summary, weekly
