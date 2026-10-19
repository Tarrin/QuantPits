"""Qlib strategy adapter for the opt-in rank-buffer policy."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, TradeDecisionWO

from quantpits.utils.strategy import _validate_rank_buffer_params


class RankBufferStrategy(TopkDropoutStrategy):
    """Top-k dropout using separate buy and sell rank thresholds."""

    def __init__(
        self,
        *,
        topk,
        n_drop,
        buy_rank,
        sell_rank,
        method_sell="bottom",
        method_buy="top",
        **kwargs,
    ):
        _validate_rank_buffer_params(topk, buy_rank, sell_rank, n_drop, required=True)
        if method_sell != "bottom" or method_buy != "top":
            raise ValueError("rank_buffer supports only method_sell='bottom' and method_buy='top'")
        super().__init__(
            topk=topk,
            n_drop=n_drop,
            method_sell=method_sell,
            method_buy=method_buy,
            **kwargs,
        )
        self.buy_rank = buy_rank
        self.sell_rank = sell_rank

    def _decision_scores(self, pred_score):
        return self._decision_scores_and_sells(pred_score)[0]

    def _decision_scores_and_sells(self, pred_score):
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if not isinstance(pred_score, pd.Series):
            pred_score = pd.Series(pred_score)

        numeric = pd.to_numeric(pred_score, errors="coerce")
        finite = numeric.notna() & np.isfinite(numeric)
        ranked = numeric[finite].sort_values(ascending=False, kind="mergesort")
        rank_by_instrument = {instrument: rank for rank, instrument in enumerate(ranked.index, 1)}

        holding_ids = list(self.trade_position.get_stock_list())
        holding_set = set(holding_ids)
        buy_ids = [
            instrument
            for instrument in ranked.head(self.buy_rank).index
            if instrument not in holding_set
        ]
        protected = []
        outside = []
        for instrument in holding_ids:
            rank = rank_by_instrument.get(instrument)
            if rank is None or rank <= self.sell_rank:
                protected.append(instrument)
            else:
                outside.append(instrument)
        outside.sort(key=rank_by_instrument.__getitem__, reverse=True)

        scores = {}
        for instrument in protected:
            scores[instrument] = 0.0
        for instrument in buy_ids + outside:
            scores[instrument] = -float(rank_by_instrument[instrument])
        return pd.Series(scores, dtype=float), tuple(outside)

    def generate_trade_decision(self, execute_result=None):
        original_signal = self.signal
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        original_n_drop = self.n_drop
        try:
            pred_score = original_signal.get_signal(
                start_time=pred_start_time,
                end_time=pred_end_time,
            )
            if pred_score is None:
                self.signal = SimpleNamespace(get_signal=lambda **_: None)
                return super().generate_trade_decision(execute_result)
            decision_score, sell_ids = self._decision_scores_and_sells(pred_score)
            if self.only_tradable:
                sell_ids = tuple(
                    instrument
                    for instrument in sell_ids
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=instrument,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                    )
                )
            self.n_drop = min(original_n_drop, len(sell_ids))
            if self.n_drop == 0:
                holding_set = set(self.trade_position.get_stock_list())
                candidate_ids = [
                    instrument for instrument in decision_score.index if instrument not in holding_set
                ]
                if self.only_tradable:
                    candidate_ids = [
                        instrument
                        for instrument in candidate_ids
                        if self.trade_exchange.is_stock_tradable(
                            stock_id=instrument,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                        )
                    ]
                if not candidate_ids:
                    return TradeDecisionWO([], self)
                decision_score = decision_score.copy()
                decision_score.loc[list(holding_set)] = 0.0
                self.n_drop = 1
            self.signal = SimpleNamespace(get_signal=lambda **_: decision_score)
            decision = super().generate_trade_decision(execute_result)
            current_count = len(self.trade_position.get_stock_list())
            actual_sell_count = sum(
                order.direction == Order.SELL for order in decision.order_list
            )
            allowed_buys = max(self.topk - (current_count - actual_sell_count), 0)
            if sum(order.direction == Order.BUY for order in decision.order_list) > allowed_buys:
                buys_left = allowed_buys
                filtered_orders = []
                for order in decision.order_list:
                    if order.direction == Order.BUY:
                        if buys_left == 0:
                            continue
                        buys_left -= 1
                    filtered_orders.append(order)
                decision.order_list = filtered_orders
            return decision
        finally:
            self.n_drop = original_n_drop
            self.signal = original_signal


__all__ = ["RankBufferStrategy"]
