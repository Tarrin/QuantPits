import importlib

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qlib.backtest.decision import Order
from qlib.backtest.signal import Signal

from quantpits.config_contracts.validators import validate_strategy_config


@pytest.fixture
def strategy_module(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    monkeypatch.setenv("QLIB_WORKSPACE_DIR", str(workspace))

    from quantpits.utils import env, strategy

    importlib.reload(env)
    return importlib.reload(strategy)


def _frames(count=25, *, scores=None, instruments=None):
    instruments = instruments or [f"I{i}" for i in range(1, count + 1)]
    scores = scores or list(range(count, 0, -1))
    pred = pd.DataFrame(
        {"score": scores},
        index=pd.MultiIndex.from_arrays(
            [[pd.Timestamp("2026-01-01")] * len(instruments), instruments],
            names=["datetime", "instrument"],
        ),
    )
    prices = pd.DataFrame(
        {
            "current_close": 10.0,
            "possible_min": 9.0,
            "possible_max": 11.0,
        },
        index=pd.Index(instruments, name="instrument"),
    )
    return pred, prices


def _rank_config(**params):
    return {
        "strategy": {"name": "rank_buffer", "params": params},
        "backtest": {"account": 100_000, "exchange_kwargs": {"deal_price": "close"}},
    }


def test_rank_buffer_generator_uses_buy_and_sell_boundaries(strategy_module):
    generator = strategy_module.TopkDropoutOrderGenerator(
        topk=5, n_drop=1, buy_suggestion_factor=1, buy_rank=10, sell_rank=20,
    )
    pred, prices = _frames()
    holdings = [{"instrument": item, "value": 100} for item in ("I10", "I11", "I20", "I21")]

    hold, sells, buys, ranked, buy_count = generator.analyze_positions(pred, prices, holdings)

    assert ranked.index[:10].tolist() == [f"I{i}" for i in range(1, 11)]
    assert sells.index.tolist() == ["I21"]
    assert hold.index.tolist() == ["I10", "I11", "I20"]
    assert buys.index.tolist() == ["I1", "I2"]
    assert buy_count == 2
    assert "I11" not in buys.index


def test_rank_buffer_generator_handles_initial_missing_and_empty_candidate_books(strategy_module):
    generator = strategy_module.TopkDropoutOrderGenerator(
        topk=5, n_drop=1, buy_suggestion_factor=1, buy_rank=10, sell_rank=20,
    )
    pred, prices = _frames(10)

    _, sells, buys, _, buy_count = generator.analyze_positions(pred, prices, [])
    assert sells.empty
    assert buys.index.tolist() == [f"I{i}" for i in range(1, 6)]
    assert buy_count == 5

    missing_holdings = [{"instrument": "MISSING", "value": 100}]
    hold, sells, buys, _, buy_count = generator.analyze_positions(pred, prices, missing_holdings)
    assert hold.index.tolist() == ["MISSING"]
    assert sells.empty
    assert buys.index.tolist() == [f"I{i}" for i in range(1, 5)]
    assert buy_count == 4

    only_held = [{"instrument": "I1", "value": 100}]
    _, sells, buys, _, buy_count = generator.analyze_positions(
        pred.iloc[:1], prices.iloc[:1], only_held,
    )
    assert sells.empty and buys.empty
    assert buy_count == 4


def test_rank_buffer_generator_sells_only_one_worst_outside_sell_rank(strategy_module):
    generator = strategy_module.TopkDropoutOrderGenerator(
        topk=5, n_drop=1, buy_suggestion_factor=1, buy_rank=10, sell_rank=20,
    )
    pred, prices = _frames()
    holdings = [{"instrument": f"I{i}", "value": 100} for i in range(21, 26)]

    result = generator.analyze_positions_with_universe(pred, prices, holdings)

    assert result.normal_sell_candidates.index.tolist() == ["I25"]
    assert result.buy_candidates.index.tolist() == ["I1"]
    assert result.target_buy_count == 1


def test_rank_buffer_generator_preserves_forced_universe_exits(strategy_module):
    from quantpits.order.execution import UniverseSnapshot

    generator = strategy_module.TopkDropoutOrderGenerator(
        topk=5, n_drop=1, buy_suggestion_factor=1, buy_rank=10, sell_rank=20,
    )
    pred, prices = _frames()
    prices.loc["X"] = [10.0, 9.0, 11.0]
    holdings = [{"instrument": "X", "value": 100}]
    universe = UniverseSnapshot.observe("demo", "2026-01-01", [f"I{i}" for i in range(1, 26)])

    result = generator.analyze_positions_with_universe(pred, prices, holdings, universe)

    assert result.forced_exit_candidates.index.tolist() == ["X"]
    assert result.normal_sell_candidates.empty
    assert result.target_buy_count == 5
    assert result.buy_candidates.index.tolist() == [f"I{i}" for i in range(1, 6)]


def test_rank_buffer_config_and_constructor_validation(strategy_module):
    valid = validate_strategy_config(_rank_config(topk=5, n_drop=1, buy_rank=10, sell_rank=20))
    assert not [message for message in valid if message.severity == "warning"]
    assert not [message for message in valid if message.severity == "error"]

    for params in (
        {"topk": 5, "n_drop": 1, "buy_rank": 10},
        {"topk": 5, "n_drop": 1, "buy_rank": True, "sell_rank": 20},
        {"topk": 5, "n_drop": 1, "buy_rank": 4, "sell_rank": 20},
        {"topk": 5, "n_drop": 1, "buy_rank": 10, "sell_rank": 9},
        {"topk": 5.0, "n_drop": 1, "buy_rank": 10, "sell_rank": 20},
    ):
        messages = validate_strategy_config(_rank_config(**params))
        assert [message for message in messages if message.severity == "error"]
        with pytest.raises(ValueError):
            strategy_module.TopkDropoutOrderGenerator(**params)

    from quantpits.utils.rank_buffer_strategy import RankBufferStrategy

    with pytest.raises(ValueError):
        RankBufferStrategy(topk=5, n_drop=1, buy_rank=10, sell_rank=20, method_buy="random")


def test_rank_buffer_factory_selects_custom_backtest_and_port_analysis(strategy_module):
    with pytest.raises(ValueError):
        strategy_module.create_order_generator(_rank_config(topk=5, n_drop=1))
    with pytest.raises(ValueError):
        strategy_module.create_order_generator(_rank_config(topk=5, n_drop=None, buy_rank=10, sell_rank=20))
    config = _rank_config(topk=5, n_drop=1, buy_rank=10, sell_rank=20, buy_suggestion_factor=1)
    generator = strategy_module.create_order_generator(config)
    assert isinstance(generator, strategy_module.TopkDropoutOrderGenerator)
    assert (generator.buy_rank, generator.sell_rank) == (10, 20)

    from quantpits.utils.rank_buffer_strategy import RankBufferStrategy

    strategy = strategy_module.create_backtest_strategy(_TestSignal(pd.Series(dtype=float)), config)
    assert isinstance(strategy, RankBufferStrategy)
    port = strategy_module.generate_port_analysis_config(config, freq="day")
    assert port["strategy"]["class"] == "RankBufferStrategy"
    assert port["strategy"]["module_path"] == "quantpits.utils.rank_buffer_strategy"
    assert port["strategy"]["kwargs"]["buy_rank"] == 10
    assert port["strategy"]["kwargs"]["sell_rank"] == 20


class _TestSignal(Signal):
    def __init__(self, value):
        self.value = value
        self.calls = []

    def get_signal(self, start_time, end_time):
        self.calls.append((start_time, end_time))
        return self.value


class _TestPosition:
    def __init__(self, holdings, *, count=1):
        self.holdings = list(holdings)
        self.count = count

    def get_stock_list(self):
        return list(self.holdings)

    def get_cash(self):
        return 100_000.0

    def get_stock_count(self, code, bar):
        return self.count

    def get_stock_amount(self, code):
        return 100.0


class _TestCalendar:
    def get_trade_step(self):
        return 0

    def get_step_time(self, step=None, shift=0):
        day = "2026-01-02" if shift == 0 else "2026-01-01"
        return pd.Timestamp(day), pd.Timestamp(day)

    def get_freq(self):
        return "day"


class _TestExchange:
    def __init__(self, *, tradable=None, block_sells=False):
        self.tradable = tradable
        self.block_sells = block_sells

    def is_stock_tradable(self, stock_id, start_time, end_time, direction=None):
        return self.tradable is None or stock_id in self.tradable

    def check_order(self, order):
        return not (self.block_sells and order.direction == Order.SELL)

    def deal_order(self, order, position):
        return order.amount, 0.0, 1.0

    def get_deal_price(self, stock_id, start_time, end_time, direction):
        return 1.0

    def get_factor(self, stock_id, start_time, end_time):
        return 1.0

    def round_amount_by_trade_unit(self, amount, factor):
        return amount


def _real_strategy(scores, holdings, *, only_tradable=False, exchange=None, count=1):
    from quantpits.utils.rank_buffer_strategy import RankBufferStrategy

    position = _TestPosition(holdings, count=count)
    return RankBufferStrategy(
        signal=_TestSignal(scores),
        topk=5,
        n_drop=1,
        buy_rank=10,
        sell_rank=20,
        only_tradable=only_tradable,
        common_infra={"trade_account": SimpleNamespace(current_position=position)},
        level_infra={"trade_calendar": _TestCalendar()},
        trade_exchange=exchange or _TestExchange(),
    )


def test_rank_buffer_real_parent_orders_and_restores_signal():
    scores = pd.Series(
        list(range(25, 0, -1)), index=[f"I{i}" for i in range(1, 26)], dtype=float,
    )
    strategy = _real_strategy(scores, ["I21"])
    original_signal = strategy.signal

    decision = strategy.generate_trade_decision()

    assert [(order.stock_id, order.direction) for order in decision.order_list] == [
        ("I21", Order.SELL),
        ("I1", Order.BUY),
        ("I2", Order.BUY),
        ("I3", Order.BUY),
        ("I4", Order.BUY),
        ("I5", Order.BUY),
    ]
    assert strategy.signal is original_signal
    assert strategy.n_drop == 1
    assert len(original_signal.calls) == 1


def test_rank_buffer_real_parent_protects_missing_and_empty_books():
    scores = pd.Series({"I1": 5.0, "I3": 3.0, "I4": 2.0, "I5": 1.0, "I6": 0.0})
    strategy = _real_strategy(scores, ["I1", "MISSING", "I3", "I4", "I5"])
    original_signal = strategy.signal

    decision = strategy.generate_trade_decision()

    assert decision.order_list == []
    assert strategy.signal is original_signal

    empty_scores = pd.Series({"I1": 5.0, "I3": 3.0, "I4": 2.0, "I5": 1.0})
    empty_strategy = _real_strategy(empty_scores, ["I1", "I3", "I4", "I5", "I6"])
    original_empty_signal = empty_strategy.signal
    assert empty_strategy.generate_trade_decision().order_list == []
    assert empty_strategy.signal is original_empty_signal


def test_rank_buffer_real_parent_does_not_buy_when_sell_is_blocked():
    scores = pd.Series(
        list(range(25, 0, -1)), index=[f"I{i}" for i in range(1, 26)], dtype=float,
    )
    strategy = _real_strategy(scores, ["I1", "I2", "I3", "I4", "I21"], exchange=_TestExchange(block_sells=True))

    decision = strategy.generate_trade_decision()

    assert decision.order_list == []


def test_rank_buffer_real_parent_filters_untradable_outside_sell():
    scores = pd.Series(
        list(range(25, 0, -1)), index=[f"I{i}" for i in range(1, 26)], dtype=float,
    )
    exchange = _TestExchange(tradable={f"I{i}" for i in range(1, 21)})
    strategy = _real_strategy(
        scores, ["I1", "I2", "I3", "I4", "I21"], only_tradable=True, exchange=exchange,
    )

    decision = strategy.generate_trade_decision()

    assert decision.order_list == []


def test_rank_buffer_decision_score_uses_absolute_buy_rank():
    from quantpits.utils.rank_buffer_strategy import RankBufferStrategy

    scores = pd.Series(
        list(range(20, 0, -1)), index=[f"I{i}" for i in range(1, 21)], dtype=float,
    )
    strategy = _real_strategy(scores, [f"I{i}" for i in range(1, 10)])
    transformed = strategy._decision_scores(scores)

    assert "I10" in transformed.index
    assert "I11" not in transformed.index
    assert all(np.isfinite(transformed.to_numpy()))
