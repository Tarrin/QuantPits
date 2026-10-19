import argparse
from dataclasses import replace
import json

import pandas as pd
import pytest

from quantpits.order.command import (
    OrderPlanError,
    OrderRunConfig,
    OrderRunOptions,
    PreparedOrderRun,
    ResolvedOrderSource,
    build_order_arg_parser,
    build_order_command_plan,
    load_order_run_config,
    options_from_namespace,
)
from quantpits.order.execution import LoadedOrderPrediction, OrderExecutionHooks, PositionAnalysisResult
from quantpits.order.service import OrderGenerationService
from quantpits.runtime import fingerprint_command_plan
from quantpits.utils.workspace import WorkspaceContext


def _config():
    return OrderRunConfig(
        merged_config={
            "market": "demo",
            "current_cash": 100.0,
            "current_holding": [
                {"instrument": "H1", "value": 100},
                {"instrument": "H2", "value": 200},
            ],
        },
        cashflow_config={"cashflows": {"2026-01-01": 25.0}},
        strategy_config={
            "strategy": {
                "name": "topk_dropout",
                "params": {"topk": 3, "n_drop": 1, "buy_suggestion_factor": 2, "sell_out_of_universe": False},
            }
        },
        ensemble_config={},
        ensemble_records={},
        train_records={},
    )


def _prepared(tmp_path, *, phase="all"):
    ctx = WorkspaceContext.from_root(tmp_path)
    options = OrderRunOptions(output_dir=(tmp_path / "output").as_posix(), dry_run=True, phase=phase)
    source = ResolvedOrderSource("ensemble", None, "demo", "rid", "Ensemble_Fusion", "ensemble")
    from quantpits.runtime import CommandPlan

    return PreparedOrderRun(
        ctx=ctx,
        options=options,
        execution_options=options,
        cli_args=("--phase", phase),
        validation_result=None,
        config=_config(),
        source=source,
        plan=CommandPlan(command="order_gen", workspace=ctx.root.as_posix(), run_id="phase-run"),
        plan_fingerprint="phase-fingerprint",
    )


class _PhaseGenerator:
    def __init__(self, calls):
        self.calls = calls

    def analyze_positions_with_universe(self, predictions, prices, holdings, universe):
        frame = pd.DataFrame(
            {
                "score": [0.95, 0.90, 0.80, 0.10],
                "current_close": [10.0, 10.0, 10.0, 10.0],
                "possible_min": [9.0, 9.0, 9.0, 9.0],
                "possible_max": [11.0, 11.0, 11.0, 11.0],
            },
            index=pd.Index(["B1", "B2", "H1", "H2"], name="instrument"),
        )
        empty = frame.iloc[0:0]
        continuing = frame.loc[["H1"]]
        normal_sell = frame.loc[["H2"]]
        buy = frame.loc[["B1", "B2"]]
        return PositionAnalysisResult(
            continuing,
            normal_sell,
            empty,
            (),
            (),
            buy,
            frame,
            2,
            None,
            None,
            0,
            1,
            1,
            1,
            1,
            2,
        )

    def generate_sell_orders(self, candidates, holdings, trade_date):
        self.calls.append(("sell", tuple(candidates.index)))
        return ([{"instrument": "H2", "estimated_amount": 10000.0}], 10000.0)

    def generate_buy_orders(self, candidates, count, cash, trade_date):
        self.calls.append(("buy", tuple(candidates.index), count, cash))
        return [{"instrument": instrument, "estimated_amount": 10.0} for instrument in candidates.index[:1]]


def _hooks(calls):
    source = ResolvedOrderSource("ensemble", None, "demo", "rid", "Ensemble_Fusion", "ensemble")
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-01-01"), "B1"), (pd.Timestamp("2026-01-01"), "B2")],
        names=["datetime", "instrument"],
    )
    prices = pd.DataFrame(
        {"current_close": [10.0] * 4, "possible_min": [9.0] * 4, "possible_max": [11.0] * 4},
        index=pd.Index(["B1", "B2", "H1", "H2"], name="instrument"),
    )
    generator = _PhaseGenerator(calls)
    return OrderExecutionHooks(
        init_qlib=lambda: None,
        get_anchor_date=lambda: "2026-01-01",
        get_next_trade_date=lambda _: "2026-01-02",
        load_predictions=lambda _: LoadedOrderPrediction(
            pd.DataFrame({"score": [0.9, 0.8]}, index=index), source, "phase source"
        ),
        get_price_data=lambda *args, **kwargs: prices,
        create_order_generator=lambda _: generator,
        get_strategy_params=lambda _: {"topk": 3, "n_drop": 1, "buy_suggestion_factor": 2, "sell_out_of_universe": False},
        build_model_opinions=lambda _: None,
        persist_artifacts=lambda _: None,
    )


def test_parser_phase_choices_and_default():
    parser = build_order_arg_parser()
    assert parser.parse_args([]).phase == "all"
    assert parser.parse_args(["--phase", "buy"]).phase == "buy"
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "invalid"])


def test_account_snapshot_requires_buy_phase():
    parser = build_order_arg_parser()
    args = parser.parse_args(["--account-snapshot", "account.json"])
    with pytest.raises(OrderPlanError, match="only with --phase buy"):
        options_from_namespace(args)


def test_account_snapshot_overlays_only_account_state(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    snapshot_path = tmp_path / "intraday_account.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "as_of": "2026-01-02T10:30:00+08:00",
                "current_cash": 456.5,
                "current_full_cash": 789.0,
                "current_holding": [{"instrument": "SNAP", "value": 12.0}],
                "ignored": "must not enter merged config",
            }
        ),
        encoding="utf-8",
    )
    base = {
        "market": "demo",
        "current_cash": 10.0,
        "current_full_cash": 20.0,
        "current_holding": [{"instrument": "OLD", "value": 1.0}],
    }
    config = load_order_run_config(
        ctx,
        OrderRunOptions(phase="buy", account_snapshot="intraday_account.json"),
        merged_config=base,
        cashflow_config={},
        strategy_config={"strategy": {"name": "topk_dropout", "params": {}}},
    )
    assert config.merged_config["current_cash"] == 456.5
    assert config.merged_config["current_full_cash"] == 789.0
    assert config.merged_config["current_holding"] == [{"instrument": "SNAP", "value": 12.0}]
    assert config.merged_config["market"] == "demo"
    assert "ignored" not in config.merged_config
    assert base["current_cash"] == 10.0
    assert base["current_holding"] == [{"instrument": "OLD", "value": 1.0}]
    assert config.account_snapshot_path == "intraday_account.json"
    assert config.account_snapshot_as_of == "2026-01-02T10:30:00+08:00"


@pytest.mark.parametrize(
    "payload",
    [
        {"current_holding": []},
        {"current_cash": -1, "current_holding": []},
        {"current_cash": "10", "current_holding": []},
        {"current_cash": 10, "current_holding": "not-a-list"},
        {"current_cash": 10, "current_holding": ["SNAP"]},
        {"current_cash": 10, "current_holding": [{"instrument": ""}]},
        {"current_cash": 10, "current_holding": [{"instrument": "SNAP"}, {"instrument": "SNAP"}]},
    ],
)
def test_invalid_account_snapshot_fails_closed_before_execution(tmp_path, payload):
    ctx = WorkspaceContext.from_root(tmp_path)
    snapshot_path = tmp_path / "bad_account.json"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrderPlanError):
        load_order_run_config(
            ctx,
            OrderRunOptions(phase="buy", account_snapshot="bad_account.json"),
            merged_config={},
            cashflow_config={},
            strategy_config={"strategy": {"name": "topk_dropout", "params": {}}},
        )


def test_account_snapshot_plan_is_fingerprinted_without_raw_account_values(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    snapshot_path = tmp_path / "account.json"
    snapshot_path.write_text(
        json.dumps({"as_of": "2026-01-02", "current_cash": 123456.75, "current_holding": [{"instrument": "SECRET"}]}),
        encoding="utf-8",
    )
    options = OrderRunOptions(phase="buy", account_snapshot="account.json", run_id="same")
    config = load_order_run_config(
        ctx,
        options,
        merged_config={"market": "demo", "current_cash": 1, "current_holding": []},
        cashflow_config={},
        strategy_config={"strategy": {"name": "topk_dropout", "params": {}}},
    )
    source = ResolvedOrderSource("ensemble", None, "demo", "rid", "Ensemble_Fusion", "ensemble")
    first = build_order_command_plan(
        ctx=ctx, options=options, cli_args=("--phase", "buy", "--account-snapshot", "account.json"),
        config=config, source=source, validation_result=None,
    )
    snapshot_ref = next(item for item in first.inputs if item.path == "account.json")
    assert snapshot_ref.required
    assert snapshot_ref.fingerprint
    assert first.metadata["account_snapshot_path"] == "account.json"
    assert first.metadata["account_snapshot_as_of"] == "2026-01-02"
    assert first.metadata["account_snapshot_holding_count"] == 1
    public = json.dumps(first.to_public_dict(), ensure_ascii=False)
    assert "123456.75" not in public
    assert "SECRET" not in public

    snapshot_path.write_text(
        json.dumps({"as_of": "2026-01-02", "current_cash": 123456.76, "current_holding": [{"instrument": "SECRET"}]}),
        encoding="utf-8",
    )
    second = build_order_command_plan(
        ctx=ctx, options=options, cli_args=("--phase", "buy", "--account-snapshot", "account.json"),
        config=config, source=source, validation_result=None,
    )
    assert fingerprint_command_plan(first) != fingerprint_command_plan(second)


def test_phase_plan_outputs_steps_and_fingerprint(tmp_path):
    ctx = WorkspaceContext.from_root(tmp_path)
    source = ResolvedOrderSource("ensemble", None, "demo", "rid", "Ensemble_Fusion", "ensemble")
    config = _config()
    all_plan = build_order_command_plan(
        ctx=ctx, options=OrderRunOptions(run_id="same", phase="all"), cli_args=(), config=config,
        source=source, validation_result=None,
    )
    sell_plan = build_order_command_plan(
        ctx=ctx, options=OrderRunOptions(run_id="same", phase="sell"), cli_args=(), config=config,
        source=source, validation_result=None,
    )
    buy_plan = build_order_command_plan(
        ctx=ctx, options=OrderRunOptions(run_id="same", phase="buy"), cli_args=(), config=config,
        source=source, validation_result=None,
    )
    assert all_plan.metadata["phase"] == "all"
    assert any("sell_suggestion" in item for item in sell_plan.metadata["preview_output_paths"])
    assert not any("buy_suggestion" in item for item in sell_plan.metadata["preview_output_paths"])
    assert not any("sell_suggestion" in item for item in buy_plan.metadata["preview_output_paths"])
    assert any("buy_suggestion" in item for item in buy_plan.metadata["preview_output_paths"])
    sell_step = next(item for item in buy_plan.steps if item.name == "generate-sell-orders")
    buy_step = next(item for item in sell_plan.steps if item.name == "generate-buy-orders")
    assert sell_step.can_skip and sell_step.skip_reason == "--phase buy"
    assert buy_step.can_skip and buy_step.skip_reason == "--phase sell"
    assert fingerprint_command_plan(all_plan) != fingerprint_command_plan(buy_plan)


@pytest.mark.parametrize(
    ("phase", "expected_sell", "expected_buy", "expected_cash"),
    [("all", 1, 1, 10125.0), ("sell", 1, 0, None), ("buy", 0, 1, 125.0)],
)
def test_service_phase_semantics(tmp_path, phase, expected_sell, expected_buy, expected_cash):
    calls = []
    summary = OrderGenerationService(_hooks(calls)).execute(_prepared(tmp_path, phase=phase))
    assert summary.phase == phase
    assert summary.sell_count == expected_sell
    assert summary.buy_count == expected_buy
    if phase == "sell":
        assert summary.target_buy_count == 0
        assert summary.planned_final_holding_count == summary.remaining_after_sell == 1
        assert calls == [("sell", ("H2",))]
    elif phase == "buy":
        assert summary.target_buy_count == 1
        assert summary.remaining_after_sell == 2
        assert summary.normal_sell_count == 0
        assert summary.forced_exit_count == 0
        assert summary.estimated_sell_amount == 0.0
        assert calls == [("buy", ("B1", "B2"), 1, expected_cash)]
    else:
        assert summary.estimated_sell_amount == 10000.0
        assert calls[0] == ("sell", ("H2",))
        assert calls[1] == ("buy", ("B1", "B2"), 1, expected_cash)
