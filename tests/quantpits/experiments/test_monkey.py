from __future__ import annotations

import math

import pandas as pd
import pytest

from quantpits.experiments.monkey import (
    MonkeyExperimentConfig,
    _position_diagnostics,
    calculate_alpha_metrics,
    calculate_regime_metrics,
    calculate_recent_period_metrics,
    calculate_trial_metrics,
    derive_trial_seeds,
    first_trade_day,
    last_trade_day,
    summarize_results,
)
from quantpits.scripts.generate_market_regimes import confirm_states
from quantpits.scripts.compare_model_to_monkey import compare
from quantpits.scripts.run_monkey_experiment import (
    RESULT_FIELDS,
    _prepare_resume_checkpoint,
    _read_completed,
)


def _strategy_config():
    return {
        "strategy": {"name": "topk_dropout", "params": {}},
        "backtest": {"account": 100.0, "exchange_kwargs": {}},
    }


def _raw_config():
    return {
        "experiment": {
            "start_time": "2024-08-29",
            "end_time": "2026-08-28",
        },
        "portfolio": {"holdings": 25, "weekly_replacements": 3},
        "simulation": {"runs": 10, "workers": 2, "chunk_size": 2},
    }


def test_trial_seeds_are_stable_and_run_specific():
    assert derive_trial_seeds(20260829, 0) == derive_trial_seeds(20260829, 0)
    assert derive_trial_seeds(20260829, 0) != derive_trial_seeds(20260829, 1)


def test_config_rejects_more_replacements_than_holdings():
    raw = _raw_config()
    raw["portfolio"]["weekly_replacements"] = 26
    with pytest.raises(ValueError, match="weekly_replacements"):
        MonkeyExperimentConfig.from_mapping(raw, strategy_config=_strategy_config())


def test_config_defaults_to_deterministic_first_trade_day_execution():
    config = MonkeyExperimentConfig.from_mapping(
        _raw_config(), strategy_config=_strategy_config()
    )
    assert config.execution_day == "first_trade_day"
    assert config.python_hash_seed == 0


def test_first_trade_day_handles_weekends_and_holidays():
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-09-02", "2024-09-03", "2024-09-06"]))
    assert first_trade_day(calendar, "2024-08-31", "2024-09-06") == pd.Timestamp("2024-09-02")
    with pytest.raises(ValueError, match="no trading day"):
        first_trade_day(calendar, "2024-09-07", "2024-09-08")
    assert last_trade_day(calendar, "2024-08-31", "2024-09-08") == pd.Timestamp("2024-09-06")


def test_calculate_trial_metrics_uses_initial_cash_and_compounds_benchmark():
    dates = pd.to_datetime(["2024-08-30", "2024-09-06", "2024-09-13"])
    report = pd.DataFrame(
        {
            "account": [100.0, 110.0, 121.0],
            "bench": [0.0, 0.05, 0.05],
            "turnover": [0.5, 0.2, 0.2],
            "cost": [0.001, 0.002, 0.003],
        },
        index=dates,
    )
    metrics = calculate_trial_metrics(
        report,
        initial_cash=100.0,
        start_time="2024-08-29",
        end_time="2026-08-28",
    )
    assert metrics["total_return"] == pytest.approx(0.21)
    assert metrics["benchmark_total_return"] == pytest.approx(0.1025)
    assert metrics["excess_total_return"] == pytest.approx(0.1075)
    assert metrics["total_turnover"] == pytest.approx(0.9)
    assert metrics["total_cost"] == pytest.approx(0.006)


def test_recent_period_metrics_preserve_prior_state_and_align_benchmark():
    dates = pd.to_datetime(
        ["2024-08-30", "2025-08-22", "2025-08-29", "2026-08-28"]
    )
    report = pd.DataFrame(
        {
            "account": [100.0, 120.0, 126.0, 138.6],
            "bench": [0.01, 0.02, 0.03, 0.04],
            "turnover": [0.1, 0.2, 0.3, 0.4],
            "cost": [1.0, 2.0, 3.0, 4.0],
        },
        index=dates,
    )
    metrics = calculate_recent_period_metrics(report, initial_cash=100.0)

    assert metrics["recent_start_date"] == "2025-08-29"
    assert metrics["recent_end_date"] == "2026-08-28"
    assert metrics["prior_return"] == pytest.approx(0.26)
    assert metrics["recent_return"] == pytest.approx(0.10)
    assert metrics["prior_benchmark_return"] == pytest.approx(
        (1.01 * 1.02 * 1.03) - 1
    )
    assert metrics["recent_benchmark_return"] == pytest.approx(0.04)
    assert metrics["recent_excess_return"] == pytest.approx(0.06)
    assert metrics["recent_total_turnover"] == pytest.approx(0.4)
    assert metrics["recent_total_cost"] == pytest.approx(4.0)


def test_alpha_metrics_recovers_weekly_intercept_and_beta():
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.00, -0.01, 0.02])
    noise = pd.Series([0.0001, -0.0002, 0.0002, -0.0001, 0.0001, -0.0001])
    portfolio = 0.001 + 1.2 * benchmark + noise
    account = 100.0 * (1.0 + portfolio).cumprod()
    report = pd.DataFrame({"account": account, "bench": benchmark})

    metrics = calculate_alpha_metrics(report, initial_cash=100.0)

    assert metrics["annualized_alpha"] == pytest.approx(0.052, abs=0.003)
    assert metrics["benchmark_beta"] == pytest.approx(1.2, abs=0.01)
    assert math.isfinite(metrics["alpha_t_stat"])
    assert 0 <= metrics["alpha_p_value"] <= 1
    assert math.isfinite(metrics["hac_alpha_t_stat"])
    assert 0 <= metrics["hac_alpha_p_value"] <= 1


def test_regime_metrics_use_ex_ante_signal_dates_and_compound_selected_weeks():
    dates = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    portfolio = pd.Series([0.01, 0.02, -0.01, -0.02, 0.005, 0.015], index=dates)
    report = pd.DataFrame(
        {
            "account": 100.0 * (1.0 + portfolio).cumprod(),
            "bench": [0.005, 0.01, -0.005, -0.01, 0.002, 0.008],
            "cost": [1.0] * 6,
            "turnover": [0.1] * 6,
        },
        index=dates,
    )
    labels = pd.DataFrame(
        {
            "trend_regime": ["bull", "bull", "bear", "bear", "sideways", "sideways"],
            "vol_regime": ["low_vol", "low_vol", "high_vol", "high_vol", "normal_vol", "normal_vol"],
        },
        index=dates,
    )
    metrics = calculate_regime_metrics(
        report, initial_cash=100.0, signal_dates=list(dates), regime_labels=labels
    )
    assert metrics["regime_labeled_weeks"] == 6
    assert metrics["bull_weeks"] == 2
    assert metrics["bull_return"] == pytest.approx((1.01 * 1.02) - 1)
    assert metrics["bear_cost"] == pytest.approx(2.0)
    assert metrics["high_vol_turnover"] == pytest.approx(0.2)


def test_market_state_confirmation_is_causal_and_requires_repetition():
    raw = pd.Series(["bull", "bull", "bear", "bull", "bear", "bear"])
    confirmed = confirm_states(raw, periods=2)
    assert confirmed.tolist() == ["sideways", "bull", "bull", "bull", "bull", "bear"]


def test_resume_archives_failures_and_retries_failed_trial_ids(tmp_path):
    results = tmp_path / "results.csv"
    attempts = tmp_path / "failed_attempts.csv"
    pd.DataFrame(
        [
            {"run_id": 0, "status": "ok", "total_return": 0.1},
            {"run_id": 1, "status": "error", "error": "temporary"},
        ]
    ).reindex(columns=RESULT_FIELDS).to_csv(results, index=False)
    _prepare_resume_checkpoint(results, attempts)
    assert _read_completed(results) == {0}
    assert pd.read_csv(results)["run_id"].tolist() == [0]
    assert pd.read_csv(attempts)["run_id"].tolist() == [1]


class _Position:
    def __init__(self, stocks):
        self.stocks = list(stocks)

    def get_stock_list(self):
        return self.stocks


class _Account:
    def get_hist_positions(self):
        return {
            pd.Timestamp("2024-08-30"): _Position(["A", "B", "C", "D"]),
            pd.Timestamp("2024-09-06"): _Position(["A", "D", "E", "F"]),
        }


class _Executor:
    trade_account = _Account()


def test_position_diagnostics_counts_set_changes():
    result = _position_diagnostics(_Executor())
    assert result["initial_holding_count"] == 4
    assert result["min_holding_count"] == 4
    assert result["max_holding_count"] == 4
    assert result["mean_sells"] == 2
    assert result["mean_buys"] == 2
    assert result["full_three_by_three_rate"] == 0


def test_summarize_results_probabilities_and_failed_rows():
    frame = pd.DataFrame(
        [
            {
                "status": "ok",
                "total_return": 0.2,
                "excess_total_return": 0.1,
                "final_nav": 120.0,
                **{name: 0.1 for name in (
                    "annualized_return", "annualized_volatility", "sharpe",
                    "max_drawdown", "calmar", "benchmark_total_return",
                    "benchmark_cagr", "excess_cagr", "full_three_by_three_rate",
                )},
            },
            {
                "status": "ok",
                "total_return": -0.1,
                "excess_total_return": -0.2,
                "final_nav": 90.0,
                **{name: 0.0 for name in (
                    "annualized_return", "annualized_volatility", "sharpe",
                    "max_drawdown", "calmar", "benchmark_total_return",
                    "benchmark_cagr", "excess_cagr", "full_three_by_three_rate",
                )},
            },
            {"status": "error", "total_return": math.nan},
        ]
    )
    summary = summarize_results(frame)
    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 1
    assert summary["positive_return_probability"] == pytest.approx(0.5)
    assert summary["beat_benchmark_probability"] == pytest.approx(0.5)


def test_model_comparison_uses_empirical_monkey_thresholds():
    monkey = pd.DataFrame(
        {
            "status": ["ok"] * 4,
            "total_return": [0.0, 0.1, 0.2, 0.3],
            "excess_total_return": [-0.1, 0.0, 0.1, 0.2],
            "annualized_alpha": [-0.02, 0.0, 0.02, 0.04],
            "information_ratio": [-0.2, 0.0, 0.2, 0.4],
        }
    )
    models = pd.DataFrame(
        [{"model_name": "demo", "total_return": 0.5, "annualized_alpha": 0.1, "alpha_p_value": 0.01}]
    )
    compared, summary = compare(monkey, models)
    assert summary["monkey_trials"] == 4
    assert compared.loc[0, "total_return_monkey_percentile"] == pytest.approx(1.0)
    assert bool(compared.loc[0, "strict_random_baseline_pass"])
