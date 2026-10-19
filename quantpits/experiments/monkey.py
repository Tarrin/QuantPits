"""Monte Carlo monkey portfolio experiment using Qlib's native backtester.

The experiment deliberately contains no predictive information.  Every trial
uses random scores and Qlib's ``TopkDropoutStrategy`` with random buy/sell
methods.  Trial seeds are derived from ``(master_seed, run_id)`` so results are
stable across worker counts and resume operations.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonkeyExperimentConfig:
    """Validated experiment settings passed to each worker."""

    name: str
    market: str
    benchmark: str
    start_time: str
    end_time: str
    initial_cash: float
    holdings: int
    weekly_replacements: int
    risk_degree: float
    runs: int
    master_seed: int
    workers: int
    chunk_size: int
    frequency: str
    execution_day: str
    python_hash_seed: int
    signal_lookback_days: int
    sample_curve_runs: int
    hac_lags: int
    regime_labels_path: str | None
    output_root: str
    strategy_config: dict[str, Any]

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        strategy_config: dict[str, Any],
    ) -> "MonkeyExperimentConfig":
        experiment = raw.get("experiment", {})
        portfolio = raw.get("portfolio", {})
        simulation = raw.get("simulation", {})
        analysis = raw.get("analysis", {})
        output = raw.get("output", {})
        config = cls(
            name=str(experiment.get("name", "monkey_csi300_weekly")),
            market=str(experiment.get("market", "csi300")),
            benchmark=str(experiment.get("benchmark", "SH000300")),
            start_time=str(experiment["start_time"]),
            end_time=str(experiment["end_time"]),
            initial_cash=float(portfolio.get("initial_cash", 100_000_000)),
            holdings=int(portfolio.get("holdings", 25)),
            weekly_replacements=int(portfolio.get("weekly_replacements", 3)),
            risk_degree=float(portfolio.get("risk_degree", 0.95)),
            runs=int(simulation.get("runs", 10_000)),
            master_seed=int(simulation.get("master_seed", 20260829)),
            workers=int(simulation.get("workers", 4)),
            chunk_size=int(simulation.get("chunk_size", 5)),
            frequency=str(simulation.get("frequency", "week")),
            execution_day=str(simulation.get("execution_day", "first_trade_day")),
            python_hash_seed=int(simulation.get("python_hash_seed", 0)),
            signal_lookback_days=int(simulation.get("signal_lookback_days", 31)),
            sample_curve_runs=int(simulation.get("sample_curve_runs", 20)),
            hac_lags=int(analysis.get("hac_lags", 4)),
            regime_labels_path=(
                str(analysis["regime_labels_path"])
                if analysis.get("regime_labels_path")
                else None
            ),
            output_root=str(output.get("root", "output/monkey_runs")),
            strategy_config=copy.deepcopy(strategy_config),
        )
        config.validate()
        return config

    def validate(self) -> None:
        start = pd.Timestamp(self.start_time)
        end = pd.Timestamp(self.end_time)
        if start >= end:
            raise ValueError("start_time must be earlier than end_time")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.holdings <= 0:
            raise ValueError("holdings must be positive")
        if not 0 <= self.weekly_replacements <= self.holdings:
            raise ValueError("weekly_replacements must be between 0 and holdings")
        if not 0 < self.risk_degree <= 1:
            raise ValueError("risk_degree must be in (0, 1]")
        if self.runs <= 0 or self.workers <= 0 or self.chunk_size <= 0:
            raise ValueError("runs, workers and chunk_size must be positive")
        if self.frequency != "week":
            raise ValueError("this experiment currently requires frequency=week")
        if self.execution_day != "first_trade_day":
            raise ValueError("execution_day must be first_trade_day")
        if self.python_hash_seed < 0:
            raise ValueError("python_hash_seed must be non-negative")
        if self.signal_lookback_days < 7:
            raise ValueError("signal_lookback_days must be at least 7")
        if self.hac_lags < 0:
            raise ValueError("hac_lags must be non-negative")

    def worker_payload(self) -> dict[str, Any]:
        return asdict(self)

    def semantic_payload(self) -> dict[str, Any]:
        """Return the state that must match when resuming a run."""

        payload = asdict(self)
        for key in ("workers", "chunk_size", "sample_curve_runs", "output_root"):
            payload.pop(key, None)
        return payload

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.semantic_payload(), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def derive_trial_seeds(master_seed: int, run_id: int) -> tuple[int, int]:
    """Derive independent signal/strategy seeds without scheduling dependence."""

    state = np.random.SeedSequence([int(master_seed), int(run_id)]).generate_state(
        2, dtype=np.uint32
    )
    return int(state[0]), int(state[1])


def chunked(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _position_diagnostics(executor_obj: Any) -> dict[str, Any]:
    positions = executor_obj.trade_account.get_hist_positions()
    snapshots: list[tuple[pd.Timestamp, set[str]]] = []
    for timestamp, position in sorted(positions.items(), key=lambda item: item[0]):
        stocks = set(position.get_stock_list())
        if stocks:
            snapshots.append((pd.Timestamp(timestamp), stocks))

    if not snapshots:
        return {
            "initial_holdings": "",
            "initial_holding_count": 0,
            "min_holding_count": 0,
            "max_holding_count": 0,
            "rebalance_periods": 0,
            "full_three_by_three_periods": 0,
            "full_three_by_three_rate": math.nan,
            "mean_sells": math.nan,
            "mean_buys": math.nan,
        }

    counts = [len(stocks) for _, stocks in snapshots]
    sold_counts: list[int] = []
    bought_counts: list[int] = []
    for (_, previous), (_, current) in zip(snapshots, snapshots[1:]):
        sold_counts.append(len(previous - current))
        bought_counts.append(len(current - previous))

    full = sum(
        sold == 3 and bought == 3
        for sold, bought in zip(sold_counts, bought_counts)
    )
    periods = len(sold_counts)
    return {
        "initial_holdings": "|".join(sorted(snapshots[0][1])),
        "initial_holding_count": counts[0],
        "min_holding_count": min(counts),
        "max_holding_count": max(counts),
        "rebalance_periods": periods,
        "full_three_by_three_periods": full,
        "full_three_by_three_rate": full / periods if periods else math.nan,
        "mean_sells": float(np.mean(sold_counts)) if sold_counts else math.nan,
        "mean_buys": float(np.mean(bought_counts)) if bought_counts else math.nan,
    }


def calculate_trial_metrics(
    report: pd.DataFrame,
    *,
    initial_cash: float,
    start_time: str,
    end_time: str,
) -> dict[str, float]:
    """Calculate scale-independent portfolio and benchmark statistics."""

    if report is None or report.empty:
        raise ValueError("Qlib returned an empty portfolio report")

    nav = pd.to_numeric(report["account"], errors="coerce").dropna()
    if nav.empty or nav.iloc[-1] <= 0:
        raise ValueError("portfolio NAV is empty or non-positive")

    first_return = nav.iloc[0] / float(initial_cash) - 1.0
    period_returns = nav.pct_change().dropna()
    period_returns = pd.concat(
        [pd.Series([first_return], index=[nav.index[0]]), period_returns]
    )
    elapsed_years = max(
        (pd.Timestamp(end_time) - pd.Timestamp(start_time)).days / 365.2425,
        1.0 / 52.0,
    )
    final_nav = float(nav.iloc[-1])
    total_return = final_nav / float(initial_cash) - 1.0
    annualized_return = (final_nav / float(initial_cash)) ** (1.0 / elapsed_years) - 1.0

    volatility = float(period_returns.std(ddof=1) * math.sqrt(52))
    if period_returns.std(ddof=1) > 0:
        sharpe = float(period_returns.mean() / period_returns.std(ddof=1) * math.sqrt(52))
    else:
        sharpe = math.nan

    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else math.nan

    if "bench" in report:
        benchmark_period = pd.to_numeric(report["bench"], errors="coerce").fillna(0.0)
        benchmark_total_return = float((1.0 + benchmark_period).prod() - 1.0)
    else:
        benchmark_total_return = math.nan
    benchmark_cagr = (
        (1.0 + benchmark_total_return) ** (1.0 / elapsed_years) - 1.0
        if pd.notna(benchmark_total_return) and benchmark_total_return > -1
        else math.nan
    )

    def column_sum(name: str) -> float:
        if name not in report.columns:
            return 0.0
        values = pd.to_numeric(report[name], errors="coerce").fillna(0.0)
        return float(values.sum())

    return {
        "final_nav": final_nav,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "benchmark_total_return": benchmark_total_return,
        "benchmark_cagr": benchmark_cagr,
        "excess_total_return": total_return - benchmark_total_return,
        "excess_cagr": annualized_return - benchmark_cagr,
        "total_turnover": column_sum("turnover"),
        "total_cost": column_sum("cost"),
    }


def _alpha_from_returns(
    portfolio_return: pd.Series,
    benchmark_return: pd.Series,
    *,
    hac_lags: int,
) -> dict[str, float]:
    """Estimate OLS and Newey-West/HAC inference from aligned returns."""

    empty = {
        "annualized_alpha": math.nan,
        "benchmark_beta": math.nan,
        "alpha_t_stat": math.nan,
        "alpha_p_value": math.nan,
        "hac_alpha_t_stat": math.nan,
        "hac_alpha_p_value": math.nan,
        "information_ratio": math.nan,
    }
    aligned = pd.concat(
        [portfolio_return.rename("portfolio"), benchmark_return.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned) < 3:
        return empty

    x = aligned["benchmark"].to_numpy(dtype=float)
    y = aligned["portfolio"].to_numpy(dtype=float)
    x_centered = x - x.mean()
    sxx = float(np.dot(x_centered, x_centered))
    beta = float(np.dot(x_centered, y - y.mean()) / sxx) if sxx > 0 else math.nan
    weekly_alpha = float(y.mean() - beta * x.mean()) if math.isfinite(beta) else math.nan

    alpha_t = math.nan
    alpha_p = math.nan
    hac_alpha_t = math.nan
    hac_alpha_p = math.nan
    if math.isfinite(beta) and sxx > 0:
        residual = y - (weekly_alpha + beta * x)
        residual_variance = float(np.dot(residual, residual) / (len(aligned) - 2))
        alpha_se = math.sqrt(
            residual_variance * (1.0 / len(aligned) + x.mean() ** 2 / sxx)
        )
        if alpha_se > 0:
            alpha_t = weekly_alpha / alpha_se
            try:
                from scipy.stats import t as student_t

                alpha_p = float(2.0 * student_t.sf(abs(alpha_t), df=len(aligned) - 2))
            except ImportError:
                alpha_p = math.nan
        try:
            import statsmodels.api as sm

            hac_fit = sm.OLS(y, sm.add_constant(x)).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": min(int(hac_lags), len(aligned) - 1)},
            )
            hac_alpha_t = float(hac_fit.tvalues[0])
            hac_alpha_p = float(hac_fit.pvalues[0])
        except (ImportError, ValueError, np.linalg.LinAlgError):
            hac_alpha_t = math.nan
            hac_alpha_p = math.nan

    active = aligned["portfolio"] - aligned["benchmark"]
    tracking_error = float(active.std(ddof=1))
    information_ratio = (
        float(active.mean() / tracking_error * math.sqrt(52))
        if tracking_error > 0
        else math.nan
    )
    return {
        "annualized_alpha": weekly_alpha * 52 if math.isfinite(weekly_alpha) else math.nan,
        "benchmark_beta": beta,
        "alpha_t_stat": alpha_t,
        "alpha_p_value": alpha_p,
        "hac_alpha_t_stat": hac_alpha_t,
        "hac_alpha_p_value": hac_alpha_p,
        "information_ratio": information_ratio,
    }


def calculate_alpha_metrics(
    report: pd.DataFrame,
    *,
    initial_cash: float,
    hac_lags: int = 4,
) -> dict[str, float]:
    """Estimate weekly CAPM-style alpha and active-return statistics.

    The regression is ``portfolio_return = alpha + beta * benchmark_return``.
    Alpha is annualized arithmetically with 52 weekly periods, matching the
    project's weekly volatility and Sharpe convention.
    """

    if report is None or report.empty or "bench" not in report:
        return _alpha_from_returns(pd.Series(dtype=float), pd.Series(dtype=float), hac_lags=hac_lags)

    nav = pd.to_numeric(report["account"], errors="coerce")
    portfolio_return = nav.pct_change()
    if len(nav):
        portfolio_return.iloc[0] = nav.iloc[0] / float(initial_cash) - 1.0
    benchmark_return = pd.to_numeric(report["bench"], errors="coerce")
    return _alpha_from_returns(
        portfolio_return,
        benchmark_return,
        hac_lags=hac_lags,
    )


REGIME_NAMES = ("bull", "bear", "sideways", "high_vol", "normal_vol", "low_vol")
REGIME_METRIC_SUFFIXES = (
    "weeks", "return", "benchmark_return", "excess_return", "annualized_alpha",
    "benchmark_beta", "hac_alpha_t_stat", "hac_alpha_p_value",
    "information_ratio", "cost", "turnover",
)


def calculate_regime_metrics(
    report: pd.DataFrame,
    *,
    initial_cash: float,
    signal_dates: Sequence[pd.Timestamp],
    regime_labels: pd.DataFrame,
    hac_lags: int = 4,
) -> dict[str, float | int]:
    """Attribute weekly trial returns to ex-ante trend and volatility states."""

    output: dict[str, float | int] = {
        f"{regime}_{suffix}": (0 if suffix == "weeks" else math.nan)
        for regime in REGIME_NAMES
        for suffix in REGIME_METRIC_SUFFIXES
    }
    if report.empty or len(report) != len(signal_dates):
        output["regime_labeled_weeks"] = 0
        output["regime_unlabeled_weeks"] = int(len(report))
        return output

    nav = pd.to_numeric(report["account"], errors="coerce")
    portfolio_return = nav.pct_change()
    portfolio_return.iloc[0] = nav.iloc[0] / float(initial_cash) - 1.0
    benchmark_return = pd.to_numeric(report.get("bench"), errors="coerce")
    signal_index = pd.DatetimeIndex(pd.to_datetime(signal_dates)).normalize()
    labels = regime_labels.sort_index().reindex(signal_index, method="ffill")
    labels.index = report.index
    labeled = labels["trend_regime"].notna() & labels["vol_regime"].notna()
    output["regime_labeled_weeks"] = int(labeled.sum())
    output["regime_unlabeled_weeks"] = int((~labeled).sum())

    for regime in REGIME_NAMES:
        column = "trend_regime" if regime in {"bull", "bear", "sideways"} else "vol_regime"
        mask = labels[column].eq(regime) & labeled
        count = int(mask.sum())
        output[f"{regime}_weeks"] = count
        if count == 0:
            continue
        portfolio_slice = portfolio_return[mask]
        benchmark_slice = benchmark_return[mask]
        portfolio_total = float((1.0 + portfolio_slice).prod() - 1.0)
        benchmark_total = float((1.0 + benchmark_slice).prod() - 1.0)
        output[f"{regime}_return"] = portfolio_total
        output[f"{regime}_benchmark_return"] = benchmark_total
        output[f"{regime}_excess_return"] = portfolio_total - benchmark_total
        alpha = _alpha_from_returns(
            portfolio_slice,
            benchmark_slice,
            hac_lags=hac_lags,
        )
        for key, value in alpha.items():
            if key in {"alpha_t_stat", "alpha_p_value"}:
                continue
            output[f"{regime}_{key}"] = value
        for source, suffix in (("cost", "cost"), ("turnover", "turnover")):
            values = pd.to_numeric(report[source], errors="coerce") if source in report else pd.Series(0.0, index=report.index)
            output[f"{regime}_{suffix}"] = float(values[mask].fillna(0.0).sum())
    return output


def calculate_recent_period_metrics(
    report: pd.DataFrame,
    *,
    initial_cash: float,
    years: int = 1,
) -> dict[str, Any]:
    """Split one complete path at the nearest weekly observation ``years`` ago.

    The recent return starts from the account value at the cutoff, so benchmark,
    turnover and cost deliberately start with the *following* report row.  This
    preserves the portfolio state accumulated during the preceding period.
    """

    if report is None or report.empty:
        raise ValueError("Qlib returned an empty portfolio report")
    if years <= 0:
        raise ValueError("years must be positive")

    ordered = report.copy()
    ordered.index = pd.DatetimeIndex(pd.to_datetime(ordered.index))
    ordered = ordered.sort_index()
    nav = pd.to_numeric(ordered["account"], errors="coerce")
    if nav.isna().any() or len(nav) < 2 or (nav <= 0).any():
        raise ValueError("portfolio NAV must contain at least two positive observations")

    target = nav.index[-1] - pd.DateOffset(years=years)
    cutoff_loc = int(nav.index.get_indexer([target], method="nearest")[0])
    if cutoff_loc < 0 or cutoff_loc >= len(nav) - 1:
        raise ValueError("the report does not contain a usable recent-period cutoff")

    cutoff_nav = float(nav.iloc[cutoff_loc])
    final_nav = float(nav.iloc[-1])
    recent_nav = nav.iloc[cutoff_loc:]
    recent_returns = recent_nav.pct_change().dropna()
    recent_return = final_nav / cutoff_nav - 1.0
    prior_return = cutoff_nav / float(initial_cash) - 1.0

    elapsed_years = max(
        (recent_nav.index[-1] - recent_nav.index[0]).days / 365.2425,
        1.0 / 52.0,
    )
    recent_annualized_return = (1.0 + recent_return) ** (1.0 / elapsed_years) - 1.0
    weekly_std = float(recent_returns.std(ddof=1))
    recent_volatility = weekly_std * math.sqrt(52)
    recent_sharpe = (
        float(recent_returns.mean() / weekly_std * math.sqrt(52))
        if weekly_std > 0
        else math.nan
    )
    recent_drawdown = recent_nav / recent_nav.cummax() - 1.0
    recent_max_drawdown = float(recent_drawdown.min())
    recent_calmar = (
        recent_annualized_return / abs(recent_max_drawdown)
        if recent_max_drawdown < 0
        else math.nan
    )

    def compounded(column: str, values: slice) -> float:
        if column not in ordered.columns:
            return math.nan
        periods = pd.to_numeric(ordered[column].iloc[values], errors="coerce").fillna(0.0)
        return float((1.0 + periods).prod() - 1.0)

    def summed(column: str, values: slice) -> float:
        if column not in ordered.columns:
            return 0.0
        periods = pd.to_numeric(ordered[column].iloc[values], errors="coerce").fillna(0.0)
        return float(periods.sum())

    prior_slice = slice(None, cutoff_loc + 1)
    recent_slice = slice(cutoff_loc + 1, None)
    recent_benchmark_return = compounded("bench", recent_slice)
    prior_benchmark_return = compounded("bench", prior_slice)

    return {
        "recent_start_date": recent_nav.index[0].date().isoformat(),
        "recent_end_date": recent_nav.index[-1].date().isoformat(),
        "recent_start_nav": cutoff_nav,
        "recent_final_nav": final_nav,
        "recent_return": recent_return,
        "recent_annualized_return": recent_annualized_return,
        "recent_benchmark_return": recent_benchmark_return,
        "recent_excess_return": recent_return - recent_benchmark_return,
        "recent_annualized_volatility": recent_volatility,
        "recent_sharpe": recent_sharpe,
        "recent_max_drawdown": recent_max_drawdown,
        "recent_calmar": recent_calmar,
        "recent_total_turnover": summed("turnover", recent_slice),
        "recent_total_cost": summed("cost", recent_slice),
        "prior_return": prior_return,
        "prior_benchmark_return": prior_benchmark_return,
        "prior_excess_return": prior_return - prior_benchmark_return,
    }


_WORKER_CONFIG: MonkeyExperimentConfig | None = None
_WORKER_SIGNAL_INDEX: pd.MultiIndex | None = None
_WORKER_EXCHANGE: Any = None
_WORKER_DAILY_CALENDAR: pd.DatetimeIndex | None = None
_WORKER_REGIME_LABELS: pd.DataFrame | None = None


def first_trade_day(
    calendar: pd.DatetimeIndex,
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
) -> pd.Timestamp:
    """Return the first actual trading day inside a closed weekly interval."""

    start = pd.Timestamp(start_time).normalize()
    end = pd.Timestamp(end_time).normalize()
    left = int(calendar.searchsorted(start, side="left"))
    if left >= len(calendar) or pd.Timestamp(calendar[left]).normalize() > end:
        raise ValueError(f"no trading day between {start.date()} and {end.date()}")
    return pd.Timestamp(calendar[left]).normalize()


def last_trade_day(
    calendar: pd.DatetimeIndex,
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
) -> pd.Timestamp:
    """Return the last actual trading day inside a closed weekly interval."""

    start = pd.Timestamp(start_time).normalize()
    end = pd.Timestamp(end_time).normalize()
    right = int(calendar.searchsorted(end, side="right")) - 1
    if right < 0 or pd.Timestamp(calendar[right]).normalize() < start:
        raise ValueError(f"no trading day between {start.date()} and {end.date()}")
    return pd.Timestamp(calendar[right]).normalize()


@contextmanager
def _disable_qlib_progress_bar():
    """Disable Qlib's tqdm without redirecting Windows console handles.

    Replacing file descriptor 2 with ``os.dup2`` is unsafe for the existing
    Windows console ``TextIOWrapper``.  tqdm.auto may select tqdm_asyncio and
    then fail during its first refresh with WinError 1.  Supplying
    ``disable=True`` keeps the progress object well-formed and avoids all I/O.
    """

    import importlib

    backtest_module = importlib.import_module("qlib.backtest.backtest")
    original_tqdm = backtest_module.tqdm

    def quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return original_tqdm(*args, **kwargs)

    backtest_module.tqdm = quiet_tqdm
    try:
        yield
    finally:
        backtest_module.tqdm = original_tqdm


def initialize_worker(payload: dict[str, Any]) -> None:
    """Initialize Qlib and immutable market data once per spawned process."""

    global _WORKER_CONFIG, _WORKER_SIGNAL_INDEX, _WORKER_EXCHANGE, _WORKER_DAILY_CALENDAR
    global _WORKER_REGIME_LABELS

    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"

    _WORKER_CONFIG = MonkeyExperimentConfig(**payload)

    from quantpits.utils import env

    env.init_qlib()
    from qlib.backtest.exchange import Exchange
    from qlib.data import D

    signal_start = (
        pd.Timestamp(_WORKER_CONFIG.start_time)
        - timedelta(days=_WORKER_CONFIG.signal_lookback_days)
    ).strftime("%Y-%m-%d")
    features = D.features(
        D.instruments(market=_WORKER_CONFIG.market),
        ["$close"],
        start_time=signal_start,
        end_time=_WORKER_CONFIG.end_time,
        freq="day",
    )
    close = pd.to_numeric(features.iloc[:, 0], errors="coerce")
    eligible = close[close.notna() & np.isfinite(close) & (close > 0)]
    if eligible.empty:
        raise RuntimeError("no eligible CSI300 observations were loaded from Qlib")
    _WORKER_SIGNAL_INDEX = eligible.sort_index().index
    _WORKER_DAILY_CALENDAR = pd.DatetimeIndex(
        pd.to_datetime(
            D.calendar(
                start_time=signal_start,
                end_time=_WORKER_CONFIG.end_time,
                freq="day",
            )
        )
    ).sort_values()
    if _WORKER_DAILY_CALENDAR.empty:
        raise RuntimeError("no daily trading calendar was loaded from Qlib")

    _WORKER_REGIME_LABELS = None
    if _WORKER_CONFIG.regime_labels_path:
        labels_path = Path(_WORKER_CONFIG.regime_labels_path)
        if not labels_path.is_absolute():
            labels_path = Path(env.ROOT_DIR) / labels_path
        labels = pd.read_csv(labels_path, parse_dates=["signal_date"])
        required = {"signal_date", "trend_regime", "vol_regime"}
        missing = required.difference(labels.columns)
        if missing:
            raise ValueError(f"regime labels are missing columns: {sorted(missing)}")
        labels = labels.set_index("signal_date").sort_index()
        labels.index = pd.DatetimeIndex(labels.index).normalize()
        _WORKER_REGIME_LABELS = labels

    all_codes = sorted(_WORKER_SIGNAL_INDEX.get_level_values("instrument").unique())
    backtest_config = _WORKER_CONFIG.strategy_config["backtest"]
    exchange_kwargs = copy.deepcopy(backtest_config["exchange_kwargs"])
    exchange_freq = exchange_kwargs.pop("freq", "day")
    _WORKER_EXCHANGE = Exchange(
        freq=exchange_freq,
        start_time=_WORKER_CONFIG.start_time,
        end_time=_WORKER_CONFIG.end_time,
        codes=all_codes,
        **exchange_kwargs,
    )


def _run_trial(run_id: int) -> dict[str, Any]:
    if (
        _WORKER_CONFIG is None
        or _WORKER_SIGNAL_INDEX is None
        or _WORKER_EXCHANGE is None
        or _WORKER_DAILY_CALENDAR is None
    ):
        raise RuntimeError("worker was not initialized")

    config = _WORKER_CONFIG
    signal_seed, strategy_seed = derive_trial_seeds(config.master_seed, run_id)
    try:
        from qlib.backtest import backtest_loop
        from qlib.backtest.account import Account
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.backtest.utils import CommonInfrastructure
        from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
        from qlib.contrib.strategy import TopkDropoutStrategy
        from quantpits.utils.backtest_utils import extract_report_df

        class DeterministicExecutionTopkDropoutStrategy(TopkDropoutStrategy):
            """TopkDropout with stable ordering and one explicit execution day.

            Qlib's weekly implementation checks tradability over the entire
            weekly interval and prices at its last valid observation.  This
            variant uses the first actual trading day's close for every order,
            so suspension and price-limit checks refer to that same day.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.execution_log: list[dict[str, Any]] = []

            def generate_trade_decision(self, execute_result=None):
                trade_step = self.trade_calendar.get_trade_step()
                trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
                pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
                execution_time = first_trade_day(
                    _WORKER_DAILY_CALENDAR,
                    trade_start_time,
                    trade_end_time,
                )
                valuation_time = last_trade_day(
                    _WORKER_DAILY_CALENDAR,
                    trade_start_time,
                    min(pd.Timestamp(trade_end_time), pd.Timestamp(config.end_time)),
                )
                signal_time = last_trade_day(
                    _WORKER_DAILY_CALENDAR,
                    pred_start_time,
                    min(
                        pd.Timestamp(pred_end_time),
                        execution_time - pd.Timedelta(days=1),
                    ),
                )
                pred_score = self.signal.get_signal(
                    start_time=pred_start_time,
                    end_time=pred_end_time,
                )
                if isinstance(pred_score, pd.DataFrame):
                    pred_score = pred_score.iloc[:, 0]
                if pred_score is None:
                    return TradeDecisionWO([], self)

                # Stable tie-breaking makes the strategy deterministic even
                # when an upstream object was constructed from a set.
                pred_score = pred_score.sort_index(kind="mergesort")

                def is_tradable(stock_id: str, direction=None) -> bool:
                    return self.trade_exchange.is_stock_tradable(
                        stock_id=str(stock_id),
                        start_time=execution_time,
                        end_time=execution_time,
                        direction=direction,
                    )

                if self.only_tradable:
                    def get_first_n(values, n, reverse=False):
                        selected = []
                        iterator = reversed(values) if reverse else values
                        for stock_id in iterator:
                            if is_tradable(stock_id):
                                selected.append(str(stock_id))
                                if len(selected) >= n:
                                    break
                        return selected[::-1] if reverse else selected

                    def get_last_n(values, n):
                        return get_first_n(values, n, reverse=True)

                    def filter_stock(values):
                        return [str(stock_id) for stock_id in values if is_tradable(stock_id)]
                else:
                    def get_first_n(values, n, reverse=False):
                        selected = [str(stock_id) for stock_id in values]
                        if reverse:
                            return selected[-n:] if n else []
                        return selected[:n]

                    def get_last_n(values, n):
                        return get_first_n(values, n, reverse=True)

                    def filter_stock(values):
                        return [str(stock_id) for stock_id in values]

                current_temp = copy.deepcopy(self.trade_position)
                cash = current_temp.get_cash()
                current_stock_list = sorted(str(code) for code in current_temp.get_stock_list())
                last = (
                    pred_score.reindex(current_stock_list)
                    .sort_values(ascending=False, kind="mergesort")
                    .index
                )

                if self.method_buy == "top":
                    today = get_first_n(
                        pred_score[~pred_score.index.isin(last)]
                        .sort_values(ascending=False, kind="mergesort")
                        .index,
                        self.n_drop + self.topk - len(last),
                    )
                elif self.method_buy == "random":
                    topk_candidates = get_first_n(
                        pred_score.sort_values(ascending=False, kind="mergesort").index,
                        self.topk,
                    )
                    candidates = [code for code in topk_candidates if code not in last]
                    count = self.n_drop + self.topk - len(last)
                    try:
                        today = [
                            str(code)
                            for code in np.random.choice(candidates, count, replace=False)
                        ]
                    except ValueError:
                        today = candidates
                else:
                    raise NotImplementedError(f"unsupported buy method: {self.method_buy}")

                combined = pred_score.reindex(
                    last.union(pd.Index(today), sort=False)
                ).sort_values(ascending=False, kind="mergesort").index
                if self.method_sell == "bottom":
                    sell = last[last.isin(get_last_n(combined, self.n_drop))]
                elif self.method_sell == "random":
                    candidates = filter_stock(last)
                    try:
                        sell = pd.Index(
                            [
                                str(code)
                                for code in (
                                    np.random.choice(candidates, self.n_drop, replace=False)
                                    if len(last)
                                    else []
                                )
                            ]
                        )
                    except ValueError:
                        sell = pd.Index(candidates)
                else:
                    raise NotImplementedError(f"unsupported sell method: {self.method_sell}")

                buy = [str(code) for code in today[: len(sell) + self.topk - len(last)]]
                sell_orders = []
                buy_orders = []
                for code in current_stock_list:
                    direction = None if self.forbid_all_trade_at_limit else OrderDir.SELL
                    if not is_tradable(code, direction=direction):
                        continue
                    if code in sell:
                        time_per_step = self.trade_calendar.get_freq()
                        if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                            continue
                        sell_order = Order(
                            stock_id=code,
                            amount=current_temp.get_stock_amount(code=code),
                            start_time=execution_time,
                            end_time=execution_time,
                            direction=Order.SELL,
                        )
                        if self.trade_exchange.check_order(sell_order):
                            sell_orders.append(sell_order)
                            trade_val, trade_cost, _ = self.trade_exchange.deal_order(
                                sell_order,
                                position=current_temp,
                            )
                            cash += trade_val - trade_cost

                value = cash * self.risk_degree / len(buy) if buy else 0.0
                for code in buy:
                    direction = None if self.forbid_all_trade_at_limit else OrderDir.BUY
                    if not is_tradable(code, direction=direction):
                        continue
                    buy_price = self.trade_exchange.get_deal_price(
                        stock_id=code,
                        start_time=execution_time,
                        end_time=execution_time,
                        direction=OrderDir.BUY,
                    )
                    buy_amount = value / buy_price
                    factor = self.trade_exchange.get_factor(
                        stock_id=code,
                        start_time=execution_time,
                        end_time=execution_time,
                    )
                    buy_amount = self.trade_exchange.round_amount_by_trade_unit(
                        buy_amount,
                        factor,
                    )
                    buy_orders.append(
                        Order(
                            stock_id=code,
                            amount=buy_amount,
                            start_time=execution_time,
                            end_time=execution_time,
                            direction=Order.BUY,
                        )
                    )

                orders = sell_orders + buy_orders
                self.execution_log.append(
                    {
                        "trade_start": pd.Timestamp(trade_start_time),
                        "trade_end": pd.Timestamp(trade_end_time),
                        "execution_time": execution_time,
                        "signal_time": signal_time,
                        "valuation_time": valuation_time,
                        "order_count": len(orders),
                        "timestamp_mismatches": sum(
                            order.start_time != execution_time or order.end_time != execution_time
                            for order in orders
                        ),
                    }
                )
                return TradeDecisionWO(orders, self)

        rng = np.random.default_rng(signal_seed)
        signal = pd.Series(
            rng.random(len(_WORKER_SIGNAL_INDEX)),
            index=_WORKER_SIGNAL_INDEX,
            name="score",
        )

        strategy_config = copy.deepcopy(config.strategy_config)
        params = strategy_config["strategy"]["params"]
        params.update(
            {
                "topk": config.holdings,
                "n_drop": config.weekly_replacements,
                "method_sell": "random",
                "method_buy": "random",
                "risk_degree": config.risk_degree,
                "only_tradable": True,
            }
        )
        params.pop("buy_suggestion_factor", None)
        params.pop("sell_out_of_universe", None)
        strategy_inst = DeterministicExecutionTopkDropoutStrategy(signal=signal, **params)

        trade_account = Account(
            init_cash=config.initial_cash,
            freq=config.frequency,
            benchmark_config={
                "benchmark": config.benchmark,
                "start_time": config.start_time,
                "end_time": config.end_time,
            },
        )
        common_infra = CommonInfrastructure(
            trade_account=trade_account,
            trade_exchange=_WORKER_EXCHANGE,
        )
        strategy_inst.reset_common_infra(common_infra)
        executor_obj = SimulatorExecutor(
            time_per_step=config.frequency,
            generate_portfolio_metrics=True,
            verbose=False,
            common_infra=common_infra,
        )

        np.random.seed(strategy_seed)
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            with _disable_qlib_progress_bar():
                raw_metrics, _ = backtest_loop(
                    start_time=config.start_time,
                    end_time=config.end_time,
                    trade_strategy=strategy_inst,
                    trade_executor=executor_obj,
                )
        report = extract_report_df(raw_metrics)
        execution_log = strategy_inst.execution_log
        if len(execution_log) == len(report):
            # Qlib labels weekly bars by their interval start.  The account is
            # actually marked at the final available daily close, so expose
            # that date to all downstream cutoff and curve calculations.
            report = report.copy()
            report.index = pd.DatetimeIndex(
                [item["valuation_time"] for item in execution_log]
            )
        metrics = calculate_trial_metrics(
            report,
            initial_cash=config.initial_cash,
            start_time=config.start_time,
            end_time=config.end_time,
        )
        recent_metrics = calculate_recent_period_metrics(
            report,
            initial_cash=config.initial_cash,
        )
        alpha_metrics = calculate_alpha_metrics(
            report,
            initial_cash=config.initial_cash,
            hac_lags=config.hac_lags,
        )
        regime_metrics: dict[str, Any] = {}
        if _WORKER_REGIME_LABELS is not None and len(execution_log) == len(report):
            regime_metrics = calculate_regime_metrics(
                report,
                initial_cash=config.initial_cash,
                signal_dates=[item["signal_time"] for item in execution_log],
                regime_labels=_WORKER_REGIME_LABELS,
                hac_lags=config.hac_lags,
            )
        diagnostics = _position_diagnostics(executor_obj)
        execution_diagnostics = {
            "execution_day_rule": config.execution_day,
            "execution_periods": len(execution_log),
            "first_execution_date": (
                execution_log[0]["execution_time"].date().isoformat()
                if execution_log
                else ""
            ),
            "last_execution_date": (
                execution_log[-1]["execution_time"].date().isoformat()
                if execution_log
                else ""
            ),
            "first_valuation_date": (
                execution_log[0]["valuation_time"].date().isoformat()
                if execution_log
                else ""
            ),
            "last_valuation_date": (
                execution_log[-1]["valuation_time"].date().isoformat()
                if execution_log
                else ""
            ),
            "order_timestamp_mismatches": int(
                sum(item["timestamp_mismatches"] for item in execution_log)
            ),
        }
        summary: dict[str, Any] = {
            "run_id": run_id,
            "status": "ok",
            "signal_seed": signal_seed,
            "strategy_seed": strategy_seed,
            **metrics,
            **recent_metrics,
            **alpha_metrics,
            **regime_metrics,
            **diagnostics,
            **execution_diagnostics,
            "error": "",
        }
        curve = None
        if run_id < config.sample_curve_runs:
            valuation_dates = [item["valuation_time"] for item in execution_log]
            execution_dates = [item["execution_time"] for item in execution_log]
            signal_dates = [item["signal_time"] for item in execution_log]
            if len(valuation_dates) != len(report):
                valuation_dates = [pd.Timestamp(value) for value in report.index]
                execution_dates = [pd.NaT] * len(report)
                signal_dates = [pd.NaT] * len(report)
            curve_labels = None
            if _WORKER_REGIME_LABELS is not None and all(pd.notna(value) for value in signal_dates):
                curve_labels = _WORKER_REGIME_LABELS.sort_index().reindex(
                    pd.DatetimeIndex(signal_dates).normalize(), method="ffill"
                )
            curve = [
                {
                    "run_id": run_id,
                    "datetime": pd.Timestamp(valuation_date).isoformat(),
                    "execution_datetime": (
                        pd.Timestamp(execution_date).isoformat()
                        if pd.notna(execution_date)
                        else ""
                    ),
                    "signal_datetime": (
                        pd.Timestamp(signal_date).isoformat()
                        if pd.notna(signal_date)
                        else ""
                    ),
                    "trend_regime": (
                        str(curve_labels.iloc[index]["trend_regime"])
                        if curve_labels is not None
                        else ""
                    ),
                    "vol_regime": (
                        str(curve_labels.iloc[index]["vol_regime"])
                        if curve_labels is not None
                        else ""
                    ),
                    "nav": float(value),
                }
                for index, (valuation_date, execution_date, signal_date, value) in enumerate(zip(
                    valuation_dates,
                    execution_dates,
                    signal_dates,
                    report["account"].to_numpy(),
                ))
            ]
        return {"summary": summary, "curve": curve}
    except Exception as exc:  # worker errors must be checkpointed, not kill the pool
        return {
            "summary": {
                "run_id": run_id,
                "status": "error",
                "signal_seed": signal_seed,
                "strategy_seed": strategy_seed,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            },
            "curve": None,
        }


def run_trial_chunk(run_ids: Sequence[int]) -> list[dict[str, Any]]:
    return [_run_trial(run_id) for run_id in run_ids]


def execute_trials(
    config: MonkeyExperimentConfig,
    run_ids: Sequence[int],
    *,
    on_batch: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Execute trial IDs with Windows-safe spawned worker processes."""

    if not run_ids:
        return []
    # Spawned workers read this value at interpreter startup.  The strategy
    # also sorts every stock collection, so workers=1 remains deterministic
    # even though changing this variable cannot alter the current interpreter.
    os.environ["PYTHONHASHSEED"] = str(config.python_hash_seed)
    payload = config.worker_payload()
    results: list[dict[str, Any]] = []
    batches = list(chunked(list(run_ids), config.chunk_size))

    if config.workers == 1:
        initialize_worker(payload)
        for batch in batches:
            batch_results = run_trial_chunk(batch)
            results.extend(batch_results)
            if on_batch is not None:
                on_batch(batch_results)
        return results

    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(payload,),
    ) as pool:
        futures = [pool.submit(run_trial_chunk, batch) for batch in batches]
        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)
            if on_batch is not None:
                on_batch(batch_results)
    return results


SUMMARY_METRICS = (
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "calmar",
    "benchmark_total_return",
    "benchmark_cagr",
    "excess_total_return",
    "excess_cagr",
    "recent_return",
    "recent_annualized_return",
    "recent_benchmark_return",
    "recent_excess_return",
    "recent_annualized_volatility",
    "recent_sharpe",
    "recent_max_drawdown",
    "recent_calmar",
    "prior_return",
    "prior_benchmark_return",
    "prior_excess_return",
    "annualized_alpha",
    "benchmark_beta",
    "alpha_t_stat",
    "alpha_p_value",
    "hac_alpha_t_stat",
    "hac_alpha_p_value",
    "information_ratio",
    "full_three_by_three_rate",
)


def summarize_results(frame: pd.DataFrame) -> dict[str, Any]:
    successful = frame[frame["status"] == "ok"].copy()
    summary: dict[str, Any] = {
        "total_rows": int(len(frame)),
        "successful_runs": int(len(successful)),
        "failed_runs": int((frame["status"] != "ok").sum()),
        "metrics": {},
    }
    if successful.empty:
        return summary

    for metric in SUMMARY_METRICS:
        if metric not in successful.columns:
            continue
        values = pd.to_numeric(successful[metric], errors="coerce").dropna()
        if values.empty:
            continue
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "max": float(values.max()),
        }

    total_return = pd.to_numeric(successful["total_return"], errors="coerce")
    excess = pd.to_numeric(successful["excess_total_return"], errors="coerce")
    summary["positive_return_probability"] = float((total_return > 0).mean())
    summary["beat_benchmark_probability"] = float((excess > 0).mean())
    summary["median_final_nav"] = float(
        pd.to_numeric(successful["final_nav"], errors="coerce").median()
    )
    return summary


def write_analysis_artifacts(results_csv: Path, output_dir: Path) -> dict[str, Any]:
    """Write aggregate JSON, Markdown, and distribution plots."""

    frame = pd.read_csv(results_csv)
    summary = summarize_results(frame)
    aggregate_path = output_dir / "aggregate_summary.json"
    aggregate_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )

    successful = frame[frame["status"] == "ok"].copy()
    if not successful.empty:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(successful["total_return"] * 100, bins=50, color="#4472C4", alpha=0.85)
        axes[0].axvline(successful["total_return"].median() * 100, color="black", linestyle="--")
        axes[0].set_title("Monkey portfolio total return")
        axes[0].set_xlabel("Total return (%)")
        axes[0].set_ylabel("Trials")
        axes[1].hist(successful["excess_total_return"] * 100, bins=50, color="#ED7D31", alpha=0.85)
        axes[1].axvline(0, color="black", linewidth=1)
        axes[1].set_title("Excess return vs CSI300")
        axes[1].set_xlabel("Excess total return (percentage points)")
        fig.tight_layout()
        fig.savefig(output_dir / "return_distributions.png", dpi=160)
        plt.close(fig)

    total_stats = summary.get("metrics", {}).get("total_return", {})
    excess_stats = summary.get("metrics", {}).get("excess_total_return", {})
    markdown = [
        "# CSI300 Monkey Experiment",
        "",
        f"- Successful runs: {summary['successful_runs']}",
        f"- Failed runs: {summary['failed_runs']}",
    ]
    if total_stats:
        markdown.extend(
            [
                f"- Median total return: {total_stats['median']:.2%}",
                f"- Mean total return: {total_stats['mean']:.2%}",
                f"- Total return P05/P95: {total_stats['p05']:.2%} / {total_stats['p95']:.2%}",
                f"- Positive-return probability: {summary['positive_return_probability']:.2%}",
            ]
        )
    if excess_stats:
        markdown.extend(
            [
                f"- Median excess return: {excess_stats['median']:.2%}",
                f"- Beat-CSI300 probability: {summary['beat_benchmark_probability']:.2%}",
            ]
        )
    markdown.extend(
        [
            "",
            "Artifacts: `results.csv`, `aggregate_summary.json`, `return_distributions.png`, `run_metadata.json`.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary
