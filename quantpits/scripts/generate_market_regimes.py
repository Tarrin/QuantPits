#!/usr/bin/env python
"""Generate causal CSI300 trend/volatility labels for monkey experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def confirm_states(raw_states: pd.Series, periods: int) -> pd.Series:
    """Switch state only after ``periods`` consecutive raw observations."""

    if periods <= 0:
        raise ValueError("confirmation periods must be positive")
    confirmed: list[str] = []
    current = "sideways"
    candidate = ""
    count = 0
    for value in raw_states.astype(str):
        if value == current:
            candidate, count = "", 0
        elif value == candidate:
            count += 1
        else:
            candidate, count = value, 1
        if count >= periods:
            current, candidate, count = candidate, "", 0
        confirmed.append(current)
    return pd.Series(confirmed, index=raw_states.index, name="trend_regime")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    from quantpits.utils import env

    env.init_qlib()
    from qlib.data import D

    benchmark = str(cfg.get("benchmark", "SH000300"))
    history_start = str(cfg.get("history_start", "2008-01-01"))
    label_start = pd.Timestamp(cfg.get("label_start", "2020-01-01"))
    label_end = pd.Timestamp(cfg.get("label_end", "2026-08-28"))
    features = D.features(
        [benchmark], ["$close"], start_time=history_start,
        end_time=label_end.strftime("%Y-%m-%d"), freq="day",
    )
    close = pd.to_numeric(features.iloc[:, 0], errors="coerce").dropna()
    if isinstance(close.index, pd.MultiIndex):
        close.index = pd.DatetimeIndex(close.index.get_level_values("datetime"))
    close = close.sort_index()
    if close.empty:
        raise RuntimeError(f"Qlib returned no close data for {benchmark}")

    ma_days = int(cfg.get("ma_days", 200))
    return_days = int(cfg.get("return_lookback_days", 126))
    vol_days = int(cfg.get("volatility_lookback_days", 63))
    daily = pd.DataFrame({"close": close})
    daily[f"ma{ma_days}"] = close.rolling(ma_days, min_periods=ma_days).mean()
    daily[f"return_{return_days}d"] = close.pct_change(return_days, fill_method=None)
    daily[f"realized_vol_{vol_days}d"] = (
        close.pct_change(fill_method=None).rolling(vol_days, min_periods=vol_days).std(ddof=1)
        * np.sqrt(252)
    )

    # Weekly observation is the actual final trading day; all inputs are known
    # before the following week's first-trading-day execution.
    weekly = daily.groupby(daily.index.to_period("W-FRI"), sort=True).tail(1).copy()
    ma_col = f"ma{ma_days}"
    ret_col = f"return_{return_days}d"
    vol_col = f"realized_vol_{vol_days}d"
    bull_threshold = float(cfg.get("bull_return_threshold", 0.05))
    bear_threshold = float(cfg.get("bear_return_threshold", -0.05))
    weekly["raw_trend_regime"] = np.select(
        [
            (weekly["close"] > weekly[ma_col]) & (weekly[ret_col] > bull_threshold),
            (weekly["close"] < weekly[ma_col]) & (weekly[ret_col] < bear_threshold),
        ],
        ["bull", "bear"],
        default="sideways",
    )
    weekly["trend_regime"] = confirm_states(
        weekly["raw_trend_regime"], int(cfg.get("confirmation_weeks", 2))
    )

    reference_start = pd.Timestamp(cfg.get("vol_reference_start", "2010-01-01"))
    reference_end = pd.Timestamp(cfg.get("vol_reference_end", "2019-12-31"))
    reference = daily.loc[reference_start:reference_end, vol_col].dropna()
    if reference.empty:
        raise RuntimeError("volatility reference period has no usable observations")
    low_q = float(cfg.get("vol_low_quantile", 0.30))
    high_q = float(cfg.get("vol_high_quantile", 0.70))
    low_threshold = float(reference.quantile(low_q))
    high_threshold = float(reference.quantile(high_q))
    weekly["vol_regime"] = np.select(
        [weekly[vol_col] >= high_threshold, weekly[vol_col] <= low_threshold],
        ["high_vol", "low_vol"],
        default="normal_vol",
    )

    result = weekly.loc[label_start:label_end].copy()
    result.index.name = "signal_date"
    result = result.reset_index()[
        ["signal_date", "trend_regime", "raw_trend_regime", "vol_regime",
         "close", ma_col, ret_col, vol_col]
    ]
    if result.empty or result[["trend_regime", "vol_regime"]].isna().any().any():
        raise RuntimeError("generated regime labels are empty or incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    metadata = {
        "benchmark": benchmark,
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "label_start": result["signal_date"].min().date().isoformat(),
        "label_end": result["signal_date"].max().date().isoformat(),
        "rows": int(len(result)),
        "trend_counts": result["trend_regime"].value_counts().to_dict(),
        "volatility_counts": result["vol_regime"].value_counts().to_dict(),
        "volatility_thresholds": {"low": low_threshold, "high": high_threshold},
        "config": cfg,
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Regime labels: {output_path} ({len(result)} weeks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
