"""Shared trailing feature engine; targets are deliberately absent from this module."""

import hashlib

import numpy as np
import pandas as pd

from stock_predictor.config import PipelineError

LIQUIDITY = ["relative_volume_20_lag1", "log_dollar_volume_20_lag1", "volume_trend_5_20_lag1"]
FEATURE_COLUMNS = (
    [f"ret_{n}d" for n in (1, 5, 20, 60)]
    + [f"ret_1d_lag_{n}" for n in range(1, 6)]
    + ["overnight_gap", "intraday_return"]
    + [f"close_to_sma_{n}" for n in (10, 20, 60)]
    + ["drawdown_60", "distance_to_high_60", "rsi_14"]
    + [f"rolling_std_{n}" for n in (5, 20, 60)]
    + ["normalized_atr_14", "range_pct"]
    + LIQUIDITY
    + [f"spy_ret_{n}d" for n in (1, 5, 20)]
    + ["spy_volatility_20", "beta_spy_60", "residual_momentum_20"]
    + [f"{name}_missing" for name in LIQUIDITY]
)
CORE_COLUMNS = [name for name in FEATURE_COLUMNS if name not in LIQUIDITY]


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    values = close.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    if len(values) <= period:
        return pd.Series(result, index=close.index)
    delta = np.diff(values)
    gain, loss = np.maximum(delta, 0), np.maximum(-delta, 0)
    average_gain, average_loss = gain[:period].mean(), loss[:period].mean()
    for i in range(period, len(values)):
        if i > period:
            average_gain = ((period - 1) * average_gain + gain[i - 1]) / period
            average_loss = ((period - 1) * average_loss + loss[i - 1]) / period
        result[i] = (
            50
            if average_gain == average_loss == 0
            else 100
            if average_loss == 0
            else 100 - 100 / (1 + average_gain / average_loss)
        )
    return pd.Series(result, index=close.index)


def local_features(block: pd.DataFrame) -> pd.DataFrame:
    """Called on one contiguous valid instrument segment, with no filled prices."""
    c, o, h, low, v = (
        block[f"{name}_basis"] for name in ("close", "open", "high", "low", "volume")
    )
    r = np.log(c / c.shift(1))
    out = pd.DataFrame(index=block.index)
    for n in (1, 5, 20, 60):
        out[f"ret_{n}d"] = np.log(c / c.shift(n))
    for n in range(1, 6):
        out[f"ret_1d_lag_{n}"] = r.shift(n)
    out["overnight_gap"] = np.log(o / c.shift(1))
    out["intraday_return"] = np.log(c / o)
    for n in (10, 20, 60):
        out[f"close_to_sma_{n}"] = c / c.rolling(n).mean() - 1
    out["drawdown_60"] = c / c.rolling(60).max() - 1
    out["distance_to_high_60"] = c / h.rolling(60).max() - 1
    out["rsi_14"] = wilder_rsi(c)
    for n in (5, 20, 60):
        out[f"rolling_std_{n}"] = r.rolling(n).std(ddof=1)
    true_range = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(
        axis=1
    )
    true_range.iloc[0] = np.nan
    out["normalized_atr_14"] = true_range.rolling(14).mean() / c
    out["range_pct"] = (h - low) / c
    preceding_volume = v.shift(1)
    out[LIQUIDITY[0]] = preceding_volume / v.shift(2).rolling(20).mean().replace(0, np.nan)
    dollars = (block.close_raw * block.volume_raw).shift(1).rolling(20).mean()
    out[LIQUIDITY[1]] = np.log1p(dollars)
    out[LIQUIDITY[2]] = (
        preceding_volume.rolling(5).mean() / preceding_volume.rolling(20).mean().replace(0, np.nan)
        - 1
    )
    out["dollar_volume_20_lag1"] = dollars
    out["contiguous_valid_sessions"] = np.arange(1, len(block) + 1)
    return out


def build_features(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    panel = panel.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    if config["benchmark"] not in set(panel.symbol):
        raise PipelineError("The declared benchmark must be present; no implicit fallback")
    technical = pd.DataFrame(index=panel.index)
    for _, instrument in panel.groupby("symbol", sort=False):
        segment_ids = (~instrument.clean_valid).cumsum()
        for _, block in instrument[instrument.clean_valid].groupby(segment_ids):
            values = local_features(block)
            technical.loc[values.index, values.columns] = values
    metadata = [
        "instrument_id",
        "symbol",
        "session_date",
        "cutoff_at",
        "session_index",
        "clean_valid",
        "raw_snapshot_key",
        "retrieved_at",
        "action_observed_at_basis",
        "quality_reason_codes",
    ]
    out = pd.concat([panel[metadata], technical], axis=1)
    benchmark = out[out.symbol == config["benchmark"]].set_index("session_date")
    for n in (1, 5, 20):
        out[f"spy_ret_{n}d"] = out.session_date.map(benchmark[f"ret_{n}d"])
    out["spy_volatility_20"] = out.session_date.map(benchmark.rolling_std_20)
    out["beta_spy_60"] = np.nan
    for _, block in out.groupby("symbol", sort=False):
        r, market = block.ret_1d, block.spy_ret_1d
        beta = r.rolling(60).cov(market) / market.rolling(60).var().replace(0, np.nan)
        out.loc[block.index, "beta_spy_60"] = beta
    out["residual_momentum_20"] = out.ret_20d - out.beta_spy_60 * out.spy_ret_20d
    for name in LIQUIDITY:
        out[name] = out[name].replace([np.inf, -np.inf], np.nan)
        out[name + "_missing"] = out[name].isna().astype("int8")
    finite_core = np.isfinite(out[CORE_COLUMNS]).all(axis=1)
    history = out.contiguous_valid_sessions.ge(
        max(config["eligibility_sessions"], config["indicator_warmup_sessions"])
    )
    liquid = out.dollar_volume_20_lag1.ge(config["minimum_dollar_volume_20"])
    benchmark_ready = out.session_date.map(benchmark.contiguous_valid_sessions).ge(
        config["indicator_warmup_sessions"]
    )
    out["feature_ready"] = out.clean_valid & history & liquid & finite_core & benchmark_ready
    out["readiness_status"] = np.select(
        [~out.clean_valid, ~history, ~benchmark_ready | ~finite_core, ~liquid],
        [
            "quarantined_bar",
            "insufficient_contiguous_history",
            "missing_core_or_benchmark",
            "below_liquidity_floor",
        ],
        default="research_ready",
    )
    out["feature_version"] = config["feature_version"]
    out["row_id"] = [
        hashlib.sha256(f"{i}|{d}|{config['feature_version']}".encode()).hexdigest()
        for i, d in zip(out.instrument_id, out.session_date, strict=True)
    ]
    out["benchmark_snapshot_key"] = out.session_date.map(benchmark.raw_snapshot_key)
    out["max_input_available_at_reconstructed"] = out.cutoff_at
    # RSI depends on earlier segment state. An expanding receive-time maximum is
    # conservative and retains evidence that reconstructed history was received later.
    stock_observed = pd.to_datetime(out.retrieved_at, utc=True).groupby(out.symbol).cummax()
    benchmark_observed = pd.to_datetime(benchmark.retrieved_at, utc=True).cummax()
    out["max_input_observed_at"] = pd.concat(
        [
            stock_observed,
            pd.to_datetime(out.action_observed_at_basis, utc=True),
            pd.to_datetime(out.session_date.map(benchmark_observed), utc=True),
            pd.to_datetime(out.session_date.map(benchmark.action_observed_at_basis), utc=True),
        ],
        axis=1,
    ).max(axis=1)
    out["availability_policy"] = config["historical_availability_policy"]
    out["production_ready"] = False
    return out[out.symbol != config["benchmark"]].reset_index(drop=True)
