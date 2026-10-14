"""Future outcomes are generated independently of model inputs."""

import numpy as np
import pandas as pd


def build_labels(panel: pd.DataFrame, features: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizon = config["horizon_sessions"]
    blocks = []
    for symbol, block in panel.sort_values(["symbol", "session_date"]).groupby("symbol"):
        if symbol == config["benchmark"]:
            continue
        block = block.reset_index(drop=True)
        labels = block[["instrument_id", "symbol", "session_date", "cutoff_at"]].copy()
        labels["target_session"] = block.session_date.shift(-horizon)
        labels["label_start_at"] = block.cutoff_at
        labels["label_end_at"] = block.cutoff_at.shift(-horizon)
        labels["label_available_at_reconstructed"] = labels.label_end_at
        labels["label_observed_at"] = pd.concat(
            [pd.to_datetime(block.retrieved_at.shift(-n), utc=True) for n in range(horizon + 1)],
            axis=1,
        ).max(axis=1)
        valid_path = pd.concat(
            [block.clean_valid.shift(-n, fill_value=False) for n in range(horizon + 1)], axis=1
        ).all(axis=1)
        mature = labels.target_session.notna()
        labels["label_ready"] = valid_path & mature
        labels["label_quality_status"] = np.select(
            [~mature, ~valid_path], ["immature", "unresolved_target_path"], default="research_ready"
        )
        labels["y_price_5d"] = np.log(block.close_basis.shift(-horizon) / block.close_basis).where(
            labels.label_ready
        )
        labels["y_exec_gross_5d"] = (
            block.close_basis.shift(-horizon) / block.open_basis.shift(-1) - 1
        ).where(labels.label_ready)
        labels["horizon_sessions"] = horizon
        labels["target_version"] = config["target_version"]
        blocks.append(labels)
    labels = pd.concat(blocks, ignore_index=True).merge(
        features[["row_id", "instrument_id", "session_date", "rolling_std_20"]],
        on=["instrument_id", "session_date"],
        validate="one_to_one",
    )
    labels["direction_threshold_5d"] = np.maximum(
        config["direction_epsilon"],
        config["direction_volatility_multiplier"] * labels.rolling_std_20 * np.sqrt(horizon),
    )
    usable = labels.label_ready & labels.direction_threshold_5d.notna()
    direction = np.select(
        [
            labels.y_price_5d.gt(labels.direction_threshold_5d),
            labels.y_price_5d.lt(-labels.direction_threshold_5d),
        ],
        ["UP", "DOWN"],
        default="FLAT",
    )
    labels["y_direction_5d"] = pd.Series(direction, index=labels.index).where(usable)
    labels["label_ready"] = usable
    labels.loc[
        ~usable & labels.label_quality_status.eq("research_ready"), "label_quality_status"
    ] = "missing_direction_threshold"
    labels["return_basis"] = "split_consistent_price_only_excludes_cash_dividends"
    labels["production_ready"] = False
    return labels.drop(columns="rolling_std_20")
