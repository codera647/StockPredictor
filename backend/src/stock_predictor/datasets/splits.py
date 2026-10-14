import numpy as np
import pandas as pd

from stock_predictor.config import PipelineError
from stock_predictor.datasets.features import CORE_COLUMNS, FEATURE_COLUMNS, LIQUIDITY

BLOCK_NAMES = ("train", "validation", "calibration", "test")
TARGET_COLUMNS = ["y_price_5d", "y_direction_5d", "direction_threshold_5d", "y_exec_gross_5d"]


def make_splits(
    features: pd.DataFrame, labels: pd.DataFrame, config: dict
) -> tuple[dict, pd.DataFrame, dict]:
    boundaries = [config[k] for k in ("validation_start", "calibration_start", "test_start")]
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise PipelineError("Dataset boundaries must be strictly chronological")
    rows = features.merge(
        labels.drop(
            columns=["instrument_id", "symbol", "session_date", "cutoff_at", "production_ready"]
        ),
        on="row_id",
        validate="one_to_one",
    )
    if rows.row_id.duplicated().any():
        raise PipelineError("Duplicate dataset row IDs")
    rows["split"] = np.select(
        [rows.session_date.lt(b) for b in boundaries], BLOCK_NAMES[:3], default="test"
    )
    rows["included"] = rows.feature_ready & rows.label_ready
    rows["split_exclusion_reason"] = np.select(
        [~rows.feature_ready, ~rows.label_ready],
        ["feature_not_ready", "label_not_ready"],
        default="included",
    )
    outputs, report = {}, {}
    for i, name in enumerate(BLOCK_NAMES):
        block_mask = rows.split.eq(name)
        if i < 3:
            crossed = block_mask & rows.included & rows.target_session.ge(boundaries[i])
            rows.loc[crossed, "included"] = False
            rows.loc[crossed, "split_exclusion_reason"] = "purged_cross_boundary_label"
        selected = (
            rows[block_mask & rows.included]
            .sort_values(["session_date", "instrument_id"])
            .reset_index(drop=True)
        )
        if selected.empty:
            raise PipelineError(f"No eligible examples in {name}; adjust scope before fitting")
        if not np.isfinite(selected[CORE_COLUMNS]).all().all():
            raise PipelineError("Nonfinite core features in a fitting block")
        if np.isinf(selected[LIQUIDITY]).any().any():
            raise PipelineError("Infinite optional features in a fitting block")
        if selected[LIQUIDITY].isna().all().any():
            raise PipelineError("A liquidity feature is entirely missing in a fitting block")
        if selected.label_end_at.le(selected.cutoff_at).any():
            raise PipelineError("Label end must follow prediction cutoff")
        if selected.max_input_available_at_reconstructed.gt(selected.cutoff_at).any():
            raise PipelineError("Reconstructed input availability crosses prediction cutoff")
        outputs[name] = {
            "X": selected[FEATURE_COLUMNS],
            "y": selected[TARGET_COLUMNS],
            "index": selected[
                [
                    "row_id",
                    "instrument_id",
                    "symbol",
                    "session_date",
                    "cutoff_at",
                    "target_session",
                    "label_end_at",
                    "label_available_at_reconstructed",
                    "max_input_observed_at",
                    "label_observed_at",
                ]
            ],
        }
        report[name] = {
            "rows": len(selected),
            "instruments": int(selected.instrument_id.nunique()),
            "dates": int(selected.session_date.nunique()),
            "first_session": selected.session_date.min(),
            "last_session": selected.session_date.max(),
            "last_target_session": selected.target_session.max(),
            "purged_rows": int(
                (block_mask & rows.split_exclusion_reason.eq("purged_cross_boundary_label")).sum()
            ),
            "class_balance": {k: int(v) for k, v in selected.y_direction_5d.value_counts().items()},
        }
    return (
        outputs,
        rows[
            [
                "row_id",
                "symbol",
                "session_date",
                "target_session",
                "split",
                "included",
                "split_exclusion_reason",
            ]
        ],
        report,
    )
