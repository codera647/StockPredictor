import numpy as np
import pandas as pd

from stock_predictor.config import PipelineError
from stock_predictor.datasets.features import CORE_COLUMNS, FEATURE_COLUMNS, LIQUIDITY
from stock_predictor.datasets.pipeline import read_table
from stock_predictor.training.metrics import CLASSES


def load_block(archive, outputs, name, bounds) -> dict:
    tables = {kind: read_table(archive, outputs[f"{kind}_{name}"]) for kind in ("X", "y", "index")}
    x, y, index = (tables[kind] for kind in ("X", "y", "index"))
    if x.empty or not len(x) == len(y) == len(index):
        raise PipelineError("Fitting table lengths differ or are empty")
    if list(x) != FEATURE_COLUMNS:
        raise PipelineError("Fitting feature ordering differs from the shared daily schema")
    if not np.isfinite(x[CORE_COLUMNS]).all().all() or np.isinf(x[LIQUIDITY]).any().any():
        raise PipelineError("Fitting block has nonfinite undeclared inputs")
    if x[LIQUIDITY].isna().all().any():
        raise PipelineError("Cannot fit a feature with no training observations")
    if not np.isfinite(y[["y_price_5d", "direction_threshold_5d", "y_exec_gross_5d"]]).all().all():
        raise PipelineError("Fitting targets must be finite and mature")
    if not set(y.y_direction_5d).issubset(CLASSES) or index.row_id.duplicated().any():
        raise PipelineError("Invalid labels or duplicate row IDs")
    start, end = bounds
    if (start and index.session_date.lt(start).any()) or (
        end and (index.session_date.ge(end).any() or index.target_session.ge(end).any())
    ):
        raise PipelineError("Fitting block crosses its purged chronological boundary")
    if index.label_end_at.le(index.cutoff_at).any():
        raise PipelineError("Target interval must end after prediction cutoff")
    if end and index.label_available_at_reconstructed.ge(pd.Timestamp(end, tz="UTC")).any():
        raise PipelineError("Fitting label is unavailable at the next block's fit boundary")
    return tables


def subset(block, mask) -> dict:
    return {kind: values.loc[mask].reset_index(drop=True) for kind, values in block.items()}


def walk_forward_blocks(block, fold) -> tuple[dict, dict, dict, dict]:
    index = block["index"]
    inner, start, end = (
        fold[k] for k in ("inner_validation_start", "evaluation_start", "evaluation_end_exclusive")
    )
    masks = [
        index.session_date.lt(inner) & index.target_session.lt(inner),
        index.session_date.ge(inner)
        & index.session_date.lt(start)
        & index.target_session.lt(start),
        index.session_date.lt(start) & index.target_session.lt(start),
        index.session_date.ge(start) & index.session_date.lt(end) & index.target_session.lt(end),
    ]
    if not inner < start < end or not all(m.any() for m in masks):
        raise PipelineError("Walk-forward fold has invalid dates or empty stages")
    return tuple(subset(block, mask) for mask in masks)
