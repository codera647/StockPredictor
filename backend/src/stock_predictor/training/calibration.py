import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss

from stock_predictor.config import PipelineError
from stock_predictor.training.metrics import validate_probabilities, validate_quantiles


def repair_quantiles(predictions) -> np.ndarray:
    # Fixed rearrangement rule, chosen before any holdout outcome is inspected.
    return np.sort(np.asarray(predictions, dtype=float), axis=1)


def fit_interval_adjustment(y, quantiles, coverage=0.8) -> dict:
    y = np.asarray(y, dtype=float)
    q = validate_quantiles(y, quantiles)
    scores = np.maximum(q[:, 0] - y, y - q[:, 2])
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * coverage)))
    padding = max(0.0, float(np.sort(scores)[rank - 1]))
    return {
        "method": "global_cqr_widen_only",
        "padding": padding,
        "nominal_coverage": coverage,
        "calibration_rows": len(y),
        "quantile_rank": rank,
        "coverage_guarantee": False,
        "dependence_note": "Dependent targets and stock/date groups prevent iid coverage claims.",
    }


def apply_interval_adjustment(quantiles, calibration) -> np.ndarray:
    result = repair_quantiles(quantiles)
    result[:, 0] -= calibration["padding"]
    result[:, 2] += calibration["padding"]
    return result


def apply_temperature(probabilities, temperature=1.0) -> np.ndarray:
    p = validate_probabilities(probabilities, len(probabilities))
    if not np.isfinite(temperature) or temperature <= 0:
        raise PipelineError("Calibration temperature must be positive and finite")
    return softmax(np.log(p) / temperature, axis=1)


def fit_temperature(labels, probabilities, bounds=(0.25, 4.0)) -> dict:
    p = validate_probabilities(probabilities, len(labels))
    raw_loss = float(log_loss(labels, p, labels=[0, 1, 2]))
    fitted = minimize_scalar(
        lambda t: log_loss(labels, apply_temperature(p, t), labels=[0, 1, 2]),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not fitted.success:
        raise PipelineError("Temperature calibration optimization failed")
    # The identity is available, so calibration never worsens its fitting objective.
    temperature = float(fitted.x) if fitted.fun < raw_loss else 1.0
    return {
        "method": "multiclass_temperature",
        "temperature": temperature,
        "bounds": list(bounds),
        "calibration_rows": len(labels),
        "raw_log_loss": raw_loss,
        "calibrated_log_loss": float(
            log_loss(labels, apply_temperature(p, temperature), labels=[0, 1, 2])
        ),
    }
