import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

from stock_predictor.config import PipelineError

CLASSES = ["DOWN", "FLAT", "UP"]
ALPHAS = np.array([0.1, 0.5, 0.9])


def encode_labels(values) -> np.ndarray:
    array = np.asarray(values)
    if not set(array).issubset(CLASSES):
        raise PipelineError("Unknown direction class in a fitting table")
    return np.array([CLASSES.index(v) for v in array], dtype=int)


def pinball_rows(y, predictions) -> np.ndarray:
    error = np.asarray(y)[:, None] - np.asarray(predictions)
    return np.maximum(ALPHAS * error, (ALPHAS - 1) * error)


def validate_quantiles(y, predictions) -> np.ndarray:
    q = np.asarray(predictions, dtype=float)
    if q.shape != (len(y), 3) or not np.isfinite(q).all():
        raise PipelineError("Quantile predictions have invalid shape or nonfinite values")
    if (np.diff(q, axis=1) < 0).any():
        raise PipelineError("Crossing quantiles must be repaired before evaluation")
    return q


def validate_probabilities(probabilities, rows: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (rows, 3) or not np.isfinite(p).all() or (p < 0).any():
        raise PipelineError("Invalid direction probability output")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-7):
        raise PipelineError("Direction probabilities must sum to one")
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(axis=1, keepdims=True)


def regression_metrics(y, predictions) -> dict:
    y = np.asarray(y, dtype=float)
    q = validate_quantiles(y, predictions)
    error = q[:, 1] - y
    pinball = pinball_rows(y, q).mean(axis=0)
    return {
        "rows": len(y),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pinball_q10": float(pinball[0]),
        "pinball_q50": float(pinball[1]),
        "pinball_q90": float(pinball[2]),
        "mean_pinball": float(pinball.mean()),
        "interval_coverage": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean()),
        "mean_interval_width": float((q[:, 2] - q[:, 0]).mean()),
    }


def reliability_table(y, probabilities, bins=10) -> list[dict]:
    result = []
    for cls, name in enumerate(CLASSES):
        for i in range(bins):
            lower, upper = i / bins, (i + 1) / bins
            mask = (probabilities[:, cls] >= lower) & (
                probabilities[:, cls] < upper if i < bins - 1 else probabilities[:, cls] <= upper
            )
            if mask.any():
                result.append(
                    {
                        "class": name,
                        "bin_low": lower,
                        "bin_high": upper,
                        "rows": int(mask.sum()),
                        "mean_probability": float(probabilities[mask, cls].mean()),
                        "observed_frequency": float((y[mask] == cls).mean()),
                    }
                )
    return result


def classification_metrics(y, probabilities) -> dict:
    y = np.asarray(y, dtype=int)
    p = validate_probabilities(probabilities, len(y))
    predicted = p.argmax(axis=1)
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    per_class = {}
    for i, name in enumerate(CLASSES):
        per_class[name] = {
            "precision": float(matrix[i, i] / max(1, matrix[:, i].sum())),
            "recall": float(matrix[i, i] / max(1, matrix[i].sum())),
            "support": int(matrix[i].sum()),
        }
    return {
        "rows": len(y),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier_score": float(np.square(p - np.eye(3)[y]).sum(axis=1).mean()),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "class_order": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "reliability_bins": reliability_table(y, p),
    }


def grouped_regression(index, y, predictions, group="symbol") -> dict:
    results = {}
    for name, positions in index.groupby(group).indices.items():
        results[str(name)] = regression_metrics(
            np.asarray(y)[positions], np.asarray(predictions)[positions]
        )
    return results


def block_bootstrap_differences(
    index, y, labels, quantiles, baseline_quantiles, probabilities, baseline_probabilities, config
) -> dict:
    y, labels = np.asarray(y), np.asarray(labels, dtype=int)
    p = validate_probabilities(probabilities, len(y))
    base_p = validate_probabilities(baseline_probabilities, len(y))
    differences = (
        pd.DataFrame(
            {
                "session_date": index.session_date.to_numpy(),
                "mae": np.abs(y - quantiles[:, 1]) - np.abs(y - baseline_quantiles[:, 1]),
                "mean_pinball": pinball_rows(y, quantiles).mean(axis=1)
                - pinball_rows(y, baseline_quantiles).mean(axis=1),
                "log_loss": -np.log(p[np.arange(len(y)), labels])
                + np.log(base_p[np.arange(len(y)), labels]),
                "rows": 1,
            }
        )
        .groupby("session_date", sort=True)
        .sum()
    )
    values = differences.to_numpy()
    length = min(config["bootstrap_block_sessions"], len(values))
    rng = np.random.default_rng(config["seed"])
    sampled = []
    for _ in range(config["bootstrap_repetitions"]):
        starts = rng.integers(0, len(values) - length + 1, size=int(np.ceil(len(values) / length)))
        positions = np.concatenate([np.arange(s, s + length) for s in starts])[: len(values)]
        sums = values[positions].sum(axis=0)
        sampled.append(sums[:-1] / sums[-1])
    point = values.sum(axis=0)
    limits = np.quantile(sampled, [0.025, 0.975], axis=0)
    return {
        "method": "moving_contiguous_date_blocks_whole_stock_panel",
        "sessions_per_block": length,
        "repetitions": config["bootstrap_repetitions"],
        "negative_favors_lightgbm": True,
        "differences": {
            name: {
                "point": float(point[i] / point[-1]),
                "lower_95": float(limits[0, i]),
                "upper_95": float(limits[1, i]),
            }
            for i, name in enumerate(("mae", "mean_pinball", "log_loss"))
        },
    }
