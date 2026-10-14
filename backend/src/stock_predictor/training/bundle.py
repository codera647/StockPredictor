import lightgbm as lgb
import numpy as np

from stock_predictor.config import PipelineError
from stock_predictor.datasets.features import FEATURE_COLUMNS
from stock_predictor.replay import read_json
from stock_predictor.training.calibration import (
    apply_interval_adjustment,
    apply_temperature,
    repair_quantiles,
)
from stock_predictor.training.estimators import predict_arima, predict_linear


def save_booster(archive, name, model) -> dict:
    from stock_predictor.storage.archive import digest

    body = model.model_to_string(num_iteration=model.best_iteration or -1).encode()
    return archive.write(f"models/{name}/{digest(body)}.txt", body)


def save_json_model(archive, name, payload) -> dict:
    from stock_predictor.storage.archive import digest, json_bytes

    body = json_bytes(payload)
    return archive.write(f"models/{name}/{digest(body)}.json", body)


class ResearchBundle:
    """Checksum-verified, pickle-free model loading shared with future serving."""

    def __init__(self, archive, spec: dict):
        if spec["feature_columns"] != FEATURE_COLUMNS:
            raise PipelineError("Model bundle uses an unsupported feature schema")
        self.spec = spec
        self.quantile_models = []
        for name in ("lightgbm_q10", "lightgbm_q50", "lightgbm_q90", "lightgbm_direction"):
            ref = spec["models"][name]
            body = archive.read(ref["key"], sha256=ref["sha256"])
            if body is None:
                raise PipelineError("Model artifact is missing")
            model = lgb.Booster(model_str=body.decode())
            if model.feature_name() != FEATURE_COLUMNS:
                raise PipelineError("Saved model feature names differ from its manifest")
            if name == "lightgbm_direction":
                self.classifier = model
            else:
                self.quantile_models.append(model)
        self.baselines = {
            name: read_json(archive, spec["models"][name])
            for name in ("naive", "ridge", "logistic", "arima")
        }

    @classmethod
    def from_manifest(cls, archive, ref):
        manifest = read_json(archive, ref)
        return cls(archive, manifest["bundle"])

    def predict(self, x, index, calibrated=True) -> dict:
        if list(x) != FEATURE_COLUMNS:
            raise PipelineError("Inference feature names/order differ from training")
        q = np.column_stack([m.predict(x, num_threads=1) for m in self.quantile_models])
        direction = self.classifier.predict(x, num_threads=1)
        naive = self.baselines["naive"]
        mean = predict_linear(self.baselines["ridge"], x)
        offsets = np.asarray(self.baselines["ridge"]["residual_quantiles"])
        ridge_q = np.column_stack([mean + offsets[0], mean, mean + offsets[2]])
        result = {
            "lightgbm": q,
            "ridge": repair_quantiles(ridge_q),
            "arima": predict_arima(self.baselines["arima"], x, index),
            "naive": np.tile(naive["quantiles"], (len(x), 1)),
            "lightgbm_direction": direction,
            "logistic": predict_linear(self.baselines["logistic"], x),
            "naive_direction": np.tile(naive["class_probabilities"], (len(x), 1)),
        }
        result["raw_quantile_crossing_fraction"] = float(
            (np.diff(q, axis=1) < 0).any(axis=1).mean()
        )
        for name in ("lightgbm", "ridge", "arima", "naive"):
            result[name] = (
                apply_interval_adjustment(result[name], self.spec["interval_calibration"][name])
                if calibrated
                else repair_quantiles(result[name])
            )
        if calibrated:
            for name in ("lightgbm_direction", "logistic", "naive_direction"):
                result[name] = apply_temperature(
                    result[name], self.spec["probability_calibration"][name]["temperature"]
                )
        return result
