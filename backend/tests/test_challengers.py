import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_training as training_tests
from stock_predictor.config import PipelineError
from stock_predictor.datasets.features import FEATURE_COLUMNS
from stock_predictor.storage.archive import Archive
from stock_predictor.training.bundle import save_booster, save_json_model
from stock_predictor.training.estimators import fit_booster, fit_linear, linear_payload
from stock_predictor.training.metrics import encode_labels
from stock_predictor_challengers.models import fit_model, load_model, predict_model, save_model
from stock_predictor_challengers.pipeline import run_comparison, validate_config


@pytest.fixture
def challenger_dataset(tmp_path):
    return training_tests.training_dataset.__wrapped__(tmp_path)


@pytest.mark.parametrize("family", ["xgboost", "catboost"])
def test_native_challenger_reload_preserves_class_order_and_probabilities(tmp_path, family):
    config = json.loads(Path("configs/challengers.daily.json").read_text())
    config.update(max_rounds=5, early_stopping_rounds=2, threads=1)
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(180, 34)), columns=FEATURE_COLUMNS)
    y = np.arange(180) % 3
    candidate = config[family + "_candidates"][0]
    model, _ = fit_model(family, x.iloc[:120], y[:120], x.iloc[120:], y[120:], candidate, config)
    archive = Archive(tmp_path)
    ref = save_model(archive, family, "test", model)
    original = predict_model(family, model, x.iloc[120:])
    loaded = predict_model(family, load_model(archive, family, ref), x.iloc[120:])
    np.testing.assert_allclose(original, loaded, atol=1e-7)
    np.testing.assert_allclose(loaded.sum(axis=1), 1)
    with pytest.raises(PipelineError):
        predict_model(family, model, x.iloc[120:][FEATURE_COLUMNS[::-1]])


def test_comparison_refuses_to_open_existing_test_by_configuration():
    config = json.loads(Path("configs/challengers.daily.json").read_text())
    config["evaluate_existing_test"] = True
    with pytest.raises(PipelineError, match="exclude"):
        validate_config(config)


def test_end_to_end_challenger_comparison_never_loads_calibration_or_test(
    challenger_dataset, monkeypatch
):
    archive, dataset, old_config, root, blocks, outputs = challenger_dataset
    train, valid = blocks["train"], blocks["validation"]
    classifier = fit_booster(
        train["X"],
        encode_labels(train["y"].y_direction_5d),
        valid["X"],
        encode_labels(valid["y"].y_direction_5d),
        old_config["lightgbm_candidates"][0],
        old_config,
    )
    logistic = fit_linear(train["X"], encode_labels(train["y"].y_direction_5d), "logistic", 0.1)
    baseline = archive.json(
        "manifests/baseline.json",
        {
            "bundle": {
                "dataset_version": "synthetic",
                "models": {
                    "lightgbm_direction": save_booster(archive, "fixture/direction", classifier),
                    "logistic": save_json_model(
                        archive,
                        "fixture/logistic",
                        linear_payload(logistic, "logistic", FEATURE_COLUMNS),
                    ),
                },
            }
        },
    )
    config = json.loads(Path("configs/challengers.daily.json").read_text())
    config.update(
        max_rounds=5,
        early_stopping_rounds=2,
        threads=1,
        logistic_cs=[0.1],
        walk_forward=old_config["walk_forward"],
        xgboost_candidates=[
            {"name": "test", "max_depth": 2, "min_child_weight": 1, "reg_lambda": 10}
        ],
        catboost_candidates=[{"name": "test", "depth": 2, "l2_leaf_reg": 10}],
    )
    forbidden = {
        outputs[f"{kind}_{block}"]["key"]
        for kind in ("X", "y", "index")
        for block in ("calibration", "test")
    }
    original_read = archive.read

    def guarded_read(key, **kwargs):
        assert key not in forbidden, "Challenger selection opened a reserved dataset block"
        return original_read(key, **kwargs)

    monkeypatch.setattr(archive, "read", guarded_read)
    result = run_comparison(archive, dataset, baseline, config, root / "challengers")
    assert not result["existing_test_evaluated"]
    assert not result["decision"]["production_approved"]
    assert result["decision"]["fresh_future_holdout_required"]
    monkeypatch.setattr(
        "stock_predictor_challengers.pipeline.select",
        lambda *args: pytest.fail("Cache must not retrain"),
    )
    assert run_comparison(archive, dataset, baseline, config, root / "cached")["cache_reused"]
