import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_pinball_loss
from stock_predictor.calendar import schedule
from stock_predictor.config import PipelineError
from stock_predictor.datasets.features import FEATURE_COLUMNS, LIQUIDITY
from stock_predictor.extraction import parquet
from stock_predictor.storage.archive import Archive
from stock_predictor.training.bundle import ResearchBundle
from stock_predictor.training.calibration import (
    apply_interval_adjustment,
    apply_temperature,
    fit_interval_adjustment,
    fit_temperature,
    repair_quantiles,
)
from stock_predictor.training.data import load_block, walk_forward_blocks
from stock_predictor.training.estimators import (
    fit_linear,
    linear_payload,
    predict_arima,
    predict_linear,
)
from stock_predictor.training.metrics import ALPHAS, regression_metrics
from stock_predictor.training.pipeline import holdout_lease, run_training


def test_quantile_metrics_and_crossing_repair():
    y = np.array([0.01, -0.02, 0.03])
    raw = np.array([[0.03, 0, 0.02], [-0.04, -0.02, 0.01], [0, 0.02, 0.05]])
    q = repair_quantiles(raw)
    metrics = regression_metrics(y, q)
    expected = [mean_pinball_loss(y, q[:, i], alpha=a) for i, a in enumerate(ALPHAS)]
    assert metrics["mean_pinball"] == pytest.approx(np.mean(expected))
    assert (np.diff(q, axis=1) >= 0).all()
    with pytest.raises(PipelineError, match="Crossing"):
        regression_metrics(y, raw)


def test_interval_calibration_widens_using_only_its_own_outcomes():
    q = np.tile([-0.01, 0, 0.01], (10, 1))
    y = np.arange(10) / 100
    calibration = fit_interval_adjustment(y, q)
    assert calibration["padding"] == pytest.approx(0.07)
    assert calibration["calibration_rows"] == 10
    adjusted = apply_interval_adjustment(q, calibration)
    np.testing.assert_array_equal(adjusted[:, 1], q[:, 1])
    assert adjusted[:, 0].max() == pytest.approx(-0.08)
    assert not calibration["coverage_guarantee"]


def test_probability_calibration_preserves_class_order_and_normalization():
    p = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]] * 10)
    labels = np.array([0, 1, 2] * 10)
    fitted = fit_temperature(labels, p)
    adjusted = apply_temperature(p, fitted["temperature"])
    assert np.isfinite(adjusted).all()
    np.testing.assert_allclose(adjusted.sum(axis=1), 1)
    np.testing.assert_array_equal(adjusted.argmax(axis=1), labels)
    assert fitted["calibrated_log_loss"] <= fitted["raw_log_loss"]


@pytest.mark.parametrize("kind", ["ridge", "logistic"])
def test_linear_json_roundtrip_and_training_only_preprocessing(kind):
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(120, 34)), columns=FEATURE_COLUMNS)
    x.loc[::3, LIQUIDITY[0]] = np.nan
    y = rng.normal(size=120) if kind == "ridge" else np.arange(120) % 3
    model = fit_linear(x, y, kind, 1)
    original_stats = model[0].statistics_.copy()
    payload = json.loads(json.dumps(linear_payload(model, kind, FEATURE_COLUMNS)))
    future = x.copy() + 1000
    expected = model.predict(future) if kind == "ridge" else model.predict_proba(future)
    np.testing.assert_allclose(predict_linear(payload, future), expected, atol=1e-12)
    np.testing.assert_array_equal(model[0].statistics_, original_stats)


def test_arima_conditional_forecast_matches_statsmodels_forecast():
    # Conditional AR(1) forecast and its aggregate variance are used without
    # refitting parameters on validation/test observations.
    from statsmodels.tsa.arima.model import ARIMA

    r = np.random.default_rng(3).normal(0, 0.01, 150)
    fitted = ARIMA(r * 100, order=(1, 0, 0), trend="c").fit()
    params = dict(zip(fitted.param_names, fitted.params, strict=True))
    payload = {
        "horizon_sessions": 5,
        "parameters": {
            "AAA": dict(
                mean=params["const"] / 100,
                phi=params["ar.L1"],
                innovation_variance=params["sigma2"] / 10000,
            )
        },
    }
    q = predict_arima(payload, pd.DataFrame({"ret_1d": [r[-1]]}), pd.DataFrame({"symbol": ["AAA"]}))
    assert q[0, 1] == pytest.approx(fitted.forecast(5).sum() / 100, abs=1e-8)
    assert q[0, 0] < q[0, 1] < q[0, 2]


def test_holdout_cannot_be_reused_for_another_selection(tmp_path):
    archive = Archive(tmp_path)
    archive.json("holdout/frozen.json", {"recipe_hash": "old"})
    holdout_lease(archive, "holdout/frozen.json", "old")
    with pytest.raises(PipelineError, match="another frozen"):
        holdout_lease(archive, "holdout/frozen.json", "new")


@pytest.fixture
def training_dataset(tmp_path):
    rng = np.random.default_rng(647)
    calendar = schedule(date(2018, 1, 2), date(2020, 12, 31)).iloc[:610].reset_index(drop=True)
    x = pd.DataFrame(rng.normal(size=(1200, 34)), columns=FEATURE_COLUMNS)
    x["ret_1d"] *= 0.01
    for name in LIQUIDITY:
        x[name + "_missing"] = 0
    returns = 0.4 * x.ret_1d + rng.normal(0, 0.025, len(x))
    y = pd.DataFrame(
        {
            "y_price_5d": returns,
            "y_direction_5d": np.select(
                [returns > 0.01, returns < -0.01], ["UP", "DOWN"], default="FLAT"
            ),
            "direction_threshold_5d": 0.01,
            "y_exec_gross_5d": np.expm1(returns),
        }
    )
    index = pd.DataFrame(
        {
            "row_id": [str(i) for i in range(1200)],
            "symbol": ["AAA", "BBB"] * 600,
            "instrument_id": ["AAA", "BBB"] * 600,
            "session_date": np.repeat(calendar.session_date.iloc[:600].to_numpy(), 2),
            "target_session": np.repeat(calendar.session_date.iloc[5:605].to_numpy(), 2),
        }
    )
    index["cutoff_at"] = np.repeat(
        (calendar.close.iloc[:600] + pd.Timedelta(minutes=30)).to_numpy(), 2
    )
    index["label_end_at"] = np.repeat(
        (calendar.close.iloc[5:605] + pd.Timedelta(minutes=30)).to_numpy(), 2
    )
    index["label_available_at_reconstructed"] = index.label_end_at
    boundaries = calendar.session_date.iloc[[300, 400, 500]].tolist()
    archive = Archive(tmp_path / "archive")
    outputs = {}
    blocks = {}
    for i, name in enumerate(("train", "validation", "calibration", "test")):
        lower = None if i == 0 else boundaries[i - 1]
        upper = boundaries[i] if i < 3 else None
        mask = (index.session_date.ge(lower) if lower else pd.Series(True, index=index.index)) & (
            index.target_session.lt(upper) if upper else pd.Series(True, index=index.index)
        )
        block = {k: t[mask].reset_index(drop=True) for k, t in dict(X=x, y=y, index=index).items()}
        blocks[name] = block
        for kind, table in block.items():
            outputs[f"{kind}_{name}"] = parquet(archive, f"fixture/{kind}_{name}", table)
    outputs["inference_features"] = parquet(archive, "fixture/inference", pd.DataFrame())
    sessions_ref = parquet(archive, "fixture/sessions", calendar)
    report = archive.json(
        "reports/dataset.json",
        {
            "research_training_ready": True,
            "production_ready": False,
            "limitations": ["Synthetic engineering fixture"],
        },
    )
    manifest = archive.json(
        "manifests/dataset.json",
        {
            "dataset_version": "synthetic",
            "outputs": outputs,
            "report": report,
            "recipe": {
                "feature_columns": FEATURE_COLUMNS,
                "inputs": {"sessions": sessions_ref},
                "config": {
                    "validation_start": boundaries[0],
                    "calibration_start": boundaries[1],
                    "test_start": boundaries[2],
                    "feature_version": "synthetic",
                },
            },
        },
    )
    config = json.loads(Path("configs/training.daily.json").read_text())
    config.update(
        max_rounds=8,
        early_stopping_rounds=3,
        threads=1,
        ridge_alphas=[10.0],
        logistic_cs=[1.0],
        arima_orders=[0, 1],
        bootstrap_repetitions=10,
        lightgbm_candidates=[
            {
                "name": "test",
                "num_leaves": 7,
                "max_depth": 3,
                "min_data_in_leaf": 15,
                "lambda_l2": 5.0,
            }
        ],
        walk_forward=[
            {
                "inner_validation_start": calendar.session_date.iloc[130],
                "evaluation_start": calendar.session_date.iloc[220],
                "evaluation_end_exclusive": calendar.session_date.iloc[270],
            }
        ],
    )
    return archive, manifest, config, tmp_path, blocks, outputs


def test_walk_forward_purges_both_inner_and_outer_boundaries(training_dataset):
    _, _, config, _, blocks, _ = training_dataset
    inner_train, inner_valid, outer_train, evaluation = walk_forward_blocks(
        blocks["train"], config["walk_forward"][0]
    )
    fold = config["walk_forward"][0]
    assert inner_train["index"].target_session.lt(fold["inner_validation_start"]).all()
    assert inner_valid["index"].target_session.lt(fold["evaluation_start"]).all()
    assert outer_train["index"].target_session.lt(fold["evaluation_start"]).all()
    assert evaluation["index"].target_session.lt(fold["evaluation_end_exclusive"]).all()


def test_wrong_feature_order_is_rejected(training_dataset):
    archive, _, _, _, _, outputs = training_dataset
    modified = copy.deepcopy(outputs)
    from stock_predictor.datasets.pipeline import read_table

    x = read_table(archive, modified["X_train"])[FEATURE_COLUMNS[::-1]]
    modified["X_train"] = parquet(archive, "fixture/bad-order", x)
    with pytest.raises(PipelineError, match="ordering"):
        load_block(archive, modified, "train", (None, None))


def test_training_end_to_end_test_is_loaded_only_after_frozen_selection(
    training_dataset, monkeypatch
):
    archive, manifest, config, root, blocks, outputs = training_dataset
    original_read = archive.read
    test_keys = {outputs[f"{kind}_test"]["key"] for kind in ("X", "y", "index")}
    accesses = []

    def guarded_read(key, **kwargs):
        if key in test_keys:
            assert list((archive.root / "checkpoints/training/selection").glob("*.json"))
            assert list((archive.root / "checkpoints/training/holdout").glob("*.json"))
            accesses.append(key)
        return original_read(key, **kwargs)

    monkeypatch.setattr(archive, "read", guarded_read)
    summary = run_training(None, archive, manifest, config, root / "summary")
    assert len(accesses) == 3
    assert summary["status"] == "trained_research_candidate" and not summary["production_approved"]
    assert summary["training_rows"] == len(blocks["train"]["X"])
    bundle = ResearchBundle.from_manifest(archive, summary["manifest"])
    predictions = bundle.predict(blocks["test"]["X"], blocks["test"]["index"])
    assert np.isfinite(predictions["lightgbm"]).all()
    assert (np.diff(predictions["lightgbm"], axis=1) >= 0).all()
    np.testing.assert_allclose(predictions["lightgbm_direction"].sum(axis=1), 1)
    monkeypatch.setattr(
        "stock_predictor.training.pipeline.select_primary",
        lambda *args: pytest.fail("Cached run must not train"),
    )
    again = run_training(None, archive, manifest, config, root / "cached")
    assert again["cache_reused"] and len(accesses) == 3
