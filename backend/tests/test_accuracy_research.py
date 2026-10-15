import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_training as training_tests
from stock_predictor.calendar import schedule
from stock_predictor.config import PipelineError
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.training.bundle import save_json_model
from stock_predictor.training.estimators import fit_linear, linear_payload
from stock_predictor.training.metrics import encode_labels
from stock_predictor_research.macro import RESEARCH_FEATURES, SERIES, macro_features
from stock_predictor_research.pipeline import run_research, validate_config
from stock_predictor_research.selective import risk_coverage, select_rule


def research_config():
    return json.loads(Path("configs/accuracy-research.daily.json").read_text())


def synthetic_macro():
    records = []
    for i, month in enumerate(pd.date_range("2016-01-01", "2021-01-01", freq="MS")):
        for series in SERIES:
            records.append(
                {
                    "series_id": series,
                    "observation_date": month.date().isoformat(),
                    "value": 100 + i if series == "CPIAUCSL" else 2 + i / 100,
                    "realtime_start": (month + pd.Timedelta(days=9)).date().isoformat(),
                    "available_at_reconstructed": month.tz_localize("UTC") + pd.Timedelta(days=10),
                    "quality_valid": True,
                }
            )
    return pd.DataFrame(records)


def test_future_macro_revision_cannot_change_an_earlier_feature():
    source = synthetic_macro()
    sessions = schedule(date(2018, 1, 2), date(2018, 3, 30))
    original = macro_features(source, sessions, research_config())
    revision = source[source.series_id.eq("CPIAUCSL")].iloc[[0]].copy()
    revision["value"] = 1e6
    revision["available_at_reconstructed"] = pd.Timestamp("2018-04-01", tz="UTC")
    revision["realtime_start"] = "2018-03-31"
    modified = macro_features(
        pd.concat([source, revision], ignore_index=True), sessions, research_config()
    )
    pd.testing.assert_frame_equal(original, modified)
    assert original.macro_max_available_at_reconstructed.le(original.macro_cutoff_at).all()
    assert len(RESEARCH_FEATURES) == 58


def test_monthly_data_waits_for_release_and_has_explicit_missing_flags():
    source = synthetic_macro()
    sessions = schedule(date(2018, 1, 2), date(2018, 1, 12))
    macro = macro_features(source, sessions, research_config())
    expected_prior = (100 + 23) / (100 + 11) - 1
    assert macro.iloc[0].macro_cpi_yoy == pytest.approx(expected_prior)
    after = macro[macro.session_date.eq("2018-01-11")].iloc[0]
    assert after.macro_cpi_yoy == pytest.approx((100 + 24) / (100 + 12) - 1)
    assert macro.macro_cpi_yoy_missing.eq(0).all()


def test_high_confidence_is_not_proof_of_ninety_percent_accuracy():
    config = research_config()
    labels = np.arange(400) % 2
    p = np.tile([0.99, 0.005, 0.005], (400, 1))
    index = pd.DataFrame(
        {
            "session_date": np.repeat(
                pd.date_range("2020-01-01", periods=200).strftime("%Y-%m-%d"), 2
            )
        }
    )
    curve = risk_coverage(labels, p, index, config)
    rule = select_rule(curve, 0.9)
    assert not rule["enabled"] and rule["threshold"] is None
    assert all(row["accuracy"] is None or row["accuracy"] == 0.5 for row in curve)


def test_zero_accepted_examples_have_unknown_accuracy():
    config = research_config()
    labels = np.arange(12) % 3
    index = pd.DataFrame({"session_date": ["2020-01-02"] * 12})
    curve = risk_coverage(labels, np.tile([0.4, 0.3, 0.3], (12, 1)), index, config)
    row = next(row for row in curve if row["threshold"] == 0.9)
    assert row["accepted"] == 0 and row["accuracy"] is None and row["coverage"] == 0


def test_final_test_reuse_is_rejected():
    config = research_config()
    config["evaluate_existing_test"] = True
    with pytest.raises(PipelineError):
        validate_config(config)


@pytest.fixture
def accuracy_dataset(tmp_path):
    return training_tests.training_dataset.__wrapped__(tmp_path)


def test_end_to_end_macro_neural_reload_and_holdout_exclusion(accuracy_dataset, monkeypatch):
    archive, manifest, old_config, root, blocks, outputs = accuracy_dataset
    source = read_json(archive, manifest)
    source["outputs"]["clean_macro_observations"] = parquet(
        archive, "fixture/macro", synthetic_macro()
    )
    enriched = archive.json("manifests/with-macro.json", source)
    logistic = fit_linear(
        blocks["train"]["X"], encode_labels(blocks["train"]["y"].y_direction_5d), "logistic", 0.1
    )
    baseline = archive.json(
        "manifests/macro-baseline.json",
        {
            "bundle": {
                "dataset_version": "synthetic",
                "models": {
                    "logistic": save_json_model(
                        archive,
                        "fixture/price-logistic",
                        linear_payload(logistic, "logistic", source["recipe"]["feature_columns"]),
                    )
                },
            }
        },
    )
    config = research_config()
    from stock_predictor.datasets.pipeline import read_table

    sessions = read_table(archive, source["recipe"]["inputs"]["sessions"])
    config.update(
        max_rounds=5,
        early_stopping_rounds=2,
        threads=1,
        neural_hidden_layers=[8, 4],
        neural_max_epochs=2,
        neural_patience=1,
        logistic_cs=[0.1],
        walk_forward=old_config["walk_forward"],
        minimum_selective_examples=10,
        minimum_selective_dates=5,
        bootstrap_repetitions=5,
        calibration_split=sessions.session_date.iloc[450],
    )
    forbidden = {outputs[f"{kind}_test"]["key"] for kind in ("X", "y", "index")}
    original_read = archive.read

    def guarded_read(key, **kwargs):
        assert key not in forbidden, "Accuracy research opened the existing final test"
        return original_read(key, **kwargs)

    monkeypatch.setattr(archive, "read", guarded_read)
    before = blocks["validation"]["y"].copy()
    result = run_research(archive, enriched, baseline, config, root / "research")
    assert not result["existing_test_evaluated"] and not result["target_achieved"]
    assert result["feature_count"] == 58
    pd.testing.assert_frame_equal(before, blocks["validation"]["y"])
    monkeypatch.setattr(
        "stock_predictor_research.pipeline.fit_components",
        lambda *args: pytest.fail("Cache must not train"),
    )
    assert run_research(archive, enriched, baseline, config, root / "cached")["cache_reused"]
