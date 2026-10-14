import json
import logging
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_pinball_loss
from threadpoolctl import threadpool_limits

from stock_predictor.calendar import schedule
from stock_predictor.config import PipelineError, Settings
from stock_predictor.datasets.features import FEATURE_COLUMNS
from stock_predictor.datasets.pipeline import code_hash, read_table
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.storage.archive import Archive, digest, json_bytes
from stock_predictor.training.bundle import ResearchBundle, save_booster, save_json_model
from stock_predictor.training.calibration import (
    fit_interval_adjustment,
    fit_temperature,
    repair_quantiles,
)
from stock_predictor.training.data import load_block, walk_forward_blocks
from stock_predictor.training.estimators import (
    fit_arima_bundle,
    fit_booster,
    fit_linear,
    linear_payload,
    predict_arima,
)
from stock_predictor.training.metrics import (
    ALPHAS,
    CLASSES,
    block_bootstrap_differences,
    classification_metrics,
    encode_labels,
    grouped_regression,
    regression_metrics,
)

LOG = logging.getLogger(__name__)


def validate_config(config):
    if (
        config.get("schema_version") != 1
        or config.get("quantiles") != [0.1, 0.5, 0.9]
        or config.get("production_ready") is not False
    ):
        raise PipelineError("Unsupported daily training contract")
    for field in (
        "max_rounds",
        "early_stopping_rounds",
        "threads",
        "bootstrap_repetitions",
        "bootstrap_block_sessions",
        "arima_minimum_sessions",
    ):
        if not isinstance(config[field], int) or config[field] < 1:
            raise PipelineError("Training budgets must be positive integers")
    if not config["lightgbm_candidates"] or len(config["lightgbm_candidates"]) > 8:
        raise PipelineError("A bounded LightGBM candidate budget is required")
    if config["nominal_coverage"] != 0.8 or not 0 < config["learning_rate"] <= 0.2:
        raise PipelineError("Unsupported interval coverage or learning rate")
    if (
        not 0 <= config["minimum_relative_improvement"] < 1
        or not 0 < config["coverage_tolerance"] < 0.2
    ):
        raise PipelineError("Invalid model gate thresholds")
    if not 0 < config["temperature_bounds"][0] < 1 < config["temperature_bounds"][1]:
        raise PipelineError("Temperature bounds must contain the identity")
    if (
        not config["ridge_alphas"]
        or not config["logistic_cs"]
        or not set(config["arima_orders"]).issubset({0, 1})
    ):
        raise PipelineError("Invalid declared baseline search")


def select_primary(train, validation, config, candidates=None):
    candidates = candidates or config["lightgbm_candidates"]
    quantiles, trials, selected = [], [], {}
    for alpha, name in zip(ALPHAS, ("q10", "q50", "q90"), strict=True):
        best_model, best_score = None, float("inf")
        for candidate in candidates:
            LOG.info("Fitting LightGBM %s: %s", name, candidate["name"])
            model = fit_booster(
                train["X"],
                train["y"].y_price_5d,
                validation["X"],
                validation["y"].y_price_5d,
                candidate,
                config,
                alpha=float(alpha),
            )
            prediction = model.predict(validation["X"], num_threads=config["threads"])
            score = float(
                mean_pinball_loss(validation["y"].y_price_5d, prediction, alpha=float(alpha))
            )
            record = {
                "model": name,
                "candidate": candidate,
                "best_iteration": model.best_iteration or model.current_iteration(),
                "validation_metric": "pinball",
                "validation_score": score,
            }
            trials.append(record)
            if score < best_score:
                best_model, best_score, selected[name] = model, score, record
        quantiles.append(best_model)
    training_labels = encode_labels(train["y"].y_direction_5d)
    validation_labels = encode_labels(validation["y"].y_direction_5d)
    classifier, best_score = None, float("inf")
    for candidate in candidates:
        LOG.info("Fitting LightGBM direction: %s", candidate["name"])
        model = fit_booster(
            train["X"], training_labels, validation["X"], validation_labels, candidate, config
        )
        prediction = model.predict(validation["X"], num_threads=config["threads"])
        score = float(log_loss(validation_labels, prediction, labels=[0, 1, 2]))
        record = {
            "model": "direction",
            "candidate": candidate,
            "best_iteration": model.best_iteration or model.current_iteration(),
            "validation_metric": "log_loss",
            "validation_score": score,
        }
        trials.append(record)
        if score < best_score:
            classifier, best_score, selected["direction"] = model, score, record
    return quantiles, classifier, selected, trials


def select_baselines(train, validation, config, session_positions):
    payloads, selected, trials = {}, {}, []
    y, validation_y = train["y"].y_price_5d, validation["y"].y_price_5d
    labels, validation_labels = (
        encode_labels(train["y"].y_direction_5d),
        encode_labels(validation["y"].y_direction_5d),
    )
    empirical = np.quantile(y, ALPHAS)
    payloads["naive"] = {
        "kind": "unchanged_price",
        "quantiles": [min(0.0, float(empirical[0])), 0.0, max(0.0, float(empirical[2]))],
        "class_probabilities": (np.bincount(labels, minlength=3) / len(labels)).tolist(),
        "class_order": CLASSES,
    }
    best = float("inf")
    for alpha in config["ridge_alphas"]:
        LOG.info("Fitting Ridge baseline: alpha=%s", alpha)
        model = fit_linear(train["X"], y, "ridge", alpha)
        score = float(mean_absolute_error(validation_y, model.predict(validation["X"])))
        record = {
            "model": "ridge",
            "alpha": alpha,
            "validation_metric": "mae",
            "validation_score": score,
        }
        trials.append(record)
        if score < best:
            best = score
            payloads["ridge"] = linear_payload(model, "ridge", FEATURE_COLUMNS)
            payloads["ridge"]["residual_quantiles"] = np.quantile(
                y - model.predict(train["X"]), ALPHAS
            ).tolist()
            selected["ridge"] = record
    best = float("inf")
    for c in config["logistic_cs"]:
        LOG.info("Fitting logistic baseline: C=%s", c)
        model = fit_linear(train["X"], labels, "logistic", c)
        score = float(
            log_loss(validation_labels, model.predict_proba(validation["X"]), labels=[0, 1, 2])
        )
        record = {
            "model": "logistic",
            "C": c,
            "validation_metric": "log_loss",
            "validation_score": score,
        }
        trials.append(record)
        if score < best:
            best, payloads["logistic"], selected["logistic"] = (
                score,
                linear_payload(model, "logistic", FEATURE_COLUMNS),
                record,
            )
    best = float("inf")
    for order in config["arima_orders"]:
        LOG.info("Fitting per-stock ARIMA(%s,0,0) on log returns", order)
        try:
            model = fit_arima_bundle(train["X"], train["index"], order, config, session_positions)
            prediction = predict_arima(model, validation["X"], validation["index"])
            score = float(mean_absolute_error(validation_y, prediction[:, 1]))
            record = {
                "model": "arima",
                "order": [order, 0, 0],
                "validation_metric": "mae",
                "validation_score": score,
                "stocks": len(model["parameters"]),
            }
            if score < best:
                best, payloads["arima"], selected["arima"] = score, model, record
        except PipelineError as exc:
            record = {
                "model": "arima",
                "order": [order, 0, 0],
                "status": "failed",
                "reason": str(exc),
            }
        trials.append(record)
    if "arima" not in payloads:
        raise PipelineError(
            "All declared ARIMA candidates failed; statistical benchmark incomplete"
        )
    return payloads, selected, trials


def evaluate(block, predictions, detailed=True):
    y = block["y"].y_price_5d.to_numpy()
    labels = encode_labels(block["y"].y_direction_5d)
    result = {
        "regression": {
            name: regression_metrics(y, predictions[name])
            for name in ("lightgbm", "ridge", "arima", "naive")
        },
        "classification": {
            name: classification_metrics(labels, predictions[name])
            for name in ("lightgbm_direction", "logistic", "naive_direction")
        },
        "raw_quantile_crossing_fraction": predictions["raw_quantile_crossing_fraction"],
    }
    if detailed:
        result["lightgbm_by_stock"] = grouped_regression(block["index"], y, predictions["lightgbm"])
        result["lightgbm_direction_by_stock"] = {
            str(symbol): classification_metrics(
                labels[positions], predictions["lightgbm_direction"][positions]
            )
            for symbol, positions in block["index"].groupby("symbol").indices.items()
        }
        result["lightgbm_by_year"] = grouped_regression(
            block["index"].assign(year=block["index"].session_date.str[:4]),
            y,
            predictions["lightgbm"],
            "year",
        )
    return result


def prediction_table(block, predictions, labels=True):
    table = block["index"].copy()
    if labels:
        table = pd.concat([table, block["y"]], axis=1)
    for name in ("lightgbm", "ridge", "arima", "naive"):
        for i, q in enumerate(("q10", "q50", "q90")):
            table[f"{name}_{q}"] = predictions[name][:, i]
    for name in ("lightgbm_direction", "logistic", "naive_direction"):
        for i, cls in enumerate(CLASSES):
            table[f"{name}_p_{cls.lower()}"] = predictions[name][:, i]
    return table


def walk_forward(train, config, archive):
    results, refs = [], {}
    for i, fold in enumerate(config["walk_forward"]):
        LOG.info(
            "Walk-forward fold %d: evaluate %s to %s",
            i + 1,
            fold["evaluation_start"],
            fold["evaluation_end_exclusive"],
        )
        inner_train, inner_valid, outer_train, outer_eval = walk_forward_blocks(train, fold)
        _, _, selected, _ = select_primary(
            inner_train, inner_valid, config, [config["lightgbm_candidates"][0]]
        )
        q_models = [
            fit_booster(
                outer_train["X"],
                outer_train["y"].y_price_5d,
                None,
                None,
                selected[name]["candidate"],
                config,
                alpha=float(alpha),
                rounds=selected[name]["best_iteration"],
            )
            for alpha, name in zip(ALPHAS, ("q10", "q50", "q90"), strict=True)
        ]
        classifier = fit_booster(
            outer_train["X"],
            encode_labels(outer_train["y"].y_direction_5d),
            None,
            None,
            selected["direction"]["candidate"],
            config,
            rounds=selected["direction"]["best_iteration"],
        )
        q = repair_quantiles(
            np.column_stack(
                [m.predict(outer_eval["X"], num_threads=config["threads"]) for m in q_models]
            )
        )
        p = classifier.predict(outer_eval["X"], num_threads=config["threads"])
        naive_q = np.quantile(outer_train["y"].y_price_5d, ALPHAS)
        naive_q[1] = 0.0
        base_q = np.tile(naive_q, (len(outer_eval["X"]), 1))
        counts = np.bincount(encode_labels(outer_train["y"].y_direction_5d), minlength=3)
        base_p = np.tile(counts / counts.sum(), (len(outer_eval["X"]), 1))
        for j, model in enumerate(q_models):
            refs[f"fold_{i + 1}_q{int(ALPHAS[j] * 100)}"] = save_booster(
                archive, f"walk-forward/{i + 1}/q{int(ALPHAS[j] * 100)}", model
            )
        refs[f"fold_{i + 1}_direction"] = save_booster(
            archive, f"walk-forward/{i + 1}/direction", classifier
        )
        results.append(
            {
                "fold": fold,
                "inner_training_rows": len(inner_train["X"]),
                "inner_validation_rows": len(inner_valid["X"]),
                "training_rows": len(outer_train["X"]),
                "evaluation_rows": len(outer_eval["X"]),
                "selected_iterations": selected,
                "lightgbm": regression_metrics(outer_eval["y"].y_price_5d, q),
                "naive": regression_metrics(outer_eval["y"].y_price_5d, base_q),
                "direction": classification_metrics(
                    encode_labels(outer_eval["y"].y_direction_5d), p
                ),
                "naive_direction": classification_metrics(
                    encode_labels(outer_eval["y"].y_direction_5d), base_p
                ),
                "calibrated": False,
                "selection_policy": (
                    "Predeclared compact candidate; stop on earlier inner validation, "
                    "then refit on outer training."
                ),
            }
        )
    return results, refs


def gate_report(validation, test, config, dataset_production_ready):
    improvement = 1 - config["minimum_relative_improvement"]
    checks = {}
    for name, metrics in (("validation", validation), ("test", test)):
        regression, classification = metrics["regression"], metrics["classification"]
        checks[name + "_pinball_beats_baselines"] = regression["lightgbm"][
            "mean_pinball"
        ] <= improvement * min(regression[b]["mean_pinball"] for b in ("naive", "ridge", "arima"))
        checks[name + "_median_mae_beats_unchanged_price"] = (
            regression["lightgbm"]["mae"] <= regression["naive"]["mae"]
        )
        checks[name + "_log_loss_beats_baselines"] = classification["lightgbm_direction"][
            "log_loss"
        ] <= improvement * min(
            classification[b]["log_loss"] for b in ("logistic", "naive_direction")
        )
        checks[name + "_brier_beats_prior"] = (
            classification["lightgbm_direction"]["brier_score"]
            <= classification["naive_direction"]["brier_score"]
        )
    checks["test_interval_coverage"] = (
        abs(test["regression"]["lightgbm"]["interval_coverage"] - config["nominal_coverage"])
        <= config["coverage_tolerance"]
    )
    return {
        "checks": {k: bool(v) for k, v in checks.items()},
        "research_model_gate_passed": bool(all(checks.values())),
        "dataset_production_ready": bool(dataset_production_ready),
        "production_approved": False,
        "promotion_status": "research_only_no_automatic_promotion",
        "failed_checks": [k for k, v in checks.items() if not v],
    }


def _verify_refs(archive, refs):
    for ref in refs.values():
        if archive.read(ref["key"], sha256=ref["sha256"]) is None:
            raise PipelineError("Training checkpoint references a missing artifact")


def holdout_lease(archive, key, recipe_hash):
    saved = archive.read(key)
    if saved and json.loads(saved)["recipe_hash"] != recipe_hash:
        raise PipelineError(
            "Holdout is already bound to another frozen recipe; "
            "perform new model selection on earlier data"
        )


def run_training(settings, archive, dataset_ref, config, summary_dir):
    validate_config(config)
    archive.check_access()
    body = archive.read(dataset_ref["key"], sha256=dataset_ref.get("sha256"))
    if body is None:
        raise PipelineError("Prepared dataset manifest is missing")
    dataset_ref = {"key": dataset_ref["key"], "sha256": digest(body)}
    dataset = json.loads(body)
    dataset_report = read_json(archive, dataset["report"])
    if (
        not dataset_report.get("research_training_ready")
        or dataset["recipe"]["feature_columns"] != FEATURE_COLUMNS
    ):
        raise PipelineError("Dataset has not passed the shared preparation checks")
    dependencies = {
        p: version(p)
        for p in ("lightgbm", "scikit-learn", "statsmodels", "scipy", "numpy", "pandas", "pyarrow")
    }
    recipe = {
        "dataset_manifest": dataset_ref,
        "dataset_version": dataset["dataset_version"],
        "config": config,
        "feature_columns": FEATURE_COLUMNS,
        "code_sha256": code_hash(),
        "dependencies": dependencies,
    }
    recipe_hash = digest(json_bytes(recipe))
    checkpoint_key = f"checkpoints/training/{recipe_hash}.json"
    cached = archive.read(checkpoint_key)
    if cached:
        summary = json.loads(cached)
        report = read_json(archive, summary["report"])
        _verify_refs(archive, report["artifacts"])
        read_json(archive, summary["manifest"])
        _write_summary(summary_dir, summary, report)
        LOG.info("Reusing complete checksummed training run; no fitting or test reevaluation")
        return {**summary, "cache_reused": True}
    outputs = dataset["outputs"]
    holdout_hash = digest(
        json_bytes({k: outputs[k]["sha256"] for k in ("X_test", "y_test", "index_test")})
    )
    holdout_key = f"checkpoints/training/holdout/{holdout_hash}.json"
    holdout_lease(archive, holdout_key, recipe_hash)
    dc = dataset["recipe"]["config"]
    validation_start, calibration_start, test_start = (
        dc[k] for k in ("validation_start", "calibration_start", "test_start")
    )
    LOG.info("Loading only training, validation and calibration; test remains unopened")
    blocks = {
        "train": load_block(archive, outputs, "train", (None, validation_start)),
        "validation": load_block(
            archive, outputs, "validation", (validation_start, calibration_start)
        ),
        "calibration": load_block(archive, outputs, "calibration", (calibration_start, test_start)),
    }
    for earlier, later in (("train", "validation"), ("validation", "calibration")):
        if set(blocks[earlier]["index"].row_id) & set(blocks[later]["index"].row_id):
            raise PipelineError("Duplicate example across chronological training blocks")
    selection_key = f"checkpoints/training/selection/{recipe_hash}.json"
    saved = archive.read(selection_key)
    if saved:
        selection = json.loads(saved)
        _verify_refs(archive, selection["artifacts"])
        LOG.info("Resuming the frozen selection/calibration checkpoint")
    else:
        with threadpool_limits(limits=config["threads"]):
            folds, artifacts = walk_forward(blocks["train"], config, archive)
            quantiles, classifier, selected, trials = select_primary(
                blocks["train"], blocks["validation"], config
            )
            sessions = read_table(archive, dataset["recipe"]["inputs"]["sessions"])
            session_positions = dict(zip(sessions.session_date, range(len(sessions)), strict=True))
            baselines, baseline_selected, baseline_trials = select_baselines(
                blocks["train"], blocks["validation"], config, session_positions
            )
        refs = {
            f"lightgbm_q{int(alpha * 100)}": save_booster(
                archive, f"daily/q{int(alpha * 100)}", model
            )
            for alpha, model in zip(ALPHAS, quantiles, strict=True)
        }
        refs["lightgbm_direction"] = save_booster(archive, "daily/direction", classifier)
        refs.update(
            {
                name: save_json_model(archive, f"daily/{name}", payload)
                for name, payload in baselines.items()
            }
        )
        artifacts.update(refs)
        bundle_spec = {
            "bundle_version": config["bundle_version"],
            "feature_columns": FEATURE_COLUMNS,
            "feature_version": dc["feature_version"],
            "class_order": CLASSES,
            "horizon_sessions": 5,
            "models": refs,
            "dataset_version": dataset["dataset_version"],
            "production_ready": False,
            "interval_calibration": {},
            "probability_calibration": {},
        }
        bundle = ResearchBundle(archive, bundle_spec)
        validation_predictions = bundle.predict(
            blocks["validation"]["X"], blocks["validation"]["index"], calibrated=False
        )
        calibration_predictions = bundle.predict(
            blocks["calibration"]["X"], blocks["calibration"]["index"], calibrated=False
        )
        # Compare reloaded model artifacts against the fitted objects before any test access.
        expected_q = repair_quantiles(
            np.column_stack(
                [
                    m.predict(blocks["validation"]["X"], num_threads=config["threads"])
                    for m in quantiles
                ]
            )
        )
        if not np.allclose(
            expected_q, validation_predictions["lightgbm"], atol=1e-12
        ) or not np.allclose(
            classifier.predict(blocks["validation"]["X"], num_threads=config["threads"]),
            validation_predictions["lightgbm_direction"],
            atol=1e-12,
        ):
            raise PipelineError("Reloaded artifacts do not reproduce fitted model predictions")
        for name in ("lightgbm", "ridge", "arima", "naive"):
            bundle_spec["interval_calibration"][name] = fit_interval_adjustment(
                blocks["calibration"]["y"].y_price_5d,
                calibration_predictions[name],
                config["nominal_coverage"],
            )
        for name in ("lightgbm_direction", "logistic", "naive_direction"):
            bundle_spec["probability_calibration"][name] = fit_temperature(
                encode_labels(blocks["calibration"]["y"].y_direction_5d),
                calibration_predictions[name],
                config["temperature_bounds"],
            )
        calibrated = bundle.predict(blocks["calibration"]["X"], blocks["calibration"]["index"])
        validation_report, calibration_report = (
            evaluate(blocks["validation"], validation_predictions),
            evaluate(blocks["calibration"], calibrated),
        )
        artifacts["predictions_validation"] = parquet(
            archive,
            "training/predictions_validation",
            prediction_table(blocks["validation"], validation_predictions),
        )
        artifacts["predictions_calibration"] = parquet(
            archive,
            "training/predictions_calibration",
            prediction_table(blocks["calibration"], calibrated),
        )
        selection = {
            "recipe_hash": recipe_hash,
            "bundle": bundle_spec,
            "selected": {**selected, **baseline_selected},
            "trials": trials + baseline_trials,
            "walk_forward": folds,
            "artifacts": artifacts,
            "validation": validation_report,
            "calibration": calibration_report,
            "calibration_metrics_are_in_sample": True,
            "artifact_reload_verified": True,
        }
        archive.json(selection_key, selection)
    bundle = ResearchBundle(archive, selection["bundle"])
    holdout_lease(archive, holdout_key, recipe_hash)
    archive.json(
        holdout_key,
        {
            "recipe_hash": recipe_hash,
            "selection_sha256": digest(json_bytes(selection)),
            "policy": "Test is bound to the frozen recipe; no tuning based on its results.",
        },
    )
    LOG.info("Selection and calibration frozen; opening the final test block once")
    test = load_block(archive, outputs, "test", (test_start, None))
    if set(test["index"].row_id) & set(pd.concat([b["index"] for b in blocks.values()]).row_id):
        raise PipelineError("Duplicate fitting example in final holdout")
    predictions = bundle.predict(test["X"], test["index"])
    test_report = evaluate(test, predictions)
    uncertainty = block_bootstrap_differences(
        test["index"],
        test["y"].y_price_5d,
        encode_labels(test["y"].y_direction_5d),
        predictions["lightgbm"],
        predictions["naive"],
        predictions["lightgbm_direction"],
        predictions["naive_direction"],
        config,
    )
    artifacts = dict(selection["artifacts"])
    artifacts["predictions_test"] = parquet(
        archive, "training/predictions_test", prediction_table(test, predictions)
    )
    recent = read_table(archive, outputs["inference_features"])
    inference_rows = 0
    if not recent.empty:
        latest = recent[recent.session_date.eq(recent.session_date.max())].reset_index(drop=True)
        latest_index = latest[
            ["row_id", "instrument_id", "symbol", "session_date", "cutoff_at"]
        ].copy()
        future = schedule(
            date.fromisoformat(latest.session_date.min()),
            date.fromisoformat(latest.session_date.max()) + timedelta(days=21),
        ).session_date.tolist()
        latest_index["target_session"] = [future[future.index(d) + 5] for d in latest.session_date]
        forecast = prediction_table(
            {"index": latest_index},
            bundle.predict(latest[FEATURE_COLUMNS], latest_index),
            labels=False,
        )
        forecast["forecast_status"] = "research_archived_cutoff_not_live"
        forecast["production_approved"] = False
        artifacts["research_predictions_latest"] = parquet(
            archive, "training/research_predictions_latest", forecast
        )
        inference_rows = len(forecast)
    gates = gate_report(
        selection["validation"], test_report, config, dataset_report["production_ready"]
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    result = {
        "status": "trained_research_candidate",
        "recipe_hash": recipe_hash,
        "recipe": recipe,
        "dataset_version": dataset["dataset_version"],
        "feature_count": len(FEATURE_COLUMNS),
        "models": selection["bundle"]["models"],
        "bundle": selection["bundle"],
        "artifacts": artifacts,
        "selected": selection["selected"],
        "trials": selection["trials"],
        "walk_forward": selection["walk_forward"],
        "validation": selection["validation"],
        "calibration": selection["calibration"],
        "calibration_metrics_are_in_sample": True,
        "test": test_report,
        "test_block_uncertainty": uncertainty,
        "gates": gates,
        "artifact_reload_verified": selection["artifact_reload_verified"],
        "training_rows": len(blocks["train"]["X"]),
        "test_rows": len(test["X"]),
        "latest_research_prediction_rows": inference_rows,
        "production_approved": False,
        "limitations": dataset_report["limitations"]
        + [
            "Final test is bound to this frozen recipe; no tuning after holdout evaluation.",
            "The latest research predictions use archived cutoffs, not refreshed live inputs.",
            "Calibration-block metrics are fitting diagnostics, not out-of-sample performance.",
            "Date-block bootstrap does not remove survivorship or availability-replay bias.",
        ],
    }
    report_ref = archive.json(f"reports/training/{run_id}.json", result)
    model_manifest = archive.json(
        f"manifests/training/{run_id}.json",
        {
            "status": result["status"],
            "recipe_hash": recipe_hash,
            "recipe": recipe,
            "bundle": selection["bundle"],
            "report": report_ref,
            "artifacts": artifacts,
            "production_approved": False,
        },
    )
    summary = {
        k: result[k]
        for k in (
            "status",
            "recipe_hash",
            "dataset_version",
            "feature_count",
            "training_rows",
            "test_rows",
            "latest_research_prediction_rows",
            "production_approved",
            "gates",
        )
    }
    summary.update(
        manifest=model_manifest,
        report=report_ref,
        test_regression=test_report["regression"],
        test_direction={
            name: {
                k: metrics[k] for k in ("log_loss", "brier_score", "accuracy", "balanced_accuracy")
            }
            for name, metrics in test_report["classification"].items()
        },
    )
    archive.json(checkpoint_key, summary)
    _write_summary(summary_dir, summary, result)
    return summary


def _write_summary(summary_dir, summary, report):
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary", summary), ("report", report)):
        (summary_dir / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def training_main(args):
    try:
        settings = Settings.load(args.env_file, upload=not args.no_upload)
        archive = Archive(args.archive_root, None if args.no_upload else settings)
        ref = (
            {"key": args.manifest}
            if args.manifest
            else json.loads(args.manifest_from_summary.read_text(encoding="utf-8"))["manifest"]
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        print(json.dumps(run_training(settings, archive, ref, config, args.summary_dir), indent=2))
        return 0
    except Exception as exc:
        message = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / "summary.json").write_text(
            json.dumps({"status": "failed", "production_approved": False, "reason": message}),
            encoding="utf-8",
        )
        LOG.error("Training failed: %s", message)
        return 1
