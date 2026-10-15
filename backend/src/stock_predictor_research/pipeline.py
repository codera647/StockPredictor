import argparse
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from stock_predictor.config import PipelineError, Settings
from stock_predictor.datasets.features import FEATURE_COLUMNS
from stock_predictor.datasets.pipeline import code_hash, read_table
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.storage.archive import Archive, digest, json_bytes
from stock_predictor.training.bundle import save_json_model
from stock_predictor.training.calibration import apply_temperature, fit_temperature
from stock_predictor.training.data import load_block, subset, walk_forward_blocks
from stock_predictor.training.estimators import fit_linear, predict_linear
from stock_predictor.training.metrics import CLASSES, classification_metrics, encode_labels
from stock_predictor_challengers.models import save_model
from threadpoolctl import threadpool_limits

from stock_predictor_research.macro import RESEARCH_FEATURES, augment, macro_features
from stock_predictor_research.models import fit_components, mix, predictions, select_ensemble
from stock_predictor_research.selective import risk_coverage, select_rule

LOG = logging.getLogger(__name__)


def source_hash():
    return digest(
        json_bytes(
            {
                p.name: digest(p.read_text(encoding="utf-8").encode())
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        )
    )


def validate_config(config):
    if (
        config.get("schema_version") != 1
        or config.get("evaluate_existing_test") is not False
        or config.get("production_ready") is not False
    ):
        raise PipelineError(
            "Accuracy research must preserve the existing final test and stay research-only"
        )
    if config.get("target_accuracy") != 0.9:
        raise PipelineError("This version has a frozen 90% accuracy target")
    if not config.get("walk_forward"):
        raise PipelineError("Earlier walk-forward evaluations are required")
    for key in (
        "threads",
        "max_rounds",
        "neural_max_epochs",
        "minimum_selective_examples",
        "minimum_selective_dates",
        "bootstrap_repetitions",
    ):
        if not isinstance(config[key], int) or config[key] < 1:
            raise PipelineError("Accuracy research budgets and evidence minimums must be positive")


def save_components(archive, components, name):
    return {
        "logistic": save_json_model(archive, f"research/{name}/logistic", components["logistic"]),
        "neural": save_json_model(archive, f"research/{name}/neural", components["neural"]),
        "catboost": save_model(archive, "catboost", f"research-{name}", components["catboost"]),
    }


def load_components(archive, refs):
    tree_ref = refs["catboost"]
    tree_body = archive.read(tree_ref["key"], sha256=tree_ref["sha256"])
    if tree_body is None:
        raise PipelineError("Macro tree artifact is missing")
    tree = CatBoostClassifier()
    tree.load_model(blob=tree_body, format="cbm")
    if tree.feature_names_ != RESEARCH_FEATURES:
        raise PipelineError("Macro model schema does not match its native artifact")
    return {
        "logistic": read_json(archive, refs["logistic"]),
        "neural": read_json(archive, refs["neural"]),
        "catboost": tree,
    }


def summarize(labels, predicted):
    return {name: classification_metrics(labels, p) for name, p in predicted.items()}


def table(block, predicted):
    result = pd.concat([block["index"], block["y"][["y_direction_5d"]]], axis=1)
    for name, p in predicted.items():
        for i, cls in enumerate(CLASSES):
            result[f"{name}_p_{cls.lower()}"] = p[:, i]
    return result


def earlier_folds(base_train, macro, config, archive):
    result, artifacts = [], {}
    for i, fold in enumerate(config["walk_forward"]):
        LOG.info(
            "Macro/neural fold: %s through %s",
            fold["evaluation_start"],
            fold["evaluation_end_exclusive"],
        )
        original_inner, original_valid, _, original_eval = walk_forward_blocks(base_train, fold)
        train, valid, evaluation = (
            augment(b, macro) for b in (original_inner, original_valid, original_eval)
        )
        # We deliberately retain the earlier inner-trained weights. The later
        # period selects only settings/epochs/mixture, never fold evaluation.
        components, selected = fit_components(train, valid, config)
        valid_predictions = predictions(components, valid["X"], config["threads"])
        weights, _ = select_ensemble(
            valid_predictions, encode_labels(valid["y"].y_direction_5d), config["ensemble_weights"]
        )
        evaluated = predictions(components, evaluation["X"], config["threads"])
        evaluated["ensemble"] = mix(evaluated, weights)
        base_model = fit_linear(
            original_inner["X"], encode_labels(original_inner["y"].y_direction_5d), "logistic", 0.1
        )
        evaluated["price_only_logistic"] = base_model.predict_proba(original_eval["X"])
        labels = encode_labels(evaluation["y"].y_direction_5d)
        refs = save_components(archive, components, f"fold-{i + 1}")
        artifacts.update({f"fold_{i + 1}_{name}": ref for name, ref in refs.items()})
        result.append(
            {
                "fold": fold,
                "training_rows": len(train["X"]),
                "tuning_rows": len(valid["X"]),
                "evaluation_rows": len(evaluation["X"]),
                "metrics": summarize(labels, evaluated),
                "ensemble_weights": weights,
                "selected": selected,
                "risk_coverage_diagnostic": risk_coverage(
                    labels,
                    evaluated["ensemble"],
                    evaluation["index"],
                    config,
                    evaluated["price_only_logistic"],
                ),
                "selection_policy": (
                    "Fit on earlier inner training; tune on inner validation; "
                    "evaluate later with frozen weights."
                ),
            }
        )
    return result, artifacts


def write_reports(directory, summary, report):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary", summary), ("report", report)):
        (directory / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_research(archive, dataset_ref, baseline_ref, config, directory):
    validate_config(config)
    archive.check_access()
    dataset = read_json(archive, dataset_ref)
    baseline = read_json(archive, baseline_ref)
    if dataset["dataset_version"] != baseline["bundle"]["dataset_version"]:
        raise PipelineError("Research and baseline datasets differ")
    if dataset["recipe"]["feature_columns"] != FEATURE_COLUMNS:
        raise PipelineError("Unsupported price feature schema")
    dataset_report = read_json(archive, dataset["report"])
    if not dataset_report["research_training_ready"]:
        raise PipelineError("Prepared source dataset is not research-ready")
    macro_ref = dataset["outputs"].get("clean_macro_observations")
    if macro_ref is None:
        raise PipelineError("The archived ALFRED vintage observations are required")
    recipe = {
        "dataset_manifest": dataset_ref,
        "baseline_manifest": baseline_ref,
        "macro_source": macro_ref,
        "config": config,
        "feature_columns": RESEARCH_FEATURES,
        "core_code_sha256": code_hash(),
        "research_code_sha256": source_hash(),
        "dependencies": {
            p: version(p) for p in ("catboost", "scikit-learn", "numpy", "pandas", "scipy")
        },
    }
    recipe_hash = digest(json_bytes(recipe))
    checkpoint = f"checkpoints/accuracy-research/{recipe_hash}.json"
    saved = archive.read(checkpoint)
    if saved:
        summary = json.loads(saved)
        report = read_json(archive, summary["report"])
        for ref in report["artifacts"].values():
            if archive.read(ref["key"], sha256=ref["sha256"]) is None:
                raise PipelineError("An accuracy-research artifact is missing")
        write_reports(directory, summary, report)
        return {**summary, "cache_reused": True}
    dc = dataset["recipe"]["config"]
    LOG.info("Replaying FRED releases and vintages; final test remains unopened")
    sessions = read_table(archive, dataset["recipe"]["inputs"]["sessions"])
    macro = macro_features(read_table(archive, macro_ref), sessions, config)
    artifacts = {"macro_features": parquet(archive, "accuracy-research/macro_features", macro)}
    train = load_block(archive, dataset["outputs"], "train", (None, dc["validation_start"]))
    validation = load_block(
        archive, dataset["outputs"], "validation", (dc["validation_start"], dc["calibration_start"])
    )
    calibration = load_block(
        archive, dataset["outputs"], "calibration", (dc["calibration_start"], dc["test_start"])
    )
    with threadpool_limits(limits=config["threads"]):
        folds, fold_artifacts = earlier_folds(train, macro, config, archive)
        artifacts.update(fold_artifacts)
        augmented_train, augmented_validation, augmented_calibration = (
            augment(b, macro) for b in (train, validation, calibration)
        )
        LOG.info("Fitting 58-feature macro logistic, tree and neural classifiers")
        components, selected = fit_components(augmented_train, augmented_validation, config)
        validation_predictions = predictions(
            components, augmented_validation["X"], config["threads"]
        )
        weights, score = select_ensemble(
            validation_predictions,
            encode_labels(validation["y"].y_direction_5d),
            config["ensemble_weights"],
        )
        validation_predictions["ensemble"] = mix(validation_predictions, weights)
    component_refs = save_components(archive, components, "daily")
    artifacts.update({"daily_" + name: ref for name, ref in component_refs.items()})
    reloaded = load_components(archive, component_refs)
    reload_predictions = predictions(reloaded, augmented_validation["X"])
    for name, expected in list(validation_predictions.items())[:3]:
        if not np.allclose(expected, reload_predictions[name], atol=1e-7):
            raise PipelineError("Reloaded macro ensemble components changed probabilities")
    baseline_payload = read_json(archive, baseline["bundle"]["models"]["logistic"])
    validation_predictions["price_only_logistic"] = predict_linear(
        baseline_payload, validation["X"]
    )
    metrics = summarize(encode_labels(validation["y"].y_direction_5d), validation_predictions)
    middle = config["calibration_split"]
    index = calibration["index"]
    fit_mask = index.session_date.lt(middle) & index.target_session.lt(middle)
    rule_mask = index.session_date.ge(middle)
    if not fit_mask.any() or not rule_mask.any():
        raise PipelineError(
            "Probability fit and confidence-rule periods must both contain examples"
        )
    probability_fit, rule_check = (subset(augmented_calibration, m) for m in (fit_mask, rule_mask))
    fit_p = mix(predictions(components, probability_fit["X"]), weights)
    temperature = fit_temperature(encode_labels(probability_fit["y"].y_direction_5d), fit_p)
    rule_p = apply_temperature(
        mix(predictions(components, rule_check["X"]), weights), temperature["temperature"]
    )
    baseline_check = predict_linear(baseline_payload, rule_check["X"][FEATURE_COLUMNS])
    curve = risk_coverage(
        encode_labels(rule_check["y"].y_direction_5d),
        rule_p,
        rule_check["index"],
        config,
        baseline_check,
    )
    rule = select_rule(curve, config["target_accuracy"])
    for name, block, predicted in (
        ("validation", augmented_validation, validation_predictions),
        ("selector", rule_check, {"ensemble": rule_p, "price_only_logistic": baseline_check}),
    ):
        artifacts[name + "_predictions"] = parquet(
            archive, "accuracy-research/" + name + "_predictions", table(block, predicted)
        )
    artifacts["research_X_train"] = parquet(
        archive, "accuracy-research/X_train", augmented_train["X"]
    )
    artifacts["research_X_validation"] = parquet(
        archive, "accuracy-research/X_validation", augmented_validation["X"]
    )
    bundle = {
        "feature_columns": RESEARCH_FEATURES,
        "macro_feature_version": config["macro_feature_version"],
        "class_order": CLASSES,
        "components": component_refs,
        "weights": weights,
        "temperature": temperature,
        "selective_rule": rule,
        "production_approved": False,
    }
    artifacts["research_bundle"] = save_json_model(archive, "research/accuracy-bundle", bundle)
    all_row_accuracy_met = metrics["ensemble"]["accuracy"] >= config["target_accuracy"]
    result = {
        "status": "accuracy_research_complete",
        "target_accuracy": config["target_accuracy"],
        "target_achieved": False,
        "all_row_validation_target_met": all_row_accuracy_met,
        "target_status": "not_established_on_fresh_holdout",
        "production_approved": False,
        "existing_test_evaluated": False,
        "recipe": recipe,
        "recipe_hash": recipe_hash,
        "feature_count": len(RESEARCH_FEATURES),
        "training_rows": len(train["X"]),
        "validation_rows": len(validation["X"]),
        "validation": metrics,
        "walk_forward": folds,
        "selected": selected,
        "ensemble_weights": weights,
        "ensemble_validation_log_loss": score,
        "probability_calibration": temperature,
        "probability_fit_rows": len(probability_fit["X"]),
        "selector_rows": len(rule_check["X"]),
        "selective_risk_coverage": curve,
        "selective_rule": rule,
        "macro_missingness": {
            n: float(augmented_train["X"][n].isna().mean()) for n in RESEARCH_FEATURES
        },
        "artifacts": artifacts,
        "bundle": bundle,
        "artifact_reload_verified": True,
        "limitations": [
            "Release availability uses conservative next-day ALFRED vintage reconstruction.",
            "Selection evidence requires a new future holdout for a performance claim.",
            "Confidence-filter accuracy includes sample counts and coverage; empty means unknown.",
            "Date-block lower bounds are approximate under financial dependence, not guarantees.",
            "Original labels are unchanged; the existing final test was not reopened.",
        ],
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_ref = archive.json(f"reports/accuracy-research/{run_id}.json", result)
    manifest = archive.json(
        f"manifests/accuracy-research/{run_id}.json",
        {"recipe": recipe, "report": report_ref, "artifacts": artifacts, "bundle": bundle},
    )
    summary = {
        k: result[k]
        for k in (
            "status",
            "target_accuracy",
            "target_achieved",
            "all_row_validation_target_met",
            "feature_count",
            "training_rows",
            "validation_rows",
            "ensemble_weights",
            "selective_rule",
            "existing_test_evaluated",
            "production_approved",
        )
    }
    summary["validation"] = {
        name: {k: m[k] for k in ("accuracy", "balanced_accuracy", "log_loss")}
        for name, m in metrics.items()
    }
    summary.update(manifest=manifest, report=report_ref)
    archive.json(checkpoint, summary)
    write_reports(directory, summary, result)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Research macro/neural ensemble accuracy with unchanged targets"
    )
    parser.add_argument("--manifest")
    parser.add_argument("--baseline-manifest")
    parser.add_argument("--config", type=Path, default=Path("configs/accuracy-research.daily.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    parser.add_argument("--summary-dir", type=Path, default=Path("artifacts/accuracy-research"))
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "botocore", "boto3", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        settings = Settings.load(args.env_file, upload=not args.no_upload)
        archive = Archive(args.archive_root, None if args.no_upload else settings)
        refs = []
        for key, file in (
            (args.manifest, "artifacts/dataset/summary.json"),
            (args.baseline_manifest, "artifacts/training/summary.json"),
        ):
            if key:
                body = archive.read(key)
                if body is None:
                    raise PipelineError("Requested research manifest is missing")
                refs.append({"key": key, "sha256": digest(body), "bytes": len(body)})
            else:
                refs.append(json.loads(Path(file).read_text(encoding="utf-8"))["manifest"])
        print(
            json.dumps(
                run_research(
                    archive,
                    *refs,
                    json.loads(args.config.read_text(encoding="utf-8")),
                    args.summary_dir,
                ),
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / "summary.json").write_text(
            json.dumps({"status": "failed", "reason": reason, "target_achieved": False}),
            encoding="utf-8",
        )
        LOG.error("Accuracy research failed: %s", reason)
        return 1
