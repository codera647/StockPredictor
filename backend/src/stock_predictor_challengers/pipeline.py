import argparse
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from stock_predictor.config import PipelineError, Settings
from stock_predictor.datasets.features import FEATURE_COLUMNS
from stock_predictor.datasets.pipeline import code_hash
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.storage.archive import Archive, digest, json_bytes
from stock_predictor.training.data import load_block, walk_forward_blocks
from stock_predictor.training.estimators import fit_linear, predict_linear
from stock_predictor.training.metrics import CLASSES, classification_metrics, encode_labels
from threadpoolctl import threadpool_limits

from stock_predictor_challengers.models import fit_model, load_model, predict_model, save_model

LOG = logging.getLogger(__name__)
FAMILIES = ("xgboost", "catboost")


def validate_config(config):
    if (
        config.get("schema_version") != 1
        or config.get("evaluate_existing_test") is not False
        or config.get("production_ready") is not False
    ):
        raise PipelineError("Challengers must exclude the existing test and remain research-only")
    if config.get("selection_metric") != "log_loss":
        raise PipelineError("The frozen selection metric must remain log loss")
    for key in ("threads", "max_rounds", "early_stopping_rounds"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise PipelineError("Challenger budgets must be positive")
    for family in FAMILIES:
        if not 1 <= len(config[family + "_candidates"]) <= 4:
            raise PipelineError("Challenger search must be bounded")


def select(family, train, validation, config):
    best, selected, trials = None, None, []
    for candidate in config[family + "_candidates"]:
        LOG.info("Fitting %s %s", family, candidate["name"])
        model, rounds = fit_model(
            family,
            train["X"],
            encode_labels(train["y"].y_direction_5d),
            validation["X"],
            encode_labels(validation["y"].y_direction_5d),
            candidate,
            config,
        )
        probabilities = predict_model(family, model, validation["X"], config["threads"])
        score = float(
            log_loss(encode_labels(validation["y"].y_direction_5d), probabilities, labels=[0, 1, 2])
        )
        trial = {
            "family": family,
            "candidate": candidate,
            "rounds": rounds,
            "validation_log_loss": score,
        }
        trials.append(trial)
        if selected is None or score < selected["validation_log_loss"]:
            best, selected = model, trial
    return best, selected, trials


def select_logistic(train, validation, config):
    best, score, chosen = None, float("inf"), None
    for c in config["logistic_cs"]:
        model = fit_linear(train["X"], encode_labels(train["y"].y_direction_5d), "logistic", c)
        loss = log_loss(
            encode_labels(validation["y"].y_direction_5d),
            model.predict_proba(validation["X"]),
            labels=[0, 1, 2],
        )
        if loss < score:
            best, score, chosen = model, loss, c
    return best, chosen


def fold_comparisons(train, config, archive):
    results, artifacts = [], {}
    for i, fold in enumerate(config["walk_forward"]):
        LOG.info(
            "Nested earlier fold %s through %s",
            fold["evaluation_start"],
            fold["evaluation_end_exclusive"],
        )
        inner_train, inner_valid, outer_train, evaluation = walk_forward_blocks(train, fold)
        labels = encode_labels(evaluation["y"].y_direction_5d)
        metrics, selection = {}, {}
        for family in FAMILIES:
            _, chosen, _ = select(family, inner_train, inner_valid, config)
            model, _ = fit_model(
                family,
                outer_train["X"],
                encode_labels(outer_train["y"].y_direction_5d),
                None,
                None,
                chosen["candidate"],
                config,
                rounds=chosen["rounds"],
            )
            probabilities = predict_model(family, model, evaluation["X"], config["threads"])
            artifacts[f"fold_{i + 1}_{family}"] = save_model(
                archive, family, f"fold-{i + 1}", model
            )
            metrics[family] = classification_metrics(labels, probabilities)
            selection[family] = chosen
        _, c = select_logistic(inner_train, inner_valid, config)
        logistic = fit_linear(
            outer_train["X"], encode_labels(outer_train["y"].y_direction_5d), "logistic", c
        )
        metrics["logistic"] = classification_metrics(
            labels, logistic.predict_proba(evaluation["X"])
        )
        counts = np.bincount(encode_labels(outer_train["y"].y_direction_5d), minlength=3)
        metrics["prior"] = classification_metrics(
            labels, np.tile(counts / counts.sum(), (len(labels), 1))
        )
        results.append(
            {
                "fold": fold,
                "training_rows": len(outer_train["X"]),
                "evaluation_rows": len(labels),
                "metrics": metrics,
                "inner_selection": selection,
                "logistic_C": c,
            }
        )
    return results, artifacts


def decide(validation_metrics, folds, config):
    baseline = validation_metrics["logistic"]
    factor = 1 - config["minimum_relative_improvement"]
    best = min(FAMILIES, key=lambda n: validation_metrics[n]["log_loss"])
    candidate = validation_metrics[best]
    weights = np.array([fold["evaluation_rows"] for fold in folds])

    def aggregate(name, metric):
        return float(np.average([f["metrics"][name][metric] for f in folds], weights=weights))

    checks = {
        "validation_log_loss_improves_logistic": candidate["log_loss"]
        <= factor * baseline["log_loss"],
        "validation_balanced_accuracy_not_worse": candidate["balanced_accuracy"]
        >= baseline["balanced_accuracy"],
        "validation_accuracy_not_worse": candidate["accuracy"] >= baseline["accuracy"],
        "walk_forward_log_loss_improves_logistic": aggregate(best, "log_loss")
        <= factor * aggregate("logistic", "log_loss"),
        "walk_forward_balanced_accuracy_not_worse": aggregate(best, "balanced_accuracy")
        >= aggregate("logistic", "balanced_accuracy"),
    }
    accepted = all(checks.values())
    return {
        "best_challenger_by_validation_log_loss": best,
        "research_switch_eligible": accepted,
        "recommended_direction_family": best if accepted else "logistic",
        "checks": {k: bool(v) for k, v in checks.items()},
        "production_approved": False,
        "existing_test_evaluated": False,
        "fresh_future_holdout_required": True,
        "weighted_walk_forward": {
            name: {
                metric: aggregate(name, metric)
                for metric in ("log_loss", "accuracy", "balanced_accuracy")
            }
            for name in (*FAMILIES, "logistic", "prior")
        },
    }


def source_hash():
    return digest(
        json_bytes(
            {
                p.name: digest(p.read_text(encoding="utf-8").encode())
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        )
    )


def write_reports(directory, summary, report):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary", summary), ("report", report)):
        (directory / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_comparison(archive, dataset_ref, baseline_ref, config, summary_dir):
    validate_config(config)
    archive.check_access()
    dataset = read_json(archive, dataset_ref)
    baseline = read_json(archive, baseline_ref)
    if dataset["dataset_version"] != baseline["bundle"]["dataset_version"]:
        raise PipelineError("Baseline and challenger dataset versions must match")
    recipe = {
        "dataset_manifest": dataset_ref,
        "baseline_manifest": baseline_ref,
        "config": config,
        "feature_columns": FEATURE_COLUMNS,
        "class_order": CLASSES,
        "core_code_sha256": code_hash(),
        "challenger_code_sha256": source_hash(),
        "dependencies": {
            p: version(p) for p in ("xgboost-cpu", "catboost", "numpy", "scikit-learn")
        },
    }
    recipe_hash = digest(json_bytes(recipe))
    key = f"checkpoints/challengers/{recipe_hash}.json"
    cached = archive.read(key)
    if cached:
        summary = json.loads(cached)
        report = read_json(archive, summary["report"])
        for ref in report["artifacts"].values():
            if archive.read(ref["key"], sha256=ref["sha256"]) is None:
                raise PipelineError("Cached challenger artifact is missing")
        write_reports(summary_dir, summary, report)
        return {**summary, "cache_reused": True}
    dc = dataset["recipe"]["config"]
    outputs = dataset["outputs"]
    LOG.info("Loading train and validation only; calibration and final test stay closed")
    train = load_block(archive, outputs, "train", (None, dc["validation_start"]))
    validation = load_block(
        archive, outputs, "validation", (dc["validation_start"], dc["calibration_start"])
    )
    with threadpool_limits(limits=config["threads"]):
        folds, artifacts = fold_comparisons(train, config, archive)
        models, selected, trials, probabilities, metrics = {}, {}, [], {}, {}
        labels = encode_labels(validation["y"].y_direction_5d)
        for family in FAMILIES:
            models[family], selected[family], records = select(family, train, validation, config)
            trials.extend(records)
            artifacts[family] = save_model(archive, family, "validation-selected", models[family])
            probabilities[family] = predict_model(
                family, models[family], validation["X"], config["threads"]
            )
            reloaded = load_model(archive, family, artifacts[family])
            if not np.allclose(
                probabilities[family], predict_model(family, reloaded, validation["X"]), atol=1e-7
            ):
                raise PipelineError("Reloaded challenger predictions differ from fitted model")
            metrics[family] = classification_metrics(labels, probabilities[family])
    logistic_payload = read_json(archive, baseline["bundle"]["models"]["logistic"])
    probabilities["logistic"] = predict_linear(logistic_payload, validation["X"])
    direction_ref = baseline["bundle"]["models"]["lightgbm_direction"]
    direction_body = archive.read(direction_ref["key"], sha256=direction_ref["sha256"])
    if direction_body is None:
        raise PipelineError("Existing direction baseline artifact is missing")
    direction = lgb.Booster(model_str=direction_body.decode())
    probabilities["lightgbm"] = direction.predict(validation["X"], num_threads=1)
    prior = np.bincount(encode_labels(train["y"].y_direction_5d), minlength=3)
    probabilities["prior"] = np.tile(prior / prior.sum(), (len(labels), 1))
    for name in ("logistic", "lightgbm", "prior"):
        metrics[name] = classification_metrics(labels, probabilities[name])
    predictions = pd.concat([validation["index"], validation["y"][["y_direction_5d"]]], axis=1)
    for name, p in probabilities.items():
        for i, cls in enumerate(CLASSES):
            predictions[f"{name}_p_{cls.lower()}"] = p[:, i]
    artifacts["validation_predictions"] = parquet(
        archive, "challengers/validation_predictions", predictions
    )
    decision = decide(metrics, folds, config)
    report = {
        "status": "direction_challengers_compared",
        "recipe": recipe,
        "recipe_hash": recipe_hash,
        "dataset_version": dataset["dataset_version"],
        "training_rows": len(train["X"]),
        "validation_rows": len(labels),
        "validation": metrics,
        "selected": selected,
        "trials": trials,
        "walk_forward": folds,
        "decision": decision,
        "artifacts": artifacts,
        "artifact_reload_verified": True,
        "existing_test_evaluated": False,
        "calibration_evaluated": False,
        "production_approved": False,
        "limitations": [
            "Validation accuracy is selection evidence, not new untouched-test evidence.",
            "A new future holdout is required to claim improved final-test performance.",
            "The dataset retains survivorship and reconstructed-availability limitations.",
            "Original return models, final test binding and model artifacts remain unchanged.",
        ],
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_ref = archive.json(f"reports/challengers/{run_id}.json", report)
    manifest = archive.json(
        f"manifests/challengers/{run_id}.json",
        {
            "recipe": recipe,
            "report": report_ref,
            "models": {family: artifacts[family] for family in FAMILIES},
            "artifacts": artifacts,
            "decision": decision,
        },
    )
    summary = {
        "status": report["status"],
        "recipe_hash": recipe_hash,
        "decision": decision,
        "validation": {
            name: {k: m[k] for k in ("accuracy", "balanced_accuracy", "log_loss", "brier_score")}
            for name, m in metrics.items()
        },
        "manifest": manifest,
        "report": report_ref,
        "existing_test_evaluated": False,
    }
    archive.json(key, summary)
    write_reports(summary_dir, summary, report)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare CPU direction challengers on earlier data"
    )
    parser.add_argument("--manifest", help="Private prepared dataset manifest key")
    parser.add_argument("--baseline-manifest", help="Private original training manifest key")
    parser.add_argument("--config", type=Path, default=Path("configs/challengers.daily.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    parser.add_argument("--summary-dir", type=Path, default=Path("artifacts/challengers"))
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "botocore", "boto3", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        settings = Settings.load(args.env_file, upload=not args.no_upload)
        archive = Archive(args.archive_root, None if args.no_upload else settings)
        refs = []
        for key, filename in (
            (args.manifest, "artifacts/dataset/summary.json"),
            (args.baseline_manifest, "artifacts/training/summary.json"),
        ):
            if key:
                body = archive.read(key)
                if body is None:
                    raise PipelineError("A requested manifest is missing")
                refs.append({"key": key, "sha256": digest(body)})
            else:
                refs.append(json.loads(Path(filename).read_text(encoding="utf-8"))["manifest"])
        summary = run_comparison(
            archive, *refs, json.loads(args.config.read_text(encoding="utf-8")), args.summary_dir
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / "summary.json").write_text(
            json.dumps({"status": "failed", "reason": reason, "production_approved": False}),
            encoding="utf-8",
        )
        LOG.error("Challenger comparison failed: %s", reason)
        return 1
