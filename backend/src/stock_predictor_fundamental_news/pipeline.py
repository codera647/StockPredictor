import argparse
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from stock_predictor.config import PipelineError, Settings
from stock_predictor.datasets.features import FEATURE_COLUMNS
from stock_predictor.datasets.pipeline import code_hash, read_table
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.storage.archive import Archive, digest, json_bytes
from stock_predictor.training.data import load_block, walk_forward_blocks
from stock_predictor.training.estimators import predict_linear
from stock_predictor.training.metrics import CLASSES, classification_metrics, encode_labels
from threadpoolctl import threadpool_limits

from stock_predictor_fundamental_news.features import (
    NEWS_COLUMNS,
    NEWS_NUMERIC,
    SEC_COLUMNS,
    SEC_NUMERIC,
    augment,
    fundamentals,
    news_features,
)
from stock_predictor_fundamental_news.models import fit, predict, save
from stock_predictor_fundamental_news.sources import collect_news

LOG = logging.getLogger(__name__)
VARIANTS = {
    "price_only": [],
    "price_sec": SEC_COLUMNS,
    "price_news": NEWS_COLUMNS,
    "price_sec_news": SEC_COLUMNS + NEWS_COLUMNS,
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


def validate(config):
    if (
        config.get("schema_version") != 1
        or config.get("evaluate_existing_test") is not False
        or config.get("production_ready") is not False
    ):
        raise PipelineError("Fundamental/news research must keep the existing test closed")
    if config.get("news_delay_hours", 0) < 24:
        raise PipelineError("Historical news needs at least the frozen 24-hour availability delay")
    if not config["news_coverage_start"] < config["news_coverage_end"]:
        raise PipelineError("Historical news coverage dates must be ordered")


def write_reports(directory, summary, report):
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in (("summary", summary), ("report", report)):
        (directory / f"{name}.json").write_text(json.dumps(body, indent=2), encoding="utf-8")


def compare(archive, train, tuning, evaluation, feature_table, config, stage):
    predictions, metrics, selected, artifacts = {}, {}, {}, {}
    labels = encode_labels(evaluation["y"].y_direction_5d)
    for variant, additions in VARIANTS.items():
        LOG.info(
            "Fitting %s / %s (%d inputs)", stage, variant, len(FEATURE_COLUMNS) + len(additions)
        )
        blocks = [
            augment(b, feature_table, additions) if additions else b
            for b in (train, tuning, evaluation)
        ]
        models, selection = fit(blocks[0], blocks[1], config)
        evaluated = predict(models, blocks[2]["X"], config["threads"])
        refs, restored = save(archive, models, f"{stage}-{variant}")
        check = predict(restored, blocks[2]["X"], config["threads"])
        for family, probabilities in evaluated.items():
            name = variant + "_" + family
            if not np.allclose(probabilities, check[family], atol=1e-10):
                raise PipelineError("Reloaded fundamental/news model changed predictions")
            predictions[name] = probabilities
            metrics[name] = classification_metrics(labels, probabilities)
            artifacts[name] = refs[family]
        selected[variant] = selection
    frame = pd.concat([evaluation["index"], evaluation["y"][["y_direction_5d"]]], axis=1)
    for name, values in predictions.items():
        for i, cls in enumerate(CLASSES):
            frame[f"{name}_p_{cls.lower()}"] = values[:, i]
    merged = evaluation["index"][["row_id"]].merge(
        feature_table, on="row_id", validate="one_to_one"
    )
    covered = merged.news_source_window_active.to_numpy(dtype=bool)
    subset_metrics = (
        {
            name: classification_metrics(labels[covered], p[covered])
            for name, p in predictions.items()
        }
        if covered.any()
        else {}
    )
    frame["news_source_window_active"] = covered
    artifacts["predictions"] = parquet(archive, "fundamental-news/" + stage + "_predictions", frame)
    result = {
        "training_rows": len(train["X"]),
        "tuning_rows": len(tuning["X"]),
        "evaluation_rows": len(evaluation["X"]),
        "metrics": metrics,
        "selected": selected,
        "news_window_rows": int(covered.sum()),
        "news_window_coverage": float(covered.mean()),
        "news_window_metrics": subset_metrics,
        "news_window_is_complete_archive_coverage": False,
        "evaluation_start": evaluation["index"].session_date.min(),
        "evaluation_end": evaluation["index"].session_date.max(),
    }
    return result, artifacts


def run(archive, dataset_ref, baseline_ref, news_ref, config, directory):
    validate(config)
    archive.check_access()
    dataset, baseline = (read_json(archive, r) for r in (dataset_ref, baseline_ref))
    news_metadata = read_json(archive, news_ref["manifest"])
    if news_metadata["table"] != news_ref["table"]:
        raise PipelineError("News table differs from its immutable source manifest")
    if dataset["dataset_version"] != baseline["bundle"]["dataset_version"]:
        raise PipelineError("Research dataset differs from the baseline bundle")
    if dataset["recipe"]["feature_columns"] != FEATURE_COLUMNS:
        raise PipelineError("Unsupported original feature schema")
    if not read_json(archive, dataset["report"])["research_training_ready"]:
        raise PipelineError("Dataset is not ready for research training")
    outputs = dataset["outputs"]
    needed = ("clean_sec_facts", "clean_sec_filings", "clean_sec_companies")
    recipe = {
        "dataset_manifest": dataset_ref,
        "baseline_manifest": baseline_ref,
        "news_source": news_ref,
        "sec_inputs": {n: outputs[n] for n in needed},
        "core_code_sha256": code_hash(),
        "experiment_code_sha256": source_hash(),
        "config": config,
        "variants": {n: FEATURE_COLUMNS + c for n, c in VARIANTS.items()},
        "dependencies": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "catboost")},
    }
    recipe_hash = digest(json_bytes(recipe))
    checkpoint = "checkpoints/fundamental-news/experiment-" + recipe_hash + ".json"
    saved = archive.read(checkpoint)
    if saved:
        summary = json.loads(saved)
        report = read_json(archive, summary["report"])
        for ref in report["artifacts"].values():
            if archive.read(ref["key"], sha256=ref["sha256"]) is None:
                raise PipelineError("A cached experiment artifact is missing")
        write_reports(directory, summary, report)
        return {**summary, "cache_reused": True}
    dc = dataset["recipe"]["config"]
    train = load_block(archive, outputs, "train", (None, dc["validation_start"]))
    validation = load_block(
        archive, outputs, "validation", (dc["validation_start"], dc["calibration_start"])
    )
    index = pd.concat([train["index"], validation["index"]], ignore_index=True)
    if index.row_id.duplicated().any():
        raise PipelineError("Research fitting blocks contain duplicate row identities")
    sessions = read_table(archive, dataset["recipe"]["inputs"]["sessions"])
    LOG.info("Replaying SEC facts, filing events and delayed historical headlines")
    facts, filings, companies = (read_table(archive, outputs[n]) for n in needed)
    sec = fundamentals(facts, filings, companies, index, sessions, config)
    news = read_table(archive, news_ref["table"])
    headlines = news_features(news, index, sessions, config)
    features = sec.merge(headlines, on="row_id", validate="one_to_one")
    joined = index[["row_id", "cutoff_at"]].merge(features, on="row_id", validate="one_to_one")
    for column in ("sec_max_available_at_reconstructed", "news_max_available_at_reconstructed"):
        if pd.to_datetime(joined[column], utc=True).gt(joined.cutoff_at).any():
            raise PipelineError("Research feature table contains future availability")
    artifacts = {
        "sec_news_features": parquet(archive, "fundamental-news/features", features),
        "filtered_news": news_ref["table"],
    }
    folds = []
    with threadpool_limits(limits=config["threads"]):
        for i, fold in enumerate(config["walk_forward"]):
            inner, tuning, _, evaluation = walk_forward_blocks(train, fold)
            result, refs = compare(
                archive, inner, tuning, evaluation, features, config, f"fold-{i + 1}"
            )
            folds.append({"fold": fold, **result})
            artifacts.update({f"fold_{i + 1}_" + n: r for n, r in refs.items()})
        # Validation selects settings and families; it is explicitly not a fresh test.
        final, refs = compare(
            archive, train, validation, validation, features, config, "validation"
        )
        artifacts.update({"validation_" + n: r for n, r in refs.items()})
    original = read_json(archive, baseline["bundle"]["models"]["logistic"])
    original_metrics = classification_metrics(
        encode_labels(validation["y"].y_direction_5d), predict_linear(original, validation["X"])
    )
    base_folds = [
        f["metrics"]["price_only_" + f["selected"]["price_only"]["family"]] for f in folds
    ]
    gates = {}
    for variant in ("price_sec", "price_news", "price_sec_news"):
        checks = []
        for f, base in zip(folds, base_folds, strict=True):
            selected = f["metrics"][variant + "_" + f["selected"][variant]["family"]]
            checks.append(
                selected["log_loss"] < base["log_loss"] and selected["accuracy"] >= base["accuracy"]
            )
        gates[variant] = bool(checks and all(checks))
    result = {
        "status": "fundamental_news_research_complete",
        "recipe": recipe,
        "recipe_hash": recipe_hash,
        "walk_forward": folds,
        "validation": final,
        "original_logistic_validation": original_metrics,
        "earlier_fold_baseline_gate": gates,
        "feature_counts": {n: len(FEATURE_COLUMNS) + len(c) for n, c in VARIANTS.items()},
        "feature_missingness": {
            n: float(features[n].isna().mean()) for n in SEC_NUMERIC + NEWS_NUMERIC
        },
        "artifacts": artifacts,
        "artifact_reloads_verified": True,
        "existing_test_evaluated": False,
        "production_approved": False,
        "target_90_established": False,
        "limitations": [
            "Release/publication times are reconstructed, not verified first receives.",
            "News outside the declared archive window remains missing.",
            "Active archive windows do not prove complete news coverage for every stock/date.",
            "Headline sentiment uses frozen lexical rules, not a financial language model.",
            "Current issuer mapping and original survivorship assumptions remain research limits.",
            "Annual comparable financial ratios are sparse; unmatched periods are not mixed.",
            "Earlier folds and validation inform research selection; no fresh test is claimed.",
        ],
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_ref = archive.json("reports/fundamental-news/" + run_id + ".json", result)
    manifest = archive.json(
        "manifests/fundamental-news/" + run_id + ".json",
        {
            "recipe": recipe,
            "report": report_ref,
            "artifacts": artifacts,
        },
    )
    summary = {
        k: result[k]
        for k in (
            "status",
            "feature_counts",
            "earlier_fold_baseline_gate",
            "existing_test_evaluated",
            "production_approved",
            "target_90_established",
        )
    }
    summary["validation"] = {
        n: {k: m[k] for k in ("accuracy", "balanced_accuracy", "log_loss")}
        for n, m in final["metrics"].items()
    }
    summary.update(manifest=manifest, report=report_ref)
    archive.json(checkpoint, summary)
    write_reports(directory, summary, result)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare SEC fundamentals and free historical news"
    )
    parser.add_argument("--manifest")
    parser.add_argument("--baseline-manifest")
    parser.add_argument("--config", type=Path, default=Path("configs/fundamental-news.daily.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    parser.add_argument("--download-dir", type=Path, default=Path("data/downloads/fnspid"))
    parser.add_argument("--summary-dir", type=Path, default=Path("artifacts/fundamental-news"))
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpcore", "botocore", "boto3", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        config = json.loads(args.config.read_text())
        validate(config)
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
                    raise PipelineError("Requested experiment input manifest is missing")
                refs.append({"key": key, "sha256": digest(body), "bytes": len(body)})
            else:
                refs.append(json.loads(Path(file).read_text())["manifest"])
        dataset = read_json(archive, refs[0])
        companies = read_table(archive, dataset["outputs"]["clean_sec_companies"])
        news_ref = collect_news(archive, set(companies.symbol), config, args.download_dir)
        if args.collect_only:
            result = news_ref
        else:
            result = run(archive, *refs, news_ref, config, args.summary_dir)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        reason = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / "summary.json").write_text(
            json.dumps({"status": "failed", "reason": reason})
        )
        LOG.error("Fundamental/news experiment failed: %s", reason)
        return 1
