import io
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from stock_predictor.config import PipelineError, Settings
from stock_predictor.datasets.auxiliary import clean_evidence
from stock_predictor.datasets.cleaning import clean_actions, clean_bars
from stock_predictor.datasets.features import FEATURE_COLUMNS, build_features
from stock_predictor.datasets.labels import build_labels
from stock_predictor.datasets.splits import make_splits
from stock_predictor.datasets.verification import fetch_split_reference
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.storage.archive import Archive, digest, json_bytes

LOG = logging.getLogger(__name__)


def read_table(archive: Archive, ref: dict) -> pd.DataFrame:
    body = archive.read(ref["key"], sha256=ref["sha256"])
    if body is None:
        raise PipelineError("A referenced dataset input is missing")
    return pd.read_parquet(io.BytesIO(body))


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("horizon_sessions") != 5:
        raise PipelineError("Only the versioned five-session daily schema is supported")
    if config.get("production_ready") is not False:
        raise PipelineError("Reconstructed data cannot be certified production-ready")
    if config.get("volume_policy") != "previous_completed_provider_day":
        raise PipelineError("Daily volume must use the declared lagged availability convention")
    if config["eligibility_sessions"] < 252 or config["indicator_warmup_sessions"] < 61:
        raise PipelineError("The declared minimum history cannot be silently shortened")
    if (
        config["minimum_dollar_volume_20"] < 0
        or config["direction_epsilon"] <= 0
        or config["direction_volatility_multiplier"] <= 0
    ):
        raise PipelineError("Invalid eligibility or label thresholds")
    boundaries = [
        date.fromisoformat(config[k])
        for k in ("validation_start", "calibration_start", "test_start")
    ]
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise PipelineError("Dataset boundaries must be chronological")


def code_hash() -> str:
    root = Path(__file__).parent.parent
    files = sorted(root.rglob("*.py"))
    # Normalize checkout line endings so Windows and Linux identify the same code.
    return digest(
        json_bytes(
            {
                p.relative_to(root).as_posix(): digest(p.read_text(encoding="utf-8").encode())
                for p in files
            }
        )
    )


def run_preparation(
    settings: Settings, archive: Archive, manifest_ref: dict, config: dict, summary_dir: Path
) -> dict:
    validate_config(config)
    archive.check_access()
    manifest_body = archive.read(manifest_ref["key"], sha256=manifest_ref.get("sha256"))
    if manifest_body is None:
        raise PipelineError("Source extraction manifest is missing")
    manifest_ref = {"key": manifest_ref["key"], "sha256": digest(manifest_body)}
    manifest = json.loads(manifest_body)
    report = read_json(archive, manifest["report"])
    inputs = report["outputs"]
    LOG.info("Reading checksummed bars, corporate actions and exchange calendar")
    bars, actions, sessions = (
        read_table(archive, inputs[name]) for name in ("bars", "actions", "sessions")
    )
    for key in ("validation_start", "calibration_start", "test_start"):
        if config[key] not in set(sessions.session_date):
            raise PipelineError("Split boundaries must be actual sessions in the source calendar")
    reference = fetch_split_reference(
        settings,
        archive,
        report["symbols"],
        date.fromisoformat(report["start"]),
        date.fromisoformat(report["end"]),
    )
    normalized_actions, action_report = clean_actions(actions, set(bars.symbol))
    clean, cleaning_report = clean_bars(
        bars, normalized_actions, read_table(archive, reference["table"]), sessions, config
    )
    LOG.info(
        "Canonical cleaning complete: %d/%d valid calendar rows",
        clean.clean_valid.sum(),
        len(clean),
    )
    features = build_features(clean, config)
    labels = build_labels(clean, features, config)
    recipe = {
        "source_manifest": manifest_ref,
        "inputs": inputs,
        "reference": reference["table"],
        "config": config,
        "code_sha256": code_hash(),
        "feature_columns": FEATURE_COLUMNS,
    }
    dataset_version = digest(json_bytes(recipe))
    features["dataset_version"] = dataset_version
    features["input_snapshot_id"] = digest(json_bytes(inputs))
    labels["dataset_version"] = dataset_version
    splits, assignments, split_report = make_splits(features, labels, config)
    LOG.info(
        "Feature/label preparation complete: %d research-ready feature rows",
        features.feature_ready.sum(),
    )
    outputs = {}
    for name, table in {
        "clean_bars": clean,
        "clean_actions": normalized_actions,
        "features": features,
        "labels": labels,
        "split_assignments": assignments,
    }.items():
        outputs[name] = parquet(archive, "datasets/" + name, table)
    cleaned_filings = None
    auxiliary_reports = {}
    names = [
        "sec_filings",
        "sec_facts",
        "macro_observations",
        "macro_vintages",
        "macro_series",
        "sec_companies",
        "company_announcements",
        "news_gkg",
        "news_articles",
    ]
    for name in names:
        if name not in inputs:
            auxiliary_reports[name] = {"status": "not_collected", "used_in_core_features": False}
            continue
        LOG.info("Cleaning auxiliary evidence: %s", name)
        table, auxiliary_reports[name] = clean_evidence(
            name, read_table(archive, inputs[name]), cleaned_filings
        )
        if name == "sec_filings":
            cleaned_filings = table
        outputs["clean_" + name] = parquet(archive, "datasets/clean_" + name, table)
    for name, tables in splits.items():
        for kind, table in tables.items():
            outputs[f"{kind}_{name}"] = parquet(archive, f"datasets/{kind}_{name}", table)
    pending_ids = labels.loc[labels.label_quality_status.eq("immature"), "row_id"]
    inference = features[features.feature_ready & features.row_id.isin(pending_ids)].copy()
    outputs["inference_features"] = parquet(archive, "datasets/inference_features", inference)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    result = {
        "status": "complete_research_dataset",
        "research_training_ready": True,
        "production_ready": False,
        "dataset_version": dataset_version,
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "source_manifest": manifest_ref,
        "recipe": recipe,
        "outputs": outputs,
        "cleaning": cleaning_report,
        "actions": action_report,
        "auxiliary": auxiliary_reports,
        "feature_rows": len(features),
        "eligible_feature_rows": int(features.feature_ready.sum()),
        "feature_readiness": {
            k: int(v) for k, v in features.readiness_status.value_counts().items()
        },
        "label_status": {k: int(v) for k, v in labels.label_quality_status.value_counts().items()},
        "inference_rows": len(inference),
        "splits": split_report,
        "coverage_by_symbol_year": clean.assign(year=clean.session_date.str[:4])
        .groupby(["symbol", "year"])
        .agg(rows=("clean_valid", "size"), valid_rows=("clean_valid", "sum"))
        .reset_index()
        .to_dict("records"),
        "limitations": [
            "Historical availability is reconstructed; observed retrieval times are retained.",
            "Same-provider split reconciliation is not independent action/identity certification.",
            "The fixed current 50-stock universe has survivorship bias.",
            "Cleaned SEC, FRED, news and IR evidence is excluded from the price/SPY model.",
            "FRED date vintages assume next-day availability, without intraday timing claims.",
            "Price-only labels exclude cash distributions and are not total returns.",
            "No imputer, scaler, model, calibration fit or trading performance has been produced.",
        ],
        "source_issues": report.get("issues", []),
    }
    report_ref = archive.json(f"reports/dataset/{run_id}.json", result)
    dataset_manifest = archive.json(
        f"manifests/dataset/{run_id}.json",
        {
            "status": result["status"],
            "dataset_version": dataset_version,
            "report": report_ref,
            "recipe": recipe,
            "outputs": outputs,
        },
    )
    summary = {
        k: result[k]
        for k in (
            "status",
            "research_training_ready",
            "production_ready",
            "dataset_version",
            "feature_count",
            "eligible_feature_rows",
            "inference_rows",
            "splits",
        )
    }
    summary.update(manifest=dataset_manifest, report=report_ref)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def preparation_main(args) -> int:
    try:
        settings = Settings.load(args.env_file, upload=not args.no_upload)
        archive = Archive(args.archive_root, None if args.no_upload else settings)
        manifest_ref = (
            {"key": args.manifest}
            if args.manifest
            else json.loads(args.manifest_from_summary.read_text(encoding="utf-8"))["manifest"]
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        print(
            json.dumps(
                run_preparation(settings, archive, manifest_ref, config, args.summary_dir), indent=2
            )
        )
        return 0
    except Exception as exc:
        # Never expose credential-bearing SDK requests in normal logs.
        message = str(exc) if isinstance(exc, PipelineError) else type(exc).__name__
        failure = {
            "status": "failed",
            "research_training_ready": False,
            "production_ready": False,
            "reason": message,
        }
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / "summary.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        LOG.error("Dataset preparation failed: %s", message)
        return 1
