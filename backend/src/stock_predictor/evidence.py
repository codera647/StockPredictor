import logging
import os
from datetime import UTC, date, datetime
from importlib.metadata import version

from stock_predictor.collectors.announcements import collect_announcements
from stock_predictor.collectors.fred import collect_fred
from stock_predictor.collectors.gdelt import collect_gdelt
from stock_predictor.collectors.sec import collect_sec
from stock_predictor.config import PipelineError, Settings
from stock_predictor.extraction import parquet
from stock_predictor.storage.archive import Archive, digest, json_bytes

LOG = logging.getLogger(__name__)


def run_evidence(
    settings: Settings,
    archive: Archive,
    instruments: list[dict],
    config: dict,
    start: date,
    end: date,
    *,
    refresh: bool,
) -> dict:
    now = datetime.now(UTC)
    run_id, scope = now.strftime("%Y%m%dT%H%M%S%fZ"), now.date().isoformat()
    symbols = {entry["symbol"] for entry in instruments if entry["kind"] == "stock"}
    aliases = {
        symbol: names for symbol, names in config["news_aliases"].items() if symbol in symbols
    }
    missing = symbols - set(aliases)
    if missing:
        raise PipelineError(f"Company news aliases missing for {sorted(missing)}")
    sources = [source for source in config["company_sources"] if source["symbol"] in symbols]
    documents = config["sec_documents_per_company"]
    if not isinstance(documents, int) or not 0 <= documents <= 100:
        raise PipelineError("SEC document limit must be an integer from 0 to 100 per company")
    config_ref = archive.json(
        f"manifests/evidence-configs/{digest(json_bytes(config))}.json", config
    )
    results, outputs, raw = {}, {}, []
    jobs = [
        (
            "sec",
            lambda: collect_sec(
                settings,
                archive,
                instruments,
                start,
                end,
                scope=scope,
                refresh=refresh,
                documents_per_company=documents,
                workers=4,
            ),
            {"facts": "sec_facts", "filings": "sec_filings", "company_metadata": "sec_companies"},
        ),
        (
            "fred_alfred",
            lambda: collect_fred(
                settings,
                archive,
                config["fred_series"],
                start,
                end,
                scope=scope,
                refresh=refresh,
            ),
            {
                "observations": "macro_observations",
                "vintage_dates": "macro_vintages",
                "series_metadata": "macro_series",
            },
        ),
        (
            "gdelt",
            lambda: collect_gdelt(
                settings,
                archive,
                aliases,
                scope=scope,
                refresh=refresh,
                lookback_days=config["news_lookback_days"],
                gkg_partitions=config["gdelt_gkg_partitions"],
                max_doc_requests=config["gdelt_max_doc_requests"],
            ),
            {"articles": "news_articles", "gkg": "news_gkg"},
        ),
        (
            "company_ir",
            lambda: collect_announcements(
                archive,
                sources,
                scope=scope,
                refresh=refresh,
            ),
            {"announcements": "company_announcements"},
        ),
    ]
    for name, collect, tables in jobs:
        LOG.info("Evidence source stage: %s", name)
        try:
            data = collect()
            raw.extend(data.pop("raw"))
            for field, category in tables.items():
                records = data.pop(field)
                outputs[category] = parquet(archive, category, records)
                data[field + "_count"] = len(records)
            results[name] = data
        except PipelineError as exc:
            results[name] = {"status": "failed", "reason": str(exc)}
        # Each source report is durable before the next potentially slow source begins.
        archive.json(f"reports/evidence/{run_id}/{name}.json", results[name])
    issues = []
    for source, data in results.items():
        if data.get("status") == "failed":
            issues.append(f"source_failed:{source}")
        for field in ("companies", "series", "results"):
            for result in data.get(field, []):
                if result.get("status") in (
                    "failed",
                    "complete_with_issues",
                    "snapshot_only_no_event_metadata",
                ):
                    identifier = (
                        result.get("symbol") or result.get("series_id") or result.get("name")
                    )
                    issues.append(f"{source}:{identifier}:{result['status']}")
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "status": "complete_with_issues" if issues else "complete",
        "training_ready": False,
        "issues": issues,
        "sources": results,
        "outputs": outputs,
        "config": config_ref,
        "coverage_policy": config["coverage_policy"],
    }
    report_ref = archive.json(f"reports/evidence/{run_id}.json", report)
    manifest = {
        "schema_version": 1,
        "pipeline_version": version("stock-predictor"),
        "code_commit": os.environ.get("GITHUB_SHA", "local-uncommitted"),
        "run_id": run_id,
        "created_at": now.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "config": config_ref,
        "outputs": outputs,
        "raw": raw,
        "report": report_ref,
        "status": report["status"],
        "training_ready": False,
    }
    manifest_ref = archive.json(f"manifests/evidence/{run_id}.json", manifest)
    return {"report": report, "report_ref": report_ref, "manifest": manifest_ref}


def full_report(daily: dict, evidence: dict, archive: Archive) -> dict:
    run_id = evidence["report"]["run_id"]
    daily_report = daily["report"]
    report = {
        "schema_version": 2,
        "run_id": run_id,
        "extraction_status": "complete_with_issues"
        if daily_report["issues"] or evidence["report"]["issues"]
        else "complete",
        "training_ready": False,
        "start": daily_report["start"],
        "end": daily_report["end"],
        "symbols": daily_report["symbols"],
        "bar_rows": daily_report["bar_rows"],
        "action_rows": daily_report["action_rows"],
        "daily": daily_report,
        "evidence": evidence["report"],
        "outputs": {**daily_report["outputs"], **evidence["report"]["outputs"]},
        "issues": daily_report["issues"] + evidence["report"]["issues"],
    }
    report_ref = archive.json(f"reports/full-extraction/{run_id}.json", report)
    manifest_ref = archive.json(
        f"manifests/full-extraction/{run_id}.json",
        {
            "schema_version": 2,
            "run_id": run_id,
            "daily_manifest": daily["manifest"],
            "evidence_manifest": evidence["manifest"],
            "report": report_ref,
            "status": report["extraction_status"],
            "training_ready": False,
        },
    )
    return {"manifest": manifest_ref, "report": report, "report_ref": report_ref}
