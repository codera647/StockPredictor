"""Publish audited source corrections while retaining every original manifest."""

import argparse
import base64
import copy
import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from stock_predictor.collectors.sec import fact_rows
from stock_predictor.config import PipelineError, Settings
from stock_predictor.extraction import parquet
from stock_predictor.storage.archive import Archive, digest, json_bytes


def read_json(archive: Archive, ref: dict) -> dict:
    body = archive.read(ref["key"], sha256=ref.get("sha256"))
    if body is None:
        raise PipelineError("Replay input is missing")
    return json.loads(body)


def replay_sec_facts(archive: Archive, settings: Settings, manifest_key: str) -> dict:
    full = read_json(archive, {"key": manifest_key})
    report = copy.deepcopy(read_json(archive, full["report"]))
    evidence_manifest = read_json(archive, full["evidence_manifest"])
    evidence = report["evidence"]
    sec = evidence["sources"]["sec"]
    failed = [
        c for c in sec["companies"] if "SEC Company Facts schema is invalid" in c.get("issues", [])
    ]
    if not failed:
        raise PipelineError("Manifest has no SEC Company Facts schema failures to replay")
    raw_refs = {ref["key"]: ref for ref in evidence_manifest["raw"]}
    scope = datetime.fromisoformat(evidence_manifest["created_at"]).date().isoformat()
    additions, corrected = [], []
    for company in failed:
        cik = company["issuer_cik"]
        spec = {
            "version": 1,
            "source": "sec",
            "url": f"{settings.sec_api_base}/api/xbrl/companyfacts/CIK{cik}.json",
            "params": {},
            "revision_scope": scope,
        }
        checkpoint = read_json(
            archive,
            {
                "key": f"checkpoints/evidence/v1/{digest(json_bytes(spec))}.json",
            },
        )
        ref = checkpoint["raw"]
        if ref["key"] not in raw_refs or raw_refs[ref["key"]]["sha256"] != ref["sha256"]:
            raise PipelineError("Cached SEC response differs from the frozen manifest")
        envelope = archive.load_snapshot(ref)
        payload = json.loads(base64.b64decode(envelope["response"]["body_base64"]))
        rows = fact_rows(
            payload,
            date.fromisoformat(report["start"]),
            date.fromisoformat(report["end"]),
            ref,
            envelope,
        )
        additions.extend(rows)
        company["facts"] = len(rows)
        company["issues"].remove("SEC Company Facts schema is invalid")
        company["status"] = "complete_with_issues" if company["issues"] else "complete"
        corrected.append({"symbol": company["symbol"], "facts": len(rows)})
        issue = f"sec:{company['symbol']}:complete_with_issues"
        if company["status"] == "complete" and issue in evidence["issues"]:
            evidence["issues"].remove(issue)
    old_ref = evidence["outputs"]["sec_facts"]
    old_body = archive.read(old_ref["key"], sha256=old_ref["sha256"])
    if old_body is None:
        raise PipelineError("Original SEC facts staging table is missing")
    frame = pd.concat(
        [pd.read_parquet(io.BytesIO(old_body)), pd.DataFrame(additions)], ignore_index=True
    )
    replacement = parquet(archive, "sec_facts", frame)
    evidence["outputs"]["sec_facts"] = replacement
    report["outputs"]["sec_facts"] = replacement
    sec["facts_count"] = replacement["rows"]
    evidence["status"] = "complete_with_issues" if evidence["issues"] else "complete"
    report["issues"] = report["daily"]["issues"] + evidence["issues"]
    report["extraction_status"] = "complete_with_issues" if report["issues"] else "complete"
    replay = {
        "source": "sec_company_facts",
        "parser_version": "numeric-cik-v2",
        "reason": "Accept valid SEC numeric-string CIK fields",
        "corrected": corrected,
        "supersedes_full_manifest": manifest_key,
        "network_source_requests": 0,
    }
    return publish_repair(archive, full, report, evidence_manifest, replay)


def publish_repair(
    archive: Archive, full: dict, report: dict, evidence_manifest: dict, replay: dict
) -> dict:
    evidence = report["evidence"]
    evidence["status"] = "complete_with_issues" if evidence["issues"] else "complete"
    report["issues"] = report["daily"]["issues"] + evidence["issues"]
    report["extraction_status"] = "complete_with_issues" if report["issues"] else "complete"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    evidence["replay"] = replay
    evidence_report_ref = archive.json(f"reports/evidence/{run_id}.json", evidence)
    evidence_manifest_ref = archive.json(
        f"manifests/evidence/{run_id}.json",
        {
            **evidence_manifest,
            "run_id": run_id,
            "outputs": evidence["outputs"],
            "report": evidence_report_ref,
            "status": evidence["status"],
            "replay": replay,
        },
    )
    report.update({"run_id": run_id, "replay": replay})
    report_ref = archive.json(f"reports/full-extraction/{run_id}.json", report)
    manifest_ref = archive.json(
        f"manifests/full-extraction/{run_id}.json",
        {
            **full,
            "run_id": run_id,
            "evidence_manifest": evidence_manifest_ref,
            "report": report_ref,
            "status": report["extraction_status"],
            "replay": replay,
        },
    )
    return {"manifest": manifest_ref, "report": report, "report_ref": report_ref}


def apply_fred_repair(archive: Archive, manifest_key: str, repair_key: str) -> dict:
    full = read_json(archive, {"key": manifest_key})
    report = copy.deepcopy(read_json(archive, full["report"]))
    manifest = read_json(archive, full["evidence_manifest"])
    replacement = read_json(archive, {"key": repair_key})
    source = replacement["source_report"]
    raw = source.pop("raw")
    # Verify every replacement output before publishing a superseding manifest.
    for ref in replacement["outputs"].values():
        if archive.read(ref["key"], sha256=ref["sha256"]) is None:
            raise PipelineError("Repaired FRED output is missing")
    report["evidence"]["sources"]["fred_alfred"] = source
    report["evidence"]["outputs"].update(replacement["outputs"])
    report["outputs"].update(replacement["outputs"])
    issues = [
        issue for issue in report["evidence"]["issues"] if not issue.startswith("fred_alfred:")
    ]
    for series in source["series"]:
        if series["status"] != "complete":
            issues.append(f"fred_alfred:{series['series_id']}:{series['status']}")
    report["evidence"]["issues"] = issues
    manifest["raw"] = list({ref["key"]: ref for ref in [*manifest["raw"], *raw]}.values())
    return publish_repair(
        archive,
        full,
        report,
        manifest,
        {
            "source": "fred_alfred",
            "reason": "Partition revision periods below the JSON vintage limit",
            "revision_window_max_days": 1500,
            "repair_report": repair_key,
            "supersedes_full_manifest": manifest_key,
        },
    )


def apply_ir_repair(archive: Archive, manifest_key: str, repair_key: str) -> dict:
    full = read_json(archive, {"key": manifest_key})
    report = copy.deepcopy(read_json(archive, full["report"]))
    manifest = read_json(archive, full["evidence_manifest"])
    repair = read_json(archive, {"key": repair_key})
    source = repair["source_report"]
    symbols = {result["symbol"] for result in source["results"]}
    old = report["evidence"]["outputs"]["company_announcements"]
    new = repair["outputs"]["company_announcements"]
    old_frame = pd.read_parquet(io.BytesIO(archive.read(old["key"], sha256=old["sha256"])))
    new_frame = pd.read_parquet(io.BytesIO(archive.read(new["key"], sha256=new["sha256"])))
    combined = pd.concat(
        [
            old_frame[~old_frame["symbol"].isin(symbols)] if not old_frame.empty else old_frame,
            new_frame,
        ],
        ignore_index=True,
    )
    output = parquet(archive, "company_announcements", combined)
    old_source = report["evidence"]["sources"]["company_ir"]
    old_source["results"] = [
        r for r in old_source["results"] if r["symbol"] not in symbols
    ] + source["results"]
    old_source["announcements_count"] = output["rows"]
    report["outputs"]["company_announcements"] = output
    report["evidence"]["outputs"]["company_announcements"] = output
    issues = [
        issue
        for issue in report["evidence"]["issues"]
        if not any(issue.startswith(f"company_ir:{symbol}:") for symbol in symbols)
    ]
    for result in source["results"]:
        if result["status"] not in ("complete",):
            issues.append(f"company_ir:{result['symbol']}:{result['status']}")
    report["evidence"]["issues"] = issues
    manifest["raw"] = list({ref["key"]: ref for ref in [*manifest["raw"], *source["raw"]]}.values())
    config_ref = archive.json(
        f"manifests/evidence-configs/{digest(json_bytes(repair['config']))}.json",
        repair["config"],
    )
    report["evidence"]["config"] = config_ref
    manifest["config"] = config_ref
    return publish_repair(
        archive,
        full,
        report,
        manifest,
        {
            "source": "company_ir",
            "reason": "Update confirmed official investor-relations URLs",
            "symbols": sorted(symbols),
            "repair_report": repair_key,
            "supersedes_full_manifest": manifest_key,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--source", choices=("sec-facts", "fred", "company-ir"), default="sec-facts"
    )
    parser.add_argument("--repair-report", help="Private FRED source-repair report key")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    parser.add_argument("--summary-dir", type=Path, default=Path("artifacts/extraction"))
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
        archive = Archive(args.archive_root, settings)
        if args.source in ("fred", "company-ir"):
            if not args.repair_report:
                raise PipelineError("Source repair requires --repair-report")
            merge = apply_fred_repair if args.source == "fred" else apply_ir_repair
            result = merge(archive, args.manifest, args.repair_report)
        else:
            result = replay_sec_facts(archive, settings, args.manifest)
        report = result["report"]
        summary = {
            "status": report["extraction_status"],
            "symbols": len(report["symbols"]),
            "bar_rows": report["bar_rows"],
            "action_rows": report["action_rows"],
            "manifest": result["manifest"],
            "report": result["report_ref"],
            "training_ready": False,
            "sources": "all",
            "additional_rows": {
                k: v.get("rows", 0)
                for k, v in report["outputs"].items()
                if k not in ("bars", "actions", "sessions", "universe")
            },
        }
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (args.summary_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 2 if report["issues"] else 0
    except PipelineError as exc:
        print(f"Replay failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
