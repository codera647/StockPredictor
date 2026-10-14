import base64
import io
import json
import logging
import os
from datetime import UTC, date, datetime
from importlib.metadata import version

import pandas as pd

from stock_predictor.calendar import request_bounds, schedule, year_partitions
from stock_predictor.collectors.actions import action_rows
from stock_predictor.collectors.alpaca import AlpacaCollector
from stock_predictor.collectors.directories import collect_directories
from stock_predictor.config import PipelineError, Settings
from stock_predictor.data_quality.bars import bar_rows, coverage_report
from stock_predictor.storage.archive import Archive, digest, json_bytes

LOG = logging.getLogger(__name__)


def parquet(archive: Archive, category: str, rows: list[dict] | pd.DataFrame) -> dict:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    body = buffer.getvalue()
    return {**archive.write(f"staging/{category}/{digest(body)}.parquet", body), "rows": len(frame)}


def run_extraction(
    settings: Settings,
    archive: Archive,
    universe: dict,
    start: date,
    end: date,
    *,
    refresh: bool = False,
    directories: bool = True,
) -> dict:
    now = datetime.now(UTC)
    request_bounds(start, end, now)  # Validate the entire window before making requests.
    archive.check_access()
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    session_frame = schedule(start, end)
    sessions = {
        row.session_date: {"open": row.open.isoformat(), "close": row.close.isoformat()}
        for row in session_frame.itertuples(index=False)
    }
    instruments = {entry["symbol"]: entry.copy() for entry in universe["instruments"]}
    symbols = sorted(instruments)
    directory_results, sec_records = (
        collect_directories(settings, archive) if directories else ([], [])
    )
    by_ticker = {}
    for row in sec_records:
        by_ticker.setdefault(row["ticker"], []).append(row)
    for symbol, entry in instruments.items():
        matches = by_ticker.get(symbol, [])
        if len(matches) == 1:
            entry["issuer_cik"] = str(matches[0]["cik"]).zfill(10)
            entry["issuer_mapping_scope"] = "current_snapshot_only"
    universe_snapshot = {**universe, "instruments": list(instruments.values())}
    universe_ref = archive.json(
        f"manifests/universes/{digest(json_bytes(universe_snapshot))}.json",
        universe_snapshot,
    )
    tasks, bar_data, action_data = [], [], []
    collector = AlpacaCollector(settings, archive)
    try:
        for lower, upper in year_partitions(start, end):
            bounds = request_bounds(lower, upper, now)
            queries = {
                "bars": {
                    "symbols": ",".join(symbols),
                    "timeframe": "1Day",
                    "start": bounds[0],
                    "end": bounds[1],
                    "feed": "sip",
                    "adjustment": "raw",
                    "asof": "-",
                    "currency": "USD",
                    "sort": "asc",
                    "limit": 10000,
                },
                "actions": {
                    "symbols": ",".join(symbols),
                    "start": lower.isoformat(),
                    "end": upper.isoformat(),
                    "sort": "asc",
                    "limit": 1000,
                    "data_quality": "all",
                    "region": "us",
                },
            }
            for kind, params in queries.items():
                LOG.info(
                    "Extracting %s %s through %s (%d symbols)", kind, lower, upper, len(symbols)
                )
                state = collector.collect(kind, params, refresh=refresh)
                count = sum(page["records"] for page in state["pages"])
                LOG.info("Archived %d %s records in %d pages", count, kind, len(state["pages"]))
                tasks.append(state)
                for ref in state["pages"]:
                    envelope = archive.load_snapshot(ref)
                    payload = json.loads(base64.b64decode(envelope["response"]["body_base64"]))
                    if kind == "bars":
                        bar_data.extend(bar_rows(payload, envelope, ref, instruments, sessions))
                    else:
                        action_data.extend(action_rows(payload, envelope, ref))
    except PipelineError as exc:
        failure = {
            "schema_version": 1,
            "run_id": run_id,
            "extraction_status": "failed",
            "training_ready": False,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": symbols,
            "feed": "sip",
            "reason": str(exc),
            "directories": directory_results,
            "completed_tasks": tasks,
            "universe": universe_ref,
        }
        exc.extraction_report = failure
        # Even a failed source run has an auditable report. A storage outage must
        # not obscure the original failure or pretend its report was uploaded.
        try:
            archive.json(f"reports/extraction/{run_id}.json", failure)
        except PipelineError:
            LOG.error("Failure report upload unavailable; writing the local report only")
        raise
    finally:
        collector.close()
    frame = pd.DataFrame(bar_data)
    coverage = coverage_report(frame, universe_snapshot, list(sessions))
    required_issues = []
    for entry in coverage["instruments"]:
        if entry["valid_unique_sessions"] == 0:
            required_issues.append(f"no_valid_bars:{entry['symbol']}")
        if entry["missing_session_count"]:
            required_issues.append(f"missing_sessions:{entry['symbol']}")
        if entry["quarantined_rows"]:
            required_issues.append(f"quarantined_bars:{entry['symbol']}")
        if entry["conflicting_revision_count"]:
            required_issues.append(f"conflicting_bar_revisions:{entry['symbol']}")
    for result in directory_results:
        if result["status"] != "complete":
            required_issues.append(f"directory_failed:{result['name']}")
    outputs = {
        "bars": parquet(archive, "bars", frame),
        "actions": parquet(archive, "actions", action_data),
        "sessions": parquet(archive, "sessions", session_frame),
        "universe": universe_ref,
    }
    limitations = [
        "Fixed current-survivor universe; historical security identity and delistings unverified.",
        "Corporate-action absence does not establish complete action coverage.",
        "Historical responses are later revisions, not archived first-publication observations.",
        "Daily volume includes extended-hours prints; close+30 feature parity is unresolved.",
    ]
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "extraction_status": "complete_with_issues" if required_issues else "complete",
        "training_ready": False,
        "issues": required_issues,
        "limitations": limitations,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "feed": "sip",
        "symbols": symbols,
        "bar_rows": len(bar_data),
        "action_rows": len(action_data),
        "directories": directory_results,
        "coverage": coverage,
        "action_types": dict(pd.Series([r["action_type"] for r in action_data]).value_counts()),
        "outputs": outputs,
    }
    # Convert numpy counts to plain JSON integers.
    report["action_types"] = {k: int(v) for k, v in report["action_types"].items()}
    report_ref = archive.json(f"reports/extraction/{run_id}.json", report)
    manifest = {
        "schema_version": 1,
        "pipeline_version": version("stock-predictor"),
        "run_id": run_id,
        "created_at": now.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "code_commit": os.environ.get("GITHUB_SHA", "local-uncommitted"),
        "dependencies": {
            name: version(name)
            for name in (
                "httpx",
                "boto3",
                "pandas",
                "pyarrow",
                "exchange_calendars",
            )
        },
        "start": start.isoformat(),
        "end": end.isoformat(),
        "universe": universe_ref,
        "tasks": tasks,
        "directories": directory_results,
        "outputs": outputs,
        "report": report_ref,
        "status": report["extraction_status"],
        "training_ready": False,
    }
    manifest_ref = archive.json(f"manifests/extraction/{run_id}.json", manifest)
    return {"manifest": manifest_ref, "report": report, "report_ref": report_ref}
