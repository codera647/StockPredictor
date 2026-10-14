import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from stock_predictor.calendar import latest_completed_session
from stock_predictor.config import PipelineError, Settings, load_universe
from stock_predictor.datasets.pipeline import preparation_main
from stock_predictor.evidence import full_report, run_evidence
from stock_predictor.extraction import run_extraction
from stock_predictor.storage.archive import Archive, digest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Stock Predictor's reproducible data pipeline")
    sub = root.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Archive actual source data and report coverage")
    extract.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 4))
    extract.add_argument(
        "--end", type=date.fromisoformat, help="Inclusive date; default last ready session"
    )
    extract.add_argument("--universe", type=Path, default=Path("configs/universe.daily.json"))
    extract.add_argument("--symbols", help="Optional comma-separated subset of the frozen universe")
    extract.add_argument("--env-file", type=Path, default=Path(".env"))
    extract.add_argument("--sources", choices=("all", "daily"), default="all")
    extract.add_argument("--evidence-config", type=Path, default=Path("configs/evidence.json"))
    extract.add_argument(
        "--daily-manifest",
        help="Reuse a completed private daily manifest instead of rebuilding it",
    )
    extract.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    extract.add_argument("--summary-dir", type=Path, default=Path("artifacts/extraction"))
    extract.add_argument("--no-upload", action="store_true", help="Explicit disk-only verification")
    extract.add_argument(
        "--skip-directories", action="store_true", help="Explicit price/action-only extraction"
    )
    extract.add_argument(
        "--refresh", action="store_true", help="Fetch new revisions of completed requests"
    )
    prepare = sub.add_parser("prepare", help="Clean, label, build features and purged time splits")
    manifest = prepare.add_mutually_exclusive_group()
    manifest.add_argument("--manifest", help="Private full or daily extraction manifest key")
    manifest.add_argument(
        "--manifest-from-summary", type=Path, default=Path("artifacts/extraction/summary.json")
    )
    prepare.add_argument("--config", type=Path, default=Path("configs/dataset.daily.json"))
    prepare.add_argument("--env-file", type=Path, default=Path(".env"))
    prepare.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    prepare.add_argument("--summary-dir", type=Path, default=Path("artifacts/dataset"))
    prepare.add_argument("--no-upload", action="store_true")
    train = sub.add_parser("train", help="Train, calibrate and evaluate the daily research bundle")
    dataset = train.add_mutually_exclusive_group()
    dataset.add_argument("--manifest", help="Private prepared dataset manifest key")
    dataset.add_argument(
        "--manifest-from-summary", type=Path, default=Path("artifacts/dataset/summary.json")
    )
    train.add_argument("--config", type=Path, default=Path("configs/training.daily.json"))
    train.add_argument("--env-file", type=Path, default=Path(".env"))
    train.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    train.add_argument("--summary-dir", type=Path, default=Path("artifacts/training"))
    train.add_argument("--no-upload", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Third-party HTTP debug logging can disclose sensitive URLs/headers; keep it quiet.
    for name in ("httpx", "httpcore", "botocore", "boto3", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if args.command == "prepare":
        return preparation_main(args)
    if args.command == "train":
        from stock_predictor.training.pipeline import training_main

        return training_main(args)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    (args.summary_dir / "summary.json").write_text(
        json.dumps({"status": "running", "sources": args.sources, "training_ready": False}),
        encoding="utf-8",
    )
    try:
        settings = Settings.load(args.env_file, upload=not args.no_upload)
        logging.info("Extraction configuration loaded; credentials will not be logged")
        selected = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
        universe = load_universe(args.universe, selected)
        archive = Archive(args.archive_root, None if args.no_upload else settings)
        logging.info("Checking private archive access and requested session window")
        if args.daily_manifest:
            manifest_body = archive.read(args.daily_manifest)
            if manifest_body is None:
                raise PipelineError("Requested daily manifest is missing")
            manifest = json.loads(manifest_body)
            if manifest.get("status") not in ("complete", "complete_with_issues"):
                raise PipelineError("Only completed daily extraction manifests can be reused")
            report_ref = manifest["report"]
            report = json.loads(archive.read(report_ref["key"], sha256=report_ref["sha256"]))
            if selected and set(selected) != set(report["symbols"]):
                raise PipelineError("Requested symbols differ from the frozen daily manifest")
            args.start, args.end = (
                date.fromisoformat(report["start"]),
                date.fromisoformat(report["end"]),
            )
            result = {
                "manifest": {
                    "key": args.daily_manifest,
                    "sha256": digest(manifest_body),
                },
                "report": report,
                "report_ref": report_ref,
            }
            logging.info("Reusing verified daily manifest: %d bars", report["bar_rows"])
        else:
            result = run_extraction(
                settings,
                archive,
                universe,
                args.start,
                args.end or latest_completed_session(),
                refresh=args.refresh,
                directories=not args.skip_directories,
            )
        if args.sources == "all":
            universe_ref = result["report"]["outputs"]["universe"]
            resolved = json.loads(archive.read(universe_ref["key"], sha256=universe_ref["sha256"]))
            evidence_config = json.loads(args.evidence_config.read_text(encoding="utf-8"))
            evidence = run_evidence(
                settings,
                archive,
                resolved["instruments"],
                evidence_config,
                args.start,
                args.end or latest_completed_session(),
                refresh=args.refresh,
            )
            result = full_report(result, evidence, archive)
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        (args.summary_dir / "report.json").write_text(
            json.dumps(result["report"], indent=2),
            encoding="utf-8",
        )
        summary = {
            "status": result["report"]["extraction_status"],
            "symbols": len(result["report"]["symbols"]),
            "bar_rows": result["report"]["bar_rows"],
            "action_rows": result["report"]["action_rows"],
            "manifest": result["manifest"],
            "report": result["report_ref"],
            "training_ready": False,
            "sources": args.sources,
            "additional_rows": {
                name: ref.get("rows", 0)
                for name, ref in result["report"]["outputs"].items()
                if name not in ("bars", "actions", "sessions", "universe")
            },
        }
        (args.summary_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return 2 if result["report"]["issues"] else 0
    except PipelineError as exc:
        args.summary_dir.mkdir(parents=True, exist_ok=True)
        failure = getattr(
            exc,
            "extraction_report",
            {
                "extraction_status": "failed",
                "training_ready": False,
                "reason": str(exc),
            },
        )
        (args.summary_dir / "report.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        (args.summary_dir / "summary.json").write_text(
            json.dumps({"status": "failed", "training_ready": False, "reason": str(exc)}, indent=2),
            encoding="utf-8",
        )
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # No traceback containing SDK credential-bearing request objects in normal CI logs.
        print(f"Extraction failed ({type(exc).__name__}); checkpoint retained", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
