import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from stock_predictor.calendar import latest_completed_session
from stock_predictor.config import PipelineError, Settings, load_universe
from stock_predictor.extraction import run_extraction
from stock_predictor.storage.archive import Archive


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
    extract.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    extract.add_argument("--summary-dir", type=Path, default=Path("artifacts/extraction"))
    extract.add_argument("--no-upload", action="store_true", help="Explicit disk-only verification")
    extract.add_argument(
        "--skip-directories", action="store_true", help="Explicit price/action-only extraction"
    )
    extract.add_argument(
        "--refresh", action="store_true", help="Fetch new revisions of completed requests"
    )
    return root


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Third-party HTTP debug logging can disclose sensitive URLs/headers; keep it quiet.
    for name in ("httpx", "httpcore", "botocore", "boto3", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        settings = Settings.load(args.env_file, upload=not args.no_upload)
        logging.info("Extraction configuration loaded; credentials will not be logged")
        selected = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
        universe = load_universe(args.universe, selected)
        archive = Archive(args.archive_root, None if args.no_upload else settings)
        logging.info("Checking private archive access and requested session window")
        result = run_extraction(
            settings,
            archive,
            universe,
            args.start,
            args.end or latest_completed_session(),
            refresh=args.refresh,
            directories=not args.skip_directories,
        )
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
