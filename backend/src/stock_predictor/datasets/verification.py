import base64
import json
import logging
from datetime import date

import pandas as pd

from stock_predictor.calendar import NEW_YORK, request_bounds, year_partitions
from stock_predictor.collectors.alpaca import AlpacaCollector
from stock_predictor.config import PipelineError, Settings
from stock_predictor.extraction import parquet
from stock_predictor.storage.archive import Archive, digest, json_bytes

LOG = logging.getLogger(__name__)


def fetch_split_reference(
    settings: Settings, archive: Archive, symbols: list[str], start: date, end: date
) -> dict:
    """Provider split-adjusted bars are audit inputs only, never model features."""
    spec = {
        "version": 1,
        "symbols": sorted(symbols),
        "start": str(start),
        "end": str(end),
        "feed": "sip",
        "adjustment": "split",
        "base_url": settings.alpaca_base,
    }
    key = f"checkpoints/dataset-reference/{digest(json_bytes(spec))}.json"
    saved = archive.read(key)
    if saved:
        ref = json.loads(saved)
        if archive.read(ref["table"]["key"], sha256=ref["table"]["sha256"]) is None:
            raise PipelineError("Split audit reference table is missing")
        return ref
    collector = AlpacaCollector(settings, archive)
    rows, tasks = [], []
    try:
        for lower, upper in year_partitions(start, end):
            lo, hi = request_bounds(lower, upper)
            LOG.info("Auditing split-adjusted reference bars: %s through %s", lower, upper)
            state = collector.collect(
                "bars",
                {
                    "symbols": ",".join(sorted(symbols)),
                    "timeframe": "1Day",
                    "start": lo,
                    "end": hi,
                    "feed": "sip",
                    "adjustment": "split",
                    "asof": "-",
                    "currency": "USD",
                    "sort": "asc",
                    "limit": 10000,
                },
            )
            tasks.append(state)
            for ref in state["pages"]:
                envelope = archive.load_snapshot(ref)
                payload = json.loads(base64.b64decode(envelope["response"]["body_base64"]))
                for symbol, bars in (payload["bars"] or {}).items():
                    if symbol not in symbols:
                        raise PipelineError("Unexpected symbol in split audit response")
                    for bar in bars:
                        stamp = pd.Timestamp(bar["t"])
                        if stamp.tzinfo is None:
                            raise PipelineError("Split reference has a naive timestamp")
                        rows.append(
                            {
                                "symbol": symbol,
                                "session_date": stamp.tz_convert(NEW_YORK).date().isoformat(),
                                **{
                                    f"{field}_split_reference": bar[field]
                                    for field in ("o", "h", "l", "c", "v")
                                },
                                "reference_snapshot_key": ref["key"],
                                "reference_hash": ref["sha256"],
                            }
                        )
    finally:
        collector.close()
    table = parquet(archive, "split_audit_reference", rows)
    result = {
        "spec": spec,
        "table": table,
        "tasks": tasks,
        "scope": "same_provider_adjustment_reconciliation_not_independent_action_certification",
    }
    archive.json(key, result)
    return result
