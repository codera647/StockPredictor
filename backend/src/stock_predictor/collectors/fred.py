import logging
import math
from datetime import date, timedelta

from stock_predictor.config import PipelineError, Settings
from stock_predictor.storage.archive import Archive

from .cache import CachedSource
from .http import SourceClient

LOG = logging.getLogger(__name__)


def revision_windows(lower: date, upper: date):
    """Each JSON request stays below the 2,000-vintage-date provider limit."""
    while lower <= upper:
        end = min(lower + timedelta(days=1499), upper)
        yield lower, end
        lower = end + timedelta(days=1)


def observation_rows(series: str, records: list, ref: dict, envelope: dict) -> list[dict]:
    result = []
    for record in records:
        if not {"date", "realtime_start", "realtime_end", "value"}.issubset(record):
            raise PipelineError("FRED observation vintage fields are missing")
        value = record["value"]
        if value == ".":
            value = None
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise PipelineError("FRED observation value is invalid") from None
            if not math.isfinite(value):
                raise PipelineError("FRED observation value is nonfinite")
        result.append(
            {
                "series_id": series,
                "observation_date": record["date"],
                "value": value,
                "value_missing": value is None,
                "realtime_start": record["realtime_start"],
                "realtime_end": record["realtime_end"],
                "vintage_precision": "date_only",
                "available_at": envelope["retrieved_at"],
                "retrieved_at": envelope["retrieved_at"],
                "availability_method": "observed_retrieval_requires_conservative_vintage_join",
                "raw_snapshot_key": ref["key"],
            }
        )
    return result


def collect_fred(
    settings: Settings,
    archive: Archive,
    series: list[str],
    start: date,
    end: date,
    *,
    scope: str,
    refresh: bool,
) -> dict:
    if not settings.fred_key or set(settings.fred_key) == {"*"}:
        raise PipelineError("Full extraction needs a real FRED_API_KEY")
    cache = CachedSource(SourceClient("fred", archive, interval=0.6), archive, refresh=refresh)
    results, observations, vintage_rows, metadata, refs = [], [], [], [], []
    lower = start - timedelta(
        days=366
    )  # Prior-year CPI observations for later year-on-year features.
    windows = list(revision_windows(lower, end))
    try:
        for series_id in series:
            LOG.info("FRED/ALFRED observations and vintages: %s", series_id)
            result = {"series_id": series_id, "status": "complete", "issues": []}
            params = {"series_id": series_id, "api_key": settings.fred_key, "file_type": "json"}
            try:
                description, ref, _ = cache.json(
                    f"{settings.fred_base}/series", params, revision_scope=scope
                )
                if not isinstance(description.get("seriess"), list):
                    raise PipelineError("FRED series metadata schema invalid")
                metadata.extend(description["seriess"])
                refs.append(ref)
                routes = [
                    (
                        "observations",
                        "observations",
                        {
                            "observation_start": lower.isoformat(),
                            "observation_end": end.isoformat(),
                            "realtime_start": window_start.isoformat(),
                            "realtime_end": window_end.isoformat(),
                            "output_type": 1,
                            "limit": 100000,
                            "sort_order": "asc",
                        },
                    )
                    for window_start, window_end in windows
                ]
                routes.append(
                    (
                        "vintagedates",
                        "vintage_dates",
                        {
                            "realtime_start": lower.isoformat(),
                            "realtime_end": end.isoformat(),
                            "limit": 10000,
                            "sort_order": "asc",
                        },
                    ),
                )
                for route, field, extra in routes:
                    offset, count = 0, None
                    while count is None or offset < count:
                        payload, ref, envelope = cache.json(
                            f"{settings.fred_base}/series/{route}",
                            {**params, **extra, "offset": offset},
                            revision_scope=scope,
                        )
                        refs.append(ref)
                        batch = payload.get(field)
                        total = payload.get("count")
                        if not isinstance(batch, list) or not isinstance(total, int) or total < 0:
                            raise PipelineError("FRED pagination schema invalid")
                        if count is not None and total != count:
                            raise PipelineError("FRED result count changed during pagination")
                        count = total
                        if not batch and offset < count:
                            raise PipelineError("FRED empty page before all records were consumed")
                        if route == "observations":
                            rows = observation_rows(series_id, batch, ref, envelope)
                            for row in rows:
                                row["request_realtime_start"] = extra["realtime_start"]
                                row["request_realtime_end"] = extra["realtime_end"]
                            observations.extend(rows)
                        else:
                            vintage_rows.extend(
                                {
                                    "series_id": series_id,
                                    "vintage_date": day,
                                    "raw_snapshot_key": ref["key"],
                                    "retrieved_at": envelope["retrieved_at"],
                                }
                                for day in batch
                            )
                        offset += len(batch)
                        if count == 0:
                            break
                    result[field] = result.get(field, 0) + offset
                result["realtime_windows"] = len(windows)
            except PipelineError as exc:
                result["status"] = "failed"
                result["issues"].append(str(exc))
            results.append(result)
    finally:
        cache.close()
    return {
        "source": "fred_alfred",
        "revision_window_max_days": 1500,
        "series": results,
        "observations": observations,
        "vintage_dates": vintage_rows,
        "series_metadata": metadata,
        "raw": refs,
    }
