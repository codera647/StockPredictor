import csv
import hashlib
import io
import logging
import re
import zipfile
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

from stock_predictor.config import PipelineError, Settings
from stock_predictor.storage.archive import Archive

from .cache import CachedSource
from .http import SourceClient

LOG = logging.getLogger(__name__)


def parse_doc(payload: dict, symbol: str, ref: dict, envelope: dict) -> list[dict]:
    if "articles" not in payload:
        raise PipelineError("GDELT response omitted articles; a failed feed is not zero news")
    articles = payload["articles"]
    if not isinstance(articles, list):
        raise PipelineError("GDELT article list schema invalid")
    rows = []
    for article in articles:
        if not isinstance(article, dict) or not article.get("url"):
            raise PipelineError("GDELT article record missing URL")
        rows.append(
            {
                "symbol": symbol,
                "url": article["url"],
                "title": article.get("title"),
                "domain": article.get("domain"),
                "language": article.get("language"),
                "source_country": article.get("sourcecountry"),
                "gdelt_seen_at": article.get("seendate"),
                "first_observed_at": envelope["retrieved_at"],
                "publication_at": None,
                "relevance_status": "candidate_requires_entity_review",
                "raw_snapshot_key": ref["key"],
                "source": "gdelt_doc",
            }
        )
    return rows


def validate_gkg_zip(body: bytes, *, expected_md5: str | None = None) -> zipfile.ZipFile:
    if len(body) > 64 * 1024 * 1024:
        raise PipelineError("GDELT GKG compressed file exceeds the configured size bound")
    if expected_md5 and hashlib.md5(body, usedforsecurity=False).hexdigest() != expected_md5:
        raise PipelineError("GDELT published file checksum mismatch")
    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size > 384 * 1024 * 1024:
            raise PipelineError("Unexpected or oversized GKG ZIP members")
        if not members[0].filename.endswith(".gkg.csv"):
            raise PipelineError("Unexpected GKG ZIP filename")
        return archive
    except zipfile.BadZipFile:
        raise PipelineError("Invalid GKG ZIP archive") from None


def gkg_rows(
    body: bytes,
    aliases: dict[str, list[str]],
    ref: dict,
    envelope: dict,
    *,
    expected_md5: str | None = None,
) -> tuple[list[dict], int]:
    patterns = {
        symbol: [re.compile(r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)") for alias in names]
        for symbol, names in aliases.items()
    }
    rows, scanned = [], 0
    with validate_gkg_zip(body, expected_md5=expected_md5) as zipped:
        with zipped.open(zipped.infolist()[0]) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8", errors="strict")
            for fields in csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(fields) != 27:
                    raise PipelineError("GKG 2.1 record must contain 27 tab-delimited fields")
                scanned += 1
                # Organization fields, not full article text, establish candidate matches.
                # GKG 2.1 columns 12/13 (zero-based 11/12) are persons;
                # organization fields are columns 14/15 (zero-based 13/14).
                organizations = (fields[13] + ";" + fields[14]).lower()
                matches = [
                    symbol
                    for symbol, terms in patterns.items()
                    if any(p.search(organizations) for p in terms)
                ]
                for symbol in matches:
                    rows.append(
                        {
                            "symbol": symbol,
                            "gkg_record_id": fields[0],
                            "gdelt_timestamp": fields[1],
                            "source_name": fields[3],
                            "url": fields[4],
                            "themes": fields[7],
                            "organizations": fields[13],
                            "organizations_v2": fields[14],
                            "tone_raw": fields[15],
                            "first_observed_at": envelope["retrieved_at"],
                            "relevance_status": "candidate_organization_match",
                            "source": "gdelt_gkg",
                            "raw_snapshot_key": ref["key"],
                        }
                    )
    return rows, scanned


def collect_gdelt(
    settings: Settings,
    archive: Archive,
    aliases: dict[str, list[str]],
    *,
    scope: str,
    refresh: bool,
    lookback_days: int,
    gkg_partitions: int,
    max_doc_requests: int,
) -> dict:
    if not 1 <= lookback_days <= 90 or not 1 <= gkg_partitions <= 96:
        raise PipelineError(
            "GDELT DOC supports a recent window; use 1-90 days and 1-96 GKG partitions"
        )
    cache = CachedSource(
        SourceClient("gdelt", archive, interval=5.1, max_attempts=2), archive, refresh=refresh
    )
    doc_rows, gkg_data, results, refs = [], [], [], []
    request_count = 0
    consecutive_failures = 0
    cutoff = datetime.strptime(scope, "%Y-%m-%d").replace(tzinfo=UTC)
    # Freeze news windows at UTC midnight for repeatability and cache reuse.
    news_start = cutoff - timedelta(days=lookback_days)
    try:
        for symbol, names in aliases.items():
            if consecutive_failures >= 3:
                results.append(
                    {
                        "symbol": symbol,
                        "status": "failed",
                        "issues": [
                            "DOC source unavailable after three failures; circuit breaker opened"
                        ],
                    }
                )
                continue
            company_requests = 0
            company_budget = max(1, max_doc_requests // max(1, len(aliases)))
            LOG.info("GDELT news metadata: %s", symbol)
            safe_terms = []
            for name in names:
                if not name or any(c in name for c in ('"', "\n", "\r")):
                    raise PipelineError("News aliases must be plain company names")
                safe_terms.append(f'"{name}"')
            query = "(" + " OR ".join(safe_terms) + ") sourcelang:english"
            pending = [(news_start, cutoff)]
            result = {
                "symbol": symbol,
                "status": "complete",
                "records": 0,
                "capped_windows": [],
                "issues": [],
            }
            while pending:
                lower, upper = pending.pop()
                if request_count >= max_doc_requests or company_requests >= company_budget:
                    result["capped_windows"].append(
                        {
                            "start": lower.isoformat(),
                            "end": upper.isoformat(),
                            "reason": "run_request_budget",
                        }
                    )
                    continue
                params = {
                    "query": query,
                    "mode": "artlist",
                    "format": "json",
                    "sort": "datedesc",
                    "maxrecords": 250,
                    "startdatetime": lower.strftime("%Y%m%d%H%M%S"),
                    "enddatetime": upper.strftime("%Y%m%d%H%M%S"),
                }
                try:
                    request_count += 1
                    company_requests += 1
                    payload, ref, envelope = cache.json(settings.gdelt_doc_url, params)
                    refs.append(ref)
                    observations = parse_doc(payload, symbol, ref, envelope)
                    consecutive_failures = 0
                    if len(observations) >= 250 and upper - lower > timedelta(hours=1):
                        midpoint = lower + (upper - lower) / 2
                        pending.extend([(lower, midpoint), (midpoint, upper)])
                        continue
                    if len(observations) >= 250:
                        result["capped_windows"].append(
                            {
                                "start": lower.isoformat(),
                                "end": upper.isoformat(),
                                "reason": "provider_record_cap",
                            }
                        )
                    doc_rows.extend(observations)
                    result["records"] += len(observations)
                except PipelineError as exc:
                    result["issues"].append(str(exc))
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
            if result["issues"] or result["capped_windows"]:
                result["status"] = "complete_with_issues" if result["records"] else "failed"
            results.append(result)
        LOG.info("GDELT GKG archive partitions: %d", gkg_partitions)
        gkg_result = {"name": "gkg", "status": "complete", "partitions": [], "issues": []}
        try:
            ref, body, _ = cache.get(
                f"{settings.gdelt_gkg_base}/lastupdate.txt", revision_scope=scope
            )
            refs.append(ref)
            advertised = []
            for line in body.decode().splitlines():
                parts = line.split()
                if len(parts) == 3 and parts[2].endswith(".gkg.csv.zip"):
                    filename = urlsplit(parts[2]).path.rsplit("/", 1)[-1]
                    if re.fullmatch(r"\d{14}\.gkg\.csv\.zip", filename):
                        advertised.append((filename, parts[1]))
            if len(advertised) != 1:
                raise PipelineError("GDELT lastupdate GKG listing invalid")
            latest = datetime.strptime(advertised[0][0][:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            for offset in range(gkg_partitions):
                stamp = latest - timedelta(minutes=15 * offset)
                filename = stamp.strftime("%Y%m%d%H%M%S") + ".gkg.csv.zip"
                try:
                    ref, body, envelope = cache.get(f"{settings.gdelt_gkg_base}/{filename}")
                    refs.append(ref)
                    observations, scanned = gkg_rows(
                        body,
                        aliases,
                        ref,
                        envelope,
                        expected_md5=advertised[0][1] if offset == 0 else None,
                    )
                    gkg_data.extend(observations)
                    gkg_result["partitions"].append(
                        {
                            "timestamp": stamp.isoformat(),
                            "scanned_records": scanned,
                            "matched_records": len(observations),
                            "raw": ref,
                        }
                    )
                except (PipelineError, UnicodeError, csv.Error) as exc:
                    gkg_result["issues"].append(
                        str(exc) if isinstance(exc, PipelineError) else "GKG encoding/TSV failure"
                    )
        except PipelineError as exc:
            gkg_result["issues"].append(str(exc))
        if gkg_result["issues"]:
            gkg_result["status"] = "complete_with_issues" if gkg_result["partitions"] else "failed"
        results.append(gkg_result)
    finally:
        cache.close()
    # The same URL may match multiple companies; retain each attribution, deduplicate within it.
    deduped = {(r["symbol"], r["url"]): r for r in doc_rows}
    return {
        "source": "gdelt",
        "news_start": news_start.isoformat(),
        "news_end": cutoff.isoformat(),
        "coverage_scope": "recent_DOC_window_and_explicit_GKG_partitions_not_ten_year_news_history",
        "requests": request_count,
        "results": results,
        "articles": list(deduped.values()),
        "gkg": gkg_data,
        "raw": refs,
    }
