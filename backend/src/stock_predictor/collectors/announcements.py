import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from stock_predictor.config import PipelineError
from stock_predictor.storage.archive import Archive

from .cache import CachedSource
from .http import SourceClient

LOG = logging.getLogger(__name__)
AGENT = "StockPredictorResearch"


def public_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise PipelineError("Announcement URL must be public HTTPS without credentials")
    if "." not in parsed.hostname or re.fullmatch(r"[0-9.]+", parsed.hostname):
        raise PipelineError("Announcement URL must use a public company domain")
    if any(name in parsed.query.lower() for name in ("token=", "api_key=", "password=")):
        raise PipelineError("Credential-bearing announcement URLs are unsupported")
    return url


def domain_family(host: str) -> str:
    # Curated company domains below use ordinary two-label public suffixes.
    return ".".join(host.split(".")[-2:])


class PageMetadata(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.feeds, self.structured = [], [], []
        self.anchor, self.script = None, None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and values.get("href"):
            self.anchor = {"href": values["href"], "title": values.get("title", ""), "text": ""}
        if tag == "link" and values.get("type") in ("application/rss+xml", "application/atom+xml"):
            if values.get("href"):
                self.feeds.append(values["href"])
        if tag == "script" and values.get("type") == "application/ld+json":
            self.script = ""

    def handle_data(self, data):
        if self.anchor is not None:
            self.anchor["text"] += data
        if self.script is not None:
            self.script += data

    def handle_endtag(self, tag):
        if tag == "a" and self.anchor is not None:
            self.links.append(self.anchor)
            self.anchor = None
        if tag == "script" and self.script is not None:
            try:
                self.structured.append(json.loads(self.script))
            except ValueError:
                pass
            self.script = None


def rss_rows(body: bytes, symbol: str, url: str, ref: dict, envelope: dict) -> list[dict]:
    if len(body) > 8 * 1024 * 1024 or b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
        raise PipelineError("RSS XML size/entities are unsupported")
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        raise PipelineError("Announcement feed returned invalid RSS/Atom XML") from None
    rows = []
    for item in root.iter():
        if item.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        values, link = {}, None
        for child in item:
            tag = child.tag.rsplit("}", 1)[-1]
            values[tag] = "".join(child.itertext()).strip()
            if tag == "link":
                link = child.attrib.get("href") or values[tag]
        if not link:
            continue
        published = values.get("pubDate") or values.get("published") or values.get("updated")
        stamp = None
        if published:
            try:
                stamp = parsedate_to_datetime(published)
            except (ValueError, TypeError):
                try:
                    stamp = datetime.fromisoformat(published.replace("Z", "+00:00"))
                except ValueError:
                    pass
        if stamp and stamp.tzinfo:
            stamp = stamp.astimezone(UTC).isoformat()
        else:
            stamp = None
        rows.append(
            {
                "symbol": symbol,
                "url": urljoin(url, link),
                "title": values.get("title"),
                "publication_at": stamp,
                "first_observed_at": envelope["retrieved_at"],
                "source": "official_company_feed",
                "raw_snapshot_key": ref["key"],
                "event_at": None,
                "event_status": "announcement_metadata_no_inferred_earnings_date",
            }
        )
    return rows


def structured_events(value):
    if isinstance(value, list):
        for item in value:
            yield from structured_events(item)
    elif isinstance(value, dict):
        types = value.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if any(kind.endswith("Event") for kind in types) and value.get("startDate"):
            yield value
        for key, item in value.items():
            if key != "@type" and isinstance(item, (list, dict)):
                yield from structured_events(item)


def collect_announcements(
    archive: Archive,
    sources: list[dict],
    *,
    scope: str,
    refresh: bool,
) -> dict:
    client = SourceClient(
        "company_ir", archive, headers={"User-Agent": AGENT}, interval=1, max_attempts=2
    )
    cache = CachedSource(client, archive, refresh=refresh)
    robots, results, observations, refs = {}, [], [], []

    def permitted(url):
        parsed = urlsplit(public_url(url))
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in robots:
            robot_url = origin + "/robots.txt"
            for _attempt in range(4):
                ref, response = client.get(
                    robot_url, acceptable_statuses=(404, 301, 302, 303, 307, 308)
                )
                if response.status_code not in (301, 302, 303, 307, 308):
                    break
                destination = urljoin(robot_url, response.headers.get("location", ""))
                public_url(destination)
                if domain_family(urlsplit(destination).hostname) != domain_family(parsed.hostname):
                    raise PipelineError("Robots policy redirects to an unconfigured domain")
                robot_url = destination
            else:
                raise PipelineError("Robots policy redirect limit reached")
            refs.append(ref)
            parser = RobotFileParser()
            parser.set_url(robot_url)
            if response.status_code == 404:
                parser.parse([])
            else:
                parser.parse(response.text.splitlines())
            robots[origin] = parser
        if not robots[origin].can_fetch(AGENT, url):
            raise PipelineError("Official site robots policy disallows this collection URL")

    def fetch(url, allowed_family):
        for _ in range(4):
            public_url(url)
            if domain_family(urlsplit(url).hostname) != allowed_family:
                raise PipelineError("Official site redirects to an unconfigured domain")
            permitted(url)
            # Public pages may redirect, but private API clients never follow redirects.
            try:
                ref, body, envelope = cache.get(url, revision_scope=scope)
                refs.append(ref)
                return url, ref, body, envelope
            except PipelineError as exc:
                if "HTTP 30" not in str(exc):
                    raise
                location = getattr(exc, "redirect_location", None)
                if not location:
                    raise PipelineError("Official site redirect omitted its destination") from None
                url = urljoin(url, location)
        raise PipelineError("Official site redirect limit reached")

    try:
        for source in sources:
            symbol, url = source["symbol"], source["url"]
            LOG.info("Official announcements snapshot: %s", symbol)
            result = {"symbol": symbol, "url": url, "status": "complete", "records": 0}
            try:
                family = domain_family(urlsplit(public_url(url)).hostname)
                fetched, ref, body, envelope = fetch(url, family)
                if source.get("kind") == "rss":
                    rows = rss_rows(body, symbol, fetched, ref, envelope)
                else:
                    page = PageMetadata()
                    page.feed(body.decode("utf-8", errors="replace"))
                    rows = []
                    for link in page.links:
                        label = (link["title"] or link["text"]).strip()
                        if label and re.search(
                            r"earnings|results|release|dividend|split|conference|annual meeting",
                            label,
                            re.I,
                        ):
                            rows.append(
                                {
                                    "symbol": symbol,
                                    "url": urljoin(fetched, link["href"]),
                                    "title": label,
                                    "publication_at": None,
                                    "first_observed_at": envelope["retrieved_at"],
                                    "source": "official_company_page",
                                    "raw_snapshot_key": ref["key"],
                                    "event_at": None,
                                    "event_status": "candidate_link_no_inferred_earnings_date",
                                }
                            )
                    for value in page.structured:
                        for event in structured_events(value):
                            rows.append(
                                {
                                    "symbol": symbol,
                                    "url": event.get("url", fetched),
                                    "title": event.get("name"),
                                    "publication_at": None,
                                    "first_observed_at": envelope["retrieved_at"],
                                    "source": "official_company_structured_event",
                                    "raw_snapshot_key": ref["key"],
                                    "event_at": str(event["startDate"]),
                                    "event_status": "source_declared_event_requires_review",
                                }
                            )
                    # Only explicitly advertised same-company-domain feeds are followed.
                    for candidate in page.feeds[:2]:
                        feed = urljoin(fetched, candidate)
                        if domain_family(urlsplit(feed).hostname or "") != family:
                            continue
                        try:
                            feed_url, feed_ref, feed_body, feed_envelope = fetch(feed, family)
                            rows.extend(
                                rss_rows(feed_body, symbol, feed_url, feed_ref, feed_envelope)
                            )
                        except PipelineError as exc:
                            result.setdefault("feed_issues", []).append(str(exc))
                observations.extend(rows)
                result.update({"records": len(rows), "fetched_url": fetched, "raw": ref})
                if not rows:
                    result["status"] = "snapshot_only_no_event_metadata"
            except PipelineError as exc:
                result.update({"status": "failed", "reason": str(exc)})
            results.append(result)
    finally:
        cache.close()
    observations = list({(r["symbol"], r["url"], r["title"]): r for r in observations}.values())
    return {
        "source": "official_company_announcements",
        "results": results,
        "coverage_scope": "prospective_snapshots_not_reconstructed_historical_calendars",
        "announcements": observations,
        "raw": refs,
    }
