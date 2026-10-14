import base64
import csv
import ftplib
import io
import json
import logging
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

from stock_predictor.config import PipelineError, Settings
from stock_predictor.storage.archive import Archive

from .http import SourceClient


def parse_nasdaq(body: bytes) -> list[dict]:
    text = body.decode("utf-8-sig")
    lines = [
        line for line in text.splitlines() if line and not line.startswith("File Creation Time")
    ]
    reader = csv.DictReader(lines, delimiter="|")
    if not reader.fieldnames or not ({"Symbol", "ACT Symbol"} & set(reader.fieldnames)):
        raise PipelineError("Unexpected Nasdaq directory schema")
    return list(reader)


def parse_sec(body: bytes) -> list[dict]:
    try:
        payload = json.loads(body)
    except ValueError:
        raise PipelineError("SEC mapping response is not JSON") from None
    if not isinstance(payload, dict):
        raise PipelineError("Unexpected SEC mapping schema")
    fields, rows = payload.get("fields"), payload.get("data")
    if not isinstance(fields, list) or not isinstance(rows, list):
        raise PipelineError("Unexpected SEC mapping schema")
    if not {"cik", "ticker", "name", "exchange"}.issubset(fields):
        raise PipelineError("SEC mapping required fields are missing")
    result = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise PipelineError("SEC mapping row length does not match fields")
        result.append(dict(zip(fields, row, strict=True)))
    return result


def ftp_snapshot(url: str, archive: Archive) -> tuple[dict, bytes]:
    parsed = urlsplit(url)
    for attempt in range(3):
        try:
            buffer = io.BytesIO()
            with ftplib.FTP() as ftp:
                ftp.connect(parsed.hostname, parsed.port or 21, timeout=20)
                ftp.login()  # Public anonymous directory; no account credentials.
                ftp.retrbinary(f"RETR {parsed.path}", buffer.write)
            body = buffer.getvalue()
            envelope = {
                "schema_version": 1,
                "source": "nasdaq",
                "retrieved_at": datetime.now(UTC).isoformat(),
                "request": {"method": "FTP_RETR", "url": url, "params": {}},
                "response": {
                    "status": 200,
                    "headers": {},
                    "body_base64": base64.b64encode(body).decode(),
                },
            }
            return archive.snapshot(envelope), body
        except ftplib.all_errors:
            if attempt == 2:
                raise PipelineError("Nasdaq FTP directory unavailable after retries") from None
            time.sleep(2**attempt)
    raise PipelineError("Nasdaq FTP retries exhausted")


def collect_directories(settings: Settings, archive: Archive) -> tuple[list[dict], list[dict]]:
    results, sec_records = [], []
    targets = [
        ("nasdaq_listed", settings.nasdaq_listed_url),
        ("nasdaq_other", settings.nasdaq_other_url),
        ("sec_mapping", settings.sec_mapping_url),
    ]
    for name, url in targets:
        configured_url = url
        parsed = urlsplit(url)
        # Same provider's verified HTTPS copies avoid blocked anonymous FTP ports
        # on hosted runners; retain the exact fetched URL in raw provenance.
        if parsed.scheme == "ftp" and parsed.hostname == "ftp.nasdaqtrader.com":
            filename = parsed.path.rsplit("/", 1)[-1]
            if parsed.path == f"/symboldirectory/{filename}" and filename in (
                "nasdaqlisted.txt",
                "otherlisted.txt",
            ):
                url = f"https://www.nasdaqtrader.com/dynamic/SymDir/{filename}"
        client = None
        logging.info("Snapshotting directory %s", name)
        try:
            if name == "sec_mapping" and (
                "@" not in settings.sec_user_agent or "example.com" in settings.sec_user_agent
            ):
                raise PipelineError(
                    "SEC_USER_AGENT needs an application name and real contact email"
                )
            if url.startswith("ftp:"):
                ref, body = ftp_snapshot(url, archive)
            else:
                client = SourceClient(
                    "sec" if name == "sec_mapping" else "nasdaq",
                    archive,
                    headers={"User-Agent": settings.sec_user_agent}
                    if name == "sec_mapping"
                    else {},
                    interval=0.25,
                )
                ref, response = client.get(url)
                body = response.content
            rows = parse_sec(body) if name == "sec_mapping" else parse_nasdaq(body)
            if name == "sec_mapping":
                sec_records = rows
            results.append(
                {
                    "name": name,
                    "status": "complete",
                    "records": len(rows),
                    "raw": ref,
                    "configured_url": configured_url,
                    "fetched_url": url,
                }
            )
        except (PipelineError, UnicodeError) as exc:
            message = str(exc) if isinstance(exc, PipelineError) else "Directory encoding invalid"
            results.append({"name": name, "status": "failed", "reason": message})
            logging.warning("Directory %s failed: %s", name, message)
        finally:
            if client:
                client.close()
    return results, sec_records
