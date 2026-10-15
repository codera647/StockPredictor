import hashlib
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd
from stock_predictor.config import PipelineError
from stock_predictor.datasets.auxiliary import canonical_url
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.storage.archive import digest, json_bytes

LOG = logging.getLogger(__name__)
NEWS_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class PartsReader(io.RawIOBase):
    """Read CSV ranges in order without preallocating or copying a second large file."""

    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.stream = None
        self.sha = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.read(len(buffer))
        buffer[:len(block)] = block
        return len(block)

    def read(self, size=-1):
        blocks = []
        remaining = size
        while size < 0 or remaining:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open("rb")
            block = self.stream.read(remaining if size >= 0 else 1024 * 1024)
            if not block:
                self.stream.close()
                self.stream = None
                continue
            blocks.append(block)
            self.sha.update(block)
            self.count += len(block)
            remaining -= len(block)
        return b"".join(blocks)

    def close(self):
        if self.stream is not None:
            self.stream.close()
        super().close()

    def validate(self, config):
        if self.count != config["news_bytes"] or self.sha.hexdigest() != config["news_sha256"]:
            raise PipelineError("FNSPID source failed its pinned full-file LFS SHA-256")


def download(config, directory):
    """Resume independently hashed ranges; validate the full digest during CSV parsing."""
    size, length = config["news_bytes"], config["news_chunk_bytes"]
    url = (
        "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/"
        + config["news_revision"]
        + "/"
        + config["news_path"]
    )
    parts = directory / (config["news_sha256"] + ".parts")
    parts.mkdir(parents=True, exist_ok=True)

    def part(start):
        end = min(start + length, size) - 1
        path, marker = parts / f"{start}.csvpart", parts / f"{start}.json"
        if marker.exists() and path.exists():
            expected = json.loads(marker.read_text())
            if path.stat().st_size == end - start + 1 and file_hash(path) == expected["sha256"]:
                return path
        for attempt in range(4):
            try:
                sha, count = hashlib.sha256(), 0
                with httpx.Client(follow_redirects=True, timeout=120) as client:
                    with client.stream(
                        "GET",
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                        params={"download": "true", "part": str(start)},
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("content-range") != (
                            f"bytes {start}-{end}/{size}"
                        ):
                            raise PipelineError(
                                "News server did not honor the requested byte range"
                            )
                        with path.open("wb") as stream:
                            for block in response.iter_bytes(chunk_size=1024 * 1024):
                                if count + len(block) > end - start + 1:
                                    raise PipelineError("News range exceeded its declared length")
                                stream.write(block)
                                sha.update(block)
                                count += len(block)
                if count != end - start + 1:
                    raise PipelineError("Incomplete news byte range")
                marker.write_text(json.dumps({"sha256": sha.hexdigest(), "bytes": count}))
                return path
            except (httpx.HTTPError, PipelineError):
                if attempt == 3:
                    raise PipelineError(
                        "News download failed; completed ranges can be resumed"
                    ) from None
                time.sleep(attempt + 1)

    starts = list(range(0, size, length))
    completed, paths = 0, {}
    with ThreadPoolExecutor(max_workers=config["news_download_workers"]) as pool:
        futures = {pool.submit(part, start): start for start in starts}
        for future in as_completed(futures):
            paths[futures[future]] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(starts):
                LOG.info("News download: %d/%d validated ranges", completed, len(starts))
    return [paths[start] for start in starts]


def clean_news(source, symbols, config):
    data = source.rename(
        columns={
            "Date": "publication_at",
            "Article_title": "headline",
            "Stock_symbol": "symbol",
            "Url": "url",
            "Publisher": "publisher",
        }
    ).copy()
    data["symbol"] = data.symbol.astype(str).str.strip().str.upper()
    data = data[data.symbol.isin(symbols)].copy()
    data["publication_at"] = pd.to_datetime(data.publication_at, utc=True, errors="coerce")
    data["headline"] = data.headline.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data["canonical_url"] = data.url.map(canonical_url)
    valid = data.publication_at.notna() & data.headline.str.len().ge(5) & data.canonical_url.notna()
    bad = int((~valid).sum())
    data = data[valid].copy()
    data["available_at_reconstructed"] = data.publication_at + pd.Timedelta(
        hours=config["news_delay_hours"]
    )
    data["availability_policy"] = "reported_timestamp_plus_24h_not_verified_first_receive"
    # The same story can have many stock links; deduplicate only within each stock.
    data = data.sort_values("publication_at").drop_duplicates(["symbol", "canonical_url"])
    data = data.drop_duplicates(["symbol", "publication_at", "headline"])
    return data.reset_index(drop=True), bad


def collect_news(archive, symbols, config, directory):
    if config.get("news_format") == "parquet_columns":
        from stock_predictor_fundamental_news.parquet_sources import collect_parquet_news

        return collect_parquet_news(archive, symbols, config, directory)
    source = {
        k: config[k]
        for k in ("news_revision", "news_path", "news_bytes", "news_sha256", "news_delay_hours")
    }
    source["symbols"] = sorted(symbols)
    checkpoint = "checkpoints/fundamental-news/source-" + digest(json_bytes(source)) + ".json"
    cached = archive.read(checkpoint)
    if cached:
        result = json.loads(cached)
        read_json(archive, result["manifest"])
        if archive.read(result["table"]["key"], sha256=result["table"]["sha256"]) is None:
            raise PipelineError("Cached filtered news is missing")
        return result
    paths = download(config, directory)
    stream = PartsReader(paths)
    retained, scanned, invalid = [], 0, 0
    try:
        for chunk in pd.read_csv(
            stream,
            usecols=NEWS_COLUMNS,
            dtype=str,
            keep_default_na=False,
            chunksize=25000,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="error",
        ):
            scanned += len(chunk)
            selected, bad = clean_news(chunk, symbols, config)
            invalid += bad
            if not selected.empty:
                retained.append(selected)
            if scanned % 500000 == 0:
                LOG.info("News CSV scanned: %d rows", scanned)
    except (pd.errors.ParserError, UnicodeError) as exc:
        raise PipelineError(
            "Pinned news CSV could not be parsed; no rows were silently skipped"
        ) from exc
    finally:
        stream.close()
    stream.validate(config)
    if not retained:
        raise PipelineError("No matching historical news records were collected")
    news = pd.concat(retained, ignore_index=True).sort_values("publication_at")
    before = len(news)
    news = news.drop_duplicates(["symbol", "canonical_url"])
    news = news.drop_duplicates(["symbol", "publication_at", "headline"]).reset_index(drop=True)
    table = parquet(archive, "fundamental-news/filtered_headlines", news)
    metadata = {
        "source": source,
        "scanned_rows": scanned,
        "invalid_matching_rows": invalid,
        "cross_chunk_duplicates": before - len(news),
        "retained_rows": len(news),
        "first_publication": news.publication_at.min().isoformat(),
        "last_publication": news.publication_at.max().isoformat(),
        "per_symbol_rows": news.symbol.value_counts().sort_index().to_dict(),
        "table": table,
        "full_source_sha256_verified": True,
        "source_rights_note": (
            "Authors' July 2025 release notice coexists with older restrictive README wording; "
            "research use only here. No upstream scraper code or article bodies are redistributed."
        ),
    }
    manifest = archive.json(
        "manifests/fundamental-news/source-" + digest(json_bytes(source)) + ".json", metadata
    )
    result = {"manifest": manifest, "table": table, "retained_rows": len(news)}
    archive.json(checkpoint, result)
    return result
