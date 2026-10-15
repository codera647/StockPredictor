"""Project headline columns from an immutable public Parquet mirror of FNSPID."""

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq
from stock_predictor.config import PipelineError
from stock_predictor.extraction import parquet
from stock_predictor.storage.archive import digest, json_bytes

from stock_predictor_fundamental_news.sources import NEWS_COLUMNS, clean_news

LOG = logging.getLogger(__name__)


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, directory, client):
        super().__init__()
        self.url, self.size, self.directory, self.client = url, size, directory, client
        self.position, self.cache, self.evidence = 0, [], {}
        directory.mkdir(parents=True, exist_ok=True)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def fetch(self, start, end):
        for a, b, body in self.cache:
            if a <= start and end <= b:
                return body[start - a : end - a]
        path = self.directory / f"{start}-{end}.bin"
        marker = path.with_suffix(".json")
        body = None
        if path.exists() and marker.exists():
            expected = json.loads(marker.read_text())
            candidate = path.read_bytes()
            if len(candidate) == end - start and digest(candidate) == expected["sha256"]:
                body = candidate
        if body is None:
            for attempt in range(4):
                try:
                    with self.client.stream(
                        "GET",
                        self.url,
                        headers={"Range": f"bytes={start}-{end - 1}"},
                        params={"download": "true", "range": f"{start}-{end}"},
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("content-range") != (
                            f"bytes {start}-{end - 1}/{self.size}"
                        ):
                            raise PipelineError("Parquet source ignored its byte range")
                        body = response.read()
                    if len(body) != end - start:
                        raise PipelineError("Incomplete Parquet column range")
                    break
                except (httpx.HTTPError, PipelineError):
                    if attempt == 3:
                        raise PipelineError(
                            "Public news column download failed; cached ranges remain"
                        ) from None
            path.write_bytes(body)
            marker.write_text(json.dumps({"sha256": digest(body)}))
        self.evidence[f"{start}:{end}"] = {"sha256": digest(body), "bytes": len(body)}
        self.cache.append((start, end, body))
        return body

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        body = self.fetch(self.position, end)
        self.position = end
        return body

    def readinto(self, buffer):
        body = self.read(len(buffer))
        buffer[: len(body)] = body
        return len(body)


def selected_groups(metadata, symbols):
    selected = []
    for i in range(metadata.num_row_groups):
        group = metadata.row_group(i)
        column = next(
            group.column(j)
            for j in range(group.num_columns)
            if group.column(j).path_in_schema == "Stock_symbol"
        )
        stats = column.statistics
        if stats is not None and stats.null_count == group.num_rows:
            continue
        if stats is not None and stats.has_min_max:
            lo, hi = stats.min, stats.max
            if isinstance(lo, bytes):
                lo, hi = lo.decode(), hi.decode()
            if not any(lo <= symbol <= hi for symbol in symbols):
                continue
        selected.append(i)
    return selected


def collect_parquet_news(archive, symbols, config, directory):
    spec_path = Path(config["news_parquet_spec"])
    spec = json.loads(spec_path.read_text())
    if (
        spec["repo"] != config["news_mirror_repo"]
        or spec["revision"] != config["news_mirror_revision"]
    ):
        raise PipelineError("Parquet source specification differs from the frozen configuration")
    source = {
        "format": "parquet_column_projection",
        "spec": spec,
        "symbols": sorted(symbols),
        "news_delay_hours": config["news_delay_hours"],
    }
    fingerprint = digest(json_bytes(source))
    checkpoint = f"checkpoints/fundamental-news/source-{fingerprint}.json"
    cached = archive.read(checkpoint)
    if cached:
        saved = json.loads(cached)
        if archive.read(saved["table"]["key"], sha256=saved["table"]["sha256"]) is None:
            raise PipelineError("Cached public news projection is missing")
        return saved
    directory = directory / fingerprint
    directory.mkdir(parents=True, exist_ok=True)

    def shard(number, entry):
        root = directory / str(number)
        ready, report_path = root / "filtered.parquet", root / "report.json"
        if ready.exists() and report_path.exists():
            report = json.loads(report_path.read_text())
            if digest(ready.read_bytes()) == report["filtered_sha256"]:
                return pd.read_parquet(ready), report
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{entry['path']}"
        with httpx.Client(timeout=90, follow_redirects=True) as client:
            stream = RangeFile(url, entry["size"], root, client)
            reader = pq.ParquetFile(stream, pre_buffer=False)
            groups = selected_groups(reader.metadata, symbols)
            # Fetch adjacent metadata columns together, excluding large article bodies.
            for i in groups:
                group = reader.metadata.row_group(i)
                columns = [
                    group.column(j)
                    for j in range(group.num_columns)
                    if group.column(j).path_in_schema in NEWS_COLUMNS
                ]
                starts = [
                    c.dictionary_page_offset if c.has_dictionary_page else c.data_page_offset
                    for c in columns
                ]
                ends = [a + c.total_compressed_size for a, c in zip(starts, columns, strict=True)]
                stream.fetch(min(starts), max(ends))
            raw = reader.read_row_groups(
                groups, columns=NEWS_COLUMNS, use_threads=False
            ).to_pandas()
            cleaned, invalid = clean_news(raw.fillna(""), symbols, config)
            report = {
                "source_file": entry,
                "metadata_rows": reader.metadata.num_rows,
                "row_groups": reader.metadata.num_row_groups,
                "selected_row_groups": len(groups),
                "read_rows": len(raw),
                "matching_rows": len(cleaned),
                "invalid_matching_rows": invalid,
                "downloaded_ranges": stream.evidence,
                "downloaded_bytes": sum(r["bytes"] for r in stream.evidence.values()),
                "full_original_file_sha256_verified": False,
            }
            stream.close()
        cleaned.to_parquet(ready, index=False)
        report["filtered_sha256"] = digest(ready.read_bytes())
        report_path.write_text(json.dumps(report))
        return cleaned, report

    tables, reports = [], []
    with ThreadPoolExecutor(max_workers=config["news_download_workers"]) as pool:
        futures = [pool.submit(shard, i, entry) for i, entry in enumerate(spec["files"])]
        for future in as_completed(futures):
            table, report = future.result()
            tables.append(table)
            reports.append(report)
            LOG.info(
                "Projected news shards: %d/%d; matching headlines in shard: %d",
                len(reports),
                len(spec["files"]),
                len(table),
            )
    news = pd.concat(tables, ignore_index=True).sort_values(
        ["publication_at", "symbol", "canonical_url"]
    )
    before = len(news)
    news = news.drop_duplicates(["symbol", "canonical_url"])
    news = news.drop_duplicates(["symbol", "publication_at", "headline"]).reset_index(drop=True)
    if news.empty:
        raise PipelineError("No matching news headlines were projected")
    table = parquet(archive, "fundamental-news/filtered_headlines", news)
    metadata = {
        "source": source,
        "shards": sorted(reports, key=lambda r: r["source_file"]["path"]),
        "metadata_rows": sum(r["metadata_rows"] for r in reports),
        "read_rows": sum(r["read_rows"] for r in reports),
        "downloaded_bytes": sum(r["downloaded_bytes"] for r in reports),
        "retained_rows": len(news),
        "cross_shard_duplicates": before - len(news),
        "invalid_matching_rows": sum(r["invalid_matching_rows"] for r in reports),
        "per_symbol_rows": news.symbol.value_counts().sort_index().to_dict(),
        "first_publication": news.publication_at.min().isoformat(),
        "last_publication": news.publication_at.max().isoformat(),
        "table": table,
        "full_original_file_sha256_verified": False,
        "source_limitations": [
            "Third-party FNSPID conversion; byte-equivalence to the original is unproven.",
            "Immutable mirror revision and downloaded column-range hashes are retained.",
            "Original full CSV/LFS hash is not claimed verified by a column projection.",
            "Research use only; upstream release and older licensing language conflict.",
        ],
    }
    manifest = archive.json(f"manifests/fundamental-news/source-{fingerprint}.json", metadata)
    result = {"manifest": manifest, "table": table, "retained_rows": len(news)}
    archive.json(checkpoint, result)
    return result
