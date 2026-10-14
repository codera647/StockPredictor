import base64
import io
import json
import zipfile
from datetime import date

import httpx
import pandas as pd
import pytest
from stock_predictor.collectors.announcements import PageMetadata, rss_rows, structured_events
from stock_predictor.collectors.cache import CachedSource
from stock_predictor.collectors.fred import collect_fred, observation_rows, revision_windows
from stock_predictor.collectors.gdelt import gkg_rows, parse_doc
from stock_predictor.collectors.http import SourceClient
from stock_predictor.collectors.sec import fact_rows, filing_url, submission_rows
from stock_predictor.config import PipelineError, Settings
from stock_predictor.storage.archive import Archive, digest, json_bytes

REF = {"key": "raw/example", "sha256": "hash"}
ENVELOPE = {"retrieved_at": "2026-10-14T15:00:00+00:00"}


def test_query_keys_and_echoes_are_redacted_and_cache_restores(tmp_path):
    archive = Archive(tmp_path)
    secret = "private-fred-key-123456789"
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.params["api_key"] == secret
        return httpx.Response(200, json={"observations": [], "echo": secret})

    source = SourceClient(
        "fred", archive, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    cache = CachedSource(source, archive)
    _, ref, _ = cache.json("https://example.org", {"api_key": secret})
    envelope = archive.load_snapshot(ref)
    assert secret not in json.dumps(envelope)
    assert secret.encode() not in base64.b64decode(envelope["response"]["body_base64"])
    assert envelope["response"]["credential_echo_redacted"]
    restored, _, _ = cache.json("https://example.org", {"api_key": secret})
    assert restored["echo"] == "[REDACTED]" and len(calls) == 1
    cache.close()


def test_size_bound_preserves_partial_response_but_not_success_checkpoint(tmp_path):
    archive = Archive(tmp_path)
    source = SourceClient(
        "gdelt",
        archive,
        max_response_bytes=10,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 20))
        ),
    )
    cache = CachedSource(source, archive)
    with pytest.raises(PipelineError, match="size bound"):
        cache.get("https://example.org/large")
    assert not list(tmp_path.glob("checkpoints/**/*.json"))
    assert len(list(tmp_path.glob("raw/gdelt/*.gz"))) == 1
    cache.close()


@pytest.mark.parametrize("cik", [320193, "0000320193"])
def test_sec_shards_keep_acceptance_and_accessions_separate_from_fact_end_date(cik):
    recent = {
        "accessionNumber": ["0000320193-26-000001"],
        "filingDate": ["2026-01-05"],
        "form": ["10-K"],
        "primaryDocument": ["aapl.htm"],
        "acceptanceDateTime": ["2026-01-05T21:00:00Z"],
    }
    assert submission_rows({"filings": {"recent": recent}}) == submission_rows(recent)
    assert filing_url(
        "https://www.sec.gov/Archives/edgar/data",
        "0000320193",
        recent["accessionNumber"][0],
        "aapl.htm",
    ).endswith("/320193/000032019326000001/aapl.htm")
    with pytest.raises(PipelineError, match="unsafe"):
        filing_url(
            "https://www.sec.gov", "0000320193", recent["accessionNumber"][0], "../other.htm"
        )
    payload = {
        "cik": cik,
        "facts": {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {
                                "val": 100,
                                "end": "2025-12-31",
                                "filed": "2026-01-05",
                                "accn": recent["accessionNumber"][0],
                            },
                            {"val": 200, "end": "2026-03-31", "filed": "2026-05-05"},
                        ]
                    }
                }
            }
        },
    }
    rows = fact_rows(payload, date(2026, 1, 1), date(2026, 2, 1), REF, ENVELOPE)
    assert len(rows) == 1 and rows[0]["period_end"] == "2025-12-31"
    assert rows[0]["available_at"] == ENVELOPE["retrieved_at"]
    assert rows[0]["accession_number"] == recent["accessionNumber"][0]


def test_sec_misaligned_arrays_fail():
    with pytest.raises(PipelineError, match="inconsistent"):
        submission_rows(
            {
                "accessionNumber": ["a"],
                "filingDate": [],
                "form": ["10-Q"],
                "primaryDocument": ["file"],
            }
        )


def test_fred_revisions_and_missing_values_keep_vintage_validity():
    rows = observation_rows(
        "CPIAUCSL",
        [
            {
                "date": "2020-01-01",
                "realtime_start": "2020-02-13",
                "realtime_end": "2021-02-09",
                "value": "100.5",
            },
            {
                "date": "2020-01-01",
                "realtime_start": "2021-02-10",
                "realtime_end": "9999-12-31",
                "value": "101.2",
            },
            {
                "date": "2020-02-01",
                "realtime_start": "2020-03-13",
                "realtime_end": "9999-12-31",
                "value": ".",
            },
        ],
        REF,
        ENVELOPE,
    )
    assert rows[0]["value"] != rows[1]["value"]
    assert rows[2]["value"] is None and rows[2]["value_missing"]
    assert all(r["vintage_precision"] == "date_only" for r in rows)


def test_fred_revision_windows_cover_the_history_without_exceeding_provider_limit():
    start, end = date(2015, 1, 3), date(2026, 10, 13)
    windows = list(revision_windows(start, end))
    assert windows[0][0] == start and windows[-1][1] == end
    assert all((upper - lower).days < 1500 for lower, upper in windows)
    for first, second in zip(windows, windows[1:], strict=False):
        assert (second[0] - first[1]).days == 1


def test_fred_offset_pagination_and_secrets(tmp_path, monkeypatch):
    archive = Archive(tmp_path)
    queries = []

    def fake(self, url, params, *, revision_scope):
        queries.append(params)
        if url.endswith("/series"):
            return {"seriess": [{"id": "DGS10", "units": "Percent"}]}, REF, ENVELOPE
        if url.endswith("vintagedates"):
            return {"count": 2, "vintage_dates": ["2020-01-01", "2020-01-02"]}, REF, ENVELOPE
        return (
            {
                "count": 2,
                "observations": [
                    {
                        "date": f"2020-01-0{params['offset'] + 1}",
                        "value": "1.0",
                        "realtime_start": "2020-01-02",
                        "realtime_end": "2020-01-03",
                    }
                ],
            },
            REF,
            ENVELOPE,
        )

    monkeypatch.setattr(CachedSource, "json", fake)
    settings = Settings(
        "https://data.alpaca.markets", "sip", "key", "secret", fred_key="fred-private"
    )
    result = collect_fred(
        settings,
        archive,
        ["DGS10"],
        date(2020, 1, 1),
        date(2020, 2, 1),
        scope="2026-10-14",
        refresh=False,
    )
    assert len(result["observations"]) == 2
    assert [p["offset"] for p in queries if "output_type" in p] == [0, 1]
    assert all(p["output_type"] == 1 for p in queries if "output_type" in p)
    assert "fred-private" not in json.dumps(result)


def test_gdelt_failed_schema_is_not_empty_news():
    with pytest.raises(PipelineError, match="failed feed"):
        parse_doc({}, "AAPL", REF, ENVELOPE)
    assert parse_doc({"articles": []}, "AAPL", REF, ENVELOPE) == []


def zipped_gkg(line):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("20261014000000.gkg.csv", line)
    return stream.getvalue()


def test_gkg_matches_organizations_without_short_ticker_false_positives():
    fields = [""] * 27
    fields[0], fields[1], fields[4] = "1", "20261014000000", "https://example.org/news"
    fields[11] = "Visa Person Name"
    fields[13] = "NVIDIA;Bank of America"
    rows, scanned = gkg_rows(
        zipped_gkg("\t".join(fields) + "\n"), {"NVDA": ["NVIDIA"], "V": ["Visa"]}, REF, ENVELOPE
    )
    assert scanned == 1 and len(rows) == 1 and rows[0]["symbol"] == "NVDA"
    with pytest.raises(PipelineError, match="27"):
        gkg_rows(zipped_gkg("invalid\trow\n"), {}, REF, ENVELOPE)
    with pytest.raises(PipelineError, match="checksum"):
        gkg_rows(zipped_gkg("\t".join(fields)), {}, REF, ENVELOPE, expected_md5="wrong")


def test_rss_and_event_metadata_do_not_invent_future_earnings_dates():
    body = (
        b"<rss><channel><item><title>Quarterly results</title>"
        b"<link>https://example.org/results</link>"
        b"<pubDate>Wed, 14 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>"
    )
    rows = rss_rows(body, "AAPL", "https://example.org/rss", REF, ENVELOPE)
    assert rows[0]["publication_at"] == "2026-10-14T10:00:00+00:00"
    assert rows[0]["event_at"] is None
    page = PageMetadata()
    page.feed(
        '<link type="application/rss+xml" href="/rss">'
        '<a href="/events">Earnings call</a><script type="application/ld+json">'
        '{"@type":"Event","startDate":"2026-11-01"}</script>'
    )
    assert page.feeds == ["/rss"] and page.links[0]["text"] == "Earnings call"
    assert list(structured_events(page.structured))[0]["startDate"] == "2026-11-01"
    with pytest.raises(PipelineError, match="entities"):
        rss_rows(b"<!DOCTYPE rss><rss/>", "AAPL", "https://example.org", REF, ENVELOPE)


def test_cli_resumes_verified_daily_manifest_without_reextracting(tmp_path, monkeypatch):
    from stock_predictor import cli

    archive = Archive(tmp_path / "archive")
    report = {
        "start": "2016-01-04",
        "end": "2026-10-13",
        "symbols": ["AAPL", "SPY"],
        "bar_rows": 2,
        "action_rows": 0,
        "extraction_status": "complete",
        "issues": [],
        "outputs": {},
    }
    ref = archive.json("reports/daily.json", report)
    archive.json("manifests/daily.json", {"status": "complete", "report": ref})
    monkeypatch.setattr(
        Settings,
        "load",
        lambda *a, **kw: Settings(
            "https://data.alpaca.markets",
            "sip",
            "key",
            "secret",
        ),
    )
    monkeypatch.setattr(
        cli, "run_extraction", lambda *a, **kw: pytest.fail("unexpected extraction")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "stock-predictor",
            "extract",
            "--sources",
            "daily",
            "--no-upload",
            "--daily-manifest",
            "manifests/daily.json",
            "--archive-root",
            str(archive.root),
            "--summary-dir",
            str(tmp_path / "summary"),
        ],
    )
    assert cli.main() == 0
    summary = json.loads((tmp_path / "summary/summary.json").read_text())
    assert summary["bar_rows"] == 2 and summary["symbols"] == 2
    (archive.root / ref["key"]).write_bytes(b"corrupted")
    assert cli.main() == 1


def test_sec_replay_adds_valid_string_cik_facts_without_rewriting_old_artifacts(tmp_path):
    from stock_predictor.extraction import parquet
    from stock_predictor.replay import read_json, replay_sec_facts

    archive = Archive(tmp_path)
    payload = {
        "cik": "0002012383",
        "facts": {
            "us-gaap": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {"val": 100, "end": "2025-12-31", "filed": "2026-01-05", "accn": "new"},
                        ]
                    }
                }
            }
        },
    }
    envelope = {
        "source": "sec",
        "retrieved_at": "2026-10-14T16:00:00+00:00",
        "response": {"body_base64": base64.b64encode(json_bytes(payload)).decode()},
    }
    raw = archive.snapshot(envelope)
    spec = {
        "version": 1,
        "source": "sec",
        "url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0002012383.json",
        "params": {},
        "revision_scope": "2026-10-14",
    }
    archive.json(f"checkpoints/evidence/v1/{digest(json_bytes(spec))}.json", {"raw": raw})
    old_rows = fact_rows(payload, date(2026, 1, 1), date(2026, 2, 1), raw, envelope)
    old_rows[0]["issuer_cik"] = "0000320193"
    old_table = parquet(archive, "sec_facts", old_rows)
    original_bytes = archive.read(old_table["key"])
    evidence = {
        "sources": {
            "sec": {
                "facts_count": 1,
                "companies": [
                    {
                        "symbol": "BLK",
                        "issuer_cik": "0002012383",
                        "status": "complete_with_issues",
                        "issues": ["SEC Company Facts schema is invalid"],
                    }
                ],
            }
        },
        "issues": ["sec:BLK:complete_with_issues", "gdelt:unavailable"],
        "outputs": {"sec_facts": old_table},
    }
    evidence_manifest = archive.json(
        "manifests/evidence/old.json",
        {
            "raw": [raw],
            "created_at": "2026-10-14T16:00:00+00:00",
        },
    )
    report = {
        "start": "2026-01-01",
        "end": "2026-02-01",
        "daily": {"issues": []},
        "evidence": evidence,
        "outputs": {"sec_facts": old_table},
    }
    report_ref = archive.json("reports/full/old.json", report)
    archive.json(
        "manifests/full/old.json", {"report": report_ref, "evidence_manifest": evidence_manifest}
    )
    settings = Settings("https://data.alpaca.markets", "sip", "key", "secret")
    result = replay_sec_facts(archive, settings, "manifests/full/old.json")
    assert result["report"]["evidence"]["sources"]["sec"]["facts_count"] == 2
    assert result["report"]["issues"] == ["gdelt:unavailable"]
    assert result["report"]["replay"]["network_source_requests"] == 0
    new_table = result["report"]["outputs"]["sec_facts"]
    assert len(pd.read_parquet(io.BytesIO(archive.read(new_table["key"])))) == 2
    assert archive.read(old_table["key"]) == original_bytes
    assert read_json(archive, report_ref)["evidence"]["sources"]["sec"]["facts_count"] == 1
