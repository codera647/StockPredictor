import base64
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest
from stock_predictor.calendar import latest_completed_session, request_bounds, schedule
from stock_predictor.collectors.alpaca import AlpacaCollector, validate_page
from stock_predictor.collectors.directories import parse_nasdaq, parse_sec
from stock_predictor.collectors.http import SourceClient
from stock_predictor.config import PipelineError, Settings, load_universe
from stock_predictor.data_quality.bars import coverage_report, normalize_bar
from stock_predictor.extraction import run_extraction
from stock_predictor.storage.archive import Archive, digest, json_bytes


@pytest.fixture
def settings():
    return Settings("https://data.alpaca.markets", "sip", "hidden-key", "hidden-secret")


@pytest.fixture
def query():
    return {"symbols": "AAPL,SPY", "feed": "sip", "timeframe": "1Day", "adjustment": "raw"}


def client(archive, handler):
    return SourceClient(
        "alpaca",
        archive,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
        interval=0,
    )


def test_pages_are_consumed_even_when_first_symbol_fills_page(tmp_path, settings, query):
    archive = Archive(tmp_path)
    calls = []

    def handler(request):
        calls.append(dict(request.url.params))
        if len(calls) == 1:
            return httpx.Response(
                200, json={"bars": {"AAPL": [{"c": 100}]}, "next_page_token": "p2"}
            )
        return httpx.Response(200, json={"bars": {"SPY": [{"c": 200}]}, "next_page_token": None})

    collector = AlpacaCollector(settings, archive, client(archive, handler))
    state = collector.collect("bars", query)
    assert state["complete"] and len(state["pages"]) == 2
    assert calls[1]["page_token"] == "p2"
    collector.collect("bars", query)
    assert len(calls) == 2  # A completed, identical request is restored without network calls.
    collector.close()


def test_resume_starts_after_last_durable_page(tmp_path, settings, query):
    archive = Archive(tmp_path)

    def failing(request):
        if "page_token" not in request.url.params:
            return httpx.Response(200, json={"bars": {"AAPL": []}, "next_page_token": "p2"})
        return httpx.Response(503)

    collector = AlpacaCollector(settings, archive, client(archive, failing))
    with pytest.raises(PipelineError, match="HTTP 503"):
        collector.collect("bars", query)
    collector.close()
    resumed = []

    def success(request):
        resumed.append(dict(request.url.params))
        return httpx.Response(200, json={"bars": {"SPY": []}, "next_page_token": None})

    collector = AlpacaCollector(settings, archive, client(archive, success))
    state = collector.collect("bars", query)
    assert resumed[0]["page_token"] == "p2"
    assert len(state["pages"]) == 2 and state["complete"]
    collector.close()


def test_looping_cursor_and_bad_schema_never_mark_complete(tmp_path, settings, query):
    archive = Archive(tmp_path)
    collector = AlpacaCollector(
        settings,
        archive,
        client(
            archive,
            lambda _: httpx.Response(
                200,
                json={"bars": {}, "next_page_token": "same"},
            ),
        ),
    )
    with pytest.raises(PipelineError, match="Repeated pagination"):
        collector.collect("bars", query)
    states = list(tmp_path.glob("checkpoints/extraction/v1/*.json"))
    assert len(states) == 1 and not json.loads(states[0].read_bytes())["complete"]
    with pytest.raises(PipelineError):
        validate_page("bars", {"bars": {}})
    collector.close()


def test_empty_page_with_next_cursor_is_not_terminal(tmp_path, settings, query):
    archive = Archive(tmp_path)
    answers = iter(
        [
            {"bars": None, "next_page_token": "p2"},
            {"bars": {"AAPL": []}, "next_page_token": None},
        ]
    )
    collector = AlpacaCollector(
        settings,
        archive,
        client(
            archive,
            lambda _: httpx.Response(
                200,
                json=next(answers),
            ),
        ),
    )
    assert len(collector.collect("bars", query)["pages"]) == 2
    collector.close()


def test_retry_retains_failed_response_and_never_logs_credentials(tmp_path):
    archive = Archive(tmp_path)
    responses = iter(
        [httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json={})]
    )
    source = client(archive, lambda _: next(responses))
    ref, _ = source.get("https://example.org/data", {"feed": "sip"})
    assert len(list(tmp_path.glob("raw/alpaca/*.gz"))) == 2
    envelope = archive.load_snapshot(ref)
    assert envelope["request"] == {
        "method": "GET",
        "url": "https://example.org/data",
        "params": {"feed": "sip"},
    }
    source.close()
    source = client(archive, lambda _: httpx.Response(401, text="hidden-secret"))
    with pytest.raises(PipelineError) as error:
        source.get("https://example.org/data")
    assert "hidden-secret" not in str(error.value)
    source.close()


def test_storage_failure_cannot_advance_cursor(tmp_path, settings, query, monkeypatch):
    archive = Archive(tmp_path)
    collector = AlpacaCollector(
        settings,
        archive,
        client(
            archive,
            lambda _: httpx.Response(
                200,
                json={"bars": {}, "next_page_token": None},
            ),
        ),
    )
    monkeypatch.setattr(
        archive, "snapshot", lambda _: (_ for _ in ()).throw(PipelineError("S3 failed"))
    )
    with pytest.raises(PipelineError, match="S3 failed"):
        collector.collect("bars", query)
    assert not list(tmp_path.glob("checkpoints/**/*.json"))
    collector.close()


def test_raw_checksum_and_immutable_collision(tmp_path):
    archive = Archive(tmp_path)
    ref = archive.snapshot({"source": "alpaca", "original": "exact body"})
    assert archive.load_snapshot(ref)["original"] == "exact body"
    (tmp_path / ref["key"]).write_bytes(b"corrupt")
    with pytest.raises(PipelineError, match="checksum"):
        archive.load_snapshot(ref)
    archive.write("example/file.bin", b"first")
    with pytest.raises(PipelineError, match="collision"):
        archive.write("example/file.bin", b"second")
    with pytest.raises(PipelineError, match="Unsafe"):
        archive.write("../outside", b"x")


def test_early_close_and_dst_control_ready_session():
    # Black Friday closes at 13:00 New York, not the usual 16:00.
    frame = schedule(date(2024, 11, 29), date(2024, 11, 29))
    assert frame.iloc[0]["close"] == pd.Timestamp("2024-11-29T18:00:00Z")
    assert latest_completed_session(datetime(2024, 11, 29, 18, 29, tzinfo=UTC)) == date(
        2024, 11, 27
    )
    assert latest_completed_session(datetime(2024, 11, 29, 18, 30, tzinfo=UTC)) == date(
        2024, 11, 29
    )
    assert latest_completed_session(datetime(2024, 7, 5, 20, 30, tzinfo=UTC)) == date(2024, 7, 5)
    with pytest.raises(PipelineError, match="close"):
        request_bounds(
            date(2024, 11, 29), date(2024, 11, 29), datetime(2024, 11, 29, 18, tzinfo=UTC)
        )


def test_historical_bounds_follow_new_york_midnight_and_free_sip_delay():
    now = datetime(2024, 7, 5, 20, 30, tzinfo=UTC)
    lower, upper = request_bounds(date(2024, 7, 3), date(2024, 7, 5), now)
    assert lower == "2024-07-03T04:00:00+00:00"
    assert datetime.fromisoformat(upper) <= now - pd.Timedelta(minutes=16)


def observation(raw=None):
    return normalize_bar(
        raw or {"t": "2024-11-29T05:00:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 0},
        "AAPL",
        {"instrument_id": "a", "exchange": "NASDAQ"},
        {"retrieved_at": "2025-01-01T00:00:00+00:00", "request": {"params": {"feed": "sip"}}},
        {"sha256": "hash", "key": "raw/key"},
        {"2024-11-29": {"close": "2024-11-29T18:00:00Z"}},
    )


def test_zero_volume_is_preserved_and_availability_is_not_backdated():
    row = observation()
    assert row["volume_raw"] == 0 and row["quality_status"] == "valid_observation"
    assert row["available_at"] == row["retrieved_at"]
    assert row["action_verification_status"] == "unverified"
    assert row["bar_end_at"] == "2024-11-29T18:00:00Z"
    assert "extended_hours" in row["volume_scope"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"h": 5}, "invalid_ohlc_order"),
        ({"c": float("inf")}, "invalid_price"),
        ({"v": 1.5}, "invalid_volume"),
        ({"v": -1}, "invalid_volume"),
        ({"t": "2024-11-29T05:00:00"}, "invalid_timestamp"),
        ({"t": "2024-11-29T18:00:00Z"}, "unexpected_daily_timestamp"),
        ({"t": "2024-11-30T05:00:00Z"}, "outside_requested_exchange_sessions"),
    ],
)
def test_invalid_values_are_quarantined_without_inventing_prices(change, reason):
    raw = {"t": "2024-11-29T05:00:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}
    row = observation({**raw, **change})
    assert row["quality_status"] == "quarantined"
    assert reason in json.loads(row["quality_reason_codes"])


def test_missing_sessions_duplicates_and_conflicting_revisions_are_distinct():
    first = observation()
    revised = observation(
        {"t": "2024-11-29T05:00:00Z", "o": 10, "h": 12, "l": 9, "c": 10.5, "v": 100}
    )
    universe = {"instruments": [{"symbol": "AAPL", "instrument_id": "a", "kind": "stock"}]}
    report = coverage_report(
        pd.DataFrame([first, first, revised]), universe, ["2024-11-27", "2024-11-29"]
    )
    entry = report["instruments"][0]
    assert entry["missing_sessions"] == ["2024-11-27"]
    assert entry["identical_duplicate_rows"] == 1
    assert entry["conflicting_revision_count"] == 1
    assert entry["valid_unique_sessions"] == 1


def test_directory_formats_and_frozen_universe():
    assert (
        parse_nasdaq(b"Symbol|Security Name\nAAPL|Apple\nFile Creation Time: 20241014||\n")[0][
            "Symbol"
        ]
        == "AAPL"
    )
    sec = parse_sec(
        json_bytes(
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[1, "Apple", "AAPL", "Nasdaq"]],
            }
        )
    )
    assert sec[0]["cik"] == 1
    with pytest.raises(PipelineError):
        parse_nasdaq(b"<html>blocked</html>")
    universe = load_universe(Path("configs/universe.daily.json"))
    assert len(universe["instruments"]) == 51
    assert len({e["instrument_id"] for e in universe["instruments"]}) == 51
    with pytest.raises(PipelineError, match="absent"):
        load_universe(Path("configs/universe.daily.json"), ["UNKNOWN"])


def test_full_extraction_writes_replayable_manifest_and_parquet(tmp_path, settings, monkeypatch):
    archive = Archive(tmp_path)
    universe = load_universe(Path("configs/universe.daily.json"), ["AAPL", "SPY"])

    def collect(self, kind, params, *, refresh=False):
        record = {"t": "2024-11-29T05:00:00Z", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}
        payload = (
            {"bars": {s: [record] for s in ("AAPL", "SPY")}, "next_page_token": None}
            if kind == "bars"
            else {
                "corporate_actions": {
                    "cash_dividends": [
                        {
                            "id": "event1",
                            "symbol": "AAPL",
                            "rate": 0.25,
                            "process_date": "2024-11-29",
                            "ex_date": "2024-11-29",
                        }
                    ]
                },
                "next_page_token": None,
            }
        )
        envelope = {
            "source": "alpaca",
            "retrieved_at": "2025-01-01T00:00:00+00:00",
            "request": {"params": params},
            "response": {"body_base64": base64.b64encode(json_bytes(payload)).decode()},
        }
        ref = archive.snapshot(envelope)
        return {"complete": True, "pages": [{**ref, "records": 2 if kind == "bars" else 1}]}

    monkeypatch.setattr(AlpacaCollector, "collect", collect)
    result = run_extraction(
        settings, archive, universe, date(2024, 11, 29), date(2024, 11, 29), directories=False
    )
    report = result["report"]
    assert report["bar_rows"] == 2 and report["action_rows"] == 1
    assert report["extraction_status"] == "complete" and not report["training_ready"]
    manifest = json.loads(archive.read(result["manifest"]["key"]))
    bars_ref = manifest["outputs"]["bars"]
    assert digest(archive.read(bars_ref["key"])) == bars_ref["sha256"]
    assert len(pd.read_parquet(tmp_path / bars_ref["key"])) == 2
    assert "hidden-secret" not in json.dumps(manifest)
