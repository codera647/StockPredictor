import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_training as training_tests
from stock_predictor.calendar import schedule
from stock_predictor.config import PipelineError
from stock_predictor.datasets.pipeline import read_table
from stock_predictor.extraction import parquet
from stock_predictor.replay import read_json
from stock_predictor.storage.archive import digest
from stock_predictor.training.bundle import save_json_model
from stock_predictor.training.estimators import fit_linear, linear_payload
from stock_predictor.training.metrics import encode_labels
from stock_predictor_fundamental_news.features import fundamentals, news_features, sentiment
from stock_predictor_fundamental_news.parquet_sources import collect_parquet_news
from stock_predictor_fundamental_news.pipeline import run, validate
from stock_predictor_fundamental_news.sources import PartsReader, clean_news, download


def config():
    return json.loads(Path("configs/fundamental-news.daily.json").read_text())


def evidence():
    facts = []
    for cik in ("1", "2"):
        for concept, value in (
            ("Assets", 100),
            ("Liabilities", 40),
            ("Revenues", 50),
            ("NetIncomeLoss", 5),
        ):
            annual = concept in ("Revenues", "NetIncomeLoss")
            facts.append(
                dict(
                    issuer_cik=cik,
                    concept=concept,
                    value=value,
                    taxonomy="us-gaap",
                    unit="USD",
                    period_start="2017-01-01" if annual else None,
                    period_end="2017-12-31",
                    available_at_reconstructed=pd.Timestamp("2018-01-03", tz="UTC"),
                    accession_number="old",
                    quality_valid=True,
                )
            )
    filings = pd.DataFrame(
        [
            dict(
                issuer_cik=cik,
                form="8-K",
                items="2.02",
                quality_valid=True,
                available_at_reconstructed=pd.Timestamp("2018-01-03 22:00:00", tz="UTC"),
            )
            for cik in ("1", "2")
        ]
    )
    companies = pd.DataFrame({"symbol": ["AAA", "BBB"], "issuer_cik": ["1", "2"]})
    return pd.DataFrame(facts), filings, companies


def index_for(sessions):
    result = sessions[["session_date"]].copy()
    result["row_id"] = result.session_date
    result["symbol"] = "AAA"
    result["cutoff_at"] = pd.to_datetime(sessions.close, utc=True) + pd.Timedelta(minutes=30)
    return result


def test_sec_release_time_restatements_and_matching_periods():
    c = config()
    sessions = schedule(date(2018, 1, 2), date(2018, 1, 10))
    index = index_for(sessions)
    facts, filings, companies = evidence()
    original = fundamentals(facts, filings, companies, index, sessions, c)
    assert original.iloc[0].sec_liabilities_assets_missing == 1
    after = original[original.row_id.eq("2018-01-03")].iloc[0]
    assert after.sec_liabilities_assets == pytest.approx(0.4)
    assert after.sec_annual_net_margin == pytest.approx(0.1)
    assert after.sec_8k_5d_missing == 1  # acceptance was after this forecast cutoff
    assert original[original.row_id.eq("2018-01-04")].iloc[0].sec_earnings_8k_5d == 1
    revision = facts.iloc[[1]].copy()
    revision["value"] = 9999
    revision["available_at_reconstructed"] = pd.Timestamp("2018-02-01", tz="UTC")
    modified = fundamentals(pd.concat([facts, revision]), filings, companies, index, sessions, c)
    pd.testing.assert_frame_equal(original, modified)
    mismatched = facts.copy()
    mismatched.loc[mismatched.concept.eq("NetIncomeLoss"), "period_end"] = "2017-12-30"
    result = fundamentals(mismatched, filings, companies, index, sessions, c)
    assert result.sec_annual_net_margin.isna().all()


def test_news_delay_future_invariance_and_unavailable_periods():
    c = config()
    c.update(news_coverage_start="2018-01-02", news_coverage_end="2018-01-08")
    sessions = schedule(date(2018, 1, 2), date(2018, 1, 12))
    index = index_for(sessions)
    news = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "headline": ["Company beats earnings estimates"],
            "available_at_reconstructed": [pd.Timestamp("2018-01-04 22:00", tz="UTC")],
        }
    )
    before = news_features(news, index, sessions, c)
    assert before[before.row_id.le("2018-01-04")].news_count_1d.isna().all()
    assert before[before.row_id.eq("2018-01-05")].iloc[0].news_count_1d == 1
    assert before[before.row_id.ge("2018-01-09")].news_count_1d.isna().all()
    future = news.assign(available_at_reconstructed=pd.Timestamp("2018-02-01", tz="UTC"))
    after = news_features(pd.concat([news, future]), index, sessions, c)
    pd.testing.assert_frame_equal(before, after)
    assert sentiment("Company beats earnings") > 0
    assert sentiment("Company does not beat earnings") < 0


def test_news_cleaning_keeps_cross_stock_links_and_delays_dates():
    source = pd.DataFrame(
        {
            "Date": ["2020-01-01 00:00:00 UTC"] * 3 + ["broken"],
            "Article_title": ["Company beats estimates"] * 4,
            "Stock_symbol": ["AAA", "AAA", "BBB", "AAA"],
            "Url": [
                "https://example.org/a#x",
                "https://example.org/a",
                "https://example.org/a",
                "https://example.org/b",
            ],
            "Publisher": ["example"] * 4,
        }
    )
    cleaned, invalid = clean_news(source, {"AAA", "BBB"}, config())
    assert len(cleaned) == 2 and invalid == 1
    assert cleaned.available_at_reconstructed.eq(pd.Timestamp("2020-01-02", tz="UTC")).all()


def test_resumable_download_validates_ranges_and_full_digest(tmp_path, monkeypatch):
    payload = b'a,b\n1,"hello\nworld"\n3,4\n'
    c = config()
    c.update(
        news_bytes=len(payload),
        news_sha256=digest(payload),
        news_chunk_bytes=5,
        news_download_workers=2,
    )
    calls = []

    def serve(request):
        calls.append(request.headers["Range"])
        a, b = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        return httpx.Response(
            206,
            content=payload[a : b + 1],
            headers={"Content-Range": f"bytes {a}-{b}/{len(payload)}"},
        )

    client = httpx.Client
    monkeypatch.setattr(
        "stock_predictor_fundamental_news.sources.httpx.Client",
        lambda **kw: client(transport=httpx.MockTransport(serve), **kw),
    )
    paths = download(c, tmp_path)
    reader = PartsReader(paths)
    assert reader.read(5) + reader.read() == payload
    reader.close()
    reader.validate(c)
    expected_calls = (len(payload) + 4) // 5
    assert download(c, tmp_path) == paths and len(calls) == expected_calls
    csv_reader = PartsReader(paths)
    frame = pd.read_csv(csv_reader)
    assert frame.iloc[0].b == "hello\nworld" and len(frame) == 2
    csv_reader.close()
    csv_reader.validate(c)
    paths[0].write_bytes(b"x" * 5)
    assert download(c, tmp_path) == paths and len(calls) == expected_calls + 1
    with pytest.raises(PipelineError, match="SHA-256"):
        reader.validate({**c, "news_sha256": "incorrect"})


def test_final_test_flag_cannot_be_enabled():
    c = config()
    c["evaluate_existing_test"] = True
    with pytest.raises(PipelineError):
        validate(c)


def test_parquet_projection_skips_unmatched_groups_and_reuses_verified_ranges(
    tmp_path, monkeypatch
):
    from stock_predictor.storage.archive import Archive

    rows = pd.DataFrame(
        {
            "Date": ["2020-01-01 00:00:00 UTC"] * 6,
            "Article_title": ["Company beats earnings " + str(i) for i in range(6)],
            "Stock_symbol": ["AAA", "AAA", "BBB", "BBB", "", ""],
            "Url": [f"https://example.org/{i}" for i in range(6)],
            "Publisher": ["example"] * 6,
            "Article": [str(i) * 80000 for i in range(6)],
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pandas(rows), sink, row_group_size=2, use_dictionary=False, compression=None
    )
    payload = sink.getvalue().to_pybytes()
    spec = {
        "repo": "example/fnspid",
        "revision": "pinned",
        "files": [
            {
                "path": "data/source.parquet",
                "size": len(payload),
                "lfs_sha256": digest(payload),
            }
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    c = config()
    c.update(
        news_mirror_repo=spec["repo"],
        news_mirror_revision=spec["revision"],
        news_parquet_spec=str(spec_path),
    )
    calls = []

    def serve(request):
        a, b = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        calls.append((a, b))
        return httpx.Response(
            206,
            content=payload[a : b + 1],
            headers={"Content-Range": f"bytes {a}-{b}/{len(payload)}"},
        )

    client = httpx.Client
    monkeypatch.setattr(
        "stock_predictor_fundamental_news.parquet_sources.httpx.Client",
        lambda **kw: client(transport=httpx.MockTransport(serve), **kw),
    )
    archive = Archive(tmp_path / "archive")
    result = collect_parquet_news(archive, {"AAA"}, c, tmp_path / "download")
    data = read_table(archive, result["table"])
    assert len(data) == 2 and data.symbol.eq("AAA").all()
    metadata = read_json(archive, result["manifest"])
    assert metadata["shards"][0]["selected_row_groups"] == 1
    assert metadata["downloaded_bytes"] < len(payload) / 2
    assert metadata["full_original_file_sha256_verified"] is False
    before = len(calls)
    assert collect_parquet_news(archive, {"AAA"}, c, tmp_path / "download") == result
    assert len(calls) == before


@pytest.fixture
def research_dataset(tmp_path):
    return training_tests.training_dataset.__wrapped__(tmp_path)


def test_end_to_end_ablation_artifacts_cache_and_test_exclusion(research_dataset, monkeypatch):
    archive, manifest, old_config, root, blocks, outputs = research_dataset
    source = read_json(archive, manifest)
    facts, filings, companies = evidence()
    for name, table in (
        ("clean_sec_facts", facts),
        ("clean_sec_filings", filings),
        ("clean_sec_companies", companies),
    ):
        source["outputs"][name] = parquet(archive, "fixture/" + name, table)
    enriched = archive.json("manifests/sec-source.json", source)
    logistic = fit_linear(
        blocks["train"]["X"], encode_labels(blocks["train"]["y"].y_direction_5d), "logistic", 0.1
    )
    baseline = archive.json(
        "manifests/sec-baseline.json",
        {
            "bundle": {
                "dataset_version": "synthetic",
                "models": {
                    "logistic": save_json_model(
                        archive,
                        "fixture/price",
                        linear_payload(logistic, "logistic", source["recipe"]["feature_columns"]),
                    )
                },
            }
        },
    )
    news = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "headline": ["Company beats earnings", "Company misses earnings"],
            "available_at_reconstructed": [pd.Timestamp("2018-01-04", tz="UTC")] * 2,
        }
    )
    news_ref = {"table": parquet(archive, "fixture/news", news)}
    news_ref["manifest"] = archive.json("manifests/news-source.json", {"table": news_ref["table"]})
    c = config()
    c.update(
        max_rounds=3,
        early_stopping_rounds=1,
        threads=1,
        logistic_cs=[0.1],
        walk_forward=old_config["walk_forward"],
    )
    forbidden = {outputs[f"{kind}_test"]["key"] for kind in ("X", "y", "index")}
    original_read = archive.read

    def guarded(key, **kwargs):
        assert key not in forbidden, "Existing test must remain closed"
        return original_read(key, **kwargs)

    monkeypatch.setattr(archive, "read", guarded)
    before = blocks["validation"]["y"].copy()
    result = run(archive, enriched, baseline, news_ref, c, root / "fundamental-news")
    assert result["feature_counts"] == {
        "price_only": 34,
        "price_sec": 62,
        "price_news": 52,
        "price_sec_news": 80,
    }
    assert not result["existing_test_evaluated"] and not result["production_approved"]
    pd.testing.assert_frame_equal(before, blocks["validation"]["y"])
    report = read_json(archive, result["report"])
    for ref in report["artifacts"].values():
        assert archive.read(ref["key"], sha256=ref["sha256"])
    assert report["artifact_reloads_verified"]
    monkeypatch.setattr(
        "stock_predictor_fundamental_news.pipeline.fit",
        lambda *args: pytest.fail("Cached experiment must not refit"),
    )
    assert run(archive, enriched, baseline, news_ref, c, root / "cached")["cache_reused"]
    features = read_table(archive, report["artifacts"]["sec_news_features"])
    assert np.isfinite(features.sec_liabilities_assets.dropna()).all()
