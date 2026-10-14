import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stock_predictor.calendar import schedule
from stock_predictor.config import PipelineError
from stock_predictor.datasets.auxiliary import clean_evidence
from stock_predictor.datasets.cleaning import clean_actions, clean_bars
from stock_predictor.datasets.features import FEATURE_COLUMNS, build_features, wilder_rsi
from stock_predictor.datasets.labels import build_labels
from stock_predictor.datasets.pipeline import run_preparation, validate_config
from stock_predictor.datasets.splits import make_splits
from stock_predictor.extraction import parquet
from stock_predictor.storage.archive import Archive


@pytest.fixture
def fixture():
    sessions = schedule(date(2016, 1, 4), date(2018, 12, 31)).iloc[:700].copy()
    config = json.loads(Path("configs/dataset.daily.json").read_text())
    config.update(
        validation_start=sessions.session_date.iloc[400],
        calibration_start=sessions.session_date.iloc[500],
        test_start=sessions.session_date.iloc[600],
    )
    rows, references = [], []
    for symbol in ("AAA", "BBB", "SPY"):
        for i, day in enumerate(sessions.session_date):
            rate = {"AAA": 0.003, "BBB": 0.0005, "SPY": 0.001}[symbol]
            c = 100 * np.exp(rate * i + 0.01 * np.sin(i / 3))
            row = dict(
                symbol=symbol,
                instrument_id=symbol,
                exchange="XNYS",
                currency="USD",
                issuer_cik="1",
                session_date=day,
                source="alpaca",
                feed="sip",
                bar_type="1Day",
                adjustment_basis="raw",
                volume_raw=1_000_000 + 1000 * i,
                record_version=f"{symbol}-{i}",
                quality_status="valid_observation",
                quality_reason_codes="[]",
                retrieved_at="2026-10-14T15:00:00Z",
                available_at="2026-10-14T15:00:00Z",
                raw_snapshot_key="raw/fixture.json.gz",
                volume_scope="provider_daily_including_extended_hours",
            )
            row.update(open_raw=c * 0.998, high_raw=c * 1.01, low_raw=c * 0.99, close_raw=c)
            rows.append(row)
            references.append(
                dict(
                    symbol=symbol,
                    session_date=day,
                    o_split_reference=row["open_raw"],
                    h_split_reference=row["high_raw"],
                    l_split_reference=row["low_raw"],
                    c_split_reference=c,
                    v_split_reference=row["volume_raw"],
                )
            )
    actions = pd.DataFrame(
        columns=[
            "event_id",
            "record_version",
            "payload_json",
            "action_type",
            "symbols_json",
            "process_date",
            "retrieved_at",
            "raw_snapshot_key",
        ]
    )
    return pd.DataFrame(rows), actions, pd.DataFrame(references), sessions, config


def prepare(fixture):
    bars, actions, reference, sessions, config = fixture
    normalized, _ = clean_actions(actions, set(bars.symbol))
    panel, report = clean_bars(bars, normalized, reference, sessions, config)
    features = build_features(panel, config)
    labels = build_labels(panel, features, config)
    return panel, features, labels, report


def test_same_date_split_and_purged_label_boundaries(fixture):
    _, features, labels, _ = prepare(fixture)
    outputs, assignments, report = make_splits(features, labels, fixture[-1])
    assert assignments.groupby("session_date").split.nunique().eq(1).all()
    assert len(FEATURE_COLUMNS) == 34
    for name, next_start in zip(
        ("train", "validation", "calibration"),
        (fixture[-1][k] for k in ("validation_start", "calibration_start", "test_start")),
        strict=True,
    ):
        assert outputs[name]["index"].target_session.lt(next_start).all()
        assert report[name]["purged_rows"] == 10  # Five sessions, each with two stocks.
    for tables in outputs.values():
        assert list(tables["X"]) == FEATURE_COLUMNS
        assert not any(name.startswith("y_") for name in tables["X"])
        assert len(tables["X"]) == len(tables["y"]) == len(tables["index"])
    assert labels.label_quality_status.eq("immature").sum() == 10
    assert features[
        features.session_date.isin(fixture[3].session_date.iloc[-5:])
    ].feature_ready.all()


def test_appended_future_data_cannot_change_earlier_features(fixture):
    panel, full, _, _ = prepare(fixture)
    prefix = build_features(panel[panel.session_index.lt(450)], fixture[-1])
    original = full[full.session_index.lt(450)].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix[FEATURE_COLUMNS], original[FEATURE_COLUMNS])


def test_today_volume_is_unavailable_and_must_not_enter_today_features(fixture):
    panel, original, _, _ = prepare(fixture)
    today = panel.session_index.eq(699)
    panel.loc[today, ["volume_raw", "volume_basis"]] *= 1000
    modified = build_features(panel, fixture[-1])
    pd.testing.assert_frame_equal(original[FEATURE_COLUMNS], modified[FEATURE_COLUMNS])


def test_real_session_gap_blocks_windows_and_future_labels(fixture):
    bars, actions, reference, sessions, config = fixture
    missing_day = sessions.session_date.iloc[400]
    bars = bars[~(bars.symbol.eq("AAA") & bars.session_date.eq(missing_day))]
    panel, features, labels, report = prepare((bars, actions, reference, sessions, config))
    gap = panel[panel.symbol.eq("AAA") & panel.session_date.eq(missing_day)].iloc[0]
    assert gap.missing_bar and not gap.clean_valid and pd.isna(gap.close_raw)
    assert report["missing_sessions"] == 1
    affected = features[features.symbol.eq("AAA") & features.session_index.between(400, 651)]
    assert not affected.feature_ready.any()
    assert features[features.symbol.eq("AAA") & features.session_index.eq(652)].feature_ready.all()
    assert not labels[
        labels.symbol.eq("AAA") & labels.session_date.isin(sessions.session_date.iloc[395:401])
    ].label_ready.any()
    assert features[features.symbol.eq("BBB") & features.session_index.eq(400)].feature_ready.all()


def test_split_unit_change_does_not_create_a_false_return(fixture):
    bars, actions, reference, sessions, config = fixture
    symbol_mask = bars.symbol.eq("AAA")
    day = sessions.session_date.iloc[450]
    after = symbol_mask & bars.session_date.ge(day)
    prices = ["open_raw", "high_raw", "low_raw", "close_raw"]
    bars.loc[after, prices] /= 2
    bars.loc[after, "volume_raw"] *= 2
    reference.loc[
        reference.symbol.eq("AAA"),
        ["o_split_reference", "h_split_reference", "l_split_reference", "c_split_reference"],
    ] /= 2
    actions = pd.DataFrame(
        [
            dict(
                event_id="split",
                record_version="v1",
                action_type="forward_splits",
                payload_json=json.dumps(dict(symbol="AAA", new_rate=2, old_rate=1, ex_date=day)),
                symbols_json='["AAA"]',
                process_date=day,
                retrieved_at="2026-10-14T15:00:00Z",
                raw_snapshot_key="raw/action.json.gz",
            )
        ]
    )
    _, f, y, report = prepare((bars, actions, reference, sessions, config))
    assert report["unresolved_split_reference_rows"] == 0
    at_split = f[f.symbol.eq("AAA") & f.session_date.eq(day)].iloc[0]
    # Return depends on price motion, not the two-for-one unit change.
    assert abs(at_split.ret_1d) < 0.01
    row = y[y.symbol.eq("AAA") & y.session_date.eq(sessions.session_date.iloc[448])].iloc[0]
    theoretical = 0.003 * 5 + 0.01 * (np.sin(453 / 3) - np.sin(448 / 3))
    assert row.y_price_5d == pytest.approx(theoretical)
    assert at_split.feature_ready


def test_missing_action_cannot_silently_assume_identity_split_factor(fixture):
    bars, actions, reference, sessions, config = fixture
    before = reference.symbol.eq("AAA") & reference.session_date.lt(sessions.session_date.iloc[450])
    reference.loc[
        before, ["o_split_reference", "h_split_reference", "l_split_reference", "c_split_reference"]
    ] /= 2
    panel, f, _, report = prepare((bars, actions, reference, sessions, config))
    assert report["unresolved_split_reference_rows"] == 450
    assert not panel[panel.symbol.eq("AAA") & panel.session_index.lt(450)].clean_valid.any()
    assert not f[f.symbol.eq("AAA")].feature_ready.any()


def test_conflicting_bar_revision_is_quarantined(fixture):
    bars, actions, reference, sessions, config = fixture
    conflict = bars.iloc[[400]].copy()
    conflict["record_version"] = "revised"
    conflict["close_raw"] *= 1.001
    panel, _, _, report = prepare(
        (pd.concat([bars, conflict], ignore_index=True), actions, reference, sessions, config)
    )
    assert report["conflicting_session_keys"] == 1
    assert not panel[panel.symbol.eq("AAA") & panel.session_index.eq(400)].clean_valid.any()


@pytest.mark.parametrize(
    ("values", "expected"),
    [(np.arange(30) + 1, 100), (np.arange(30)[::-1] + 1, 0), (np.ones(30), 50)],
)
def test_rsi_edge_cases(values, expected):
    assert wilder_rsi(pd.Series(values)).iloc[-1] == expected


def test_macro_vintage_preserves_revisions_and_open_end():
    source = pd.DataFrame(
        [
            dict(
                series_id="X",
                observation_date="2020-01-01",
                value=1,
                realtime_start="2020-02-01",
                realtime_end="2020-03-01",
            ),
            dict(
                series_id="X",
                observation_date="2020-01-01",
                value=2,
                realtime_start="2020-03-02",
                realtime_end="9999-12-31",
            ),
        ]
    )
    cleaned, _ = clean_evidence("macro_observations", source)
    assert len(cleaned) == 2 and cleaned.quality_valid.all()
    assert cleaned.realtime_end.iloc[-1] == "9999-12-31"
    assert cleaned.available_at_reconstructed.iloc[0] == pd.Timestamp("2020-02-02", tz="UTC")


def test_news_uses_first_observed_time_not_a_backdated_publication():
    source = pd.DataFrame(
        [
            dict(
                symbol="AAA",
                url="https://example.com/story#top",
                publication_at="2020-01-01T00:00:00Z",
                first_observed_at="2026-10-14T00:00:00Z",
            )
        ]
    )
    cleaned, _ = clean_evidence("company_announcements", source)
    assert cleaned.available_at_observed.iloc[0].year == 2026
    assert cleaned.canonical_url.iloc[0] == "https://example.com/story"


def test_sec_conflicting_fact_values_are_retained_and_quarantined():
    base = dict(
        issuer_cik="1",
        taxonomy="us-gaap",
        concept="Revenue",
        unit="USD",
        period_start="2020-01-01",
        period_end="2020-03-31",
        filed_date="2020-05-01",
        accession_number="acc",
    )
    cleaned, report = clean_evidence(
        "sec_facts", pd.DataFrame([{**base, "value": 1}, {**base, "value": 2}])
    )
    assert len(cleaned) == 2 and not cleaned.quality_valid.any()
    assert report["conflicting_value_rows"] == 2


def test_reconstructed_data_cannot_claim_production_readiness(fixture):
    config = {**fixture[-1], "production_ready": True}
    with pytest.raises(PipelineError, match="production-ready"):
        validate_config(config)


def test_end_to_end_archive_artifacts_are_separate_and_checksummed(fixture, tmp_path, monkeypatch):
    bars, actions, reference, sessions, config = fixture
    archive = Archive(tmp_path / "archive")
    inputs = {
        name: parquet(archive, name, frame)
        for name, frame in dict(bars=bars, actions=actions, sessions=sessions).items()
    }
    report = archive.json(
        "reports/source.json",
        dict(
            outputs=inputs,
            symbols=["AAA", "BBB", "SPY"],
            start=sessions.session_date.iloc[0],
            end=sessions.session_date.iloc[-1],
            issues=[],
        ),
    )
    manifest = archive.json("manifests/source.json", dict(report=report))
    monkeypatch.setattr(
        "stock_predictor.datasets.pipeline.fetch_split_reference",
        lambda *args: {"table": parquet(archive, "reference", reference)},
    )
    result = run_preparation(None, archive, manifest, config, tmp_path / "summary")
    assert result["research_training_ready"] and not result["production_ready"]
    recorded = json.loads((tmp_path / "summary/report.json").read_text())
    for ref in recorded["outputs"].values():
        assert archive.read(ref["key"], sha256=ref["sha256"]) is not None
    assert recorded["outputs"]["inference_features"]["rows"] == 10
