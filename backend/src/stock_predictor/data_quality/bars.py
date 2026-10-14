import json
import math
from collections import Counter
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from stock_predictor.config import PipelineError
from stock_predictor.storage.archive import digest

NY = ZoneInfo("America/New_York")


def numeric(value: object, *, integer: bool = False) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    if integer:
        return int(value) if value >= 0 and value == int(value) else None
    return float(value) if value > 0 else None


def normalize_bar(
    raw: dict, symbol: str, instrument: dict, envelope: dict, ref: dict, sessions: dict[str, dict]
) -> dict:
    reasons = []
    timestamp, session, bar_end = None, None, None
    try:
        stamp = datetime.fromisoformat(raw["t"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("naive timestamp")
        local = stamp.astimezone(NY)
        timestamp = stamp.astimezone(UTC).isoformat()
        session = local.date().isoformat()
        if any((local.hour, local.minute, local.second, local.microsecond)):
            reasons.append("unexpected_daily_timestamp")
        if session not in sessions:
            reasons.append("outside_requested_exchange_sessions")
        else:
            bar_end = sessions[session]["close"]
    except (KeyError, TypeError, ValueError, AttributeError):
        reasons.append("invalid_timestamp")
    values = {
        name: numeric(raw.get(key))
        for name, key in (
            ("open_raw", "o"),
            ("high_raw", "h"),
            ("low_raw", "l"),
            ("close_raw", "c"),
        )
    }
    volume = numeric(raw.get("v"), integer=True)
    if any(value is None for value in values.values()):
        reasons.append("invalid_price")
    else:
        o, h, low, c = (values[k] for k in ("open_raw", "high_raw", "low_raw", "close_raw"))
        if h < max(o, c) or low > min(o, c) or h < low:
            reasons.append("invalid_ohlc_order")
    if volume is None:
        reasons.append("invalid_volume")
    # A source's day aggregate can contain extended-hours volume. It is not an RTH bar.
    # Historical first-publication availability is deliberately not invented here.
    return {
        "instrument_id": instrument["instrument_id"],
        "symbol": symbol,
        "exchange": instrument["exchange"],
        "currency": "USD",
        "issuer_cik": instrument.get("issuer_cik"),
        "session_date": session,
        "source_timestamp": timestamp,
        "bar_end_at": bar_end,
        "available_at": envelope["retrieved_at"],
        "retrieved_at": envelope["retrieved_at"],
        "availability_method": "observed_retrieval",
        **values,
        "volume_raw": volume,
        "source": "alpaca",
        "feed": envelope["request"]["params"]["feed"],
        "bar_type": "provider_1Day",
        "volume_scope": "provider_daily_includes_extended_hours",
        "adjustment_basis": "raw",
        "adjustment_version": "unadjusted",
        "record_version": digest(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()),
        "response_hash": ref["sha256"],
        "raw_snapshot_key": ref["key"],
        "quality_status": "quarantined" if reasons else "valid_observation",
        "quality_reason_codes": json.dumps(reasons),
        "action_verification_status": "unverified",
        "identity_verification_status": "fixed_symbol_history_unverified",
    }


def bar_rows(
    data: dict, envelope: dict, ref: dict, instruments: dict, sessions: dict
) -> list[dict]:
    rows = []
    for symbol, observations in (data["bars"] or {}).items():
        if symbol not in instruments:
            raise PipelineError("Alpaca returned a symbol outside the frozen universe")
        for raw in observations:
            rows.append(normalize_bar(raw, symbol, instruments[symbol], envelope, ref, sessions))
    return rows


def coverage_report(frame: pd.DataFrame, universe: dict, session_dates: list[str]) -> dict:
    coverage = []
    for instrument in universe["instruments"]:
        symbol = instrument["symbol"]
        rows = frame[frame["symbol"] == symbol] if not frame.empty else frame
        valid = rows[rows["quality_status"] == "valid_observation"] if not rows.empty else rows
        dates = sorted(set(valid["session_date"])) if not valid.empty else []
        missing = sorted(set(session_dates) - set(dates))
        revisions, duplicates = 0, 0
        if not rows.empty:
            grouped = rows.groupby("session_date", dropna=False)["record_version"]
            unique = grouped.nunique()
            duplicates = int((grouped.size() - unique).sum())
            revisions = int((unique - 1).clip(lower=0).sum())
        yearly = Counter(day[:4] for day in dates)
        coverage.append(
            {
                "symbol": symbol,
                "instrument_id": instrument["instrument_id"],
                "kind": instrument["kind"],
                "raw_rows": len(rows),
                "valid_unique_sessions": len(dates),
                "first_session": dates[0] if dates else None,
                "last_session": dates[-1] if dates else None,
                "quarantined_rows": len(rows) - len(valid),
                "identical_duplicate_rows": duplicates,
                "conflicting_revision_count": revisions,
                "expected_sessions": len(session_dates),
                "missing_session_count": len(missing),
                "missing_sessions": missing,
                "sessions_by_year": dict(sorted(yearly.items())),
            }
        )
    reasons = Counter()
    if not frame.empty:
        for codes in frame["quality_reason_codes"]:
            reasons.update(json.loads(codes))
    return {"instruments": coverage, "quarantine_reasons": dict(reasons)}
